"""Evasive velocity command: discrete, abruptly switching SE(2) commands with emergency patterns.

Why: the reactive avoidance layer on top of this controller emits step-changing body-velocity commands
(vx in {-2, 0, 2} m/s, vy in {-0.8, 0, 0.8} m/s, yaw rate from a heading P-controller). A policy trained with Isaac Lab's
UniformVelocityCommand (one sample held for 10 s) never sees a 2 -> -2 m/s reversal or a 0 -> 0.8 m/s lateral jump.
This term samples from the discrete set, holds commands for a mixture of short (0.1-0.4 s) and long (0.8-3 s)
durations, and injects emergency patterns relative to the previous command (full stop, reversal, lateral jump).
An optional linear curriculum ramps the aggressiveness (``curriculum_steps``; it counts env steps of the CURRENT env,
so every new env -- e.g. an evaluation run -- must set it explicitly).

Command layout is the parent's: ``(vx, vy, wz)`` in the base frame. Extra state for reward terms: ``hold_time``,
``prev_command_b`` and ``elapsed`` (seconds since the last switch).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from .utils import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import EvasiveVelocityCommandCfg


class EvasiveVelocityCommand(UniformVelocityCommand):
    """Discrete, abruptly switching SE(2) velocity commands with emergency patterns.

    Command layout is unchanged from the parent: ``(vx, vy, wz)`` in the base frame, so every
    observation / reward term written for ``UniformVelocityCommand`` keeps working.
    Extra state exposed for reward terms:

    * ``hold_time``   -- duration (s) of the current command, sampled at the last switch
    * ``prev_command_b`` -- the command that was active before the last switch
    * ``elapsed``     -- seconds since the last switch (property)
    """

    cfg: EvasiveVelocityCommandCfg

    def __init__(self, cfg: EvasiveVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._vx_choices = torch.tensor(cfg.lin_vel_x_choices, device=self.device, dtype=torch.float32)
        self._vy_choices = torch.tensor(cfg.lin_vel_y_choices, device=self.device, dtype=torch.float32)
        self.hold_time = torch.full((self.num_envs,), cfg.long_hold_range[1], device=self.device)
        self.prev_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        # metrics: how many switches per episode and the tracking error inside the transition window
        self.metrics["num_switches"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_xy_transition"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "EvasiveVelocityCommand:\n"
        msg += f"\tvx choices: {self.cfg.lin_vel_x_choices}\n"
        msg += f"\tvy choices: {self.cfg.lin_vel_y_choices}\n"
        msg += f"\twz range: {self.cfg.ranges.ang_vel_z} (P(wz=0)={self.cfg.rel_zero_yaw})\n"
        msg += f"\thold: short {self.cfg.short_hold_range} s w.p. {self.cfg.short_hold_prob}, long {self.cfg.long_hold_range} s\n"
        msg += f"\temergency prob: {self.cfg.emergency_prob} (stop / reverse / lateral jump)\n"
        msg += f"\tcurriculum steps: {self.cfg.curriculum_steps} (speed starts at {self.cfg.speed_curriculum_start:.2f}x)\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    # ------------------------------------------------------------------ properties
    @property
    def elapsed(self) -> torch.Tensor:
        """Seconds since the current command was issued. Shape (num_envs,)."""
        return (self.hold_time - self.time_left).clamp_(min=0.0)

    @property
    def curriculum_fraction(self) -> float:
        if self.cfg.curriculum_steps <= 0:
            return 1.0
        step = float(getattr(self._env, "common_step_counter", 0))
        return min(1.0, step / float(self.cfg.curriculum_steps))

    # ------------------------------------------------------------------ overrides
    def _update_metrics(self):
        super()._update_metrics()
        in_window = (self.elapsed < self.cfg.transition_window).float()
        err = torch.norm(self.vel_command_b[:, :2] - as_torch(self.robot.data.root_lin_vel_b)[:, :2], dim=-1)
        max_command_step = self.cfg.long_hold_range[1] / self._env.step_dt
        self.metrics["error_vel_xy_transition"] += in_window * err / max_command_step

    def _resample(self, env_ids: Sequence[int]):
        """Hold-time mixture instead of the parent's single uniform ``resampling_time_range``."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        if n == 0:
            return
        frac = self.curriculum_fraction
        short_prob = self.cfg.short_hold_prob * frac
        is_short = torch.rand(n, device=self.device) < short_prob
        t_long = torch.empty(n, device=self.device).uniform_(*self.cfg.long_hold_range)
        t_short = torch.empty(n, device=self.device).uniform_(*self.cfg.short_hold_range)
        hold = torch.where(is_short, t_short, t_long)
        self.time_left[env_ids] = hold
        self.hold_time[env_ids] = hold
        self.command_counter[env_ids] += 1
        self.metrics["num_switches"][env_ids] += 1.0
        self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        prev = self.vel_command_b[env_ids].clone()
        self.prev_command_b[env_ids] = prev
        # -- curriculum: speed scale ramps speed_curriculum_start -> 1 over curriculum_steps (1.0 = no speed ramp)
        frac = self.curriculum_fraction
        speed = self.cfg.speed_curriculum_start + (1.0 - self.cfg.speed_curriculum_start) * frac
        # -- independent draw from the discrete sets
        vx = self._vx_choices[torch.randint(len(self._vx_choices), (n,), device=self.device)] * speed
        vy = self._vy_choices[torch.randint(len(self._vy_choices), (n,), device=self.device)] * speed
        # -- emergency patterns relative to the previous command
        is_emerg = torch.rand(n, device=self.device) < self.cfg.emergency_prob * frac
        pattern = torch.randint(0, 3, (n,), device=self.device)
        stop = is_emerg & (pattern == 0)
        reverse = is_emerg & (pattern == 1)
        lateral = is_emerg & (pattern == 2)
        vy_max = float(self._vy_choices.abs().max()) * speed
        lat_sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        vx = torch.where(stop, torch.zeros_like(vx), vx)
        vy = torch.where(stop, torch.zeros_like(vy), vy)
        vx = torch.where(reverse, -prev[:, 0], vx)
        vy = torch.where(reverse, prev[:, 1], vy)
        vx = torch.where(lateral, prev[:, 0], vx)
        vy = torch.where(lateral, lat_sign * vy_max, vy)
        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 1] = vy
        # -- yaw rate: continuous (goal-heading P-controller on the adapter side), often zero
        wz = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.ang_vel_z)
        zero_wz = torch.rand(n, device=self.device) < self.cfg.rel_zero_yaw
        self.vel_command_b[env_ids, 2] = torch.where(zero_wz, torch.zeros_like(wz), wz)
        # -- heading / standing envs exactly like the parent
        r = torch.empty(n, device=self.device)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
