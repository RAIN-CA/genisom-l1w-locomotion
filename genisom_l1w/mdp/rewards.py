"""Reward terms of the GENISOM L1-W tasks.

Two groups:
  * evasive-response shaping (ours): fast convergence after an abrupt command switch;
  * standing / posture terms adapted from robot_lab (https://github.com/fan-ziqi/robot_lab,
    Copyright (c) 2024-2026 Ziqi Fan, Apache-2.0): "zero velocity" has infinitely many postures, and only a
    posture-level anchor makes the policy's standing equal q_default. Each of them is gated by uprightness
    (clamp(-g_z, 0, 0.7) / 0.7), so a fallen robot is not also punished for its posture while getting up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .utils import as_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


# ---------------------------------------------------------------------------------------------- evasive response


def track_lin_vel_xy_transition_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    window: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extra xy tracking reward, active only during the first ``window`` seconds after a command switch.

    Pays for converging quickly to the new command (reversal, stop, lateral jump). Needs the command term to expose
    ``elapsed`` (:class:`EvasiveVelocityCommand`); returns zero otherwise.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    elapsed = getattr(env.command_manager.get_term(command_name), "elapsed", None)
    if elapsed is None:
        return torch.zeros(env.num_envs, device=env.device)
    active = (elapsed < window).float()
    err = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - as_torch(asset.data.root_lin_vel_b)[:, :2]),
        dim=1,
    )
    return active * torch.exp(-err / std**2)


class track_lin_vel_xy_convergence(ManagerTermBase):
    """Rate at which the xy tracking error shrinks: (|e_{t-1}| - |e_t|) / dt, clipped to +-max_rate (m/s^2).

    Informative at any error magnitude (the exp kernel is flat right after a 2 -> -2 m/s reversal). The step on which
    the command itself jumps and the first step after a reset are masked out.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_err = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.valid[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        max_rate: float = 8.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        err = torch.norm(
            env.command_manager.get_command(command_name)[:, :2] - as_torch(asset.data.root_lin_vel_b)[:, :2], dim=1
        )
        rate = ((self.prev_err - err) / env.step_dt).clamp(-max_rate, max_rate)
        active = self.valid.clone()
        elapsed = getattr(env.command_manager.get_term(command_name), "elapsed", None)
        if elapsed is not None:
            active &= elapsed > 1.0e-6  # the command switched on this very step: the error jump is not the policy's
        reward = torch.where(active, rate, torch.zeros_like(rate))
        self.prev_err = err.detach().clone()
        self.valid[:] = True
        return reward


def lin_vel_xy_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Dense |v_cmd - v| (m/s); keeps a gradient where the exp kernel is flat. Use with a negative weight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.norm(
        env.command_manager.get_command(command_name)[:, :2] - as_torch(asset.data.root_lin_vel_b)[:, :2], dim=1
    )


def ang_vel_z_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Dense |wz_cmd - wz| (rad/s). Use with a negative weight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(env.command_manager.get_command(command_name)[:, 2] - as_torch(asset.data.root_ang_vel_b)[:, 2])


def action_term_l2(env: ManagerBasedRLEnv, term_name: str = "joint_vel") -> torch.Tensor:
    """L2 of the raw actions of one action term. Keeps the disconnected wheel channels near zero in the StepFirst
    stage (wheel scale 0), so the warm start into the rolling stage begins with ~0 wheel velocity targets."""
    return torch.sum(torch.square(as_torch(env.action_manager.get_term(term_name).raw_actions)), dim=1)


# ------------------------------------------------------------------------------------ standing / posture (robot_lab)


def _upright_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-as_torch(env.scene["robot"].data.projected_gravity_b)[:, 2], 0, 0.7) / 0.7


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical joint power sum(|qdot * tau|). Use with a negative weight."""
    asset: Articulation = env.scene[asset_cfg.name]
    qd = as_torch(asset.data.joint_vel)[:, asset_cfg.joint_ids]
    tau = as_torch(asset.data.applied_torque)[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qd * tau), dim=1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Offset from the default joint positions while the command is ~zero. Use with a negative weight."""
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return reward * _upright_gate(env)


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """||q - q_default||, multiplied by ``stand_still_scale`` while standing (small command and small speed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(as_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
    running = torch.linalg.norm(
        as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids] - as_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids],
        dim=1,
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold), running, stand_still_scale * running
    )
    return reward * _upright_gate(env)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """Asymmetry between mirrored joint groups (diagonal leg pairs, trot-like symmetry)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if getattr(env, "joint_mirror_joints_cache", None) is None:
        # find_joints returns (ids, names) sorted by asset index; abad < hip < knee holds inside every leg,
        # so the two groups line up element-wise in any asset joint order
        env.joint_mirror_joints_cache = [[asset.find_joints(name) for name in pair] for pair in mirror_joints]
    joint_pos = as_torch(asset.data.joint_pos)
    reward = torch.zeros(env.num_envs, device=env.device)
    for pair in env.joint_mirror_joints_cache:
        reward += torch.sum(torch.square(joint_pos[:, pair[0][0]] - joint_pos[:, pair[1][0]]), dim=-1)
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward * _upright_gate(env)


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot touch-downs while the command is ~zero (plants the feet when standing)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = as_torch(contact_sensor.compute_first_contact(env.step_dt))[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    return reward * _upright_gate(env)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(1 - g_z)^2: ~4 upright, ~1 on the side. Acts as a strong alive bonus."""
    asset = env.scene[asset_cfg.name]
    return torch.square(1 - as_torch(asset.data.projected_gravity_b)[:, 2])
