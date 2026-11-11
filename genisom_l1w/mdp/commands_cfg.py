"""Configuration of the evasive velocity command.

Only the config lives here; the runtime class is referenced lazily (``"{DIR}.commands:EvasiveVelocityCommand"``,
resolved by the command manager when the env is built), the same split Isaac Lab uses for its own commands.
"""

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .commands import EvasiveVelocityCommand


@configclass
class EvasiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`EvasiveVelocityCommand`.

    ``ranges.lin_vel_x`` / ``ranges.lin_vel_y`` are ignored (kept for API compatibility); the discrete
    ``*_choices`` are used instead. ``ranges.ang_vel_z`` is still the continuous yaw-rate range.
    ``resampling_time_range`` is not used for holding times; they come from the mixture below.
    """

    class_type: type["EvasiveVelocityCommand"] | str = "{DIR}.commands:EvasiveVelocityCommand"

    lin_vel_x_choices: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    """Discrete forward-velocity set (m/s). HiSAF adapter uses {-2, 0, 2}; the intermediate values help interpolation."""
    lin_vel_y_choices: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    """Discrete lateral-velocity set (m/s). HiSAF adapter uses {-0.8, 0, 0.8}."""
    rel_zero_yaw: float = 0.5
    """Probability that the yaw-rate command is exactly zero."""

    short_hold_prob: float = 0.3
    """Probability that a command is held only for ``short_hold_range`` (rapid toggling)."""
    short_hold_range: tuple[float, float] = (0.1, 0.4)
    """Short hold duration range (s)."""
    long_hold_range: tuple[float, float] = (0.8, 3.0)
    """Long hold duration range (s)."""

    emergency_prob: float = 0.3
    """Probability that a new command is an emergency pattern w.r.t. the previous one (stop / reversal / lateral jump)."""
    transition_window: float = 0.5
    """Seconds after a switch that count as the transition window (used by metrics and the transition reward)."""
    curriculum_steps: int = 0
    """If > 0, ``short_hold_prob`` and ``emergency_prob`` ramp linearly from 0 to their value over this many env steps
    (``env.common_step_counter``: one count per ``env.step()``, i.e. 24 per rsl_rl iteration), and the discrete
    linear-velocity magnitudes ramp from ``speed_curriculum_start`` x to 1 x."""
    speed_curriculum_start: float = 1.0
    """Fraction of the full discrete vx/vy magnitudes at step 0 when ``curriculum_steps`` > 0 (1.0 = no speed ramp)."""
