"""Velocity-tracking tasks of the GENISOM 钢镚 L1-W (Isaac Lab 3.0, manager-based, PhysX).

Interface (the deployment runtime reproduces exactly this):
  * policy obs  = 57 x 10 = 570: six terms per frame, 10-frame history flattened PER TERM (term-major, oldest first);
                  no base linear velocity -- the history is the velocity estimator
  * critic obs  = 60, single frame, noise-free, + ground-truth base linear velocity (asymmetric actor-critic)
  * actions     = 16 = 12 leg position targets + 4 wheel velocity targets; joint order fixed by preserve_order
  * control     = 50 Hz policy (PhysX 200 Hz, decimation 4), 20 s episodes

Stages (task IDs registered in ``genisom_l1w/__init__.py``):
  1. StepFirstEnvCfg -- wheels braked: the robot must learn to STEP in every direction first (a free-wheel policy
     from scratch rolls longitudinally and ignores lateral commands -- a rolling-only local optimum).
  2. RollingEnvCfg   -- wheels free, warm-started from stage 1.
  *  StableEnvCfg    -- stage-1 fine-tune: attitude first (tilt priced, earlier termination), livelier gait,
                        stronger pushes, and the standing repair (stand_still / joint_pos_penalty) -- model v2.2.

Isaac Lab 3.0 notes: the velocity base defaults to Newton, so PhysX is pinned below (every checkpoint was trained on
PhysX); the base's add_base_mass became multiplicative, so the additive -1..+4 kg randomization is written out.
"""

import copy
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import TerminationsCfg as BaseTerminationsCfg

from . import mdp
from .robot import GENISOM_L1W_CFG

# Joint order of the obs / action blocks. Legs grouped per leg (abad, hip, knee) in the SDK leg order
# FR (FAR), FL (FBL), BR (RAR), BL (RBL); the four wheels are appended at positions 12-15.
# fmt: off
LEG_JOINT_NAMES = [
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
]
WHEEL_JOINT_NAMES = ["FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT"]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
# fmt: on

HISTORY_LENGTH = 10  # 0.2 s at 50 Hz


##
# MDP
##


@configclass
class CommandsCfg:
    """Discrete, abruptly switching commands from the avoidance layer's action set."""

    base_velocity = mdp.EvasiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.8, 3.0),  # only used by the parent's bookkeeping; holds come from the mixture
        rel_standing_envs=0.10,
        heading_command=False,
        debug_vis=True,
        lin_vel_x_choices=(-2.0, -1.0, 0.0, 1.0, 2.0),
        lin_vel_y_choices=(-0.8, -0.4, 0.0, 0.4, 0.8),
        rel_zero_yaw=0.5,
        short_hold_prob=0.3,
        short_hold_range=(0.1, 0.4),
        long_hold_range=(0.8, 3.0),
        emergency_prob=0.3,
        transition_window=0.5,
        curriculum_steps=0,  # StepFirst sets 24_000; any evaluation must keep 0
        speed_curriculum_start=1.0,
        ranges=mdp.EvasiveVelocityCommandCfg.Ranges(lin_vel_x=(-2.0, 2.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.5, 1.5)),
    )


@configclass
class ActionsCfg:
    """12 leg position targets (scale: ABAD 0.125, hip/knee 0.25, offset q_default) + 4 wheel velocity targets."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={".*_ABAD_JOINT": 0.125, "^(?!.*_ABAD_JOINT).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,  # v_des = 5 * a (rad/s); wheel drive kp 0 / kd 0.7
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Policy: 57 per frame x 10 frames. Critic: 60, single frame, privileged."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Term order defines the deployment layout -- do not reorder."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH  # every term keeps 10 frames, flattened per term, oldest first
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged, single frame, noise-free: + ground-truth base linear velocity."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Rolling-stage reward table (the StepFirst and Stable stages modify it below)."""

    # -- velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- evasive response: fast convergence right after a command switch
    track_lin_vel_xy_transition = RewTerm(
        func=mdp.track_lin_vel_xy_transition_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 1.0, "window": 0.5},
    )
    track_lin_vel_xy_convergence = RewTerm(
        func=mdp.track_lin_vel_xy_convergence, weight=0.05, params={"command_name": "base_velocity", "max_rate": 8.0}
    )
    lin_vel_xy_error_l1 = RewTerm(func=mdp.lin_vel_xy_error_l1, weight=-0.25, params={"command_name": "base_velocity"})
    ang_vel_z_error_l1 = RewTerm(func=mdp.ang_vel_z_error_l1, weight=-0.1, params={"command_name": "base_velocity"})
    # -- base
    upward = RewTerm(func=mdp.upward, weight=1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # -- joints
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-9, params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)}
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    joint_power = RewTerm(
        func=mdp.joint_power, weight=-2e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
    )
    # -- standing: posture anchored to q_default whenever ||cmd|| < 0.1
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK")},
    )
    # -- diagonal-pair symmetry (kept weak so lateral side-steps are not suppressed)
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FAR_(ABAD|HIP|KNEE).*", "RBL_(ABAD|HIP|KNEE).*"],
                ["FBL_(ABAD|HIP|KNEE).*", "RAR_(ABAD|HIP|KNEE).*"],
            ],
        },
    )
    # -- actions: reactions must be allowed to be abrupt
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # -- contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_FOOT_LINK).*"), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"), "threshold": 100.0},
    )


@configclass
class TerminationsCfg(BaseTerminationsCfg):
    """time_out + base_contact (stock) + bad_orientation: self-righting is not trained, so a fall ends the episode."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


##
# Environments
##


@configclass
class RollingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Stage 2 (and the base of every stage): flat ground, wheels free, full command aggressiveness."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # physics: PhysX (the 3.0 velocity base defaults to Newton)
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        # robot
        self.scene.robot = GENISOM_L1W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # flat ground, no height scan, no terrain curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "BASE_LINK"
        # domain randomization (the lab unit carries extra compute and a 360-degree LiDAR on its back)
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.9)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 4.0)
        self.events.add_base_mass.params["operation"] = "add"  # kg, additive (3.0 base default: "scale")
        self.events.add_base_mass.params["distribution"] = "uniform"  # (3.0 base default: "log_uniform")
        self.events.add_base_mass.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "BASE_LINK"
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="BASE_LINK"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (0.0, 0.08)},
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )
        # reset: tame (exact q_default, small tilt); falls terminate instead of training self-righting
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.05), "roll": (-0.1, 0.1),
                           "pitch": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.3, 0.3),
                               "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5)},
        }

    def play_mode(self):
        """Applied by the official play entry point: no pushes / noise, full command range, 50 envs by default."""
        super().play_mode()  # removes pushes and the external wrench
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.curriculum_steps = 0


@configclass
class StepFirstRewardsCfg(RewardsCfg):
    """Rolling table + a stepping incentive + quiet (disconnected) wheel channels."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_LINK"),
            "threshold": 0.5,
        },
    )
    wheel_action_l2 = RewTerm(func=mdp.action_term_l2, weight=-0.01, params={"term_name": "joint_vel"})


@configclass
class StepFirstEnvCfg(RollingEnvCfg):
    """Stage 1: wheels braked, the robot learns to step in every direction first."""

    rewards: StepFirstRewardsCfg = StepFirstRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # brake the wheels: velocity target always 0, firm damping (0.7 would still free-wheel under contact)
        self.actions.joint_vel.scale = 0.0
        robot_cfg = copy.deepcopy(self.scene.robot)  # never mutate the shared GENISOM_L1W_CFG
        robot_cfg.actuators["wheels"].damping = 5.0
        self.scene.robot = robot_cfg
        # bootstrap curriculum: command speed 0.25x -> 1x, short holds / emergencies 0 -> full over 24k env steps
        # (24 env steps per PPO iteration -> the first ~1000 iterations)
        self.commands.base_velocity.curriculum_steps = 24_000
        self.commands.base_velocity.speed_curriculum_start = 0.25


@configclass
class StableRewardsCfg(StepFirstRewardsCfg):
    """StepFirst table + an explicit price on tilt (20 degrees ~ -0.29 per step)."""

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)


@configclass
class StableEnvCfg(StepFirstEnvCfg):
    """Stage-1 fine-tune (v2.1 attitude + gait, v2.2 standing repair). Warm start from a stage-1 checkpoint."""

    rewards: StableRewardsCfg = StableRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # attitude first: damp pitch/roll rates, end near-rollover states early (0.7 rad = 40 degrees)
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.terminations.bad_orientation.params["limit_angle"] = 0.7
        # stop paying for slamming the brakes (the exposure to emergency commands is unchanged)
        self.rewards.track_lin_vel_xy_transition.weight = 0.5
        self.rewards.lin_vel_xy_error_l1.weight = -0.15
        # livelier gait
        self.rewards.feet_air_time.weight = 0.4
        self.rewards.joint_mirror.weight = -0.01
        # recovery practice in more postures
        self.events.push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        # warm start: no bootstrap ramp
        self.commands.base_velocity.curriculum_steps = 0
        self.commands.base_velocity.speed_curriculum_start = 1.0
        # v2.2 standing repair (both gated on ||cmd|| < 0.1)
        self.rewards.stand_still.weight = -3.5
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 8.0
