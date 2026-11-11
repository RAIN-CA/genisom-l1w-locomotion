"""Articulation config of the GENISOM 钢镚 L1-W wheeled-legged robot (12 leg joints + 4 wheels).

Physical model: ``assets/usd/zsl1w/zsl1w.usd``, converted with the Isaac Lab URDF importer from
``assets/urdf/zsl1w_sim.urdf`` (the vendor URDF with continuous wheel joints and cylinder wheel colliders, see
``assets/README.md``). Masses / inertias / limits are the vendor's; total mass 20.56 kg, wheel radius 0.0704 m.

Control interface (the deployment runtime must use the same gains):
  * legs: explicit DC-motor model, kp 20 / kd 0.7, 28 N*m peak torque, 28 rad/s; position targets.
  * wheels: velocity-controlled implicit drive, kp 0 / kd 0.7 (= velocity gain), 28 N*m, 28 rad/s.
  * default posture q_default: ABAD 0, HIP 0.8, KNEE -1.5 on all four legs.

Set ``GENISOM_L1W_USD`` to load the robot from another USD file.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.environ.get("GENISOM_L1W_USD", os.path.join(_REPO_ROOT, "assets", "usd", "zsl1w", "zsl1w.usd"))

GENISOM_L1W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # self-collision ON: without it the legs pass through each other and policies learn a crossed-leg gait
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={".*_ABAD_JOINT": 0.0, ".*_HIP_JOINT": 0.8, ".*_KNEE_JOINT": -1.5, ".*_FOOT_JOINT": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            actuator_effort_limit=28.0,
            saturation_effort=28.0,
            actuator_velocity_limit=28.0,
            stiffness=20.0,
            damping=0.7,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT_JOINT"],
            joint_effort_limit=28.0,
            joint_velocity_limit=28.0,  # vendor URDF value; wheel speed saturates near 2 m/s (28 * 0.0704)
            stiffness=0.0,
            damping=0.7,
        ),
    },
)
"""GENISOM L1-W: position-controlled legs, velocity-controlled wheels."""
