"""GENISOM 钢镚 L1-W wheeled-legged locomotion for Isaac Lab 3.0.

Importing this package registers the Gym task IDs below. Only configuration code is imported (the MDP namespace is
lazy), so this is safe before the simulator starts -- the Isaac Lab 3.0 entry points build the task config first and
launch Kit afterwards.
"""

import gymnasium as gym

# task id -> (env cfg class in env_cfg.py, runner cfg class in agents.py)
TASKS = {
    "Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-v0": ("StepFirstEnvCfg", "StepFirstPPORunnerCfg"),
    "Isaac-Velocity-Flat-Genisom-L1W-V2-Rolling-v0": ("RollingEnvCfg", "RollingPPORunnerCfg"),
    "Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-Stable-v0": ("StableEnvCfg", "StablePPORunnerCfg"),
}

for _task, (_env_cfg, _runner_cfg) in TASKS.items():
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:{_runner_cfg}",
            "default_agent": "rsl_rl",
        },
    )
