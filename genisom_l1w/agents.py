"""rsl_rl PPO runner configs (rsl-rl-lib 5.x, Isaac Lab 3.0 layout).

Actor [512, 256, 128] ELU on the 570-dim history, critic [512, 256, 128] ELU on the 60-dim privileged state,
scalar-std Gaussian (init 1.0), no observation normalization (the deployment runtime feeds raw observations).
train.sh passes the same PPO hyper-parameters on the command line, so they can be changed without editing this file.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class StepFirstPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "genisom_l1w_v2"
    run_name = "stage1_stepfirst"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RollingPPORunnerCfg(StepFirstPPORunnerCfg):
    run_name = "stage2_rolling"


@configclass
class StablePPORunnerCfg(StepFirstPPORunnerCfg):
    max_iterations = 2500
    run_name = "stage1c_stand"
