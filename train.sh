#!/usr/bin/env bash
# Train the GENISOM L1-W locomotion policy: train.py + the hyper-parameters of the v2 recipe (and play / export).
#
#   bash train.sh                                     # stage 1 (StepFirst) from scratch: 4096 envs, 5000 iterations
#   NUM_ENVS=2048 MAX_ITERATIONS=3000 bash train.sh   # every variable below can be overridden from the environment
#   TASK=Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-Stable-v0 MAX_ITERATIONS=2500 \
#     CHECKPOINT=checkpoints/v2.0_stage1/model_4999.pt bash train.sh       # fine-tune from a checkpoint
#   bash train.sh --video env.rewards.feet_air_time.weight=0.3             # extra arguments go to train.py as they are
#   bash train.sh --play --num_envs 16 --video --video_length 500          # play + export CHECKPOINT (default: v2.2)
#   DRY_RUN=1 bash train.sh                           # print the command instead of running it
#
# Isaac Lab 3.0 runs headless unless a visualizer is requested (e.g. --viz kit).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAY=0
for arg in "$@"; do if [ "$arg" = "--play" ]; then PLAY=1; fi; done

# ---- task ------------------------------------------------------------------------------------------------------------
TASK=${TASK:-Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-v0}  # ...-StepFirst-v0 | ...-Rolling-v0 | ...-StepFirst-Stable-v0
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-5000}
SEED=${SEED:-20261014}
# training: warm start from an rsl-rl 5.x model_*.pt (empty = from scratch); --play: the policy to play (default v2.2)
CHECKPOINT=${CHECKPOINT:-}
if [ "$PLAY" = 1 ] && [ -z "$CHECKPOINT" ]; then CHECKPOINT=$REPO/checkpoints/v2.2_stage1c/model_9997.pt; fi

# ---- PPO (rsl-rl 5.x): the values every released checkpoint was trained with ------------------------------------------
NUM_STEPS_PER_ENV=${NUM_STEPS_PER_ENV:-24}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
LEARNING_RATE=${LEARNING_RATE:-1.0e-3}
SCHEDULE=${SCHEDULE:-adaptive}
DESIRED_KL=${DESIRED_KL:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.01}
CLIP_PARAM=${CLIP_PARAM:-0.2}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-5}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-4}
GAMMA=${GAMMA:-0.99}
LAM=${LAM:-0.95}
VALUE_LOSS_COEF=${VALUE_LOSS_COEF:-1.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---- environment -----------------------------------------------------------------------------------------------------
# Isaac Lab 3.0 checkout whose uv venv (.venv) holds Isaac Sim 6.1; without one the active python environment is used.
ISAACLAB=${ISAACLAB:-$REPO/IsaacLab}
# rsl_rl writes <LOG_ROOT>/logs/rsl_rl/genisom_l1w_v2/<date>_<run>/ (checkpoints, tensorboard, params/); --play with a
# released checkpoint writes <LOG_ROOT>/logs/play/<name>/. Default: this repository (logs/ is git-ignored and may be a
# symlink to a data disk).
LOG_ROOT=${LOG_ROOT:-$REPO}

if [ -n "$CHECKPOINT" ]; then CHECKPOINT=$(realpath "$CHECKPOINT"); fi  # resolve before changing directory

if [ "$PLAY" = 1 ]; then
  # play mode of the task (no pushes, no observation noise, 50 envs unless --num_envs) + TorchScript / ONNX export;
  # the training hyper-parameters do not apply
  ARGS=(--task "$TASK")
  case " $* " in *" --checkpoint "* | *" --checkpoint="*) ;; *) ARGS+=(--checkpoint "$CHECKPOINT") ;; esac
else
  ARGS=(--task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" --seed "$SEED")
  if [ -n "$CHECKPOINT" ]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
  ARGS+=(
    agent.num_steps_per_env="$NUM_STEPS_PER_ENV"
    agent.save_interval="$SAVE_INTERVAL"
    agent.algorithm.learning_rate="$LEARNING_RATE"
    agent.algorithm.schedule="$SCHEDULE"
    agent.algorithm.desired_kl="$DESIRED_KL"
    agent.algorithm.entropy_coef="$ENTROPY_COEF"
    agent.algorithm.clip_param="$CLIP_PARAM"
    agent.algorithm.num_learning_epochs="$NUM_LEARNING_EPOCHS"
    agent.algorithm.num_mini_batches="$NUM_MINI_BATCHES"
    agent.algorithm.gamma="$GAMMA"
    agent.algorithm.lam="$LAM"
    agent.algorithm.value_loss_coef="$VALUE_LOSS_COEF"
    agent.algorithm.max_grad_norm="$MAX_GRAD_NORM"
  )
fi
ARGS+=("$@")

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "cd $LOG_ROOT"
  printf 'python %q' "$REPO/train.py"
  printf ' %q' "${ARGS[@]}"
  echo
  exit 0
fi

if [ -f "$ISAACLAB/.venv/bin/activate" ]; then
  # a uv venv must not inherit a conda environment (PYTHONPATH / CONDA_* break its imports)
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL
  set +u
  # shellcheck disable=SC1091
  source "$ISAACLAB/.venv/bin/activate"
  set -u
fi
# running Isaac Sim means accepting the NVIDIA Omniverse license (EULA); these skip the interactive prompt
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}

mkdir -p "$LOG_ROOT"
cd "$LOG_ROOT"
exec python "$REPO/train.py" "${ARGS[@]}"
