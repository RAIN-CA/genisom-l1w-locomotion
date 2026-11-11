#!/usr/bin/env python3
"""Train -- or play / export -- the GENISOM L1-W locomotion policy with Isaac Lab 3.0 and rsl_rl.

    python train.py --task Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-v0 --num_envs 4096 --max_iterations 5000
    python train.py --play --task Isaac-Velocity-Flat-Genisom-L1W-V2-StepFirst-v0 \
        --checkpoint checkpoints/v2.2_stage1c/model_9997.pt --num_envs 16 --video --video_length 500

This file only registers this repository's tasks and then hands over to the official Isaac Lab 3.0 entry points
(``isaaclab_rl.entrypoints.run_train_cli`` / ``run_play_cli``), so every official option works unchanged:
``--checkpoint`` (warm start), ``--video``, ``--device``, ``--distributed``, ``physics=...``, ``env.<field>=...``,
``agent.<field>=...``. ``train.sh`` holds the full training command with the hyper-parameters.

``--play`` loads a checkpoint, runs the policy with the play-mode overrides of the task, and exports TorchScript /
ONNX. The official play writes ``exported/`` and ``videos/`` next to the checkpoint; a checkpoint from this
repository's ``checkpoints/`` is therefore copied to ``./logs/play/<name>/`` first, so the tracked files stay as
they are and the outputs land in the current directory.
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
sys.path.insert(0, ROOT)

import warp as wp  # noqa: E402

wp.config.enable_backward = False  # as in the official scripts: no autodiff kernels, faster cold start

# Register Isaac Lab's own tasks first and this repository's tasks second, so ours are the ones in the gym registry
# even if an older copy of them was installed into the Isaac Lab checkout (gymnasium warns and keeps the later one).
import isaaclab_tasks  # noqa: E402,F401
import genisom_l1w  # noqa: E402,F401
from isaaclab_rl.entrypoints import run_play_cli, run_train_cli  # noqa: E402


def _play_copy_of_repo_checkpoint(argv: list[str]) -> list[str]:
    """Replace a --checkpoint from ``checkpoints/`` by a copy under ./logs/play/<parent dir>/."""
    for i, arg in enumerate(argv):
        if arg == "--checkpoint" and i + 1 < len(argv):
            idx, path = i + 1, argv[i + 1]
        elif arg.startswith("--checkpoint="):
            idx, path = i, arg.split("=", 1)[1]
        else:
            continue
        src = os.path.abspath(path)
        if os.path.isfile(src) and os.path.commonpath([CHECKPOINTS, src]) == CHECKPOINTS:
            dst_dir = os.path.join(os.getcwd(), "logs", "play", os.path.basename(os.path.dirname(src)))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            argv[idx] = dst if idx == i + 1 else f"--checkpoint={dst}"
            print(f"[train.py] playing a copy of {src}\n           -> {dst} (exported/ and videos/ are written there)")
        break
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    play = "--play" in argv
    if play:
        argv.remove("--play")
        argv = _play_copy_of_repo_checkpoint(argv)
    if not any(a == "--rl_library" or a.startswith("--rl_library=") for a in argv):
        argv = ["--rl_library", "rsl_rl"] + argv
    return run_play_cli(argv) if play else run_train_cli(argv)


if __name__ == "__main__":
    # as the official train.py: under torchrun, report the failing rank's traceback (no-op otherwise)
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
