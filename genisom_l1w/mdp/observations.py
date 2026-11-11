"""Observation terms of the GENISOM L1-W tasks.

``joint_pos_rel_without_wheel`` is adapted from robot_lab (https://github.com/fan-ziqi/robot_lab,
Copyright (c) 2024-2026 Ziqi Fan, Apache-2.0). Change: the wheel columns are located by joint identity, so the
term stays correct whatever public joint order the physics backend exposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .utils import as_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

_WHEEL_COLS: dict[tuple, list[int]] = {}


def _ids_list(ids, n: int) -> list[int]:
    if isinstance(ids, slice):
        return list(range(n))[ids]
    return [int(i) for i in ids]


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions relative to the default posture (``asset_cfg`` order), wheel columns set to zero.

    A continuous wheel angle carries no information for the policy, so those columns are always 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = as_torch(asset.data.joint_pos)
    default_joint_pos = as_torch(asset.data.default_joint_pos)
    n_asset = joint_pos.shape[1]
    sel = _ids_list(asset_cfg.joint_ids, n_asset)
    key = (tuple(sel), tuple(_ids_list(wheel_asset_cfg.joint_ids, n_asset)))
    cols = _WHEEL_COLS.get(key)
    if cols is None:
        wheel = set(key[1])
        cols = [k for k, j in enumerate(sel) if j in wheel]
        _WHEEL_COLS[key] = cols
        print(f"[genisom_l1w] asset joint order: {list(getattr(asset, 'joint_names', []))}", flush=True)
        print(f"[genisom_l1w] joint_pos_rel_without_wheel: asset ids {sel} -> wheel columns {cols}", flush=True)
    joint_pos_rel = joint_pos[:, sel] - default_joint_pos[:, sel]
    joint_pos_rel[:, cols] = 0
    return joint_pos_rel
