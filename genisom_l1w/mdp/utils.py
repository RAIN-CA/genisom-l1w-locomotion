"""Small helpers shared by the MDP terms."""

from __future__ import annotations

import torch


def as_torch(x) -> torch.Tensor:
    """Torch view of an Isaac Lab 3.0 data buffer.

    Asset / sensor ``.data.*`` fields and ``ContactSensor.compute_first_contact()`` return proxy arrays whose torch
    view is ``.torch`` (the official idiom). Plain tensors pass through; a bare ``warp.array`` is wrapped zero-copy.
    """
    if isinstance(x, torch.Tensor):
        return x
    t = getattr(x, "torch", None)
    if isinstance(t, torch.Tensor):
        return t
    import warp as wp

    return wp.to_torch(x)
