"""MDP terms of the GENISOM L1-W tasks plus the stock Isaac Lab velocity mdp.

Lazy namespace, exactly like ``isaaclab_tasks.core.velocity.mdp``: ``lazy_export()`` reads ``__init__.pyi``, imports a
local name on first access and resolves every other name (``base_ang_vel``, ``JointPositionActionCfg``,
``randomize_rigid_body_mass``, ...) from the stock velocity mdp. Nothing is imported eagerly, which keeps the task
configs safe to build before Kit starts.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
