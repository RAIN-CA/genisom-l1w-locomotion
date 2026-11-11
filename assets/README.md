# Robot assets: GENISOM 钢镚 L1-W (ZSL-1W)

| Path | Content | Origin |
|---|---|---|
| `usd/zsl1w/` | `zsl1w.usd` + `configuration/` layers, the model used for training (`genisom_l1w/robot.py`) | converted from `urdf/zsl1w_sim.urdf` |
| `urdf/zsl1w_sim.urdf` | simulation URDF | derived from `urdf/ZSL-1W.urdf` |
| `urdf/ZSL-1W.urdf` | vendor URDF, unmodified | vendor model package, `zsl-1w/urdf/` |
| `meshes/*.STL` | vendor meshes referenced by the two URDFs, unmodified | vendor model package, `zsl-1w/meshes/` |
| `LICENSE` | vendor license (BSD-3-Clause) | vendor model package |

## Simulation URDF

`zsl1w_sim.urdf` differs from the vendor URDF in two points only; masses, inertias, joint limits and the other
collision shapes are the vendor's:

- the four wheel joints (`*_FOOT_JOINT`, revolute with ±999999 rad limits) are `continuous`;
- each wheel collider is a cylinder (radius 0.0704 m, length 0.048 m, offset 0.0475 m along the axle; measured
  from the wheel meshes) instead of the wheel mesh.

In this copy the mesh paths point to `../meshes/`.

## USD conversion

The USD was produced on 2026-10-02 with the Isaac Lab 2.3 URDF converter (`scripts/tools/convert_urdf.py`) and
loads unchanged in Isaac Sim 6.1. Settings (recorded in `usd/zsl1w/config.yaml`, whose absolute paths refer to the
machine it was converted on):

| Setting | Value |
|---|---|
| fixed base | no |
| merge fixed joints | no |
| joint drive | force, position target, stiffness 25, damping 0.5 (replaced by the actuator models in `genisom_l1w/robot.py`) |
| mesh colliders | convex hull (only `BASE_LINK` still uses its mesh) |
| self-collision | off in the file, enabled when the robot is spawned |
| instanceable | yes |

Set `GENISOM_L1W_USD=/path/to/robot.usd` to train with a different conversion.

## License

The vendor files and the files derived from them (`zsl1w_sim.urdf`, `usd/zsl1w/`) are Copyright (c) 2025, ZsiBot,
and are redistributed under the BSD 3-Clause License in [`LICENSE`](LICENSE).
