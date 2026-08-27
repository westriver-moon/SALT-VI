# SALT-VI-pact-data-20260827 worktree retirement audit

Date: 2026-08-27

## Decision

The experimental worktree `/home/lab929/ybj/SALT-VI-pact-data-20260827` is eligible for removal after this synchronization commit is retained. Its reusable implementation has been migrated to `/home/lab929/ybj/SALT-VI-plugin-mainline-20260827`, while its exact historical state and experiment payload remain independently archived.

## Migration map

| Old worktree content | Canonical destination |
|---|---|
| Final-image geometry, person-fit, YOLO11x pose and shared cache | `person_preprocessing/` |
| PACT anatomy and token-mask derivation | `plugins/pact/` |
| Qwen consumption of shared person assets | `plugins/qwen_imagination/` |
| SALT training/evaluation bridge for prepared images | `src/salt_vi/person_assets.py` and dataset/loader integration |
| C3 ellipse attention regularizer | `src/salt_vi/models/vision_adapter.py`, `vision_transformer.py`, and `training/recipes.py` |
| Layer 2 / Layer 4 ellipse configurations and test | `configs/stage_a/safe_tricks/` and `src/salt_vi/tests/test_ellipse_attention.py` |
| Experiment report and registry records | `docs/history/stage_a/`, `docs/plans/`, and `reports/experiment_registry/experiment_registry.csv` |

The old `src/salt_vi/data/prepared/` prototype and its early Qwen adapters are intentionally not copied verbatim: commit `a596f4c1` replaced them with the one-way `person_preprocessing -> Qwen/PACT -> SALT` dependency design. Prototype-only configs, sample reports and validation artifacts remain available in the immutable source snapshot below.

## Independent preservation

- Exact dirty source snapshot: `/home/lab929/ybj/experiments/archive/_shared/source/SALT-VI-worktree-20260827-pact-ellipse`
- Snapshot source digest: `7b8219451d450095648c53f24e53fab4bef8232563592f79be701c71d5b61efb`
- Completed experiment archive: `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-c3-ellipse-20260827-gpu12`
- Final audit: `/home/lab929/ybj/experiments/archive/stage_a/c3_ellipse_audit_20260827.json` (`ok=true`)
- The original experiment run tree is retained separately at `/home/lab929/ybj/experiments/c3-ellipse-20260827-gpu12`.

## Verification

- Targeted ellipse/person-assets/PACT/Qwen boundary tests: 9 passed.
- Full local repository test selection: 179 passed.
- Experiment archive audit: 20/20 payload files matched, 504 archived inventory entries verified, and the shared source snapshot inventory verified.
- The pluginized mainline was clean before synchronization; all synchronized changes are recorded in one follow-up commit.

Removing the old worktree must not be treated as permission to delete its Git branch, source snapshot, experiment archive, or original run tree.
