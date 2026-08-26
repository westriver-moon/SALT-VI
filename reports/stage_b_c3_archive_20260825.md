# Stage-B C3 Archive Record — 2026-08-25

## Scope

This record covers the completed C3 Stage-B runs that were launched from the
Stage-A C3 camera-diverse + cosine-classifier warm start.  The source code and
training configurations are pinned to the main-worktree commit
`1dceb9a7080b41bb9861878db851aa10c0e2915e`.

All archives use the fixed TVILFM-compatible SYSU-MM01 protocol:

- all-search, single-shot, 10 gallery trials;
- infrared-image query plus visible identity-caption query;
- visible-image gallery;
- identity-text retrieval backend;
- no reranking and no test-time augmentation.

The original run roots are retained.  Each archive contains the complete
original payload, source/configuration manifests, environment information,
results, logs, checkpoints, and hash verification.

## Archive index

| Experiment | Archive | Result | Best checkpoint |
|---|---|---|---|
| C3 aligned baseline | `/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-c3-stageb-aligned-20260823` | Rank-1 83.4262%, mAP 80.7980%, mINP 71.0409%, epoch 27 | `model_output/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_parameter_add_id,cross_modal_hard_Fix_Visual/models/model_Fusion_epoch_27.pth` |
| C3 safe-tricks grid | `/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-c3-stageb-safe-tricks-20260823` | Best variant b5: Rank-1 84.2966%, mAP 81.6982%, mINP 71.9119%, epoch 6 | `b5_hard_loss_ramp/model_output/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_parameter_add_id,cross_modal_hard_Fix_Visual/models/model_Fusion_epoch_6.pth` |
| C3 B5 tricks grid | `/home/lab929/ybj/experiments/archive/stage_b/SALT-VI-c3-b5-tricks-grid-20260824` | Best variant ema_0999: Rank-1 85.1354%, mAP 82.4802%, mINP 72.9672%, epoch 6 | `ema_0999/model_output/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_parameter_add_id,cross_modal_hard_Fix_Visual/models/model_Fusion_epoch_6.pth` |

The six B5 grid variants all completed: `camera_only`, `cosine_only`,
`label_smoothing_010`, `ema_0999`, `rgb_fusion_025`, and `rgb_fusion_050`.
The safe-tricks record explicitly preserves `b2=failed` and `b3=skipped/
queued`; that grid is closed and is not planned for resumption.  The two
failed launch diagnostics are retained under
`provenance/launch_diagnostics/`.

## Reproduction materials

- Exact repository source snapshot:
  `/home/lab929/ybj/experiments/archive/_shared/source/SALT-VI-worktree-20260825-1dceb9a7`
- Snapshot manifest:
  `.../snapshot_manifest.json`
- Snapshot source inventory:
  `.../source_inventory.sha256`
- Git archive and repository bundle are under the snapshot's
  `git_archive/` and `repository.bundle`.
- Per-run source/configuration manifests are under each archive's
  `provenance/source_files_manifest.json`.
- Per-run result and archive manifests are under
  `provenance/final_metrics.json` and `provenance/archive_manifest.json`.
- Per-run payload verification is recorded in
  `provenance/verification.json`; all three runs have `ok=true` and zero
  missing, extra, or hash-mismatched payload files.

The B5 grid source is also tracked in the main worktree at:

- `configs/pipelines/sysu_stage_b_c3_b5_tricks_grid_20260824.yaml`
- `configs/stage_b/b5_tricks_grid_20260824/`
- `scripts/experiments/run_stage_b_c3_b5_tricks_grid.py`

The exact shared source snapshot and the per-run manifests are authoritative
for replaying the historical runs after the experiment worktree is removed.
