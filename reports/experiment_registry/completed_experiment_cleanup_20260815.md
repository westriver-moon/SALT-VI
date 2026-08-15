# Completed Experiment Archive and Cleanup — 2026-08-15

## Scope

This pass archived completed SALT-VI experiments on the selected `lab929-3090`
profile while preserving the active `r0_switch20_triangle` route. The 4090
profile was not selected, so its host-local runtime output was not modified.

## Newly archived routes

| Route | Rank-1 | mAP | mINP | Checkpoint retention |
|---|---:|---:|---:|---|
| `r1_switch24_triangle` | 0.8361293674 | 0.7901898390 | 0.6634736902 | Removed; below group best |
| `r2_switch28_triangle` | 0.8346042633 | 0.7892631383 | 0.6628064145 | Removed; below group best |
| `r3_switch24_fullpairs` | 0.8375493288 | 0.7927363722 | 0.6682530138 | Retained as group best |

The retained r3 checkpoint is:

`checkpoints/stage_b/experiments/stage_b_rn50_two_stage_grid_40_20260815/r3_switch24_fullpairs/best/model_Fusion_epoch_6.pth`

SHA-256: `a7f2713e374066227af32cc6ad51837aed7245fdeaa1f25d66de22ed0903e0e0`

For all three routes, metrics, route state, exact commands, resolved phase
configurations, raw event streams, train logs and TensorBoard records remain
available. Training/resume checkpoints and non-selected result weights were
removed after route-status, process-use and selected-checkpoint hash checks.

## Storage cleanup

- Permanently removed non-retained route weights: 7,058,712,805 bytes.
- Moved the retained r3 result weight into the canonical checkpoint tree.
- Moved raw phase logs and TensorBoard events out of run directories into
  `logs/raw/experiments/stage_b_rn50_two_stage_grid_40_20260815/`.
- Removed stale lock, PID and empty launcher files for completed routes.
- Preserved every file under the active `r0_switch20_triangle` route.

## Dead code and redundant configuration cleanup

- Archived the completed RN50 direct Stage-B one-off entrypoint under its
  experiment `source/` directory; the active `scripts/training/` copy was removed.
- Moved the completed LLM-Aug configuration into
  `configs/experiments/reproduction/archived_configs/`.
- Moved the historical `sysu_textv1_fgap2_p1_seed0.yaml` configuration out of
  the active experiment-config root and into the archived-config area.
- Removed the redundant `configs/experiments/templates/a3_e4_hpt_l025_template.yaml`;
  the canonical source-core template remains at
  `configs/stage_b/reproduction/source_core/a3_e4_hpt_l025_template.yaml`.
- Kept the RN50 Stage-B base config and two-stage runner because the active r0
  process still depends on them.

## Registry changes

The unified experiment registry now contains authoritative records for r1, r2
and r3, including same-epoch metrics, lifecycle, logs, checkpoint disposition
and provenance. Existing records were updated for the archived LLM-Aug config
and the RN50 direct experiment's archived entrypoint/dependency lifecycle.

## Deferred items

- `r0_switch20_triangle`: still running on GPU 0 and intentionally untouched.
- 4090-local TVI-LFM original Stage-B runtime output: not modified because the
  selected remote profile remained `lab929-3090`.
