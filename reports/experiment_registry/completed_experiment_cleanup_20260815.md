# Completed Experiment Archive and Cleanup — 2026-08-15

## Scope

This pass archived completed SALT-VI experiments on the selected `lab929-3090`
profile. The final `r0_switch20_triangle` route was stopped by the user and
archived after its best result failed to exceed the retained group best. The
4090 profile was not selected, so its host-local runtime output was not modified.

## Newly archived routes

| Route | Rank-1 | mAP | mINP | Checkpoint retention |
|---|---:|---:|---:|---|
| `r0_switch20_triangle` | 0.8362607956 | 0.7902120977 | 0.6633535114 | Removed; stopped and below group best |
| `r1_switch24_triangle` | 0.8361293674 | 0.7901898390 | 0.6634736902 | Removed; below group best |
| `r2_switch28_triangle` | 0.8346042633 | 0.7892631383 | 0.6628064145 | Removed; below group best |
| `r3_switch24_fullpairs` | 0.8375493288 | 0.7927363722 | 0.6682530138 | Retained as group best |

The retained r3 checkpoint is:

`checkpoints/stage_b/experiments/stage_b_rn50_two_stage_grid_40_20260815/r3_switch24_fullpairs/best/model_Fusion_epoch_6.pth`

SHA-256: `a7f2713e374066227af32cc6ad51837aed7245fdeaa1f25d66de22ed0903e0e0`

For all three routes, metrics, route state, exact commands, resolved phase
configurations, raw event streams, train logs and TensorBoard records remain
available. Training/resume checkpoints and non-selected result weights were removed after
route-status and process-use checks; retained comparison checkpoints use their
already recorded hashes.

## Storage cleanup

- Permanently removed non-retained route weights: 9,530,908,029 bytes.
- Moved the retained r3 result weight into the canonical checkpoint tree.
- Moved raw phase logs and TensorBoard events out of run directories into
  `logs/raw/experiments/stage_b_rn50_two_stage_grid_40_20260815/`.
- Removed stale lock, PID and empty launcher files for completed routes.
- Archived r0 metrics, commands, resolved configs and logs; removed its four non-retained weights (2,472,195,224 bytes).

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
- Archived the completed RN50 Stage-B base config, the full two-stage grid
  configuration set and the grid runner; no active experiment depends on them.

## Registry changes

The unified experiment registry now contains authoritative records for all four
grid routes, including same-epoch metrics, lifecycle, logs, checkpoint disposition
and provenance. Existing records were updated to the immutable archived config paths.

## Deferred items

- 4090-local TVI-LFM original Stage-B runtime output was not modified because
  the selected remote profile remained `lab929-3090`.
