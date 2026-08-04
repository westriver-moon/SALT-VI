# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj_cleanup_20260720.md`  
> Original SHA-256: `07f12ec3c122c6e94c141c94e859a34046fc9f70e1ea1a464f082922e0b595a7`  
> This is read-only experiment evidence, not an active runtime instruction.

# ybj storage cleanup record — 2026-07-20

Scope: `/home/cgv841/ybj` only. Current `PMSR-A1`, `PMSR-A2`, and `PMSR-A3`
training outputs and processes were excluded from cleanup.

## Preserved research assets

- All metrics, configurations, logs, TensorBoard events, reports, plots, and result manifests.
- The selected Stage A sampling/mining checkpoint
  `h5_pk8x4_auto_replace_crossmodal_hard/model_IR_23.pth`.
- The selected Stage A MBPatch checkpoint
  `stage_a_vision_text_encoder_recipe_288x144_768_mbpatch_run1/model_IR_27.pth`.
- Every Stage B checkpoint.
- Every checkpoint in `experiments/tvilfm-result-archive-20260710`.
- Every checkpoint retained by the earlier curated `metric_boost` cleanup.
- All current SALT-VI vision-text baseline SR checkpoints. A0 `best.pth` and `latest.pth` remain at
  both paths but now share one inode after byte-for-byte verification.

## Removed low-risk artifacts

- One GPU smoke checkpoint; its configuration, metrics, and logs remain.
- One checkpoint from a failed `return_code=-15` attempt; its status and logs remain.
- Four registered worktrees whose commits were already merged into `main` and which
  had no active process. One was clean, one contained 105 files byte-identical to
  files in `main`, and two contained only pending/smoke preparation provenance with
  no metrics or checkpoints.
- Python bytecode/test caches and an empty `.tmp` directory. The zero-byte `%ln`
  root file was retained after final verification showed that Git tracks it.

## Removed reconstructable feature caches

The eight directories named `feature_cache` below had no open files and were not
listed in `manifest.json`, `artifact_hashes.json`, or result-hash manifests. Their
derived metrics, CSV files, reports, and runtime provenance remain. They can be
regenerated from the retained code, data, and selected checkpoints.

- `SALT-VI/reports/feature_text_dataset_compare/pair_v1/original/feature_cache`
- `SALT-VI/reports/feature_text_dataset_compare/pair_v1/corrected/feature_cache`
- `SALT-VI/reports/feature_domain_gap/TRAIN-3-H1/feature_cache`
- `SALT-VI/reports/feature_domain_gap/E4/feature_cache`
- `SALT-VI/reports/external_feature_screen/TRAIN-3-H1/feature_cache`
- `SALT-VI/reports/external_feature_screen/TRAIN-3-H1-llcm-only/feature_cache`
- `SALT-VI/reports/external_feature_screen/TRAIN-3-H1-threshold-2p0/feature_cache`
- `SALT-VI/reports/external_feature_screen/TRAIN-3-H1-threshold-2p0-llcm-only/feature_cache`

## Removed non-selected historical checkpoints

Only the binary weights were removed. All corresponding logs, configs, TensorBoard
events, and tabulated metrics remain. No configuration, script, running process, or
stored hash record referenced these files.

| Run | SHA-256 before deletion | Reason |
|---|---|---|
| `stage_a_vision_text_encoder_recipe_256x128_run2/model_IR_23.pth` | `0bec5b1f2c31efab8a4b8f01cb57a5d6fbabf22a548a75b47fd1d3a3d1f8e3d8` | superseded size control |
| `stage_a_vision_text_encoder_recipe_288x144_run2/model_IR_23.pth` | `5367348c68b9ff7c9d50a20d89be95534181aca43cfc0735df27c57fcb30e2cf` | superseded recipe control |
| `stage_a_vision_text_encoder_recipe_288x144_768_run1/model_IR_21.pth` | `6d1fac02c225a740a4c935d7407721a286a2f1c65f3583211fcf0b001bdcb690` | superseded by MBPatch |
| `stage_a_rn50_ori_control/model_IR_107.pth` | `c47252655d8018108f253fb5116e7fd44733308b664742b4039d042c54b510de` | obsolete RN50 control |
| `stage_a_vision_text_encoder/model_IR_31.pth` | `4638c55c3187830dcd809f0c72b80c37353cdee063232c5967972a7e4d9c027c` | failed recipe integration |
| `sampling_mining_ablation/h1_pk8x4_auto_replace_wrt/model_IR_19.pth` | `097b32ad72780259f20fe4a4a13d7789ed95fe14d1869a5af30fc4c5838ad349` | failed mining control |
| `sampling_mining_ablation/s0_pk8x4_current_replace_hard/model_IR_21.pth` | `70a0f461cdb9602108139b06b6d700cbf91046fe5fd61987e1b07c1eee9c71ec` | historical baseline, not selected |
| `sampling_mining_ablation/s1_pk8x4_auto_replace_hard/model_IR_23.pth` | `81b8002416b8bbf5d13369a7d15a3ac25f9fd59d7d04242bdd8af5458d4e436a` | sampler control, not selected |
| `sampling_mining_ablation/s2_pk16x2_auto_replace_hard/model_IR_21.pth` | `d8535cc48ea8374a3b1a52528f57db69b14afe35127f25b914274d4c559d9975` | negative PK control |
| `sampling_mining_ablation/s3_pk4x8_auto_replace_hard/model_IR_23.pth` | `a1f52b9d6d61493972f296978b2c24a144d6ce7522bd016036f04354f8a06947` | non-selected PK control |

The incomplete and invalidated `TEXTV1-FGAP2-P1-seed0` checkpoint was also removed.
Its event history through epoch 12, including the epoch-1 best metrics, remains.
Pre-deletion SHA-256:
`01b53f5dde4e9a2a389b38397fecece162c9220548b3f7e28d9cfe8c7403b0f7`.

## Recoverability

Hard-link deduplication is transparent and requires no recovery. Deleted cache files
are reproducible but not directly recoverable. Deleted checkpoint binaries are not
recoverable from this directory; their hashes and experimental evidence remain.

## Deliberately retained candidates

- `Single-experiment` contains two independent historical best checkpoints (RGB
  LastViT and IR SCHP), not duplicate cache data; these were retained.
- `SALT-VI/reports/metric_boost` contains the previously curated set of independent
  checkpoints; these were retained.
- The three incomplete PMT-SR runs retain both `best.pth` and `latest.pth` because
  their resume and best states are part of the still-unfinished experiments.
