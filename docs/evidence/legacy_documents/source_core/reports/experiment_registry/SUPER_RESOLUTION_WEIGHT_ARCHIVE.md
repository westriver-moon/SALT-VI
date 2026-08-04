# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/SUPER_RESOLUTION_WEIGHT_ARCHIVE.md`  
> Original SHA-256: `03887f10c504c7a5558a761a3946c2c5e0032e9b2d2012402331248fbed38281`  
> This is read-only experiment evidence, not an active runtime instruction.

# PMSR Super-Resolution Weight Archive

This is the authoritative retention record for the five completed PMSR ablations. Metrics are SYSU-MM01 all-search, single-shot gallery, 10 trials. Each row reports the epoch with the highest Rank-1 and the mAP/mINP measured at that same epoch.

| Experiment | Input / enhancement | Seed | Selected epoch | Rank-1 | mAP | mINP | Runtime config snapshot | Checkpoint status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PMSR-A0-original-256 | Native 256x128 | 0 | 24 | 66.229% | 64.554% | 51.810% | `configs/experiments/reproduction/archived_configs/pmsr_a0_original_256.yaml` | Pruned |
| PMSR-A1-bicubic-x2 | Bicubic x2, 512x256 | 0 | 24 | 68.701% | 67.405% | 55.803% | `configs/experiments/reproduction/archived_configs/pmsr_a1_bicubic_x2.yaml` | Pruned |
| PMSR-A1-bicubic-x2-tail-e32 | A1 continuation | 0 | 26 | 68.601% | 67.394% | 55.927% | `configs/experiments/reproduction/archived_configs/pmsr_a1_bicubic_x2_tail_e32.yaml` | Pruned |
| **PMSR-A3-swinir-both-x2** | **Offline SwinIR x2 on RGB + IR, 512x256** | **0** | **22** | **69.519%** | **67.711%** | **55.895%** | `configs/experiments/reproduction/archived_configs/pmsr_a3_swinir_both_x2.yaml` | **Retained: `SALT-VI/checkpoints/super_resolution/PMSR-A3-swinir-both-x2_best.pth`** |
| PMSR-A3-swinir-both-x2-tail-e32 | A3 continuation | 0 | 30 | 69.677% | 68.244% | 56.912% | `configs/experiments/reproduction/archived_configs/pmsr_a3_swinir_both_x2_tail_e32.yaml` | Pruned |

## Retention rule

Only the standard A3 checkpoint is retained. The A3 tail result remains documented for reproducibility but is a separate continuation experiment and its checkpoint is deliberately not retained. The retained file is 1,057,509,525 bytes and is stored outside `SALT-VI vision-text baseline`.

## Relation to the 84.078% main result

The current best `SALT_R_TEXT_VISUAL` model consumes the precomputed `SYSU-MM01-swinir-x2-pmt256-v1` RGB+IR images. It does not load this PMSR checkpoint at inference; this checkpoint is retained as the trained PMSR-A3 ablation record and provenance artifact.
