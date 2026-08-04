# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: a3 e4 hpt stage3 runbook

> Source document ID: `source_core:docs/a3_e4_hpt_stage3_runbook.md`  
> Original SHA-256: `32bb16bf3211ffbf3eb49a61b8a9c8b253ea0a4170cdab0f3fea3bff03a9271b`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# A3-E4-SR HPT Stage-3 runbook

Stage-3 tests four static cross-modal pair-weight structures while fixing the
Stage-2 winner's hyperparameters: `lr_txt=7.5e-6` and
`cross_modal_hard_weight=1.25`.

All four runs start from the same A3-E4 epoch-21 checkpoint. The Stage-2
`L075-W125` epoch-14 checkpoint is a selection result, not an initialization.
The all-equal group is rerun as a same-commit paired control.

GPU 0 is excluded. The scheduler dynamically fills physical GPUs 1, 2, and 3:
one idle GPU starts one pending experiment, two idle GPUs start two, and three
idle GPUs start three. Each assignment is protected by the shared GPU lock and
is rechecked against memory, utilization, and compute-process state immediately
before launch. When a run finishes, the next pending experiment may use that GPU.

Preflight:

```bash
python scripts/metric_boost/run_a3_e4_hpt_stage3.py --preflight
```

Launch dynamically on currently idle GPUs 1, 2, and 3:

```bash
python scripts/metric_boost/run_a3_e4_hpt_stage3.py \
  --run --gpu-indices 1,2,3 --max-parallel 3 \
  --confirm-launch I_UNDERSTAND_A3_E4_HPT_STAGE3_WILL_START
```

Status:

```bash
python scripts/metric_boost/run_a3_e4_hpt_stage3.py --status
```

Results are exploratory because the SYSU test protocol is used during
hyperparameter selection. Selection is by highest Rank-1, with mAP and mINP
reported from the same epoch.


---

## Migrated source: a3 e4 hpt stage3 results

> Source document ID: `source_core:docs/a3_e4_hpt_stage3_results.md`  
> Original SHA-256: `5734619749aa2c2c1484f5ef9dac657de45e57b7be9b0e2e1a3419ac02c0ad57`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# A3-E4 HPT Stage-3 results

Stage-3 tested four cross-modal pair-weight structures while holding the Stage-2 winner's optimization point fixed (`lr_txt=7.5e-6`, `cross_modal_hard_weight=1.25`). Every candidate started independently from the common A3-E4 epoch 21 checkpoint; no candidate inherited another Stage-3 or Stage-2 trained checkpoint.

The run was archived on 2026-07-23 after the three completed candidates and after PAIR-NOTEXT finished epoch 13. PAIR-NOTEXT was stopped by user request.

| Experiment | Progress | Rank-1-best epoch | Rank-1 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAIR-EQUAL | 20/20 | 14 | **83.65%** | **81.24%** | **71.69%** |
| PAIR-MILD | 20/20 | 14 | 83.29% | 80.87% | 71.25% |
| PAIR-STRONG | 20/20 | 1 | 83.11% | 80.54% | 70.66% |
| PAIR-NOTEXT | 14/20, stopped | 1 | 82.76% | 80.21% | 70.27% |

The table reports all three metrics at each run's Rank-1-best checkpoint. PAIR-NOTEXT's metric-specific best mINP was 70.31% at epoch 13.

PAIR-EQUAL is the Stage-3 winner and reproduces the Stage-2 L075-W125 optimum. Down-weighting text-related pairs did not improve retrieval performance, and stronger suppression caused progressively larger regressions. The Stage-3 result therefore does not justify replacing equal pair weights.

The server-side archive retains complete event histories, configs, logs, provenance fingerprints, and representative checkpoints. Redundant metric-specific checkpoints were removed only after their hashes and scalar histories were recorded in the archive manifest.
