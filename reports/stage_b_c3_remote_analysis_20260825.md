# C3 Stage-B remote-analysis index

This file is intentionally self-contained so a reviewer or web-based GPT can
analyze the C3 Stage-B experiments from the Git repository without access to
the 3090 server's filesystem.

## Reproducibility identity

- Repository branch: `codex/pmt-mscm-phased-pasd-20260821-v2`
- Main archive/source-registration commit: `93028a5cd0d3fe2ebc56dacfa4c9c9be86ffc8bd`
- Training-code commit: `1dceb9a7080b41bb9861878db851aa10c0e2915e`
- Experiment-worktree source commit before cherry-pick: `bd6d67cee607df79e4fc3255468aa6d68c5d2b3d`
- Dataset: SYSU-MM01
- Stage-A warm start: C3 camera-diverse sampler + cosine classifier
- Stage-A checkpoint SHA-256: `00c581ff6965a93088ac4c807c5937db270378336702d16f38720949693e8171`

The committed training entry points and configurations are:

- `configs/stage_b/a3_e4_stageb.yaml`
- `configs/stage_b/safe_tricks/`
- `configs/pipelines/sysu_stage_b_c3_b5_tricks_grid_20260824.yaml`
- `configs/stage_b/b5_tricks_grid_20260824/`
- `scripts/experiments/run_safe_tricks_pipeline.py`
- `scripts/experiments/run_stage_b_c3_b5_tricks_grid.py`

## Fixed evaluation protocol

All reported numbers use the same protocol: all-search, single-shot, 10
gallery trials; infrared image query plus visible identity-caption query;
visible-image gallery; identity-text retrieval backend; Fusion test modality.
There is no reranking and no test-time augmentation.  Model selection is by
best Rank-1, with mAP and mINP reported from that same epoch.

## Complete result ledger

Percentages are shown to four decimal places.  `latest` is the last completed
epoch and is included to distinguish a selected checkpoint from the terminal
training state.

### C3 aligned baseline

| Variant | Status | Best epoch | Best Rank-1 | Best mAP | Best mINP | Latest epoch | Latest Rank-1 | Latest mAP | Latest mINP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | completed | 27 | 83.4262% | 80.7980% | 71.0409% | 29 | 83.2133% | 80.7041% | 70.9924% |

Configuration: `configs/stage_b/a3_e4_stageb.yaml`.

### Safe-tricks grid

| Variant | Status | Best epoch | Best Rank-1 | Best mAP | Best mINP |
|---|---|---:|---:|---:|---:|
| b0 baseline | completed | 4 | 84.1231% | 81.5740% | 71.9212% |
| b2 unfreeze-last2 | failed, epoch 2 retained | 2 | 80.9598% | 78.3788% | 67.8288% |
| b3 unfreeze-last2 + LLRD | skipped/queued, never launched | — | — | — | — |
| b4 QBN freeze-6 | completed | 4 | 83.7286% | 81.5968% | 72.2073% |
| b5 hard-loss ramp | completed | 6 | **84.2966%** | **81.6982%** | 71.9119% |
| b6 RGB consistency | completed | 4 | 84.1231% | 81.5740% | **71.9212%** |

The safe-tricks grid is closed. `b2=failed` and `b3=skipped`; there is no
resume plan.  The complete safe-tricks configuration directory is committed
at `configs/stage_b/safe_tricks/`.

### B5 trick grid

| Variant | Isolated change | Status | Best epoch | Best Rank-1 | Best mAP | Best mINP |
|---|---|---|---:|---:|---:|---:|
| camera_only | camera-diverse sampling only | completed | 5 | 84.5727% | 82.0499% | 72.6046% |
| cosine_only | cosine classifier only | completed | 26 | 83.3737% | 80.8636% | 71.2727% |
| label_smoothing_010 | label smoothing ε=0.10 | completed | 4 | 83.1081% | 80.3887% | 70.2804% |
| ema_0999 | EMA decay 0.999 | completed | 6 | **85.1354%** | **82.4802%** | **72.9672%** |
| rgb_fusion_025 | RGB-Fusion hard alignment weight 0.25 | completed | 6 | 84.2861% | 81.6873% | 71.8844% |
| rgb_fusion_050 | RGB-Fusion hard alignment weight 0.50 | completed | 6 | 84.2545% | 81.6644% | 71.8519% |

All six variants completed.  The configuration and runner are committed at
`configs/pipelines/sysu_stage_b_c3_b5_tricks_grid_20260824.yaml`,
`configs/stage_b/b5_tricks_grid_20260824/`, and
`scripts/experiments/run_stage_b_c3_b5_tricks_grid.py`.

## Analysis-ready conclusions

1. `ema_0999` is the best tested Stage-B setting: +0.8388 Rank-1 points over
   the safe-tricks b5 result and +1.7092 points over the aligned baseline.
2. The camera-diverse sampler alone helps, but the cosine classifier alone does
   not; this isolates why the Stage-A pair should not be interpreted as two
   independently transferable improvements.
3. Label smoothing ε=0.10 is harmful in this run.  RGB-Fusion alignment at
   0.25/0.50 is competitive with b5 but below EMA.
4. These are single-shot retrieval results under a fixed protocol.  No
   post-hoc reranking, test-time augmentation, or test-set-specific inference
   transform is part of the reported gains.

## Artifact and archive evidence

The large checkpoints, event logs, and full payloads remain on the 3090
archive storage rather than in GitHub.  The repository contains their complete
metrics and provenance summary; this is deliberate to avoid committing
multi-gigabyte binary artifacts.  The verified archive inventories are:

| Archive group | Original/archived payload files | Inventory SHA-256 |
|---|---:|---|
| C3 aligned | 14 / 14 | `238ebee68c76640b03a78915f3b610a5bb1250f57de62d1be46b32a7d82bcf5d` |
| Safe tricks | 53 / 53 | `38bae348ec7c00ae74cf892fde6dd36fe894186e2496cbb0364a4237c41c9acd` |
| B5 trick grid | 60 / 60 | `746e3dffc4fe5ca6219e3771b451b5e5babb08481a018b88e400f65cb61b427a` |

All three archives were verified with zero missing files, extra payload files,
or payload hash mismatches.  The canonical machine-readable experiment table
is `reports/experiment_registry/experiment_registry.csv`; the archive
contract summary is `reports/stage_b_c3_archive_20260825.md`.

## Reproduction boundary

To replay training, check out the pinned training-code commit and provide the
SYSU-MM01 data root, the derived SwinIR-x2/PMT256 data root, the retained C3
Stage-A checkpoint, and the recorded Python environment.  The committed YAMLs
and runners define the training behavior; the metrics above define the
historical selection outcome.  The raw checkpoint and event-log paths are
provenance references to the separately retained archive, not required for a
web reviewer to understand or compare the reported results.
