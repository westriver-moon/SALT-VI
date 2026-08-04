# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/a3_e4_hpt_stage3/archive_summary.md`  
> Original SHA-256: `d916d4ae1ef3c9470f488cc11e13b54dcfef8dd8fa5d3d42d6f2651e13f3eb59`  
> This is read-only experiment evidence, not an active runtime instruction.

# A3-E4 HPT Stage-3 archive

- Archived at: 2026-07-23T01:21:23Z
- Stop reason: user-requested early stop after the three completed runs
- Source branch: `codex/a3-e4-hpt-stage3`
- Source commit: `933c055e2bb1b1e2495065bd8b0c64174bc63f53`
- Common initialization: A3-E4 epoch 21
- Common initialization SHA256: `7cd15b7b12ba138ccf6590f850dd72a11bfea16d7723a4ce101b98b0a3b1996c`

## Results

| Experiment | Terminal state | Progress | Rank-1-best epoch | Rank-1 | mAP at that checkpoint | mINP at that checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PAIR-EQUAL | succeeded | 20/20 | 14 | 0.8364974856 | 0.8124074193 | 0.7169447781 |
| PAIR-MILD | succeeded | 20/20 | 14 | 0.8328951001 | 0.8087087462 | 0.7125271636 |
| PAIR-STRONG | succeeded | 20/20 | 1 | 0.8311069608 | 0.8054009220 | 0.7066198504 |
| PAIR-NOTEXT | stopped by user | 14/20 (epochs 0-13) | 1 | 0.8275834322 | 0.8020699433 | 0.7027152560 |

PAIR-NOTEXT reached its metric-specific best mINP, 0.7030879038, at epoch 13. The epoch 13 checkpoint is retained together with its Rank-1-best epoch 1 checkpoint.

## Retention policy

Retained for every run: `events.jsonl`, launcher/training logs, runtime and resolved configs, manifests, environment and dataset fingerprints, command lines, source state, and artifact hashes.

Retained checkpoints:

- PAIR-EQUAL epoch 14 (overall winner)
- PAIR-MILD epoch 14 (Rank-1-best)
- PAIR-STRONG epoch 1 (Rank-1-best)
- PAIR-NOTEXT epoch 1 (Rank-1-best)
- PAIR-NOTEXT epoch 13 (last completed epoch and metric-specific mINP-best)

Removed as redundant, while their complete scalar histories remain in `events.jsonl`:

- PAIR-MILD epoch 12 (metric-specific mINP checkpoint)
- PAIR-STRONG epoch 14 (metric-specific mAP/mINP checkpoint)

## Interpretation

PAIR-EQUAL remains the winner and exactly reproduces the three reported Stage-2 L075-W125 metrics, but not the checkpoint binary. Reducing text-related pair weights did not improve the result; stronger suppression produced a larger regression. PAIR-NOTEXT was already below PAIR-EQUAL when stopped after epoch 13.

## Reproduction scope

The reported Rank-1, mAP, and mINP values are identical. The model binaries are not: Stage-2 SHA256 is `bc419e4e...d34fbb`, while Stage-3 PAIR-EQUAL SHA256 is `a1c3747e...60303f`. This is metric reproduction, not byte-for-byte model reproduction.
