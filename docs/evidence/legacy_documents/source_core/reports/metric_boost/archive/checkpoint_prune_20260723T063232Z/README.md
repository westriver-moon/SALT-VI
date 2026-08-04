# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/metric_boost/archive/checkpoint_prune_20260723T063232Z/README.md`  
> Original SHA-256: `8d50d41a6a96d5d44892a81b33c74239d0b17caedcbf16d1fa7aa47dd5c1d444`  
> This is read-only experiment evidence, not an active runtime instruction.

# Checkpoint prune archive

- Completed: `2026-07-23T06:32:33.782208+00:00`
- Experiments pruned: **17**
- Checkpoints pruned: **18**
- Bytes released from model files: **10789200851**
- Retained: status, resolved config, metric events, launcher logs, and provenance files where present.

## Pruned experiments

- `FGAP-P1-asym-hard` — 598957443 bytes
- `FGAP-P2-asym-hard-u2` — 598957438 bytes
- `FGAP2-P1-h1-qbn` — 598999988 bytes
- `FGAP2-P3-h1-qbn-llm05` — 1197998734 bytes
- `PAIRWISE-1-hard-id15` — 598957887 bytes
- `PAIRWISE-1-llm05-id15` — 598957993 bytes
- `TRAIN-1-U1` — 598957378 bytes
- `TRAIN-2-learnable-pa` — 598957716 bytes
- `TRAIN-3-H2` — 598957874 bytes
- `TRAIN-3-H3` — 598957472 bytes
- `TRAIN-5-llm-0p25` — 598957374 bytes
- `TRAIN-5-llm-0p5` — 598957377 bytes
- `TRAIN-6-id05` — 598957401 bytes
- `TRAIN-6-id15` — 598957378 bytes
- `TRAIN-6-wrt05` — 598957373 bytes
- `TRAIN-8-cls-gem` — 598958015 bytes
- `A3E4-S2-L050-W100` — 606796010 bytes

## Retained checkpoint experiments

- `FGAP-P3-asym-hard-qbn` — selected FGAP stage reference
- `FGAP2-P2-h1-qbn-freeze` — retained epoch 19 because it is the mINP-best checkpoint (Rank-1 0.8124113083, mAP 0.7937119468, mINP 0.6941603916). The run's Rank-1/mAP-best checkpoint was epoch 1 and is not retained.
- `PAIRWISE-1-hard-llm05` — best pairwise result and FGAP2 historical reference
- `TRAIN-3-H1` — selected training baseline and downstream reference
- `TRAIN-4-seed-1` — required to reproduce archived seed ensemble
- `TRAIN-4-seed-42` — best seed replicate and required by seed ensemble
