# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/archive/retired_rel_metrics/README.md`  
> Original SHA-256: `dff7a500afa61b6869039ef8b3b559e53112ad5a462e7f30d66ff5661218be89`  
> This is read-only experiment evidence, not an active runtime instruction.

# Retired REL metrics archive

- Archived: 2026-07-23 (Asia/Shanghai)
- Dataset/protocol: SYSU-MM01, all-search, single-shot, 10 trials; infrared query, visible gallery.
- Metric values in the CSV files are ratios in `[0, 1]`.
- REL1 screening completed 20 epochs per run. REL2 was stopped early at the user's request.
- This archive intentionally contains metrics only. No model weights, implementation, experiment design, runtime configuration, logs, environment snapshots, or dataset fingerprints are retained.

`best_metrics.csv` selects the epoch with the highest Rank-1 for each experiment; mAP and mINP are taken from that same epoch.
