# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/ARCHIVE_README.md`  
> Original SHA-256: `21233fbd08b3fca5b61545f2761a8d9878fb7c55b5fd81b777c6df5c62d44e56`  
> This is read-only experiment evidence, not an active runtime instruction.

# SALT-VI Result Archive

This registry records the completed token-to-token ablation and the completed
token-aware warm-start/direct-supervision runs.

The generated result view is `experiment_results.csv`. The finalized external
source rows are kept in `archived_results.csv`, and their YAML snapshots are in
`configs/experiments/reproduction/archived_configs/`. The overlap is intentional: `archived_results.csv` is an
input to the registry builder, while `experiment_results.csv` is its normalized
output. Automated validation requires the shared rows to remain identical.

This registry is scoped to the legacy Stage-A and early Stage-B consolidation.
It is not the global project leaderboard and does not include metric-boost,
A3-E4 HPT Stage-2/3, multistage-text, or RegDB results.

On the server, each archived run keeps only:

- the best checkpoint when it is still retained, otherwise an explicitly empty
  checkpoint field and a metrics-only note;
- the YAML snapshot used to launch the run;
- a compact metrics/status manifest;
- the source path and SHA-256 values recorded in the archive manifest.

The original training logs, TensorBoard files, and non-best output directories
are removed only after the archive verification passes. The checkpoint archive
is intentionally outside Git because the model files are large.
