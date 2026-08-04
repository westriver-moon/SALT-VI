# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/README.md`  
> Original SHA-256: `16f062899a65d0ebf46c7d53e3df0ba69cbe85abccf8591094beeff46244f6b6`  
> This is read-only experiment evidence, not an active runtime instruction.

# Historical Experiment Archives

These folders are retained as non-active provenance material alongside the experiment registry. They are not inputs to the current SALT-VI/SALT-VI training or evaluation path.

| Archive | Contents | Weight status |
| --- | --- | --- |
| `ybj2_sysu_multiseed_20260717/` | Four-seed SYSU reproduction plans, metrics, environment/provenance records, scripts, and small logs copied from the retired `ybj2` workspace. | No model weights retained. |
| `ybj_inactive_experiments_20260720T2233/` | Inactive-experiment manifests, cleanup ledger, and metadata snapshot. | No independent model weights; compatibility links may be unresolved after checkpoint cleanup. |
| `ybj_cleanup_20260720.md` | Original workspace-cleanup ledger from 2026-07-20. It is retained as historical evidence only; some references predate subsequent cleanup and relocation. | No model weights. |

The active normalized result table is one level above in `experiment_results.csv`; archived configurations are in the sibling `configs/experiments/reproduction/archived_configs/` directory.
