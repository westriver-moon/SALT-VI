# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj_inactive_experiments_20260720T2233/README.md`  
> Original SHA-256: `5751dfb7610c6436a831534f9afb0da58dd49affa7897275cb4dd489c281fc57`  
> This is read-only experiment evidence, not an active runtime instruction.

# Inactive experiment archive

This archive excludes the then-live `SALT-VI/reports/a3_e4_hpt_l025` pipeline.
Metadata is retained in `metadata.tar.gz`. The original archive used content-addressed hardlinks in `checkpoint_blobs/`; after the 2026-07-25 workspace-wide deduplication, those 33 compatibility paths are relative symbolic links to the canonical retained checkpoints. They do not consume a second copy of model storage, and all targets remain inside `/home/cgv841/ybj`.

`manifest.json` and `cleanup.json` are immutable historical ledgers: their checkpoint counts and byte totals describe the archive at creation time, not the current number of regular files in `checkpoint_blobs/`. PMT `epoch_*` intermediate snapshots are recorded by SHA-256 but are not retained as blobs after the cleanup ledger is completed.
