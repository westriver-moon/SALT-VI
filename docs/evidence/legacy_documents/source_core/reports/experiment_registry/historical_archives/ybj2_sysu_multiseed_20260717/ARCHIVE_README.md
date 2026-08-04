# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/ARCHIVE_README.md`  
> Original SHA-256: `158ca1ebbd3423a3ce5c1f9ebf20a3d1eb7fe5f759e83a4b554a82172a2b0dcf`  
> This is read-only experiment evidence, not an active runtime instruction.

# ybj2 SYSU multi-seed reproduction archive

Archived from `/home/cgv841/ybj2` on 2026-07-19 after the isolated four-seed
SALT-VI SYSU reproduction completed.

## Retained

- `reproduction/sysu_multiseed/`: reports, metrics, manifests, provenance,
  runtime configurations, scripts, patches and logs.
- `artifacts/official/author_train.log`: the small official training log
  referenced by the global manifest.
- `data/sysu_text_adapted/PATH_ADAPTATION_MANIFEST.json`: path-adaptation
  provenance.
- Top-level reproduction plans and isolation notes/scripts.

## Deliberately excluded

- All `*.pth`, `*.pt` and `*.ckpt` files (17 files, approximately 5.0 GB).
- The old upstream `SALT-VI` clone.
- Official model-weight copies and the pretraining cache.
- Adapted dataset payloads, Python bytecode caches, PID and lock files.

Historical manifests retain their original `/home/cgv841/ybj2/...` paths so
their recorded bytes and hashes remain unchanged. Those paths are provenance,
not live dependencies. Checkpoint references are intentionally unresolved in
this report-only archive.

The Conda environment `/home/cgv841/anaconda3/envs/tvilfm-ybj2` is separate
from the removed folder and was not modified.
