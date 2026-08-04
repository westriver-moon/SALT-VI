# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/metric_boost/plans/fgap-asym-hard-20260714/ARCHIVE.md`  
> Original SHA-256: `2ad0c824d3126d5686b1f9f82d40c703b211135743ca9537c2f066c486bd862f`  
> This is read-only experiment evidence, not an active runtime instruction.

# FGAP asymmetric hard-pair archive

This archive preserves the complete analysis and reproducibility record for:

- `FGAP-P1-asym-hard`
- `FGAP-P2-asym-hard-u2`
- `FGAP-P3-asym-hard-qbn`

The Git archive includes the immutable pre-launch provenance bundle, resolved
runtime configuration, status, structured event stream, human log, parsed
epoch tables, final summaries, and all loss/metric figures. The implementation
is represented by Git commit `68fd381aee9e60960f38b93de084a8b3887a3446`.

Checkpoint binaries are intentionally not committed because each is about
572 MiB and exceeds GitHub's normal file-size limit. They remain on the server;
`checkpoint_inventory.json` records their exact paths and SHA-256 digests.
The event streams also contain the selected checkpoint and digest.

Canonical curve archive:

`logs/raw/source_core/reports/metric_boost/curves/fgap-asym-hard-20260714/`

Canonical run archives:

`logs/raw/source_core/reports/metric_boost/runs/<experiment-id>/`

Only `model_output/` is excluded from Git. No manifest or provenance file was
rewritten during archiving.
