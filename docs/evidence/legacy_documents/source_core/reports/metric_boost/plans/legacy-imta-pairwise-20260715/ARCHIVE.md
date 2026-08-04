# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/metric_boost/plans/legacy-imta-pairwise-20260715/ARCHIVE.md`  
> Original SHA-256: `9b959ef586d29b3562ec0177b6f886cc2f4305e41a341b670f41b56160e32dd8`  
> This is read-only experiment evidence, not an active runtime instruction.

# Legacy IMTA and PAIRWISE archive

Archived at: 2026-07-15T05:40:16.013051+00:00

Six completed legacy runs were retrospectively upgraded to the current manifest/provenance contract.

The failed `IMTA-M1-prototype` attempt was superseded by `IMTA-M1-prototype-retry1`; its CUDA-OOM cause is retained in the retry manifest before removal of the failed run-specific files.

| Experiment | Epoch | Rank-1 | mAP | mINP |
|---|---:|---:|---:|---:|
| IMTA-M1-prototype-retry1 | 1 | 0.810518 | 0.782549 | 0.671391 |
| IMTA-M2-relation | 1 | 0.812175 | 0.783963 | 0.673317 |
| IMTA-M2-relation-light | 1 | 0.814041 | 0.784910 | 0.674007 |
| PAIRWISE-1-hard-id15 | 1 | 0.821194 | 0.791903 | 0.683478 |
| PAIRWISE-1-hard-llm05 | 21 | 0.822140 | 0.795934 | 0.692515 |
| PAIRWISE-1-llm05-id15 | 1 | 0.821956 | 0.791610 | 0.681421 |
