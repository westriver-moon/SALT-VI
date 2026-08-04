# SALT-VI migrated evidence document

> Source document ID: `source_core:reports/experiment_registry/historical_archives/ybj2_sysu_multiseed_20260717/reproduction/sysu_multiseed/reproduction_report.md`  
> Original SHA-256: `07902e1ba3c6102e4f6b80868b5548a3e4af98a391b9412e09b2271c41aa020b`  
> This is read-only experiment evidence, not an active runtime instruction.

# SALT-VI SYSU four-seed reproduction

## Per-seed results

| Seed | Status | no-MER R1 | no-MER mAP | no-MER mINP | MER R1 | MER mAP | MER mINP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | complete | 81.24 | 77.04 | 64.34 | 80.65 | 76.70 | 64.07 |
| 2 | complete | 81.97 | 77.60 | 65.22 | 81.48 | 77.32 | 64.91 |
| 3 | complete | 81.06 | 77.18 | 65.05 | 80.53 | 76.93 | 64.75 |
| 4 | complete | 81.42 | 77.36 | 64.83 | 81.98 | 77.97 | 65.72 |

## Aggregate

### no MER

- rank1: mean 81.42, sample std 0.40, min 81.06, max 81.97, n=4
- map: mean 77.29, sample std 0.24, min 77.04, max 77.60, n=4
- minp: mean 64.86, sample std 0.38, min 64.34, max 65.22, n=4

### MER

- rank1: mean 81.16, sample std 0.69, min 80.53, max 81.98, n=4
- map: mean 77.23, sample std 0.55, min 76.70, max 77.97, n=4
- minp: mean 64.86, sample std 0.68, min 64.07, max 65.72, n=4

## Reference

- Author seed-1 no-MER: R1 84.17, mAP 80.72, mINP 70.02.
- Paper MER: R1 84.90, mAP 81.47, mINP 70.85.
- No numerical pass/fail threshold is applied.

## Provenance

- Seed 1: `/home/cgv841/ybj2/reproduction/sysu_multiseed/runs/seed_1/attempt_06/manifest.json` — valid
- Seed 2: `/home/cgv841/ybj2/reproduction/sysu_multiseed/runs/seed_2/attempt_06/manifest.json` — valid
- Seed 3: `/home/cgv841/ybj2/reproduction/sysu_multiseed/runs/seed_3/attempt_06/manifest.json` — valid
- Seed 4: `/home/cgv841/ybj2/reproduction/sysu_multiseed/runs/seed_4/attempt_06/manifest.json` — valid
