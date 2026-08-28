# SALT person preprocessing

This package materializes the single canonical 512x256 person-fit dataset from
the audited V1 + YOLO26 quality gate. Earlier V2/V3 fallback overlays are
retired and are not part of the supported interface.

The quality gate combines YOLO26x detection and pose, a YOLO11x detector
cross-check, SCHP-LIP human parsing, temporal agreement, and the original V1
localization. Passing records are rendered from the audited person bbox.
Fallback records preserve the full source frame with the same aspect-preserving
geometry, so no inventory item is dropped.

The current audited release contains 99,752 images:

| Dataset | Pass | Fallback | Fallback rate |
|---|---:|---:|---:|
| SYSU | 44,031 | 714 | 1.5957% |
| RegDB | 7,161 | 1,079 | 13.0947% |
| LLCM | 45,065 | 1,702 | 3.6393% |
| Total | 96,257 | 3,495 | 3.5037% |

The materializer refuses duplicate or missing keys, invalid boxes, count drift,
and the observed upper-body-only failure pattern. It writes one immutable
contract and one complete manifest per dataset.

```bash
python -m person_preprocessing \
  --config person_preprocessing/configs/person_assets_512x256.yaml \
  --datasets sysu regdb llcm \
  --workers 32
```

Training uses `/home/lab929/ybj/datasets/person-assets-512x256/person_fit`
through the three canonical Stage A `*_person_fit.yaml` configs.
