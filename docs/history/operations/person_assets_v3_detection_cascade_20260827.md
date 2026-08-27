# Person Assets V3 Detection Cascade Audit (2026-08-27)

## Outcome

V3 is a sparse overlay on top of the V2 asset store. It revisits only the 2,031 V2
full-frame fallbacks and records a person detection box for every one of them. The
merged V3 inventory therefore has no remaining `fallback_full_frame` localization
records.

The detector coverage target and the rendering quality target are separated on
purpose:

- every revisited item has a pose-model `raw_bbox` and `tertiary_detected_person`
  provenance;
- a crop is rendered only when an independent detection model agrees with the pose
  model and the geometry/keypoint checks pass;
- otherwise, the V1-safe full-frame render is preserved exactly while the newly
  detected box remains available in metadata.

Implementation commit: `97fe7b82` (`feat: add quality-guarded person assets v3 overlay`).

## Inputs and outputs

- Repository: `/home/lab929/ybj/SALT-VI`
- Parent V2 store: `/home/lab929/ybj/datasets/person-assets-512x256-v2-overlay`
- Final V3 overlay: `/home/lab929/ybj/datasets/person-assets-512x256-v3-overlay`
- Run logs: `/home/lab929/ybj/experiments/person_preprocessing/person-fit-v3-cascade-20260827`
- Final contact sheets: `/home/lab929/ybj/experiments/person_preprocessing/person-fit-v3-cascade-20260827/quality-audit`
- Pose model: `/home/lab929/ybj/models/qri-v1/checkpoints/yolo11x-pose.pt`
- Independent validation model: `/home/lab929/ybj/models/qri-v1/checkpoints/yolo11x.pt`

## Detection cascade

Each unresolved V2 fallback is processed by the following sequence until a pose
box is found:

1. `relaxed_960`: original image, confidence `0.01`, image size `960`;
2. `floor_1280`: original image, confidence `0.001`, image size `1280`;
3. `autocontrast_1280`: autocontrast transform, confidence `0.001`, image size
   `1280`.

The final candidate is subjected to geometry and keypoint checks, then compared
with the independent YOLO11x detection head. Trusted rendering requires detector
confidence at least `0.10` and pose/detector IoU at least `0.50`, in addition to
the pose-quality checks. This dual-model gate prevents relaxed thresholds from
turning chairs, desktops, partial limbs, or thermal artifacts into destructive
person crops.

## Coverage and render policy

| Dataset | V2 fallbacks | Relaxed 960 | Floor 1280 | Autocontrast 1280 | Trusted crops | Full-frame guards |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 527 | 470 | 56 | 1 | 75 | 452 |
| RegDB | 532 | 509 | 16 | 7 | 8 | 524 |
| LLCM | 972 | 815 | 130 | 27 | 98 | 874 |
| **Total** | **2,031** | **1,794** | **202** | **35** | **181** | **1,850** |

Detection coverage is `2,031 / 2,031` (100%). All 2,031 records are accepted into
the sparse V3 overlay with detection provenance. The conservative quality gate
uses 181 boxes for actual crops and preserves the safe parent render for the other
1,850 records.

## Integrity verification

The final merged store was loaded through `PersonAssetStore` and every referenced
image was opened and checked.

| Dataset | Merged inventory | Missing | Corrupt | Wrong size | Guard pixel mismatch | Trusted gate failure |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 44,745 | 0 | 0 | 0 | 0 | 0 |
| RegDB | 8,240 | 0 | 0 | 0 | 0 | 0 |
| LLCM | 46,767 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **99,752** | **0** | **0** | **0** | **0** | **0** |

All 1,850 quality-guard outputs are pixel-identical to their parent safe renders.
All 181 trusted crops satisfy the stored dual-model thresholds.

## Tests and visual audit

The implementation passed:

```text
PYTHONPATH=person_preprocessing /home/lab929/miniconda3/envs/pasd/bin/python \
  -m pytest -q \
  person_preprocessing/tests/test_refine_v3.py \
  person_preprocessing/tests/test_refine.py \
  person_preprocessing/tests/test_pipeline.py

6 passed in 0.19s
```

`git diff --check` also completed without errors.

The final contact sheets inspected two adversarial selections per dataset:

- trusted crops with the lowest independent-validator IoU/confidence;
- guarded records with the highest pose confidence.

The trusted selections were genuine people with usable crops. The guarded
selections included ambiguous people and false-positive hazards; preserving the
full frame was the correct quality decision.

## Reproducibility and quarantined trials

The active Stage-A V3 configs point only to the final V3 overlay. Earlier
experimental outputs are retained for recovery and comparison, but are not
referenced by active configs:

- `/home/lab929/ybj/datasets/person-assets-512x256-v3-overlay-first-pass-quarantine-20260827`
- `/home/lab929/ybj/datasets/person-assets-512x256-v3-overlay-single-model-quarantine-20260827`
- `/home/lab929/ybj/experiments/person_preprocessing/person-fit-v3-cascade-first-pass-quarantine-20260827`
- `/home/lab929/ybj/experiments/person_preprocessing/person-fit-v3-cascade-single-model-quarantine-20260827`

The quarantine runs demonstrate why the independent validation model is required:
the relaxed pose model alone can produce geometrically plausible boxes on
non-person structures.
