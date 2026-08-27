# Person-assets v2 fallback refinement (2026-08-27)

## Scope

The v2 asset is a sparse overlay on the immutable v1 `person_fit` dataset. It
does not copy, link, or reprocess any non-fallback image. Only the 3,745 v1
records marked `fallback_full_frame` receive a second YOLO11x-pose pass.

- Base: `/home/lab929/ybj/datasets/person-assets-512x256-v1/person_fit`
- Overlay: `/home/lab929/ybj/datasets/person-assets-512x256-v2-overlay`
- Run: `/home/lab929/ybj/experiments/person_preprocessing/person-fit-v2-fallback-refinement-20260827`
- Config: `person_preprocessing/configs/person_assets_512x256_v2.yaml`
- Implementation commit: `0f07c21831b9a1a5a09c4be9d1f7b82bfc4f18d1`

## Acceptance rule

The second pass uses the official `yolo11x-pose.pt`, `rect=False`, image size
640, and detection confidence 0.10. A proposed crop is accepted only if all of
the following hold:

- box area fraction is within `[0.08, 0.95]`;
- box height is at least 0.35 of the source height;
- box height/width is at least 0.75;
- normalized center distance is at most 0.50;
- at least five keypoints have confidence 0.15 or higher;
- at least one shoulder/hip keypoint has confidence 0.15 or higher.

Rejected proposals retain the v1 full-frame fallback at read time. Accepted
images are stored in the overlay. `PersonAssetStore` presents a complete merged
view without materializing the unchanged v1 payload again.

## Results

| Dataset | v1 fallback | v2 accepted | retained fallback | Acceptance |
|---|---:|---:|---:|---:|
| SYSU | 901 | 374 | 527 | 41.5% |
| RegDB | 1,094 | 562 | 532 | 51.4% |
| LLCM | 1,750 | 778 | 972 | 44.5% |
| Total | 3,745 | 1,714 | 2,031 | 45.8% |

The effective fallback rate across all 99,752 images falls from 3.75% to 2.04%.
The sparse overlay contains 1,714 PNG files and occupies approximately 132 MB.
All three refinement manifests cover their complete v1 fallback inventories.

## Quality audit

The audit sampled 12 accepted and 12 rejected records per dataset, stratified
by modality. No clearly background-only false acceptance was observed among
the 36 accepted samples. Two accepted SYSU examples were marginal because the
person was heavily occluded or only partially visible. Rejected examples show
that the gate is intentionally conservative: several real RegDB full-body
thermal images remain fallback because their pose keypoints are insufficient.

Audit artifacts:

`/home/lab929/ybj/experiments/person_preprocessing/person-fit-v2-fallback-refinement-20260827/quality-audit`
