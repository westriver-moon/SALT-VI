# SALT person preprocessing

This repository-owned package creates the immutable visual assets shared by
SALT-VI, Qwen annotation, and PACT. It has no dependency on any of those
consumers.

The ordering is deliberate:

1. consume the inventoried super-resolved source image;
2. deterministically resize or person-fit it to the exact ViT input geometry;
3. run one canonical pose annotation on that final image;
4. bind image and pose products with immutable contracts.

The cache contains only generic artifacts: the final RGB image, geometry,
person bbox, COCO-17 keypoints, detection confidence, and provenance. PACT
anatomical masks and Qwen semantic ROIs belong to their own plugins.

For `person_fit`, an earlier localization pass is needed to define the crop.
That localization is recorded as geometry provenance; the reusable pose is
still generated once on the final ViT-sized image.

The canonical configuration uses `no_person_fallback: full_frame_fit`. If the
localizer misses a person, the sample is not discarded: the complete source
frame is fitted with the same aspect-preserving geometry and the record is
marked `fallback_full_frame`. This keeps the prepared dataset inventory exact
while exposing every fallback for later quality analysis.

Geometry preparation is intentionally offline. Training reads the immutable
prepared images through `prepared_data_root` and applies only stochastic data
augmentation online; it never reruns YOLO or changes the crop between epochs.

`person_assets_512x256_v2.yaml` creates a sparse overlay containing only v1
`fallback_full_frame` records. It runs a square-padded (`rect=False`)
YOLO11x-pose second pass at confidence 0.10, then accepts a crop only when box
area, height, aspect, center distance, and keypoint/torso evidence all pass
explicit gates. No non-fallback image is copied or processed. `PersonAssetStore`
combines accepted overlay images with the immutable v1 base at read time. Run
the refinement with:

```bash
python -m person_preprocessing.refine \
  --config person_preprocessing/configs/person_assets_512x256_v2.yaml \
  --datasets sysu regdb llcm --device 0
```
