# Shared person assets, Qwen, and PACT

This worktree keeps all components in the SALT repository while enforcing a
one-way dependency graph:

```text
person_preprocessing
  final ViT image + geometry + raw COCO-17 pose
             |                         |
             v                         v
plugins/qwen_imagination          plugins/pact
SCHP/SAM semantic ROI             anatomy/token masks; future training adapter
             |                         |
             +------------+------------+
                          v
                   SALT orchestration
```

`person_preprocessing` is a standalone package and cannot import `salt_vi`,
`qwen_imagination`, or `pact`. It first produces the deterministic final ViT
image and then runs the canonical pose model on that exact image. For
`person_fit`, an earlier localization pass exists only to define geometry.

Qwen prepared annotation reads the cached bbox and keypoints by `source_key`.
It no longer loads YOLO. Its result signature embeds `pose.contract.json`, so a
weight, threshold, geometry, or source change invalidates annotations.

PACT is a standalone plugin. It converts the same raw pose into method-specific
head, torso, arms, legs, background, residual, and token-grid masks. These
artifacts live outside the shared cache because their definitions are research
choices, not general preprocessing facts.

SALT core has one optional bridge, `salt_vi.person_assets`, used only when a
training config sets `prepared_data_root`. Legacy datasets remain unchanged.

Training-time random crop and horizontal flip must be applied jointly to the
image and PACT pixel masks, followed by recomputation of token masks. Random
erasing must update regional visibility. This remains part of the PACT training
adapter, not shared preprocessing.

The method hypothesis and planned intervention-layer search are documented in
[`plugins/pact/docs/research_plan.md`](../../plugins/pact/docs/research_plan.md).
