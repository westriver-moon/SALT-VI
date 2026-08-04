# Archived 288x144 Stage-B configurations

These ten configurations were moved from `configs/stage_b/` on 2026-08-04. They are retained for metrics and log provenance, not for current training.

All require a 288x144 single-branch PMT-ViT Stage-A warm start. No such retained checkpoint exists. The current canonical A3 checkpoint is 512x256 with a multi-branch patch embedding and must not be used as a substitute.

The training loader now fails before model construction when no topology-compatible warm start is available. Registry rows use lifecycle `archived_unreproducible` and point to this directory.
