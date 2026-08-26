# QRI imagination control experiments

This directory contains category-agnostic experiments for VLM-planned local
imagination. The eyeglasses example is a validation EditSpec, not an object-
specific implementation.

- `edit_program.py` validates and rasterizes generic ellipse, rectangle,
  polygon and polyline primitives into creation/transition/preservation maps.
- `controlled_inpaint_benchmark.py` keeps high diffusion strength in the
  creation region while optionally restoring source latents after every
  denoising step elsewhere.
- `adaptive_writeback_preview.py` audits coherent-region and evidence-based
  write-back without rerunning diffusion.
- `pasd_layout_control_benchmark.py` tests the repository's existing PASD
  ControlNet either with an abstract layout or with a photo-like semantic
  proposal. The accepted route is the latter: SD1.5 proposes clear content and
  PASD restores pedestrian structure and surveillance style.
- `sam_semantic_writeback.py` is an audited fallback. It is not selected for
  the thin-transparent-object sample because SAM included protected eye pixels.
- `test_edit_program.py` checks map geometry and value bounds without loading a
  diffusion model.

For the current fixed sample, the selected artifact is recorded in
`reports/qri_imagination_control/eyeglasses_cam1_0001_selection.json`. Direct
layout stenciling, layout-draft initialization, constant source locking, and
SAM write-back are retained as explained negative results, not defaults.

Large images and metrics must be written outside the repository under
`/home/lab929/ybj/experiments/qri_imagination_control/`.
