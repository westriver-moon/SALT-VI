# Architecture

## Offline auxiliary path

The 512x256 SwinIR view is isolated from the image backbone. It has two
consumers:

1. YOLO26x person detection plus RTMPose-L WholeBody under six TTA transforms.
   pose_posterior.py inverse-maps detections, renders dense support, computes
   image reliability and pools pose/fallback support to the 364-token grid.
2. Qwen ROI annotation. qwen/zoom.py creates tight/context boards;
   qwen/hypotheses.py performs eight validated draws and complete-link
   clustering per region; qwen/sampling.py creates 64 joint draws and retains
   at most eight empirical worlds. Track-anchor selection shares semantics
   while leaving ROI geometry image-specific.

Both outputs are serializable preprocessing artifacts. Neither path changes
the pixels seen by the ViT.

## Image model path

The original 288x144 RGB/IR image is converted to a 28x13 patch grid. The
token_dropout module evaluates P_eff and P_drop for each layer and returns
fixed-position masks. backbone.py applies those masks inside the transformer
while preserving the patch positions. The model returns one global feature
sequence plus block-2 CLS attention in both training and evaluation modes.

recipe.py combines global identity/triplet terms with CAL and the separately
validated soft ellipse attention loss. No local or intermediate auxiliary
feature objective exists in the implementation package.

## Optional Stage-B training bridge

stage_b.py carries a model-independent Stage-B schedule: bidirectional hard
mining across RGB/IR/text/fusion features and an epoch-3-to-7 hard-loss ramp.
Qwen regional worlds may be converted to offline text candidates, encoded
before training and reduced by their selected weights. They never invoke Qwen
within the online model path.

## File ownership

- backbone.py: single global ViT
- token_dropout.py: final equations, per-layer calibration and hard guard
- pose_posterior.py: TTA geometry, dense support, fallback and structural score
- losses.py: exact CAL and validated superellipse-family loss
- recipe.py: training objective assembly
- qwen/: ROI boards, prompts, validation, clustering, worlds and HTTP backend
- stage_b.py: Stage-B hard-mining schedule, reusable without a legacy runner
- configs/: one fixed method configuration
- tests/: deterministic component and integration contracts
