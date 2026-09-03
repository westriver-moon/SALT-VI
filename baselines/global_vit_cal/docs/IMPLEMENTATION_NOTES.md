# Audited implementation notes

The following server sources were inspected on 2026-09-02. They remain
read-only; this repository is an independent migration.

## Center aggregation loss

The released CenterAggregationLoss implementation was audited and migrated
with its original paired-center denominator reduction. No branch architecture,
intermediate feature head or unrelated objective was migrated with it.

## Successful ellipse-family loss

Run:

    /home/lab929/ybj/experiments/c3-ellipse-radius-layer2-20260829/r045_050

The exact successful contract was layer 2, radii 0.45/0.50, temperature 0.12,
tolerance 0.08, weight 0.1 and a three-epoch warm-up. Best epoch 19 recorded
Rank-1 0.7080725431 and mAP 0.6912178607. The exponent used by the actual code
was 2, not the untrained p=4 shape from earlier geometric analyses.

## Qwen ROI and empirical sampling

Root:

    /home/lab929/ybj/SALT-VI/plugins/qwen_imagination

Authoritative files:

- qwen_imagination/regional/visual_context.py
- qwen_imagination/text_annotation/reasoner.py
- qwen_imagination/text_annotation/empirical.py
- qwen_imagination/text_annotation/track_anchor.py
- configs/text_annotation_sysu_yolo26x_rtmpose_wholebody.yaml

The migrated sampler is empirical-atomic-v1: eight independent atoms per ROI,
complete-link clustering, then 64 empirical joint draws compressed to eight
worlds. The later QRI-v6 one-shot/uniform-world proposal is intentionally not
used because it removes the requested regional repeated-hypothesis mechanism.

## Pose posterior dropout

Geometry and multimodal validation source:

    /home/lab929/ybj/pose_posterior_pruning

The baseline contract uses the final image-level fallback, removes the ineffective
per-token reliability power, retains the 15% rate, and adds the empirically
necessary R_img/TTA hard guard.
