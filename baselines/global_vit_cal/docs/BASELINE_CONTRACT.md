# Normative method contract

## Token confidence dropout

For token i, combine dense pose support and the geometric fallback by image
reliability:

    P_eff_i = R_img P_pose_i + (1 - R_img) P_ellipse_i

At transformer layer l, the final probability is:

    P_drop_i,l = min[p_max, rho_l (1 - P_eff_i)^gamma]

There is intentionally no per-token R_i^beta multiplier. rho_l calibrates the
expected layer rate; deterministic contract execution drops the highest
probability 15% per eligible image (32 of 210 tokens or 55 of 364 tokens).
If R_img < 0.30 or fewer than four TTA passes succeed, that image is not
pruned. CLS is never dropped. Patch positions are never removed.

P_pose is built from six inverse-mapped TTA WholeBody results. Each result is
rendered as confidence-weighted joint Gaussians, body-bone capsules, a torso
polygon and head support, fused by noisy-or. The dense mean is pooled onto the
288x144, patch-16/stride-10 grid. R_img is the geometric mean of structural-AE
reliability, bounding-box association stability and TTA availability. The
ellipse fallback is pooled to the identical 28x13 (364-token) grid.

## Global backbone and CAL

Use one global token sequence throughout the transformer. The backbone emits
only the final global sequence and the layer-2 CLS-to-patch attention required
by the geometric objective. There are no spatial-band branches, intermediate
feature heads or auxiliary feature losses.

CAL receives aligned concatenation [RGB, IR], with identical identity order in
both halves. The denominator follows the released code exactly, including the
single subtraction of the B paired-center distances from the two 2B negative
distance sums.

## Superellipse-family attention loss

The successful setting is the p=2 member of the superellipse family:

    d = |(x-0.5)/0.45|^2 + |(y-0.5)/0.50|^2
    M = sigmoid((1-d)/0.12)
    L = 0.1 * mean(max(sum_i A_i(1-M_i) - 0.08, 0))

It is applied to layer-2 CLS-to-patch attention and linearly warmed over the
first three epochs. Other exponents are exposed only for ablation and are not
claimed as validated.

## Qwen regional multiple hypotheses

The main ViT never sees the super-resolved image. A 512x256 SwinIR image is
used only to form a 512x512 board for each ROI: a tight crop and a crop with a
0.75 relative margin. Each selected ROI receives eight independent atomic
Qwen draws. A valid draw has exactly one category/state/value/location and no
probability, confidence, score or second candidate.

Valid draws are grouped by complete-link semantic clustering at cosine 0.85.
Cluster weights are empirical valid-sample frequencies; no self-reported Qwen
confidence is used. Across regions, sample 64 independent cluster tuples,
count identical tuples, retain at most eight by descending count and
renormalize retained counts into selected_weight. This regional-hypothesis to
joint-world chain is mandatory.
