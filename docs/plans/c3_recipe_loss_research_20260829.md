# C3 recipe/loss research pipeline

This worktree fixes the Stage-A input to the canonical C3 configuration:

- SYSU-MM01;
- classical SwinIR-M x2 RGB and IR arrays;
- source size 256x128 and training size 512x256;
- direct resize geometry;
- camera-diverse identity sampling;
- normalized cosine classifier with scale 30;
- batch size 32 per modality and `num_pos=4`.

No variant may override a data, geometry, sampler, classifier, batch, seed, or
evaluation-protocol field.  The runner rejects such manifest changes.

## Research sequence

The manifest contains 13 single-seed training runs plus two checkpoint
diagnostics.  All 15 jobs use the same fixed C3 input contract.

1. **Schedule (3 runs).** Reproduce current C3, shorten the gray stage, then
   decouple RGB activation from the auxiliary metric losses and ramp the latter.
2. **Loss necessity (4 runs).** Remove or isolate MSEL/DCL before tuning mining;
   the lower-value symmetric 0.25/0.25 point is omitted.
3. **Mining/margin (3 runs).** Compare cross-modal hard mining at margins
   0.05, 0.10, and 0.20 against the retained PMT-hard controls.
4. **Selective relation (3 runs).** Replace pointwise MSEL/DCL alignment with a
   relation-preserving loss and retain a small modality-internal triplet term.
   The aggressive 0.20 relation weight is omitted from the first screen.

The two diagnostics run first, one on each GPU:

1. **Loss-gradient conflict.** At the archived canonical C3 checkpoint, measure
   pairwise gradient cosine among ID, triplet, MSEL, and DCL on the final two
   shared PMT blocks.
2. **Modality information.** On identical batches, compare PMT initialization
   with the trained C3 checkpoint using identity-center relation geometry and
   an identity-stratified linear modality probe on identity-residual features.

The second diagnostic does not claim that modality predictability is always
beneficial.  It measures whether current PMT training removes, preserves, or
increases sensor-specific residual structure; the gradient diagnostic supplies
the complementary optimization evidence.

Every scheduled evaluation uses the official SYSU-MM01 all-search, single-shot,
ten-gallery-trial aggregate.  The reported checkpoint is the evaluated epoch
with the highest Rank-1; the later epoch wins an exact Rank-1 tie, matching the
trainer's checkpoint rule.  The selection rule is fixed before results are inspected.
Only shortlisted control/method pairs should be repeated with additional seeds.

## Relation-preserving selective alignment

For every identity in a batch, the implementation forms an RGB center and an
IR center.  It matches the two within-modality cosine-relation matrices:

`smooth_l1(sim(C_rgb, C_rgb), sim(C_ir, C_ir))`.

Unlike DCL, it does not pull RGB and IR observations into one joint center.
Cross-modal hard triplet supplies the retrieval alignment, while a small
modality-internal triplet term preserves identity structure that is useful
inside each sensor domain.  The relation term cannot prevent collapse alone;
it is therefore never used without ID classification and triplet supervision.

## Commands

Validate without starting training:

```bash
python scripts/experiments/run_c3_recipe_loss_research.py plan
python -m pytest -q src/salt_vi/tests/test_c3_recipe_loss_research.py
```

Formal training requires a separate explicit launch.  Example:

```bash
python scripts/experiments/run_c3_recipe_loss_research.py run \
  --gpus 2,3
```

The scheduler refuses GPUs other than physical 2 and 3, verifies both are idle,
runs one diagnostic first on each card, then assigns the 13 trainings
round-robin.  Each job has a separate launcher log, while
`scheduler/state.json` and `results.partial.json` are updated atomically.
Both diagnostics are startup gates: if either fails, no training starts.  Once
the gates pass, an individual training failure is recorded and does not
suppress the remaining independent training jobs.
