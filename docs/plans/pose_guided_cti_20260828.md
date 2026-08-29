# Pose-guided Counterfactual Token Intervention (CTI)

Date: 2026-08-28
Branch: `codex/pose-guided-cti-20260828`

## 1. Goal

Add a training-only counterfactual branch to the existing C3/SALT-VI pipeline. The original C3 objective and inference graph remain intact. At a configurable ViT depth, the counterfactual branch physically removes a fixed number of randomly sampled human-region patch tokens and requires the true-identity evidence to decrease by a margin.

This is token deletion, not a soft attention mask and not a replacement for C3.

## 2. Motivation

Deleting only one preselected token can encourage the network to move identity evidence into another token. Sampling a fresh fixed-size subset from the full pose-defined human region on every training visit makes that shortcut substantially harder: identity evidence must remain distributed and locally meaningful instead of being concentrated at one stable position.

The first implementation deliberately does not combine CTI with fixed rounded-rectangle corner pruning. CTI is evaluated on the normal full patch grid so the intervention has a clear causal interpretation.

## 3. Forward paths

Let `T_l` be the patch-token sequence after ViT block `l`, with the CLS token kept separately. For one training image:

1. Run the ordinary full-token prefix to intervention layer `l`.
2. Continue the full sequence through the remaining blocks and obtain ordinary features/logits.
3. From the pose-derived human token set `H`, sample exactly `K` distinct patch-token indices without replacement.
4. Physically gather the complement of those tokens, keep CLS, and run the shortened sequence through the same remaining blocks.
5. Compute the true-ID evidence for both paths.

The sampled indices are redrawn on every call. They are sorted only for the gather operation so the surviving token order is unchanged.

If an asset contains fewer than `K` valid human tokens, that sample is excluded from CTI loss but remains in the original C3 batch. It is never padded by deleting background tokens.

## 4. Objective

For classifier logits `s` and ground-truth identity `y`, define true-ID evidence as:

```text
G(s, y) = s_y - logsumexp({s_j | j != y})
```

The counterfactual margin loss is:

```text
L_CTI = mean_valid max(0, margin - G_full + G_deleted)
L_total = L_C3 + lambda(epoch) * L_CTI
```

`L_C3` is the existing, unmodified sum of ID, triplet, MSEL and DCL terms selected by the current recipe. CTI is additive. A delayed start and linear weight ramp are used so early training first establishes the original C3 representation:

```text
lambda(epoch) = 0                                           epoch < start
              = lambda_max * progress                       during warm-up
              = lambda_max                                  afterwards
```

Counterfactual classifier scoring must not update BatchNorm running statistics; only the full path retains the existing classifier behavior.

## 5. Pose-token asset contract

The loader consumes exported anatomy files rather than importing the PACT plugin at runtime:

```text
<cti_anatomy_root>/<dataset>/anatomy/<source_key>.npz
```

Each file contains `token_masks` with shape `[P, H_tokens, W_tokens]`. The human mask is the union/max over valid anatomical parts. The expected initial grid is `42 x 21` for 512 x 256 input.

The token mask follows the exact horizontal flip applied independently to each image view. RGB-original, RGB-augmented and IR views therefore receive separate synchronized masks. Photometric transforms and random erasing change only pixels, not token membership.

The current canonical `person-assets-512x256/person_fit` manifest does not retain enough keypoint coordinates to reconstruct these masks directly. Before launching a full experiment, pose/anatomy assets must be regenerated for the canonical source keys. Missing or malformed masks are a configuration/data error; silently treating the whole image as human is forbidden.

## 6. Initial configuration

The implementation exposes:

```yaml
cti_enabled: true
cti_anatomy_root: /path/to/exported/pact-assets
cti_intervention_layer: 9
cti_delete_count: 32
cti_human_threshold: 0.05
cti_margin: 0.5
cti_weight: 0.2
cti_start_epoch: 6
cti_warmup_epochs: 3
```

Initial values are experiment defaults, not claims of optimality. The implementation rejects CTI together with fixed token pruning and validates layer/count/threshold/weight ranges.

## 7. Invariants

- CTI disabled: model outputs and C3 losses are unchanged.
- Evaluation/inference: only the ordinary full-token path runs.
- Exactly `K` distinct human tokens are deleted for every valid CTI sample.
- CLS is never deleted.
- Surviving patch tokens keep their original order and positional information.
- Full and deleted suffixes share all ViT parameters.
- Counterfactual scoring does not mutate classifier BN statistics.
- Samples invalid for CTI still contribute normally to C3.
- Pose masks are synchronized with each view's horizontal flip.

## 8. Quality gates

Unit tests must cover:

1. fixed-count deletion, uniqueness, human-region membership and CLS preservation;
2. per-sample physical sequence shortening and preserved survivor order;
3. invalid samples with fewer than `K` candidates;
4. CTI margin/evidence calculation against a hand-computed example;
5. no BN running-stat mutation during counterfactual scoring;
6. horizontal-flip synchronization for pose masks;
7. config rejection for incompatible token pruning or invalid CTI values;
8. CTI-off regression: the original training path and losses are unchanged.

A smoke test should run one forward/backward step with finite gradients through both suffix paths. A real experiment is launchable only after an asset audit confirms complete source-key coverage and the expected token-grid shape.

## 9. Practical interpretation

CTI tests whether pose-selected human tokens carry identity evidence: removing randomly chosen human evidence should reduce the correct-ID advantage. It does not prove that each deleted token is individually causal, and it does not require a background-token control for the first experiment. A later ablation may add matched random background deletion to separate generic token-count effects from human-region effects, but that is outside this first implementation.
