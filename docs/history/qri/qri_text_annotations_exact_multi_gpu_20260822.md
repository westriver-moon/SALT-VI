# Exact per-image Qwen annotation on three RTX 3090 GPUs

Date: 2026-08-22

## Semantics

`exact` means that every source image creates one independent Qwen 3.8 vision
request. No caption, probability, or regional hypothesis is propagated from
another frame. Every complete record has `direct_vlm=true`,
`semantic_scope=source_image`, and `roi_geometry_scope=source_image`.

The accelerated profile keeps the authoritative LR image, precomputed SwinIR
proposal, three four-tile ROI boards, global caption, observations, world
knowledge, two or three mutually exclusive hypotheses, probabilities, and
joint text-world sampling. Qwen returns compact JSON aliases; the plugin
expands them to the canonical persisted schema.

The former 12-SwinIR-TTA ranking returned exactly zero for every ROI on the
reference eyeglasses frame. The production exact profile therefore reuses the
canonical per-image SwinIR result, protects the eye ROI, and selects the other
two ROIs by regional blur. This removes 10.5 seconds per sample on the measured
frame without discarding a working uncertainty signal.

## Measured smoke results

Legacy detailed exact, RGB `cam1/0001/0001.jpg`:

- Qwen: 157.61 seconds;
- completion: 1,752 tokens;
- eyeglasses: 0.85.

Compact exact with full TTA, same image:

- pipeline: 81.84 seconds;
- Qwen: 63.74 seconds;
- completion: 810 tokens;
- eyeglasses: 0.70.

Two-replica fast exact smoke on GPUs 0 and 3:

- RGB `cam1/0001/0001.jpg`: 63.75 pipeline seconds, eyeglasses 0.60;
- IR `cam3/0001/0001.jpg`: 36.38 pipeline seconds;
- both records complete, independent, and `direct_vlm=true`;
- no visible-spectrum color was asserted for the IR record;
- both workers returned zero and wrote distinct shard manifests.

Using the slower RGB timing as a conservative constant, 44,745 images require
about 33 GPU-days. Three independent 3090 replicas reduce this to about 11
continuous days. A 30-60 image pilot should replace this two-image estimate
with a latency distribution before the production launch.

## Production commands

The wrapper supplies the verified qri-v1 Python/CUDA library environment and
refuses GPUs using more than 1 GiB unless explicitly overridden.

Dry run:

```bash
/home/lab929/ybj/SALT-VI/scripts/experiments/qri_text_annotations/run_exact_multi_gpu_3090.sh \
  --gpu-ids 0,1,2 --split train --dry-run
```

Production train pass:

```bash
/home/lab929/ybj/SALT-VI/scripts/experiments/qri_text_annotations/run_exact_multi_gpu_3090.sh \
  --gpu-ids 0,1,2 --split train
```

Production evaluation pass:

```bash
/home/lab929/ybj/SALT-VI/scripts/experiments/qri_text_annotations/run_exact_multi_gpu_3090.sh \
  --gpu-ids 0,1,2 --split evaluation
```

Complete records are reused on rerun. Failed records are retried. Servers are
started with offline mode and stopped automatically after every worker exits.
The production full pass was not launched during this implementation.
