# Centralized Qwen imagination plugins

This directory is the sole source location for versioned Qwen imagination
implementations used by SALT-VI. SALT training code calls the stable bridge in
src/salt_vi/imagination.py and does not import QRI internals directly.

The qwen_imagination/api.py file defines the request and result contract.
The registry selects qri-v1, qri-v2, or a future version lazily.
The versions directory contains one adapter per version.
The regional directory contains shared regional engine code.
The configs and tests remain inside this plugin area.

Model weights, third-party source repositories, and experiment outputs are
runtime assets, not plugin source. They remain outside this directory.

The existing QRI checkout remains a migration baseline until both versions
pass the contract and regional tests.

## Dataset-scale text annotation

`salt-qwen-text-annotation` is the image-generation-free annotation mode. It
reuses the regional plugin's SYSU-MM01 source traversal, human ROI stack,
SwinIR comparison data and four-tile ROI comparison board. The default
`track_anchor` strategy performs source-specific ROI geometry and fast
SwinIR-residual/blur scoring for every image, selects a representative frame
for each `(split, camera, identity)` track, and makes one Qwen 3.8 request per
track for the global caption and selected Top-3 regional annotations. Every
source record states its anchor and whether it was directly seen by Qwen.

`--strategy exact` retains the expensive diagnostic mode: 12-view SwinIR
instability and one Qwen request per source. It is useful for spot checks, not
the full SYSU-MM01 annotation run. Qwen is always the configured Qwen 3.8
vision model; a smaller text-only LLM is intentionally deferred to the later
caption-composition stage.

Each source record contains:

- one evidence-grounded global person description;
- every candidate ROI and its uncertainty score;
- Top-3 regional observations, world knowledge, mutually exclusive hypotheses
  and normalized probabilities;
- deterministic joint text-world samples;
- latency and token telemetry without persisted chain of thought.

It never imports PASD or diffusers and does not materialize candidate images.
Records are written atomically below one output root:

```text
<output_root>/
├── metadata/<camera>/<identity>/<image>.json
├── prescan/<camera>/<identity>/<image>.json
└── manifests/
    ├── <split>.shard-<index>-of-<count>.jsonl
    └── <split>.shard-<index>-of-<count>.summary.json
```

Resolve the environment variables used by
`configs/text_annotation_sysu_v1.yaml`, start the existing remote Qwen server,
and preflight before a run:

```bash
salt-qwen-text-annotation \
  --config plugins/qwen_imagination/configs/text_annotation_sysu_v1.yaml \
  preflight
```

Run one deterministic shard, or use `--num-shards N` with distinct
`--shard-index` values for parallel workers. Track-anchor sharding keeps every
camera/identity track on one worker; `--limit` counts tracks in this mode and
individual sources in exact mode:

```bash
salt-qwen-text-annotation \
  --config plugins/qwen_imagination/configs/text_annotation_sysu_v1.yaml \
  run --split all --num-shards 8 --shard-index 0
```

Completed records with the same run signature are reused. Failed records are
retained with their exception and retried on the next run. Per-image prescans
are cached separately, so a failed Qwen request only rebuilds its representative
frame. `--overwrite` explicitly regenerates completed records. Caption
compilation, probability sampling for training, and CLIP tokenization are
deliberately downstream concerns; this mode preserves the complete global and
regional annotation first.
