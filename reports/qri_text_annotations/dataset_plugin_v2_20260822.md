# Qwen 3.8 dataset text annotation plugin v2

Date: 2026-08-22
Host profile: `lab929-3090`
Physical GPU: NVIDIA RTX 3090 GPU0
Repository: `/home/lab929/ybj/SALT-VI`

## Decision

The plugin now supports the requested offline-first order:

1. traverse the complete SYSU-MM01 train and evaluation image population;
2. produce source-specific human ROI geometry and uncertainty metadata;
3. use Qwen 3.8 to produce a global person description plus probabilistic
   descriptions for the selected Top-3 semantic ROIs;
4. save all annotation inputs before any training-time caption composition;
5. leave probability sampling, caption composition and CLIP tokenization to a
   separate downstream text-only stage.

The annotation plugin contains no PASD or diffusion dependency and writes no
candidate images.

## Two execution strategies

`track_anchor` is the default dataset strategy. Every source image receives an
independent ROI prescan using the canonical precomputed SwinIR result, fast
SwinIR-versus-bicubic high-frequency disagreement, and regional blur. Images
are grouped by `(split, camera, identity)`. One representative frame is chosen
from each group and sent to Qwen 3.8. The resulting global and regional
probability annotation is propagated to every source record with explicit
`anchor_source_key`, `direct_vlm`, `semantic_scope`, and
`roi_geometry_scope` provenance.

`exact` retains the diagnostic path: 12 SwinIR TTA restorations and one Qwen
3.8 request for every source image. It remains useful for quality spot checks,
but is not the recommended full-dataset path.

## Dataset scale

| Population | Images | Camera/identity tracks |
|---|---:|---:|
| Train | 34,167 | 1,753 |
| Evaluation | 10,578 | 490 |
| Total | 44,745 | 2,243 |

The mean track contains about 20 images, so track anchoring reduces Qwen calls
from 44,745 to 2,243 without replacing Qwen 3.8 with a smaller visual model.

## Verification

Automated tests:

- 9 text-annotation tests;
- 20 existing regional and plugin-registry regression tests;
- 29/29 total passed;
- Python compilation and `git diff --check` passed.

Exact-mode reference sample (`cam1/0001/0001.jpg`):

- Qwen elapsed: 157.61 s;
- completion: 1,752 tokens;
- 13 ROI candidates and exactly three selected regions;
- each regional probability sum: 1.0;
- no persisted reasoning content and no image output.

Track-anchor RGB smoke (`cam1/0001`, 29 frames):

- 29 complete records, zero failures;
- 29 distinct per-source ROI geometries;
- one direct Qwen 3.8 request, anchor `cam1/0001/0023.jpg`;
- Qwen elapsed: 208.64 s, 1,621 completion tokens;
- end-to-end elapsed: 310.70 s;
- selected regions: `left_pocket`, `upper_torso`, `right_pocket`;
- all regional probability sums: 1.0;
- full cached rerun: 1.56 s, zero Qwen requests.

Track-anchor IR smoke (`cam3/0001`, 20 frames):

- 20 complete records, zero failures;
- 20 distinct per-source ROI geometries;
- one direct Qwen 3.8 request, anchor `cam3/0001/0014.jpg`;
- Qwen elapsed: 194.69 s, 1,461 completion tokens;
- end-to-end elapsed: 271.77 s;
- no visible-spectrum hue term in the global or regional annotation;
- all regional probability sums: 1.0.

Prescans are atomically cached below `prescan/`. If a Qwen request fails, the
next run reuses every completed source prescan and reconstructs only the
representative frame needed by Qwen.

## Single-GPU feasibility estimate

The exact strategy would require about 81.6 GPU-days for Qwen calls alone at
the measured exact-sample latency.

Using the two track smoke runs, the default strategy is estimated at:

- about 1.9 days for 44,745 per-image ROI prescans;
- about 5.2 days for 2,243 Qwen 3.8 track requests;
- about 7.1 continuous GPU0 days in total.

This is an engineering estimate from two tracks, not a guaranteed completion
time. Before the full launch, a 20-50-track pilot should be used to estimate
the latency distribution and failure rate. Deterministic track sharding and
resume are already implemented.

An eight-shard audit produced 280-281 tracks and 5,500-5,789 images per shard.
The measured metadata and prescan sizes project to about 1.25 GiB of JSON for
the full dataset, excluding logs.

## Production commands

Resolve the paths in `plugins/qwen_imagination/runtime_paths.example.env`, set
a production output root, and start the configured Qwen 3.8 server. Then:

```bash
salt-qwen-text-annotation \
  --config plugins/qwen_imagination/configs/text_annotation_sysu_v1.yaml \
  preflight

salt-qwen-text-annotation \
  --config plugins/qwen_imagination/configs/text_annotation_sysu_v1.yaml \
  run --split train --num-shards 1 --shard-index 0 --device cuda:0

salt-qwen-text-annotation \
  --config plugins/qwen_imagination/configs/text_annotation_sysu_v1.yaml \
  run --split evaluation --num-shards 1 --shard-index 0 --device cuda:0
```

Re-running either command resumes from complete records. In track-anchor mode,
`--limit N` means N complete camera/identity tracks. Caption assembly is not
part of these commands.
