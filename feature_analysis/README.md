# SALT-VI Feature Analysis

This directory is an isolated, reusable feature-analysis subsystem for SALT-VI.
It does not modify training code and does not assume one particular checkpoint or
retrieval protocol.

## Directory contract

```text
feature_analysis/
├── configs/                 # Versioned analysis specifications
├── scripts/                 # Stable command-line entry point
├── src/salt_feature_analysis/
│   ├── config.py            # Schema validation and path resolution
│   ├── salt_adapter.py      # SALT model, checkpoint, and dataset adapter
│   ├── extraction.py        # Exhaustive feature extraction
│   ├── storage.py           # Safe NPZ/JSON artifact format
│   ├── statistics.py        # Geometry and drift measurements
│   ├── plotting.py          # Optional PNG figures
│   ├── reporting.py         # CSV/Markdown report generation
│   └── cli.py               # validate/extract/analyze/all commands
├── tests/                   # Synthetic, checkpoint-free unit tests
└── artifacts/
    ├── features/            # Compressed feature matrices and metadata
    ├── tables/              # Machine-readable JSON/CSV statistics
    ├── figures/             # PNG plots only
    ├── reports/             # Human-readable Markdown conclusions
    └── manifests/           # Reproducibility catalog and hashes
```

Generated artifacts are ignored by Git. Each run is isolated by `run_id`; the
tool refuses to overwrite an existing feature file unless `overwrite: true` is
explicitly set.

## Supported samples

- every SYSU query sample;
- every gallery sample in any or all ten gallery trials;
- every unique training RGB sample;
- every unique training IR sample;
- every configured PASD training view when multiview storage is used.

## Supported representations

- `image`: image-only feature (`rgb` or `ir`);
- `text`: caption-only feature;
- `fusion`: image-caption feature (`rgb` or `ir` fusion);
- `protocol_query`: the checkpoint config's actual query representation;
- `protocol_gallery`: the checkpoint config's actual gallery representation.

Representations can be captured before BN (`pre_bn`) or after the SALT
classifier BN and L2 normalization (`post_bn`). Protocol encoders always use the
canonical evaluation path and therefore produce post-BN features.

## Commands

Run from the SALT-VI repository root with the training environment:

```bash
/home/cgv841/anaconda3/envs/clipreid/bin/python \
  feature_analysis/scripts/run.py validate \
  --config feature_analysis/configs/example_all_samples.yaml

/home/cgv841/anaconda3/envs/clipreid/bin/python \
  feature_analysis/scripts/run.py extract \
  --config feature_analysis/configs/my_analysis.yaml

/home/cgv841/anaconda3/envs/clipreid/bin/python \
  feature_analysis/scripts/run.py analyze \
  --config feature_analysis/configs/my_analysis.yaml

# extract followed by analyze
/home/cgv841/anaconda3/envs/clipreid/bin/python \
  feature_analysis/scripts/run.py all \
  --config feature_analysis/configs/my_analysis.yaml
```

`matplotlib` is optional. Without it, feature extraction and all numeric tables
still work; only PNG generation is skipped with a recorded warning.

## Reproducibility guarantees

Every feature artifact records the resolved training config, checkpoint path and
SHA-256, representation definition, sample identifiers, labels, cameras, feature
dimension, and extraction seed. NPZ files contain arrays only and are loaded with
`allow_pickle=False`.

The analysis reports include feature norms, finite-value checks, anisotropy,
within/between-identity cosine similarity, centroid compactness, nearest-centroid
accuracy, covariance effective rank, checkpoint drift, linear CKA, orthogonal
Procrustes residual, and label-centroid alignment.

To evaluate one checkpoint under a protocol different from the config it was
trained with, list the same checkpoint again under another `model.id` and use
`models[].overrides` to change `retrieval_backend`, `test_modality`, and caption
manifest settings. This keeps weight drift separate from protocol drift.
