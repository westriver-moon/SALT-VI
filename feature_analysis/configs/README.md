# Analysis configuration

Each YAML file describes one isolated run. Multiple checkpoints may be listed in
one file so that identical sample IDs can be compared directly.

Important fields:

- `run_id`: filesystem-safe unique run name;
- `models`: SALT config/checkpoint pairs plus optional top-level config overrides;
- `splits`: exhaustive query/gallery/train selection;
- `representations`: vectors to extract for each split;
- `comparisons`: explicit artifact pairs for protocol or modality analysis;
- `representations[].models`: optional model-ID filter for protocol-specific vectors;
- `analysis.auto_checkpoint_comparisons`: compare the same representation across
  all listed checkpoints automatically.

Artifact selectors use `model_id::split_tag::representation`, for example
`old_model::query::protocol_query` or
`new_model::gallery_trial_00::protocol_gallery`.
