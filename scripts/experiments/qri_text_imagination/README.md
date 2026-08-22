# QRI text-only imagination

This experiment retains enlarged ROI reinspection and probability-sampled
semantic worlds while removing every image-generation and image-writeback step.

The VLM response separates authoritative observations, world-knowledge
abstractions, mutually exclusive hypotheses, and unresolved details. Candidate
probabilities are normalized and sampled locally; the runner never inserts a
generic positive candidate that the VLM did not return.

Run the unit tests:

```bash
python -m pytest -q \
  scripts/experiments/qri_text_imagination/test_text_only_planner.py
```

Run the matched thinking ablation after starting the existing local Qwen
llama.cpp server on physical GPU0:

```bash
python scripts/experiments/qri_text_imagination/text_only_planner.py \
  --config configs/experiments/qri_text_imagination/thinking_ablation_gpu0.yaml
```

Large ROI boards and JSON records are written below
`/home/lab929/ybj/experiments/qri_text_imagination/`, not into the repository.

The measured production default is `no_thinking`: it already preserves the
world-knowledge and mutually-exclusive-hypothesis instructions, while the
matched GPU0 ablation was 3.40x faster on average. Reserve high-effort thinking
for selected ambiguous ROIs where probability calibration is worth the extra
latency; do not use it merely to obtain longer captions.
