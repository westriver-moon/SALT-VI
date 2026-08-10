# Reproduction configuration archive

This directory contains configurations retained as experiment provenance. They are not current training entry points.

## `stage_b_288_no_matching_stage_a/`

Ten historical 288x144 Stage-B configurations. Their original image-only Stage-A warm starts were pruned during checkpoint consolidation. The retained canonical A3 checkpoint is a 512x256 multi-branch model and is deliberately rejected by the topology guard. Historical metrics, raw logs, and any retained Stage-B weights remain indexed in `reports/experiment_registry/experiment_registry.csv`.

## `super_resolution_legacy_source_core/`

A legacy 288x144 source-core super-resolution configuration. Its Stage-B warm-start file was pruned, so this file is evidence only.

Other files at this level are prior historical YAML snapshots. A configuration may be promoted back to an active tree only after its exact initialization checkpoint and data contract are restored and a GPU smoke run succeeds.

## Recent completed runs (2026-08-10)

The `*_20260809.yaml` files are immutable provenance snapshots for the completed
original-config reproduction, low-learning-rate continuation, and IR-to-RGB+Text
Fusion-84 experiments. Their resolved configurations, source snapshots, logs,
metrics, and retained best-Rank-1 checkpoints are indexed by
`reports/experiment_registry/experiment_registry.csv`.
