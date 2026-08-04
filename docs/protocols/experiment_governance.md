# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: experiment archive policy

> Source document ID: `source_core:docs/experiment_archive_policy.md`  
> Original SHA-256: `7428446bf4285c87447d971474de58acd6f83baac60a83e25fdda42eb54afada`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Experiment archive policy

This policy applies to experiment reports committed under `reports/`.

## Required semantics

- Terminal run statuses use `succeeded`, `failed`, `stopped_by_user`, `cancelled`, `blocked`, or `skipped`.
- A user-authorized signal stop is `stopped_by_user`, not `failed`, and requires an explicit `user_stop.json` marker for future automated runs.
- Selection archives partition runs into completed, reused, and excluded sets. The sets must be disjoint and explain every candidate.
- Multi-metric results report Rank-1, mAP, and mINP from the same selected epoch. Metric-specific optima may be recorded separately.
- Equal reported metrics do not imply equal model binaries. Checkpoint SHA-256 values define binary identity.
- A pruned checkpoint has `checkpoint: null` plus `checkpoint_original_path` and `checkpoint_sha256`. It must record either `checkpoint_archive_blob`, or `checkpoint_disposition: deleted_without_archive` together with a `checkpoint_prune_ledger`.
- Registry rows use a non-empty `checkpoint` only for a server-side file that is currently retained. Metrics-only rows whose binaries were pruned leave `checkpoint` empty and explain the disposition in `notes`.
- Historical cleanup manifests may reference deleted paths when the entry explicitly records a delete/prune action. Current-state summaries and fields named `latest` or `verified` must distinguish immutable snapshot state from live filesystem state.

## Compact Git archive

Keep one group-level dataset fingerprint and environment description when runs share them. Each run manifest references those files and retains its runtime config, config diff, command, source state, metric events, and artifact hashes.

Do not commit empty `code.patch` files, repeated full `configs.yaml` snapshots, `vis_logs`, PID/lock files, feature caches, or full per-trial feature-analysis exports. Large raw analysis files belong in an external server archive with a committed SHA-256 inventory. Git retains a compact summary, result tables, provenance, and selected figures.

## Statistical reporting

`multi_kernel_mmd2` is an unbiased signed estimator. Report raw signed MMD² and its bootstrap interval. Do not divide it by a near-zero or negative internal estimate; such an MMD ratio is unstable and has no supported interpretation.

## Validation

Run from the SALT-VI directory:

```bash
python scripts/validate_experiment_archives.py
pytest -q
```


---

## Migrated source: experiment logging curve contract

> Source document ID: `source_core:docs/experiment-logging-curve-contract.md`  
> Original SHA-256: `8ead0966f67cbb097d7716480ee8bd00949c2bd72849922a1a5778e32e6ce174`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SALT-VI Experiment Logging and Curve Contract

Version: 1

This contract applies to future experiments launched below
`/home/cgv841/ybj`, including runs executed from registered git worktrees. Its
purpose is to make loss and retrieval-metric curves reproducible without
guessing experiment boundaries or scraping arbitrary processes.

## Experiment boundary

Every scheduler or experiment batch must publish an explicit plan containing
the complete set of `experiment_id` values it owns. "All experiments are
finished" means every ID in that plan is terminal. It does not mean that no
a legacy training process exists anywhere under `/home/cgv841/ybj`.

The plot watcher must never recursively treat every historical log in `ybj` as
one batch. DataLoader workers, another scheduler, archived runs, and other
users' jobs are outside the plan unless explicitly listed.

## Canonical paths

New SALT-VI metric-boost runs use:

```text
logs/raw/source_core/reports/metric_boost/runs/<experiment-id>/
  manifest.json
  status.json
  events.jsonl
  launcher.log
  runtime_config.yaml
  model_output/
```

Batch-level curve outputs use:

```text
logs/raw/source_core/reports/metric_boost/curves/<plan-id>/
  parsed/
  per_experiment/
  comparisons/
  summary.csv
  summary.md
  report-status.json
```

Historical `logs/raw/source_core/logs/**/logs/log.log` and current `launcher.log` remain valid human
logs, but new producers should also emit `events.jsonl`.

## Manifest requirements

`manifest.json` is immutable after launch and contains:

```json
{
  "logging_contract_version": 1,
  "experiment_id": "IMTA-M2-relation",
  "plan_id": "imta-20260713",
  "stage": "IMTA-1",
  "config_path": "configs/metric_boost/imta_experiments.yaml",
  "runtime_config": "logs/raw/source_core/reports/metric_boost/runs/IMTA-M2-relation/runtime_config.yaml",
  "git_commit_sha": "...",
  "dataset": "sysu",
  "protocol": "all-search-single-shot-10-trial",
  "seed": 0,
  "max_epoch": 33,
  "epoch_index_origin": 0,
  "planned_metric_names": ["Rank-1", "mAP", "mINP"],
  "planned_loss_names": ["id_loss", "wrt_loss", "imta_proto_loss"],
  "selection_validity": "standard"
}
```

## Status requirements

`status.json` is updated atomically and includes `experiment_id`, `status`,
`updated_at`, `start_time`, `end_time`, `runner_pid`, `gpu`,
`last_completed_epoch`, `return_code`, `error`, final metrics, `best_epoch`,
and `checkpoint` when available.

Allowed transitions:

```text
pending -> running -> completed_pending_summary -> succeeded
                   \-> failed
pending/running     \-> blocked | cancelled | skipped
```

Terminal states are `succeeded`, `skipped`, `failed`, `blocked`, and
`cancelled`. `completed_pending_summary` is not terminal. A final plot is
authorized only after every ID in the curve plan is terminal and its scheduler
children have exited.

## Structured events

`events.jsonl` is UTF-8, append-only, flushed after each line, and never
rewritten. Every line includes `schema_version`, unique `event_id`,
`experiment_id`, `event_type`, `timestamp`, and `attempt`. Epoch events also
include `epoch`.

### Training event

```json
{
  "schema_version": 1,
  "event_id": "IMTA-M2-relation:train:2:1",
  "experiment_id": "IMTA-M2-relation",
  "event_type": "train_epoch",
  "timestamp": "2026-07-13T03:20:00+08:00",
  "attempt": 1,
  "epoch": 2,
  "losses": {
    "total_loss": 0.31,
    "id_loss": 0.08,
    "wrt_loss": 0.16,
    "imta_proto_loss": 0.04,
    "imta_dual_loss": 0.02,
    "imta_rel_loss": 0.01
  },
  "scalars": {
    "accuracy": 0.98,
    "learning_rate": 0.00001,
    "temperature": 0.07
  },
  "duration_seconds": 590.2,
  "amp_skipped_steps": 0
}
```

The `losses` and `scalars` objects are open mappings. Future losses require no
plotter code change. Non-finite and unavailable values are omitted, never
written as NaN or substituted with zero.

### Evaluation event

```json
{
  "schema_version": 1,
  "event_id": "IMTA-M2-relation:eval:2:1",
  "experiment_id": "IMTA-M2-relation",
  "event_type": "eval_epoch",
  "timestamp": "2026-07-13T03:24:00+08:00",
  "attempt": 1,
  "epoch": 2,
  "dataset": "sysu",
  "protocol": "all-search-single-shot-10-trial",
  "test_mode": "Fusion_RGB",
  "metrics": {"Rank-1": 0.8219, "mAP": 0.7932, "mINP": 0.6849},
  "best_so_far": {"Rank-1": 0.8219, "mAP": 0.7932, "mINP": 0.6849},
  "is_new_best": true
}
```

`metrics` is always the raw result at that epoch. `best_so_far` is cumulative
state. Curve code must not plot `best_so_far` as if it were the raw series.
Stored metrics are fractions in `[0, 1]`; percentage formatting is a report
concern.

Other supported events are `run_started`, `checkpoint_saved`,
`amp_overflow`, and `run_finished`. Checkpoint events record path, epoch,
SHA-256, selection metric, and `is_best`.

## Legacy SALT-VI parser

Until all trainers emit JSONL, the watcher may parse the current text form:

```text
Time: ...; Epoch: N; <name>_loss: V; total_loss: V; acc: V;
Time: ...; Dataset: sysu, Test Mode: Fusion_RGB,
mINP: V
mAP: V
Rank: [R1 ...]
```

The evaluation block belongs to the most recently completed epoch. A line
starting with `Best Fusion_RGB` is cumulative best state, not the current raw
measurement. The adapter must report truncated blocks, duplicate epochs,
unknown tokens, and parse failures in `report-status.json`.

## Curve plan and completion

A curve plan declares:

- stable `plan_id`
- exact experiment IDs
- baseline ID or fixed baseline metrics
- status and log paths
- poll interval and output root
- whether preview rendering is enabled

The watcher uses a singleton file lock, performs CPU-only work, never acquires
GPU leases, never changes training status, and writes reports atomically. It
may refresh previews while training is active. It sets `finalized: true` only
after the plan is terminal.

## Required outputs

For each experiment:

- raw total-loss curve
- dynamically discovered component-loss curves
- training accuracy and learning rate when present
- raw Rank-1, mAP, and mINP against evaluation epoch
- optional cumulative-best curves shown separately
- best-epoch annotation

For each plan:

- comparable experiment overlays
- E4 or declared baseline horizontal lines
- `summary.csv` with status, best epoch, final/best metrics, duration, GPU,
  checkpoint, and parse warnings
- `summary.md` describing missing data, failures, smoothing, and validity

Missing observations create gaps. Smoothing never replaces the unsmoothed
series. Experiments with different protocols are not drawn on one comparison
axis unless the distinction is explicit.

## Scientific validity

Plotting and summarization do not change selection validity. Test-set-tuned
MER, re-ranking, or hyperparameter searches remain marked exploratory. Failed
and cancelled runs remain visible in the summary and are not silently dropped.
