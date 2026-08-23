# Autoresearch diagnostic archive — 2026-08-14/15

This archive closes the unregistered Stage-B autoresearch attempts found while
checking all SALT-VI branches and worktrees on 2026-08-20. No run directory,
checkpoint, log, target snapshot, provenance file, or result file was deleted
or moved. The original controller run roots remain under
`/home/lab929/ybj/autoresearch-v2/runs/`.

## Registered records

| Experiment | Status | Primary result | Retained evidence |
|---|---|---:|---|
| `SALTVI-STAGEB-CAPTION-IDENTITY-RANK-20260814` | result present; controller baseline promotion failed | identity-balanced Rank-1 `0.506256935976` | `stage-b-caption-identity-rank-20260814/artifacts/w1/iter-0002/` |
| `SALTVI-STAGEB-TEXT-TO-RGB-V2-20260814` | result present; controller baseline promotion failed | Rank-1 `0.50007886` | `stage-b-best-text-to-image-test-v2-20260814/artifacts/w1/iter-0001/` |
| `SALTVI-STAGEB-TEXT-TO-RGB-20260814` | failed before execution | — | `stage-b-best-text-to-image-test-20260814/state.json`, target, program, outcome, runner log |
| `SALTVI-STAGEB-RN50-PASD-R-TEXT-VISUAL-30-20260815-FAILED` | failed before execution | — | `stage-b-rn50-pasd-r-text-visual-30-20260815/state.json`, target, program, outcome, runner log |

The two result-bearing rows are diagnostic evidence only. They are not
promoted to the formal training leaderboard because the autoresearch driver
did not complete baseline promotion. Their structured `metrics.json` files
remain the sole metric source.

## Scope checks

- Existing Stage-A safe-trick, Stage-A structural-trick, PASD-x4, E36 and QRI
  archive manifests were already represented in
  `reports/experiment_registry/experiment_registry.csv`.
- The `semantic-imagination-qwen36-int4-20260814` directory contains only a
  controller lock and no state, target result or provenance; it is not an
  executed experiment and was not fabricated into the registry.
- `salt-readonly-framework-20260810` and `doctor` are framework/access checks,
  not research experiments.

## Retention rule

Keep the four original autoresearch run roots, including `state.json`,
`spec/program.md`, `spec/target.yaml`, `outcome.json`, `provenance.json` when
present, `metrics.json` when present, detailed diagnostic JSON when present,
and runner/process logs. No checkpoint copy was created and no SHA-256 was
recomputed by this archival pass.
