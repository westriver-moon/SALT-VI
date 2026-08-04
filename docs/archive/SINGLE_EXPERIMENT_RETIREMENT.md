# Single-experiment Retirement

`Single-experiment/` was removed from the active branch because it duplicated
the TVI-LFM implementation, carried a conflicting Python/PyTorch environment,
and contributed 719 tracked files (about 195 MiB) to normal searches and
checkouts. No active source or automation outside that directory imported it.

The deletion is recoverable. The last retained tree is
`dfc773505eca3d53ec826e6a0ff31ed7ca724c56` in commit
`9208ee3ef1e505bbe498503d39539b83de688747`.

Inspect it without restoring it to the active branch:

```bash
git ls-tree -r 9208ee3e -- Single-experiment
git show 9208ee3e:Single-experiment/README.md
```

If a historical reproduction genuinely requires the full tree, create a
separate worktree or branch from that commit. Do not copy the legacy source
back under the current main worktree, because its Python 3.10/PyTorch 2.0.1
environment conflicts with the server-supported TVI-LFM Python 3.8/PyTorch
1.8.1 stack.

The former nested SCHP repository provenance remains recorded in the Git
history and in the external backup described by `GIT_PROVENANCE.md`.

## Server runtime assets

On 2026-07-24, the 67 remaining untracked runtime assets (about 2.5 GiB) were
moved atomically out of the searchable source tree to:

```text
/home/cgv841/ybj/archives/Single-experiment-retired-20260724/
```

This archive includes experiment checkpoints, the SCHP pretrained weight, and
generated text features. It was not deleted and can be moved back if a
historical reproduction needs it. The `archives/` directory is intentionally
ignored by Git.
