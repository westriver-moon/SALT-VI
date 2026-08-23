# SALT-VI 当前状态

最后核验：2026-08-24（Asia/Shanghai），主机 `cgv841-SYS-7049GP-TRT`。

## 代码与工作树

- 主仓库分支：`codex/pmt-mscm-phased-pasd-20260821-v2`，本轮清理基线为 `6b60bee5`；
- 本次提交整合 QRI V4 经验原子采样改造，以及文档集中化、旧代码归档和缓存清理记录；
- 已登记工作树 `.codex-worktrees/stageb-c3-safe-tricks-20260823` 未移动或改写；
- 新增活动工作树 `.codex-worktrees/stageb-b5-tricks-grid-20260824` 明确排除于清理范围。

## 运行状态

- 最终核验时，Stage-B B5 tricks grid 调度器正在运行，`camera_only` 与 `cosine_only` 已启动，其余任务排队；
- 该活动调度器、两个训练任务、输出目录和新工作树均未被清理或测试改写；
- 一个从 2026-08-17 遗留、无训练子进程且匹配表达式错误的 watcher 已在本轮停止；
- 其他账号和 `/home/lab929/ybj` 之外的训练/推理任务不属于本次清理范围。

## 结果与证据入口

- 2026-08-21/22 实验结果快照：[experiment_status_20260822.md](experiment_status_20260822.md)；
- 唯一结构化总表：`reports/experiment_registry/experiment_registry.csv`；
- 文档总入口：[`../README.md`](../README.md)。

除非新增实验经过完成门禁，不应把历史快照中的 running/deferred 字段解释为实时进程状态。
