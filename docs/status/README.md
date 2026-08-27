# SALT-VI 当前状态

QRI-v6 机制更新：2026-08-26（Asia/Shanghai），主机 `cgv841-SYS-7049GP-TRT`。
下述运行状态保留 2026-08-24 快照，不代表实时进程状态。

## 代码与工作树

- 当前仓库为 `/home/lab929/ybj/SALT-VI`，所在分支为
  `codex/pmt-mscm-phased-pasd-20260821-v2`；
- 2026-08-26 的均匀候选机制直接修改现有 v6，未创建或切换分支、worktree；
- 版本名和 schema 保持 `qri-v6` / 6；历史 V4/V5 产物不重标为 v6。

## 运行状态

- 最终核验时，Stage-B B5 tricks grid 调度器以及 `camera_only`、`cosine_only` 训练进程正在运行；
- 该活动调度器、两个训练任务、输出目录和新工作树均未被清理或测试改写；
- 一个从 2026-08-17 遗留、无训练子进程且匹配表达式错误的 watcher 已在本轮停止；
- 其他账号和 `/home/lab929/ybj` 之外的训练/推理任务不属于本次清理范围。

## QRI-v6 状态

- 当前入口：`plugins/qwen_imagination/versions/qri-v6/plugin.yaml`；
- 当前机制：VLM 一次生成联合候选、结构检查、complete-link 语义去重、
  全部代表世界以 `1/K` 等权输出并逐个 LLM 改写；VLM、编码器和 LLM 仅通过接口注入；
- 默认一次请求 8 个候选；移除重复抽样、簇频率、频率 Top-K 与 Wilson 区间；
- V2/V5 text-annotation 配置已移入 `plugins/qwen_imagination/configs/legacy/`；
- V6 尚未绑定真实模型或运行 SYSU 生产数据，不存在模型效果或 ReID 收益结论。

## 结果与证据入口

- 2026-08-21/22 实验结果快照：[experiment_status_20260822.md](experiment_status_20260822.md)；
- 唯一结构化总表：`reports/experiment_registry/experiment_registry.csv`；
- V6 接口与可复现契约：[`../reference/qri_v6_contract.md`](../reference/qri_v6_contract.md)；
- 文档总入口：[`../README.md`](../README.md)。

除非新增实验经过完成门禁，不应把历史快照中的 running/deferred 字段解释为实时进程状态。
