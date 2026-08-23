# SALT-VI 当前状态

最后核验：2026-08-24（Asia/Shanghai），主机 `cgv841-SYS-7049GP-TRT`。

## 代码与工作树

- 清理分支 `codex/pmt-mscm-phased-pasd-20260821-v2` 已固定在 `392c8e11`；
- 当前 QRI 开发分支为 `codex/qri-v6-semantic-imagination-20260824`，统一版本名为
  `qri-v6`。清理提交中的 text-annotation 代码实际是 V5 过渡实现，内部残留的
  V4 标签已纠正；V4 只指 2026-08-22 的历史 smoke 产物；
- 活动工作树 `.codex-worktrees/stageb-b5-tricks-grid-20260824` 明确排除于本次改动范围；
- 独立 V6 工作树 `.codex-worktrees/qri-v6-semantic-imagination-20260824` 承载本分支改动。

## 运行状态

- 最终核验时，Stage-B B5 tricks grid 调度器以及 `camera_only`、`cosine_only` 训练进程正在运行；
- 该活动调度器、两个训练任务、输出目录和新工作树均未被清理或测试改写；
- 一个从 2026-08-17 遗留、无训练子进程且匹配表达式错误的 watcher 已在本轮停止；
- 其他账号和 `/home/lab929/ybj` 之外的训练/推理任务不属于本次清理范围。

## QRI-v6 状态

- 当前入口：`plugins/qwen_imagination/versions/qri-v6/plugin.yaml`；
- 当前机制：VLM 联合世界抽样、complete-link 语义聚类、簇频率权重和逐代表世界
  LLM 改写；VLM、编码器和 LLM 仅通过接口注入；
- V2/V5 text-annotation 配置已移入 `plugins/qwen_imagination/configs/legacy/`；
- V6 尚未绑定真实模型或运行 SYSU 生产数据，不存在模型效果或 ReID 收益结论。

## 结果与证据入口

- 2026-08-21/22 实验结果快照：[experiment_status_20260822.md](experiment_status_20260822.md)；
- 唯一结构化总表：`reports/experiment_registry/experiment_registry.csv`；
- V6 接口与可复现契约：[`../reference/qri_v6_contract.md`](../reference/qri_v6_contract.md)；
- 文档总入口：[`../README.md`](../README.md)。

除非新增实验经过完成门禁，不应把历史快照中的 running/deferred 字段解释为实时进程状态。
