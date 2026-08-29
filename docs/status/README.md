# SALT-VI 当前状态

更新：2026-08-29（Asia/Shanghai），主机 `lab929-3090`。

## 代码与运行

- 当前主线工作区：`/home/lab929/ybj/SALT-VI`，分支 `main`；
- Rounded token pruning、Pose-guided CTI、原生 SwinIR ×4 image-tree 输入及其测试
  已整合到主线工作区；
- 六个 token 干预训练均已完成 24 epochs，GPU 0–3 已释放；
- 两组训练产物已从实验 worktree 原子迁入统一 archive 并通过逐文件 SHA-256 校验；
- 实验 worktree 只等待主线测试、提交和推送完成，之后即可删除。

## 当前数据资产

统一根目录为 `/home/lab929/ybj/datasets/person-assets-512x256/`：

- `person_fit/`：审核后的 V1 + YOLO26 防拉伸输入；
- `resize/`：相同 native realSR ×4 上游的直接 512×256 resize 对照；
- 两套 SYSU 数据均包含 pose 与 anatomy sidecar；
- `upstream/swinir-real-sr-x4-v1/`：SYSU、RegDB、LLCM 全量原生 ×4 PNG、
  manifest、contract、benchmark 和日志。

精确计数、哈希与兼容路径只查
[`../reference/data_asset_registry.md`](../reference/data_asset_registry.md)及其机器可读清单。

## 结果与证据入口

- Token pruning/CTI 归档说明：
  [`../history/stage_a/stage_a_c3_token_interventions_20260828.md`](../history/stage_a/stage_a_c3_token_interventions_20260828.md)；
- 统一实验总表：`reports/experiment_registry/experiment_registry.csv`；
- 机器可读资产表：`reports/assets/person_assets_20260829.json`；
- 旧 2026-08-21/22 状态快照：
  [`experiment_status_20260822.md`](experiment_status_20260822.md)；
- QRI-v6 契约：[`../reference/qri_v6_contract.md`](../reference/qri_v6_contract.md)。

历史快照中的 running/deferred 字段不代表当前进程状态；实时状态以上述当前页和
服务器进程/GPU 检查为准。
