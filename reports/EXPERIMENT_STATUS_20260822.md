# SALT-VI 当前实验状态与归档总览

最后更新：2026-08-22 19:28 CST（本次交付明确排除 H3-35）
服务器：`lab929-3090`（实验账号 `lab929`；所有者侧资产核验账号 `cgv841`）
仓库分支：`codex/pmt-mscm-phased-pasd-20260821-v2`
归档源码快照基线：`27ff03a16f2d0d99c1511a484e9aa0c891365207`

本文是 2026-08-21/22 这批实验的当前状态入口。指标只引用结构化
`events.jsonl`、调度器 `results.json` 或实验权威 `metrics.json`；旧专项报告仍保留，
但不再承担“当前状态”或跨实验排行榜职责。

本次归档交付范围为除独立 H3-35 延长训练之外的已稳定实验。H3-35 按用户指令明确
不轮询、不干预、不封存；其原始目录、启动诊断和历史中间状态仍作为后续证据保留。
QRI text annotations 归档已同步首次归档后新增的 73 个文件，但当前 v4 priority smoke
进程仍在原目录写入，因此该新增尾部不被当前稳定审计误报为最终完成。

## Stage-A PMT-MSCM 混合损失

统一评估协议为 SYSU-MM01、IR-to-visible、all-search、single-shot、10 个 gallery trials，
seed 0。表中“最高”按 Rank-1 选择，三项指标来自同一评估 epoch；四组 24 epoch 实验的
最终 epoch 为 23。实际权重以对应 YAML 和 resolved `configs.yaml` 为准，不能用旧的
“epoch 6 后只启用 QCT”文字替代。

| 实验 | MSEL | DCL | QCT | 最高/最终 epoch | Rank-1 | mAP | mINP | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H1 balanced | 0.25 | 0.25 | 0.25 | 23 | 70.31% | 68.20% | 56.00% | 已完成 |
| H2 DCL-0.50 | 0.25 | 0.50 | 0.25 | 23 | 69.78% | 68.61% | 57.25% | 已完成 |
| H3 QCT-0.40 | 0.25 | 0.25 | 0.40 | 23 | 70.23% | 68.69% | 56.99% | 已完成 |
| H4 Full-aux | 0.50 | 0.50 | 0.25 | 23 | 69.71% | 68.47% | 57.13% | 已完成 |
| H3 QCT-0.40 e35 | 0.25 | 0.25 | 0.40 | 明确排除 | — | — | — | 本次交付按用户指令不归档；仅保留历史 epoch 9 快照与 watcher 说明 |

H1–H4 的原始运行目录均保留；四组 24 epoch 结果和对应 Stage-A 归档均已通过稳定
校验。H3-35 是独立的 35 epoch 延长复现，不覆盖 H3 的 24 epoch 结果；本次交付不等待
其完成，也不把其中间 checkpoint 或历史 epoch 9 指标写成最终结果。

代码与配置：

- `configs/stage_a/plugins/hybrid_loss/common.yaml`
- `configs/stage_a/plugins/hybrid_loss/h1_balanced_025.yaml`
- `configs/stage_a/plugins/hybrid_loss/h2_dcl_050.yaml`
- `configs/stage_a/plugins/hybrid_loss/h3_qct_040.yaml`
- `configs/stage_a/plugins/hybrid_loss/h4_full_aux.yaml`
- `configs/stage_a/plugins/hybrid_loss/h3_qct_040_e35.yaml`

归档目标：

- `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-pmt-mscm-hybrid-loss-20260821`
- `/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-pmt-mscm-hybrid-loss-h3-qct-040-e35-20260822`

第一个 Stage-A 归档已通过 `verification.json`（165 项检查，`ok=true`）；第二个需待训练
完成后再封存，不能把 epoch 9 中间状态写成最终归档。

## QRI 2026-08-21/22 专项实验

这些实验是 Qwen 想象力/文本标注/超分辅助研究的独立证据，不是 Stage-A ReID 排行榜。
“已完成”表示 smoke、诊断或专项验证完成，不表示已经启动全数据集生产任务。

| 实验 | 结果状态 | 归档 |
|---|---|---|
| QRI fast search | 加速与质量审计完成；保留负面/边界证据 | `archive/qri/qri-fast-search-gpu0-20260821` |
| QRI glasses | 单图 PASD、Inpainting、后融合验证完成；推荐 blend 0.25 | `archive/qri/qri-glasses-gpu0-20260821` |
| QRI imagination control | 像素控制失败诊断与文本级收缩路线均保留 | `archive/qri/qri-imagination-control-20260822` |
| QRI text annotations | v3/compact 及历史 smoke 已封存；首次归档后同步 73 个文件；v4 priority smoke 仍在运行，未列入稳定最终审计 | `archive/qri/qri-text-annotations-smoke-20260822` |
| QRI text imagination | no-thinking/thinking 六次受控消融完成 | `archive/qri/qri-text-imagination-thinking-ablation-gpu0-20260822` |

四个稳定 QRI 对象已完成原目录保留、载荷清单、源码/配置引用、强外部资产注册表和
最终稳定审计。QRI text annotations 的归档自身 inventory 已在同步 73 个新增文件后通过，
但原目录仍有 v4 priority smoke 进程写入，故它及其新增尾部不纳入本次稳定最终审计。
QRI 的详细事实仍以各专项报告及归档中的 `metrics.json`、`summary.json`、`selection.json`
为准；不能将 smoke 结果升级为全量训练结果。

## 本次交付的审计证据

- 稳定对象审计：`/home/lab929/ybj/experiments/archive/final_audit_20260822_excluding_h3_stable.json`，
  包含 Stage-A H1–H4 矩阵、QRI fast search、QRI glasses、QRI imagination control、QRI text imagination，
  顶层 `ok=true`。
- QRI text annotations 的活动态审计：`/home/lab929/ybj/experiments/archive/audit_20260822_excluding_h3_qri_text_active.json`，
  明确记录原目录在首次归档后继续产生新载荷，不能把该快照当作当前运行的最终归档。
- H3-35 不在上述审计 ID 中；其排除是范围决策，不代表训练失败或指标结论。

## 复现与资产边界

- 共享 dirty-worktree 源码快照位于 `/home/lab929/ybj/experiments/archive/_shared/source/`，
  包括源码树、tar、Git bundle、二进制安全 patch、环境和外部资产指纹。
- 强外部资产注册表位于 `/home/lab929/ybj/experiments/archive/_shared/assets/`。
- `cgv841` 账号仅用于只读核验 SYSU-MM01 完整树和 SwinIR 资产；没有修改或搬移数据。
- 数据集、模型、checkpoint 和原始训练日志不进入 Git；Git 保存配置、代码、报告、归档
  规范和可重建的外部资产引用。

## 历史文档规则

`reports/stage_a_*`、`reports/qri_*`、`reports/autoresearch_*` 是历史专项证据，保留其
失败实验、负面结果、路径和当时协议，但不再表述为当前默认配置。当前项目入口只使用：

1. 本文的当前实验状态；
2. `reports/experiment_registry/experiment_registry.csv` 的结构化总表；
3. `docs/README.md` 的架构与运行说明；
4. 各归档目录中的原始日志、解析配置、事件和 provenance。
