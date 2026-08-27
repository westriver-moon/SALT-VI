# Stage-A C3 椭圆注意力层位实验记录

日期：2026-08-27
实验组：`stage_a_c3_ellipse_attention_20260827`
基线：Stage-A C3（MBPatch + 12-layer ViT，输入 512×256）

## 设计

在 C3 主干中加入较宽的行人椭圆约束，仅改变约束注入层位：

| 变体 | 椭圆损失层 | 权重 | 半径 `(x,y)` | warm-up |
|---|---:|---:|---:|---:|
| Layer 2 | 第 2 层后 | 0.10 | `(0.58, 0.55)` | 3 epochs |
| Layer 4 | 第 4 层后 | 0.10 | `(0.58, 0.55)` | 3 epochs |

两组使用相同的 C3 配方、数据、采样和 24-epoch 预算；有效 batch size 为 64（单卡 batch 32）。Layer 2 使用 GPU 1，Layer 4 使用 GPU 2，正式评测为 SYSU-MM01 all-search、single-shot、10 gallery trials。

## 结果

| 变体 | 最后/最佳 epoch | Rank-1 | mAP | mINP |
|---|---:|---:|---:|---:|
| C3 + ellipse Layer 2 | 23 | 70.6179% | 68.8428% | 57.4742% |
| C3 + ellipse Layer 4 | 23 | 70.3892% | 68.8766% | 57.5001% |

两组训练均正常完成；Layer 2 的 Rank-1 略高，Layer 4 的 mAP/mINP 略高。该层位比较用于选择后续研究候选，不据此宣称椭圆约束已经优于所有 C3 基线；应在固定预算下补充同初始化的 C3 对照和机制消融。

## 复现与归档

- 原始运行树：`/home/lab929/ybj/experiments/c3-ellipse-20260827-gpu12`
- 正式归档：`/home/lab929/ybj/experiments/archive/stage_a/SALT-VI-c3-ellipse-20260827-gpu12`
- 两组最终模型：各自 `models/model_IR_23.pth`
- 两组完整训练状态：各自 `checkpoint/checkpoint_latest.pth`
- 起点 PMT 权重已存在 Stage-A 共享归档：`/home/lab929/ybj/experiments/archive/stage_a/_shared/weights/jx_vit_base_p16_224-80ecf9dd.pth`，不重复复制。
- 完整 dirty worktree、Git bundle、binary patch、解析配置、事件日志和环境/数据指纹位于归档 `provenance/`。
