# SALT_R_TEXT_VISUAL 正式默认方案

状态：正式默认
生效日期：2026-08-08
适用阶段：SALT-VI Stage B

## 决策

`SALT_R_TEXT_VISUAL` 正式取代 Stage-3 `PAIR-EQUAL`，成为 SALT-VI 的
默认 Stage-B 训练与结果引用方案。权威配置为：

```text
configs/stage_b/r_text_visual_20260729.yaml
```

默认启动命令：

```bash
python scripts/train.py --config_select configs/stage_b/r_text_visual_20260729.yaml
```

## 方案边界

- 冻结视觉分支，训练 RGB/IR/Text Stage-B 模块。
- 使用离线 SwinIR x2 的 RGB 与 IR 512x256 输入。
- 保留 RGB-IR、RGB-Text、IR-Text 的 cross-modal hard pair loss。
- RGB-Fusion、IR-Fusion、Fusion-Text 的 pair loss 权重为 0。
- 使用双分支 PMT patch embedding 与可学习 PatchGeM。
- 不启用 LLM caption augmentation。

## 晋升证据

评估协议为 SYSU-MM01 all-search、single-shot、10 trials。保留 checkpoint
来自 epoch 23：

| 方案 | Rank-1 | mAP | mINP |
| --- | ---: | ---: | ---: |
| Stage-3 PAIR-EQUAL | 83.6497% | 81.2407% | 71.6945% |
| SALT_R_TEXT_VISUAL | 84.0783% | 81.4334% | 71.7899% |

权威 checkpoint SHA-256：
`f92ea24e0808fa37234cba04e796dbfe8695185b13a1ac6f4ac099b848a50106`。

## 验证声明

本次晋升是明确的项目决策，不要求补充多随机种子实验。现有结果来自单种子
SYSU 实验，且超参数选择使用了 SYSU 测试协议；后续论文或对外报告必须保留
这一限制说明，不得把当前证据表述为多种子统计结论。

`PAIR-EQUAL`、Qwen 文本增强和其他 SALT 消融继续作为历史对照保留，但不再
作为默认训练入口。
