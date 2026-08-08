# 配置

运行时配置直接服务于 `src/salt_vi`。

- `src/salt_vi/config/default.yaml`：包内默认训练配置
- `stage_a/`：视觉编码器与采样消融
- `stage_b/`：融合、token interaction 与适配器消融
- `metric_boost/`：指标增强实验计划
- `experiments/`：可复现实验基础配置
- `super_resolution/`：超分辨率配置
- `vision_text/`：独立视觉-文本基线配置

## 正式默认配置

Stage-B 正式默认方案为 `stage_b/r_text_visual_20260729.yaml`，实验标识
`SALT_R_TEXT_VISUAL`。`stage_b/a3_e4_stageb.yaml` 和 Stage-3 `PAIR-EQUAL`
继续作为对照与历史主线证据保留，不再是默认训练入口。

默认启动命令：

```bash
python scripts/train.py --config_select configs/stage_b/r_text_visual_20260729.yaml
```

非对称检索与 RGB PASD 五视图配置：

```bash
python scripts/train.py --config_select configs/stage_b/ir_to_rgb_text_pasd_rgb_multiview_20260808.yaml
```

历史配置已按实验语义迁入各类 reproduction 子目录，不作为默认运行入口。checkpoint/pretrained 路径按迁移边界保持不动。
