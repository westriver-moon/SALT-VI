# 配置

运行时配置直接服务于 `src/salt_vi`。

- `default.yaml`：默认训练配置
- `stage_a/`：视觉编码器与采样消融
- `stage_b/`：融合、token interaction 与适配器消融
- `metric_boost/`：指标增强实验计划
- `experiments/`：可复现实验基础配置
- `super_resolution/`：超分辨率配置
- `vision_text/`：独立视觉-文本基线配置

历史配置已按实验语义迁入各类 reproduction 子目录，不作为默认运行入口。checkpoint/pretrained 路径按迁移边界保持不动。
