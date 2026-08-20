# Qwen Regional Imagination v2

QRI-v2 是与 QRI-v1 并行的 imagination-first 区域语义世界生成插件。两者共享 SwinIR、
ROI、PASD direct-rewrite、校准和 manifest 基础设施，但拥有独立配置、策略类、输出环境变量、
Stage-A 配置、实验 ID 和 pipeline registry；V2 不读取或覆盖 V1 的缓存与实验目录。

## 推理策略

1. LR 是权威观测，SwinIR 仅为高频 proposal。
2. 每个入选 ROI 以一张 512×512 四宫格提供 LR tight/context 与 SwinIR tight/context；
   所有格子保持长宽比，LR 使用 nearest-neighbour 放大。
3. proposer 运行三轮，最大化互斥合理世界的候选召回率。每个区域必须覆盖至少一个正向
   interpretation、允许时的 `absent` 和 `unresolved`；不可见不得转换为不存在。
4. 16 次联合采样先执行候选 coverage schedule，再进行高温自由采样。每个 world 分别
   记录 coverage/free sample count；组合频率只叫 scheduled proposal mass，不解释为现实后验。
5. critic 只排除与 LR、身份、姿态、几何或联合语义冲突的世界。标签为
   `strong_pixel_supported`、`weak_pixel_supported`、`prior_plausible`、
   `contradicted`、`unresolved`；弱证据和先验世界在不冲突时保留。
6. 最多选择五个世界：保留 no-edit baseline，并尽量确保每个高不确定区域至少有一个
   正向编辑世界。
7. 每个正向 assignment 单独截取保持 1:2 长宽比的上下文 ROI，并放大到 PASD 的
   256×512 canonical 画布。PASD 只接收一个具体、可绘制的区域假设，使用
   `guidance_scale=7.0`、`conditioning_scale=0.75`；生成结果映射回原坐标后再由该区域的
   soft mask 融合。不同区域依次融合，mask 外始终保留 SwinIR。
8. V2 的局部负向提示不含 `new accessories` 或 `altered face`，避免正向 eyewear、
   headwear、wrist accessory 假设被通用身份保护词反向压制。身份、姿态、体型和区域外
   内容仍然受到保护。
9. manifest 同时记录 critic 前的 `u_qwen_proposal` 和排除矛盾后的
   `u_qwen_compatible`；兼容性熵才是区域语义不确定性的正式值。
10. LR cycle、C3 identity drift 和 edit penalty 把 proposal mass 校准为 posterior weight。
   训练按 paired world 采样，测试对 top-5 世界做特征边缘化。

## 隔离边界

- 生成配置：`configs/qri_v2_imaginative_sysu.yaml`
- PASD 配置：`configs/pasd_sysu_qri_v2.yaml`
- 输出环境：`QRI_V2_OUTPUT_ROOT`
- Stage-A 配置：`../configs/stage_a/semantic_imagination_v2/`
- pipeline registry：`../configs/pipelines/sysu_qri_v2.yaml`
- launcher：`../scripts/experiments/run_qri_v2_pipeline.py`
- 训练环境：`SALT_QRI_V2_DATA_ROOT`、`SALT_QRI_V2_VIEW_MANIFEST`、
  `SALT_QRI_V2_EXPERIMENT_ROOT`

QRI-v1 继续使用其原有路径和 `SALT_QRI_*` 环境变量。manifest consolidation 会拒绝将
不同 plugin version 写入同一份输出。

## 运行

```text
salt-qri serve --config semantic_imagination/configs/qri_v2_imaginative_sysu.yaml
QRI_V2_OUTPUT_ROOT=.../smoke32 salt-qri run \
  --config semantic_imagination/configs/qri_v2_imaginative_sysu.yaml \
  --device cuda:1 --limit 32 --fail-fast
python scripts/experiments/run_qri_v2_pipeline.py list
```

正式生成仍先运行 32-source fail-fast smoke 和 512-source train-only pilot，再固定 pilot
category statistics 生成 `--split all`。自动晋级保持关闭，Stage-A 只报告固定 epoch 23。

## 回归不变量

- 背面或遮挡眼镜区域必须允许正向 eyewear world，不得只产生 `absent`。
- `unresolved`、`absent` 和正向候选是不同状态。
- proposer/sampler 不含 `Prefer abstention`。
- 每次 ROI PASD 调用必须为 identity-coordinate `direct_rewrite`；聚合记录的模式为
  `roi_direct_rewrite`。只有尺寸相同但经过 person-fit、padding 或背景恢复的输出必须拒绝。
- 每个 ROI 的 crop box、control/output checksum、单独 caption、seed、采样强度和 PASD
  geometry 都必须进入 world realization provenance。
- 局部负向提示不得包含 `new accessories` 或 `altered face`。
- 世界图像、caption 和 mask 始终 paired。

## 真实模型冒烟

启动本地 llama.cpp server 后，可以对一个实际眼部 ROI 验证模型是否真正提出正向世界：

```text
python scripts/experiments/smoke_qri_v2_imagination.py \
  --lr /path/to/original.jpg \
  --swin /path/to/canonical-256x512-swin.jpg \
  --bbox 69,20,201,71
```

如果只有自动正向兜底而 Qwen 自身没有提出正向 interpretation，命令会失败；因此该检查
不能被 coverage contract 的 fallback 假阳性绕过。
