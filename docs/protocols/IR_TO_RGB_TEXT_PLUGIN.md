# SYSU IR → RGB+Text 非对称检索协议

## 目标

查询端只使用红外图像，图库端融合可见光图像及其图像级 BLIP caption：

```text
IR query → IR encoder
RGB gallery + per-image RGB caption → RGB-Text fusion encoder
```

该协议由 `retrieval_backend: ir_to_rgb_text` 启用，不改变 SALT 的 legacy
`IR`、`Fusion` 或 `Text` 评测路径。

## 数据契约

- 训练仅加载 RGB caption，不加载 IR caption。
- 查询数据加载器不产生文本字段。
- 图库 caption 来自 `gallery_caption_manifest`，按原始图片路径查找。
- 禁止通过测试身份标签从 identity caption pool 选择图库文本。
- RGB 与 IR 均使用配置指定的 SwinIR array 数据。

## 训练目标

插件构造三个特征：IR、RGB，以及 RGB+Text。身份分类只作用于 IR 与
RGB+Text；跨模态 hard loss 包含主项 `IR ↔ RGB+Text` 和辅助项
`IR ↔ RGB`。`gallery_text_dropout` 按样本将融合特征退化为 RGB 特征，
避免图库描述质量成为单点依赖。

## 评测与模型选择

SYSU all-search single-shot 的查询特征由 IR 编码器生成，十次图库试验的
图库特征由 RGB+Text 融合编码器生成。结果键为 `IR-RGBText`，训练期间
按该结果的 Rank-1、mAP 与 mINP 保存 Fusion 头 checkpoint。

权威配置：

```text
configs/stage_b/ir_to_rgb_text_20260808.yaml
```
