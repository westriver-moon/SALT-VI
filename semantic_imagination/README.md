# SALT Semantic Imagination

`semantic_imagination.v6` 是当前 QRI-v6 的唯一联合语义候选核心。它把一个
低质量观测转换为有限个等权的联合世界，并通过注入的文本改写接口生成每个
代表世界的完整 caption。该包不导入具体 VLM、LLM、PASD 或 SALT 训练代码。

## V6 数据流

1. `VLMBackend.observe` 返回全局 caption 与逐 ROI 稳定事实；
2. `VLMBackend.generate_joint_worlds` 一次返回最多 `candidate_world_count`
   个完整联合候选，默认请求 8 个；不循环调用单世界抽样接口；
3. 检查候选的 ROI 覆盖、类别与非空字段，记录无效候选，不补抽；
4. `SemanticEncoder` 编码完整世界，在相同 state signature 内以
   complete-link 和 value/location 语义距离去重；
5. 保留全部 K 个代表世界，各赋权 `selected_weight=1/K`；
6. `RewriteBackend` 将共同观测和每个代表世界改写成完整 caption；
7. manifest 记录候选来源、诊断计数、调用遥测和 generation contract。

不再进行重复联合抽样、簇频率估计、频率 Top-K 或二次世界采样。
均匀权重是候选使用策略，不是现实真值概率。

VLM、语义编码器和 LLM 改写器都必须提供 `BackendDescriptor`。descriptor 包含
后端 ID、模型 ID、权重或代码 revision，以及 prompt、temperature、top-p、
token 限制等影响结果的参数。descriptor 与算法配置共同进入 run signature；
修改任一生成参数都会使缓存失效；旧频率型 v6 缓存不再复用。

```python
from semantic_imagination.v6 import V6Pipeline, load_v6_config

pipeline = V6Pipeline(
    load_v6_config("plugins/qwen_imagination/configs/qri_v6_semantic_sysu.yaml"),
    vlm=my_vlm_backend,
    encoder=my_semantic_encoder,
    rewriter=my_rewrite_backend,
)
result = pipeline.run(source)
```

输出只写入配置指定的仓库外 `output_root/metadata/`。测试使用 pytest 临时目录。
接口和 manifest 字段见
[`qri_v6_contract.md`](../docs/reference/qri_v6_contract.md)，数学解释见
[`semantic_imagination_mathematical_spec.md`](../docs/reference/semantic_imagination_mathematical_spec.md)。

## 历史兼容层

包根目录的 `sampling.py`、`clustering.py`、`validator/` 与 `pasd.py` 保留
V3 原子边缘分布和旧 PASD record 的读取/复现能力。它们不是 QRI-v6 的实现，
也不得生成标记为 `qri-v6` 的 manifest。新代码只能从
`semantic_imagination.v6` 导入当前接口。
