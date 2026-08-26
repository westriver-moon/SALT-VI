# SALT Semantic Imagination

`semantic_imagination.v6` 是当前 QRI-v6 的唯一语义采样核心。它把一个低质量
观测转换为带经验质量的联合语义世界，并通过注入的文本改写接口生成每个代表
世界的完整 caption。该包不导入具体 VLM、LLM、PASD 或 SALT 训练代码。

## V6 数据流

1. `VLMBackend.observe` 返回全局 caption 与逐 ROI 稳定事实；
2. `VLMBackend.sample_joint_world` 在一次请求中返回所有目标 ROI 的联合赋值；
3. `SemanticEncoder` 编码完整联合世界；
4. 框架在相同 state signature 内执行 complete-link 聚类，value/location
   的语义距离真实参与分簇；
5. 簇频率直接形成经验质量，不再进行第二次随机世界组合；
6. `RewriteBackend` 将共同观测和代表世界改写成完整 caption；
7. manifest 原样记录失败质量、保留质量、全部调用遥测和可哈希 contract。

VLM、语义编码器和 LLM 改写器都必须提供 `BackendDescriptor`。descriptor 包含
后端 ID、模型 ID、权重或代码 revision，以及 prompt、temperature、top-p、
token 限制等影响结果的参数。descriptor 与算法配置共同进入 run signature；
修改任一采样参数都会使缓存失效。

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
