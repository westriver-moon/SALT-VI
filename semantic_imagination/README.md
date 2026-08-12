# SALT Semantic Imagination

该离线插件把一幅模糊行人图像转换为带经验权重的语义假设集合，并将其导出为 PASD source records。它不导入 SALT 训练代码或 PASD 模型代码。

权威数学语义和实现不变量见 [`MATHEMATICAL_SPEC.md`](MATHEMATICAL_SPEC.md)。实现可以替换 VLM、扰动、文本嵌入和聚类后端，但不能改变其中定义的概率解释。

## 工作流

1. 从输入图像提取稳定可见的共同观测；
2. 对语义保持扰动进行多次随机想象采样；
3. 用文本嵌入把样本聚成语义等价簇；
4. 选择簇内真实样本的 medoid，簇频率作为 `hypothesis_weight`；
5. 将共同观测与代表假设组合成 caption；
6. 导出动态视图 PASD record。

```python
from pathlib import Path

from semantic_imagination import build_hypothesis_manifest, to_pasd_record

manifest = build_hypothesis_manifest(
    image=Path("person.jpg"),
    source_key="cam1/0001/person.jpg",
    backend=my_backend,
    instruction="Describe only plausible details that are not sufficiently observed.",
    sample_count=20,
    seed=20260811,
    similarity_threshold=0.85,
)
record = to_pasd_record(manifest, output_dir="images/cam1/0001/person")
```

后端负责 `observe`、`perturb`、`imagine` 和 `embed`；插件负责采样合同、聚类、medoid、经验质量和 record 导出。

## 与 PASD/SALT 的接口

- PASD 生成配置使用 `views_per_source: 0` 表示每个源图像具有动态假设数。
- SALT 配置使用 `sysu_sr_views_per_image: 0`。
- 每个 source 的权重必须为正且总和为 1。
- 旧 manifest 缺少权重时可按均匀分布读取，但不能把该兼容值解释为 VLM 经验质量。
- 扩散噪声重复样本不是新的语义簇，不能重复计算概率质量。

## 当前状态

插件、PASD records 和 SALT 加权 sampler 的接口已经实现并有单元测试，但尚未选择正式 VLM/扰动/embedding backend，也没有活跃训练 YAML 使用动态视图。当前 geometry-matched Stage-A 数据仍是一视图、权重 1。因此本模块是下一阶段研究接口，而非当前运行实验的组成部分。
