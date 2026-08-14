# SALT Semantic Imagination

该离线插件把一幅模糊行人图像转换为带经验权重的语义假设集合，并将其导出为 PASD source records。它不导入 SALT 训练代码或 PASD 模型代码。

权威数学语义和实现不变量见 [`MATHEMATICAL_SPEC.md`](MATHEMATICAL_SPEC.md)。原单链接机制的实测缺陷、修正算法和剩余限制见 [`KNOWN_LIMITATIONS_AND_CORRECTED_MECHANISM.md`](KNOWN_LIMITATIONS_AND_CORRECTED_MECHANISM.md)。实现可以替换 VLM、扰动、文本嵌入和聚类后端，但不能改变其中定义的概率解释。

## 工作流

1. 从输入图像提取稳定可见的共同观测；
2. 对语义保持扰动进行多次随机想象采样；
3. 每次生成一个带 `category/canonical_state/value/location` 的原子细节；正式
   概率簇由受控 `category + canonical_state` 定义，旧自由文本使用完全链接；
4. 选择簇内真实样本的 medoid，簇频率及其条件 Wilson 95% 区间作为统计输出；
5. 将共同观测与代表假设组合成 caption；
6. 导出动态视图 PASD record。

```python
from pathlib import Path

from semantic_imagination import (
    build_hypothesis_manifest,
    to_pasd_record,
    validate_atomic_response,
)

manifest = build_hypothesis_manifest(
    image=Path("person.jpg"),
    source_key="cam1/0001/person.jpg",
    backend=my_backend,
    instruction="Describe only plausible details that are not sufficiently observed.",
    sample_count=20,
    seed=20260811,
    similarity_threshold=0.85,
    cluster_linkage="complete",
    sampling_strata=("eyewear", "wrist_accessory", "headwear"),
    validator=validate_atomic_response,
    max_attempts=4,
    validation_failure_policy="exclude",
)
record = to_pasd_record(manifest, output_dir="images/cam1/0001/person")
```

后端负责 `observe`、`perturb`、单次原始 `imagine` 和 `embed`；框架负责分层
调度、带失败反馈的重试、失败溯源、受控状态校验、状态分组（或旧文本的完全链接）、medoid、
经验质量、类别内条件质量、条件区间和 record 导出。`cluster_linkage="single"`
只应用于旧结果复现。校验耗尽样本保留在 manifest 中，但不进入语义聚类；模型
主动返回的 `no_additional_detail` 仍是合法样本，两者不可混同。

## v3 模块边界

- `taxonomy.py`：八类状态表和 state/value 证据规则；
- `schema.py`：原子假设与校验结果；
- `validator/parser.py`：结构解析和可唯一确定的表面修复；
- `validator/semantic.py`：state 主导语义、明确冲突和观测重复校验；
- `validator/feedback.py`：将失败代码转换为下一次重试提示；
- `validation.py`：旧导入路径兼容层；
- `sampling.py`：分层采样、带反馈的确定性重试和诊断；
- `clustering.py`：精确状态分组与旧文本完全链接；
- `pasd.py`：PASD record 适配；
- `plugin.py`：旧调用方兼容门面。

## 与 PASD/SALT 的接口

- PASD 生成配置使用 `views_per_source: 0` 表示每个源图像具有动态假设数。
- SALT 配置使用 `sysu_sr_views_per_image: 0`。
- `hypothesis_empirical_mass` 保留校验失败后未归一化的有效语义质量；
- PASD 使用的 `hypothesis_weight` 只在有效视图内部归一化，仍严格总和为1；
- record 级 `imagination_validation_failure_rate` 保留失败样本率；若某个分层类别
  完全没有有效样本，`imagination_unrepresented_category_mass` 另外记录其未覆盖
  的设计先验质量。二者都不能生成伪视图。
- 旧 manifest 缺少权重时可按均匀分布读取，但不能把该兼容值解释为 VLM 经验质量。
- 扩散噪声重复样本不是新的语义簇，不能重复计算概率质量。

## 当前状态

插件、PASD records 和 SALT 加权 sampler 的接口已经实现并有单元及集成测试。
当前研究 backend 为本地 InternVL2.5-8B，实验入口为
`experiments/run_internvl_sampling.py`。它仍是离线生成插件，没有活跃训练 YAML
自动启用动态视图；当前 geometry-matched Stage-A 数据仍是一视图、权重1。
旧校验器的原始结果已清理，关键结论保留在
`KNOWN_LIMITATIONS_AND_CORRECTED_MECHANISM.md`；v3 的设计、N=512 离线重放与真实
smoke 验收见 `VALIDATOR_V3_AUDIT.md` 和 `reports/20260814_validator_v3_smoke/`。
