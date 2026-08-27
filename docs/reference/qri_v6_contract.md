# QRI-v6 接口与可复现契约

`qri-v6` 保持当前版本名。机制改为一次生成联合候选、语义去重、等权输出；
移除重复抽样、簇频率估计与频率 Top-K。V4/V5 历史产物仍不是当前 v6 输出。

## 模块位置

- 核心：`semantic_imagination/semantic_imagination/v6/`
- 适配器：`plugins/qwen_imagination/qwen_imagination/versions/qri_v6.py`
- 默认配置：`plugins/qwen_imagination/configs/qri_v6_semantic_sysu.yaml`
- 插件入口：`plugins/qwen_imagination/versions/qri-v6/plugin.yaml`

## 后端接口

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `VLMBackend.observe` | 源图与 ROI | 共同 caption 和逐 ROI 稳定事实 |
| `VLMBackend.generate_joint_worlds` | 源图、共同观测、count、seed | 一次响应中的完整联合世界列表 |
| `SemanticEncoder.encode` | 完整联合世界文本 | 去重所需的语义向量 |
| `RewriteBackend.rewrite` | 共同观测、代表世界、seed | 完整自然语言 caption |

生成后端必须用一次请求返回最多 count 个联合候选，不在内部循环执行独立抽样。
每个候选完整覆盖 ROI，保留共同可见事实，不独立拼接区域边缘候选，不自报概率。
如需完全关闭解码采样，具体后端应采用确定性解码并记录实际参数。

后端仍携带 `BackendDescriptor`：backend_id、model_id、revision、
prompt 版本和完整解码参数。框架不绑定具体模型，也不猜测后端配置。

## 配置、检查与去重

- `candidate_world_count: 8` 指一次请求的目标候选数；实际允许返回 1 到该数量。
- 旧 `joint_sample_count` 与 `max_worlds` 配置已移除，不保留静默兼容路径。
- 按 source ROI 顺序规范化候选；重复/缺失区域、类别不符和空字段记为 invalid。
- 少返回或剔除无效候选后不补抽。无有效候选则失败，不写 complete manifest。
- 相同 state signature 内保留原 complete-link 和相似度阈值，仅作语义去重。
- 全部去重组均取实际成员 medoid 并改写，不按频率排序或截断。
- 实际得到 K 个世界后，各 `selected_weight=1/K`；K=1 时权重为 1。

## Manifest

顶层记录 observation、generation_seed、candidates、worlds、
generation_contract 及其摘要、run_signature、generation_diagnostics 和 telemetry。
`world_weighting` 固定为 `uniform_over_unique_worlds`。

每个 world 包含：

- world_id、representative_candidate_index、member_candidate_indices；
- assignments、caption、selected_weight。

每个 candidate 记录 candidate_index、valid/invalid 状态，以及规范化赋值和
world_id，或无效原因。诊断只记录 requested_candidate_count、
returned_candidate_count、valid_candidate_count、invalid_candidate_count、
duplicate_candidate_count 与 world_count。

不再输出 samples、sampling_diagnostics、sample_count、empirical_mass、
valid_weight、valid_weight_interval_95 或保留质量。重复次数不参与权重。

## 请求与遥测

正常路径为一次观测、一次联合生成、一次编码和 K 次改写，合计 K+3 次后端操作。
异常请求可按 request_max_attempts 重试原操作，并使用相同 seed；这不用于增加
候选数量或估计频率。生成请求耗尽重试则整次失败。
telemetry 按 vlm_observation、vlm_joint_generation、semantic_encoding、
llm_rewrite 记录 operation、attempt、retry、耗时与 usage。

## 缓存与迁移

run signature 覆盖新算法契约和全部 backend descriptor。generation contract
另含源图身份、source key、模态与 ROI 几何。缓存仅在 complete、版本、源图、
run signature 和 generation_contract 摘要一致时复用。

版本仍为 qri-v6，但旧频率型 sampling_contract 不再满足新缓存条件，会重新生成。
后端实现需将 sample_joint_world 替换为批量 generate_joint_worlds；
不提供把旧接口循环调用多次的适配器。旧 manifest 字段也不伪装成均匀权重。

输出位置保持为 `<QRI_V6_OUTPUT_ROOT>/metadata/<source_key-with-json-suffix>`，
必须在仓库外。测试使用临时目录。

## 当前边界

框架、接口、语义去重、均匀权重、重试与缓存可独立测试；具体 VLM、编码器和
改写模型尚未绑定。结构检查不承担视觉事实验证，本次修改未运行生产生成或
验证 ReID 收益。
