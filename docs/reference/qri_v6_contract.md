# QRI-v6 接口与可复现契约

`qri-v6` 是当前超分语义想象力接口的唯一版本名。V4 仅指 2026-08-22 的
Swin-only smoke 产物，V5 仅指未完成的 text-annotation 过渡实现；它们都不能
写出 `plugin_version: qri-v6`。

## 模块位置

- 核心：`semantic_imagination/semantic_imagination/v6/`
- Qwen 适配器：`plugins/qwen_imagination/qwen_imagination/versions/qri_v6.py`
- 插件 manifest：`plugins/qwen_imagination/versions/qri-v6/plugin.yaml`
- 默认配置：`plugins/qwen_imagination/configs/qri_v6_semantic_sysu.yaml`

## 后端接口

V6 不绑定具体模型，只定义三个接口：

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `VLMBackend.observe` | 源图与 ROI | 共同 caption 和逐 ROI 稳定事实 |
| `VLMBackend.sample_joint_world` | 源图、共同观测、seed | 一次完整联合世界抽样 |
| `SemanticEncoder.encode` | 联合世界 canonical text | 有限、同维语义向量 |
| `RewriteBackend.rewrite` | 共同观测、代表世界、seed | 完整自然语言 caption |

所有后端必须携带 `BackendDescriptor`：`backend_id`、`model_id`、`revision`
和完整 `parameters`。`parameters` 必须记录 prompt 版本、解码温度、top-p、token
限制及其他影响分布的设置；框架不从运行时对象猜测这些值。

## 采样与聚类

- 每个 sample 是一次 VLM 联合抽样，必须恰好包含每个目标 ROI 的一个原子赋值。
- ROI 之间的相关性来自同一次联合抽样，不使用区域边缘分布的独立乘积。
- 不同 `region/category/state` signature 是硬边界；同一 signature 内使用完整
  value/location 文本向量执行 complete-link。
- 相似度阈值真实参与分簇；V6 不提供 single-link 或“只按 state 合并”开关。
- 簇频率直接计算 `empirical_mass=count/scheduled` 与
  `valid_weight=count/valid`。筛选 Top-K 后只对保留簇计算 `selected_weight`；
  不再执行第二次 Monte Carlo 世界抽样。

## 请求失败与遥测

观察、联合抽样、编码和改写都使用同一个 `request_max_attempts`。联合抽样重试
耗尽时保留 `request_failed` sample，并继续其他独立 sample；其他阶段重试耗尽
则不能形成完整 manifest。遥测按 phase 记录实际 attempt、成功调用、额外重试、
耗时和后端 usage，不能用“一张图一次请求”代替真实调用数。

## 缓存与输出

run signature 哈希覆盖算法配置和三个 backend descriptor。完整 sampling contract
还覆盖 source key、图像 SHA-256、模态以及 ROI 类别与坐标。缓存只有在以下信息
全部一致时复用：

1. `plugin_version == qri-v6`；
2. source key、图像 SHA-256、模态和 ROI 几何一致；
3. 完整 run signature 一致；
4. 完整 sampling contract 哈希一致；
5. record 状态为 `complete`。

记录固定写入配置指定的仓库外：

```text
<QRI_V6_OUTPUT_ROOT>/metadata/<source_key-with-json-suffix>
```

仓库内不保存 smoke、pytest cache、模型响应或临时 manifest。

## 当前边界

V6 已实现框架、接口、缓存、聚类、权重、重试和遥测；具体 VLM、编码器与 LLM
改写后端尚未绑定。因此 V6 当前是“接口与机制完成、真实模型实验未运行”，不得
登记为生产数据完成或 ReID 收益成立。
