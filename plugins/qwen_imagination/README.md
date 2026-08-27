# Qwen imagination plugins

本目录是 SALT-VI 的版本化 QRI 插件入口。当前接口版本统一为 `qri-v6`；
`qri-v1` 与 `qri-v2` 只用于历史结果复现。V4 是 2026-08-22 的 smoke
产物版本，V5 是未完成的 text-annotation 过渡实现，两者都不是当前插件版本。

## QRI-v6

标准入口如下：

- 版本适配器：`qwen_imagination/versions/qri_v6.py`
- 插件 manifest：`versions/qri-v6/plugin.yaml`
- 默认配置：`configs/qri_v6_semantic_sysu.yaml`
- 权威核心：`semantic_imagination/semantic_imagination/v6/`
- 契约说明：`docs/reference/qri_v6_contract.md`

V6 只要求运行方注入三个带完整 descriptor 的后端接口：

1. `VLMBackend`：一次稳定观测和一次批量联合语义候选生成；
2. `SemanticEncoder`：编码完整联合世界，用于完全链接语义去重；
3. `RewriteBackend`：把共同观测与一个代表世界改写为完整 caption。

插件不内置或猜测具体 VLM/LLM。模型 ID、权重/代码 revision、prompt
版本、temperature、top-p、token 限制等影响分布的参数由 descriptor 进入
run signature 和 generation contract。`load_plugin("qri-v6")` 返回未绑定插件；
调用 `bind(vlm=..., encoder=..., rewriter=...)` 后得到可运行的 `V6Pipeline`。

V6 调用 `generate_joint_worlds(source, observation, count, seed)`，
一次返回最多 `candidate_world_count` 个完整联合候选，默认请求 8 个。
不从区域边缘分布独立组合，也不循环执行单世界抽样。
候选通过结构检查与 complete-link 语义去重后，全部 K 个代表世界等权输出，
`selected_weight=1/K`。每个代表只调用一次改写接口。

重复联合抽样、簇频率权重、频率 Top-K 和 Wilson 区间已移除；
旧 `joint_sample_count`、`max_worlds` 配置及频率型 v6 缓存不再使用。
具体后端若需要关闭 token 采样，应使用确定性解码并在 descriptor 中记录。

默认输出根由 `QRI_V6_OUTPUT_ROOT` 显式提供，记录固定写入：

```text
<output_root>/metadata/<source_key-with-json-suffix>
```

测试只使用 pytest 的临时目录，不在仓库内生成 smoke 或缓存产物。

## 历史版本

- `qri-v1`、`qri-v2`：保留的 regional QRI 适配器与配置；
- `qwen_imagination/text_annotation/`：V2/V5 历史数据标注实现，不属于 V6；
- `configs/legacy/text_annotation_sysu_v2_track_anchor.yaml`：历史 track-anchor 配置；
- `configs/legacy/text_annotation_sysu_v5_exact.yaml`：历史 V5 exact 配置。

历史 launcher 仍可复现实验，但不得把其 per-ROI 独立组合、自报概率或
`empirical-atomic-v1` 输出登记为 QRI-v6。
