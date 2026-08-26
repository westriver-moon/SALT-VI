# QRI Swin-only 分离文本标注 v4 审计

日期：2026-08-22
仓库：`/home/lab929/ybj/SALT-VI`
实测输出：`/home/lab929/ybj/experiments/qri_text_annotations/sysu_swin_separated_rgb5_smoke_v4`

## 结论

当前 exact 管线已经按目标拆开：Qwen 只接收完整 SwinIR 图和 SwinIR-only ROI boards；全局 caption 与每个 ROI 的描述/两条互斥候选独立输出；Qwen 不拼接最终 caption，不添加 `may/might`，也不报告概率。

旧提示词要求 Qwen 为两条候选直接填写概率是不正确的。权威数学规范要求通过重复独立采样和语义聚类计算经验质量：

\[
w_k=\frac{|C_k|}{M}.
\]

该值不是现实真值概率，也不是 VLM 自报置信度。当前 v4 把概率阶段标记为 `deferred_empirical`，每张标注中不存在 hypothesis probability 或 `sampled_text_worlds`。

## 当前流程

1. 从隔离的预计算缓存读取 2× SwinIR 图；Qwen 请求中不加入 LR 图。
2. ROI 生成器给出最多 13 个候选区域：eyes、head、双腕、双臂、upper torso、双口袋、双 carried、双脚。
3. 计算 ROI 选择分，而不是概率：

   `selection_score = min(1, normalized_blur + category_priority_boost)`

   当前 boost 为 carried object `+0.25`、wrist accessory `+0.10`、headwear `+0.05`。eyes 始终保留；所有达到 `0.60` 的区域均保留；不足 3 个时按分数补齐；最多 8 个。
4. Qwen 一次直接看完整 SwinIR 图和每个入选区域的 SwinIR tight/context board，输出：
   - 一个 22–35 词的独立全局 caption；
   - 六个 grammar-ready 全局属性槽；
   - 每个 ROI 的独立短描述；
   - 每个 ROI 恰好两条不带概率的候选解释。
5. 全局 caption 若仅词数越界，触发一次 text-only Qwen 压缩；只改 `g.c`，不重新看图、不改属性、不接触 ROI。5 张 v4 实测均未触发。
6. 保存分离结果。未来的小型纯文本 LLM 再负责把全局 caption、被采样的 ROI 假设以及 `may/might` 等语法组合起来；该组合阶段当前尚未实现或运行。
7. 未来概率阶段对每个原子假设做 M 次受控独立采样、语义聚类和频率统计；当前未伪造概率。

## 当前主 Qwen prompt

动态 ROI 列表会附在末尾；核心 prompt 如下：

```text
Annotate a SwinIR-restored low-resolution person image for person re-identification.
The SwinIR full image and SwinIR ROI boards are the sole visual inputs. Produce two
independent outputs and never merge them.

First, write one natural English global caption from the full SwinIR image. Target
26-30 words and enforce a hard limit of 22-35 words; count words before returning
JSON. Use this single-sentence shape without repeating slots: Person with [head]
wears [upper], [lower], and [footwear]; [carried item]; [distinctive detail]. It must
cover head/hair/headwear; upper garment color, type, sleeves and visible design;
lower garment color, type and length; footwear; carried item; and distinctive detail.
If a slot is not visible, state that it is not clearly visible. The global caption
must not contain ROI hypotheses, modal qualifiers, or any final composed regional
text. For body sides, use the person's anatomical left or right; when orientation is
unclear, say one wrist or one hand instead of guessing a side.

Second, for every listed ROI, independently return one 4-14 word visual caption and
exactly two mutually exclusive 2-8 word candidate hypotheses. Describe the ROI
itself; do not rewrite the whole person caption. Candidate descriptions must be
direct neutral phrases without modal qualifiers. Both candidates must answer the
same ROI question and cannot both be true. Never split an object and its attribute
into separate candidates; for example use thin eyeglasses versus facial shadow
without eyewear.

Do not assign probabilities or confidence scores; probability estimation is a
separate repeated-sampling and semantic-clustering stage. Do not rewrite or append
the global caption using ROI content. Output JSON only and no reasoning or prose.
For RGB input, describe only colors visible in the SwinIR image.

Return only the keys in this minimal contract:
{"g":{"c":"22-35 word independent global caption","a":{"hd":"head phrase",
"up":"upper phrase","lo":"lower phrase","ft":"footwear phrase",
"ca":"carried-item phrase","ds":"distinctive phrase"}},"r":[{"id":"region_id",
"s":"4-14 word independent ROI caption","h":[{"d":"candidate one"},
{"d":"candidate two"}]}]}. Omit every other key. Keep each global attribute phrase
at most 8 words. ROIs: <dynamic ROI list>
```

词数修复 prompt：

```text
Repair only an overlength person caption. Return JSON with exactly one key c.
Rewrite the supplied caption into 26-30 English words while preserving all six
supplied attribute facts. Use one sentence, do not add facts, do not include ROI
content, and do not use modal qualifiers. JSON only.
```

概率计算和下游组合目前没有在线 prompt；前者服从数学规范，后者留待后续纯文本 LLM 设计。

## 5 张 RGB 实测

| 图像 | Qwen 全局词数 | ROI 数 | 总耗时 | TRI 词数 |
|---|---:|---:|---:|---:|
| cam1/0001/0001.jpg | 34 | 8 | 36.24 s | 36 |
| cam1/0001/0002.jpg | 27 | 7 | 86.48 s | 26 |
| cam1/0001/0003.jpg | 31 | 8 | 109.24 s | 32 |
| cam1/0001/0004.jpg | 32 | 8 | 126.17 s | 33 |
| cam1/0001/0005.jpg | 27 | 7 | 114.64 s | 27 |

均值：30.2 个全局词、7.6 个 ROI、713.6 completion tokens、88.82 s Qwen、94.55 s 全管线。

结构核验：5/5 complete；5/5 SwinIR-only；5/5 无 self-reported probability；5/5 无 sampled worlds；所有 ROI 均恰好两条候选；39 项回归测试全部通过。

## 与原 SYSU TRI caption 的细粒度比较

总体上，两者的全局长度几乎相同；区别主要是信息分布，而不是字数：

- 新 Qwen 全局 caption 更稳定地覆盖眼镜、cargo pocket、鞋带/露趾结构、胸前红色图案等细节；ROI 层进一步保留这些细节的替代解释。
- TRI 对 carried item 有时更具体：0004 的 `holding a bottle` 与视觉中的浅色/粉色手持物一致；Qwen 只在 ROI 候选中给出 `holding a small object`，具体类别不足。
- 0003 的 TRI 写 `mobile in the right hand`，但 LR 和 SwinIR 中没有足够清晰的手机证据；Qwen 的 `no carried item` 属于可辩护的保守观测，不能直接判为劣于 TRI。
- 手表侧别存在标注坐标/人体解剖侧混淆。v4 已要求不确定时写 `one wrist`，但 0003 仍输出了 `right wrist`，而 TRI 写 left hand。
- 0005 全局 distinctive slot 写成 `indoor setting`，不如 TRI 的 wristwatch 有区分度；不过腕部 ROI 单独正确保留了 `metallic wristwatch` 与 `plain bare wrist` 两条候选。这恰好说明全局与 ROI 分离后需要下游文本 LLM 做受约束组合。
- 候选互斥性明显好于旧版，例如 `thin-framed eyeglasses` / `facial shadow without eyewear`、`empty hand` / `holding a small object`。仍有少数弱互斥对，如 0005 的 `wristwatch on arm` / `empty hand`，后续组合器或采样校验器应拒绝这类不共享同一判别轴的 pair。

结论：新标注不是简单“比 TRI 更长”；它在服饰和局部结构上更细，在小型携带物的具体命名上仍可能不如 TRI。全局 caption + ROI 假设的互补结构是成立的，但下游组合和经验概率阶段仍是必要组件。

## 全数据集算力估算

SYSU 当前索引包含 22,258 张 RGB train 和 11,909 张 IR train。按这 5 张 v4 的 94.55 s/图外推：

- 单张 3090 跑完整 RGB train：约 24.36 天；
- 三张 3090 并行跑完整 RGB train：约 8.12 天；
- 三张 3090 并行跑 RGB+IR train：约 12.46 天。

这只是同一 identity 的 5 张小样本外推，实际吞吐受同时运行任务和 llama.cpp 长上下文速度影响。相比 6-ROI v3（约 62.18 s/图），7–8 ROI v4 的平均总耗时增加约 52%，换来 carried/wrist ROI 的覆盖。若最终预算更紧，可把 max 从 8 降为 7，或只对语义优先区域启用第 8 个 ROI。

## 概率规范定位

权威文件：`/home/lab929/ybj/SALT-VI/docs/reference/semantic_imagination_mathematical_spec.md`

- 第 58–67 行：M 次独立 VLM 样本；
- 第 69–125 行：语义等价类、聚类和 medoid；
- 第 127–154 行：`w_k=|C_k|/M`，明确不是 VLM 自报置信度，也不是现实真值概率；
- 第 197–211 行：观测 caption 与被选假设的后续组合。
