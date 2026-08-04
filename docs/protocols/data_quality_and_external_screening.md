# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: external feature screen spec

> Source document ID: `source_core:docs/external_feature_screen_spec.md`  
> Original SHA-256: `f494ef2e9e222ff6647b7ba94aa3b9cc24788575cf5935860bfd10f12dcfc1a8`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# SALT-VI 外部 VI-ReID 数据特征筛选方案（修订版）

## 1. 目标与边界

使用 `TRAIN-3-H1` 模型在不训练、不使用 SYSU 测试身份的前提下，对外部
VI-ReID 官方训练集进行确定性特征提取、目标域校准、身份级筛选和
目标覆盖选择，生成可供后续联合预训练使用的 JSONL manifest。

本阶段只生成特征、指标、筛选清单、图表和溯源记录，不执行训练，
不将筛选指标解释为最终模型效果。

## 2. 当前服务器可执行范围

| 数据集 | 路径 | 本阶段状态 | 筛选主单位 |
|---|---|---|---|
| SYSU-MM01 | `/home/cgv841/datasets/SYSU-MM01` | 仅作目标域校准 | identity |
| LLCM | `/home/cgv841/datasets/LLCM` | 启用 | identity |
| RegDB | `/home/cgv841/datasets/RegDB` | 启用，默认 trial 1 | identity |
| HITSZ-VCM | `/home/cgv841/datasets/HITSZ-VCM` | 启用 | identity（tracklet 为子单位） |
| BUPTCampus | 当前服务器不存在 | 禁用，数据到位后再适配 | identity |

`SALT-VI/datasets/llcm` 和 `SALT-VI/datasets/regdb` 只保存文本，不是图像根目录。

## 3. 不可变约束

1. 只读取各数据集官方训练 split；不实例化会附带读取 query/gallery 的完整 `Loader`。
2. 图像使用与当前模型测试路径等价的确定性 Resize/Normalize，不使用随机训练增强。
3. 模型只从 `logs/raw/source_core/reports/metric_boost/runs/TRAIN-3-H1/status.json` 解析；若其中保留的
   旧 worktree 绝对路径已退役，只允许按 `logs/raw/source_core/reports/metric_boost/runs/...` 相对部分
   重定位到声明的 `target.workspace_root=/home/cgv841/ybj/SALT-VI`，并在 manifest 记录
   最终路径和 SHA-256；不重写模型内部融合公式。
4. 筛选主空间为 `post_bn`，同时保存 `pre_bn`。距离计算前执行 L2 normalize。
5. 缺失 caption 时只输出 RGB/IR，不生成、伪造或随机替代文本。
6. HITSZ-VCM 的单条 tracklet 是单模态的，不对 tracklet 强制同时存在 RGB/IR。
7. 正式运行前完整生成 provenance bundle，数据指纹必须包括 split、caption、
   原始 manifest 和质量分文件。

## 4. 数据适配

### 4.1 LLCM 和 RegDB

直接解析官方训练索引：

- LLCM: `idx/train_vis.txt` 和 `idx/train_nir.txt`；
- RegDB: `idx/train_visible_<trial>.txt` 和 `idx/train_thermal_<trial>.txt`。

适配器只保存路径和元数据，图像在 batch 提取时流式解码，不将整个数据集
预先转换为内存数组。BLIP caption 从主工作区的绝对 `text_root`
`/home/cgv841/datasets/<name>/Text` 解析；这些大文本产物未跟随 Git worker
复制，因此不依赖 worker 内的相对路径。

### 4.2 HITSZ-VCM

使用 `info/train_name.txt` 和 `info/track_train_info.txt`。当前服务器数据中有 39 个
`track_train_info` 区间与文件名中的 PID/模态边界不一致，共涉及 745 帧。
直接照该区间加载会混合身份。因此适配器会审计 `track_train_info` 的范围，
但以官方 `train_name.txt` 中的 `(PID, M, D, T)` 四元组重建单模态 tracklet，
避免静默的标签污染。`M1` 表示 IR，`M2` 表示 RGB。

每个 tracklet 按时间分成最多 `frames_per_tracklet=8` 个连续区间：

- 有有限质量分时选每段最高分，并以较早帧解决平局；
- 无质量分时选中心帧；
- 短轨迹不复制帧；
- 保存 source index、frame index、tracklet ID 和原始轨迹帧数。

特征先形成 tracklet 单模态原型，再在同一 PID 内按模态形成 identity 原型。
筛选和 RGB–IR 一致性仅在 identity 级进行；tracklet 级只输出同模态目标域距离。

## 5. 特征提取

保留当前模型接口：

```python
raw_image = model.extract_global_feat(model.encode_image_featmap(images, modality.lower()))
post_image = model.classifier(raw_image, modality)
raw_text = model.encode_text_feat(tokens)
post_text = model.classifier(raw_text, "Text")
raw_fusion = model.encode_fusion(tokens, ir_images, mode="ir")
post_fusion = model.classifier(raw_fusion, "Fusion")
```

每个身份最多选择 `captions_per_identity` 条去重后 caption。Fusion 对每张 IR 图像
使用该身份的所有入选 RGB caption，对模型输出求均值，从而避免随机配对。

## 6. SYSU 目标域校准

使用 `reports/feature_domain_gap/TRAIN-3-H1/feature_cache/train_*_post_bn.npz`。当前缓存
覆盖 395 个官方训练身份，特征维数为 768。

对每个模态，先构建 395 个身份原型，然后进行 full-reference leave-one-out：

\[
d_{i,m}^{LOO}=\min_{t\ne i}\left(1-\cos(p_{i,m},p_{t,m})\right)
\]

输出 q50/q90/q95/q99/mean/std。外部身份同样与全部 SYSU 训练原型比较，
从而避免 50% reference 和 100% reference 导致的最近邻尺度偏差。

对 SYSU 相同 PID 计算 RGB–IR、RGB–Text、RGB–Fusion 和 IR–Fusion 距离分布。

## 7. 身份级评分与过滤

外部身份原型到 SYSU 同模态身份原型的最近距离为 `d`，以 SYSU LOO q95
校准：

\[
z_{g,m}=\frac{d_{g,m}}{Q_m^{LOO}(0.95)+\epsilon}
\]

默认候选要求：

- identity 同时具有 RGB 和 IR；
- 每个模态至少 2 个代表样本；
- `rgb_normalized_distance <= 1.25`；
- `ir_normalized_distance <= 1.25`；
- `rgb_ir_normalized_distance <= 1.25`；
- 先在 record 级删除 synthetic，不因身份历史上存在 synthetic 而删除其真实样本。

文本和 Fusion 只作附加字段，不影响默认视觉候选资格。

## 8. 目标覆盖选择

为保证所有候选在同一目标函数下可比，默认覆盖模态固定为 RGB+IR，
Fusion 不参与主选择。候选 `g` 到 SYSU 身份 `t` 的代价为：

\[
c(g,t)=\frac{1}{2}\sum_{m\in\{RGB,IR\}}
\frac{1-\cos(p_{g,m},p_{t,m})}{Q_m^{LOO}(0.95)+\epsilon}
\]

\[
J(S)=\operatorname{mean}_t\min_{g\in S}c(g,t)
+\operatorname{P95}_t\min_{g\in S}c(g,t)
\]

第一步直接选择使 `J({g})` 最小的候选；后续每步最大化 `J` 降低量。
平局时依次使用平均 normalized distance、与已选候选的最小距离、质量分和
稳定字典序。

默认输出 eligible pool 的 25%/50%/75%/100%。`selected_100` 表示全部 eligible 身份；
100% 不另造无意义的随机对照。其余随机对照保持各数据集身份数与正式选择一致。

## 9. 数据集级指标

使用身份原型，不使用全部帧，计算 center cosine、MMD²、FFD、交叉验证
域分类 AUC、NN 分布和 kNN mixing。输出：

```text
mmd2_raw
mmd2_nonnegative = max(mmd2_raw, 0)
domain_auc_symmetric = max(auc, 1-auc)
domain_separability = 2*abs(auc-0.5)
```

外部身份与 SYSU 身份没有正配对，因此不计算伪造的跨数据集 hard-negative
positive margin。改为输出最近和次近 SYSU 原型之间的 target-assignment margin。

## 10. Manifest 语义

图像级数据在身份入选后输出该身份的全部官方训练图像；HITSZ-VCM 只输出
确定性选中的代表帧。每条记录必须包含：

```text
dataset, split, dataset_root, path, resolved_path,
source_pid, global_pid, modality, camid, tracklet_id,
frame_index, source_index, selected_frame, tracklet_frame_count,
caption, caption_path, quality_score, is_synthetic,
selection_unit, selection_budget, selection_rank, coverage_gain
```

`global_pid_mapping.json` 单独保存并参与产物哈希，不依赖运行时隐式偏移。

## 11. 阶段与验收

```text
prepare: 只解析官方训练 split，生成标准记录
extract: 流式提取 pre_bn/post_bn 特征
score:   SYSU LOO 校准、数据集/身份/tracklet 评分
select:  RGB+IR 身份级覆盖选择和 JSONL 输出
all:     依次执行上述阶段
```

必须通过：校准、稳定性、身份 PID 隔离、train-only、HITSZ 确定性抽帧、
tracklet 单模态不被误删、缺失 Text 视觉选择、覆盖贪心和缓存维度测试。

## 12. 正式运行溯源

每次正式运行必须在进程启动前原子生成：

```text
manifest.json, design.md, status.json, events.jsonl, launcher.log,
command.txt, runtime_config.yaml, config_diff.yaml, code.patch,
source_state.json, environment.json, dataset_fingerprint.json,
artifact_hashes.json
```

`manifest.json` 启动后不可改写；阶段进度只写入 `status.json` 和 `events.jsonl`。


---

## Migrated source: llcm caption quality screening

> Source document ID: `source_core:docs/llcm_caption_quality_screening.md`  
> Original SHA-256: `6381524e5cb0b5685e268c5b9520519d8b2d0776e7ca01f7fac9024f3ec88923`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# LLCM 文本描述质量筛选机制设计

## 1. 目标与原则

本机制用于在 LLCM 跨域训练前识别明显错误、内部矛盾或低可信的文本描述。核心方法为：

> 结构化属性抽取 + 模态分层的身份共识 + 可靠性加权离群检测 + 图文复核

第一阶段以规则和统计为主，不直接删除原始数据，只生成质量等级、异常原因和可复现的过滤清单。严格版与放宽版 LLCM 数据必须使用同一套属性字典、权重和判定阈值，避免文本清洗成为额外实验变量。

## 2. 输入单位与输出单位

每条输入记录至少包含：

- `dataset`、`source_pid`、`modality`；
- 图像路径、原始描述；
- 摄像头、帧号等可用元数据；
- 可选的图像质量、遮挡和可见性分数。

输出保留原始字段，并追加：

- 结构化属性及抽取置信度；
- 图像与模态可靠性；
- 身份共识及共识强度；
- 属性级冲突、总异常分数和原因码；
- `normal`、`suspicious` 或 `high_risk` 等级；
- 建议动作：保留、降权、图文复核或排除。

## 3. 结构化属性体系

### 3.1 属性分组

| 组别 | 属性 | 稳定性 | 默认重要性 |
|---|---|---:|---:|
| 人体核心 | 性别表达 | 高 | 1.00 |
| 人体核心 | 发型/头发长度 | 中 | 0.45 |
| 主体服装 | 上衣类型 | 高 | 0.90 |
| 主体服装 | 上衣主色 | 高（RGB） | 0.80 |
| 主体服装 | 下装类型 | 高 | 0.85 |
| 主体服装 | 下装主色 | 高（RGB） | 0.75 |
| 局部属性 | 鞋子类型 | 中 | 0.45 |
| 局部属性 | 鞋子颜色 | 中（RGB） | 0.35 |
| 携带物 | 背包、手提包 | 中 | 0.55 |
| 瞬时物品 | 手机、瓶子、雨伞等 | 低 | 0.20 |

颜色、衣物和携带物允许多标签；无法可靠判断时必须输出 `unknown`，不得强制猜测。

### 3.2 归一化

建立固定本体和同义词表，例如：

- `grey`、`gray` 统一为 `gray`；
- `trousers`、`pants` 统一到上位类 `pants`，同时保留细分类；
- `handbag`、`shoulder bag`、`sling bag` 统一到 `bag` 的子类；
- `hoodie jacket` 可同时标记 `hoodie` 与 `jacket`，避免层级词被误判为冲突。

属性抽取结果推荐采用如下形式：

```json
{
  "upper_type": {"value": ["jacket", "hoodie"], "confidence": 0.91},
  "upper_color": {"value": ["gray"], "confidence": 0.87},
  "gender": {"value": "male", "confidence": 0.76}
}
```

初版可使用规则、词典和模板解析；复杂句子可由受约束的语言模型补充解析，但必须输出置信度并经过 JSON Schema 校验。

## 4. 可靠性建模

对描述 (i) 的属性 (k)，定义有效可靠性：

\[
r_{ik}=c^{text}_{ik}\cdot q^{image}_{ik}\cdot q^{modality}_{k}\cdot q^{visibility}_{ik}
\]

其中：

- (c^{text}_{ik})：属性抽取置信度；
- (q^{image}_{ik})：图像清晰度、尺寸和遮挡质量；
- (q^{modality}_{k})：模态对该属性的可靠性；
- (q^{visibility}_{ik})：对应身体区域是否可见。

建议初始模态可靠性如下，后续用人工标注校准：

| 属性 | RGB | IR |
|---|---:|---:|
| 性别表达 | 1.00 | 0.60 |
| 发型 | 0.85 | 0.40 |
| 上/下装类型 | 1.00 | 0.85 |
| 上/下装颜色 | 1.00 | 0.00 |
| 鞋子 | 0.75 | 0.60 |
| 大型包具 | 0.85 | 0.70 |
| 小型瞬时物品 | 0.50 | 0.30 |

IR 中的颜色值可以记录，但不参与颜色共识和冲突惩罚。

## 5. 身份共识建立

### 5.1 分层共识

共识分三层计算：

1. RGB 身份共识：颜色、服装类型和携带物的主要依据；
2. IR 身份共识：只用于衣物轮廓、鞋子、大型包具等非颜色属性；
3. 跨模态共识：仅对两个模态都可靠的属性进行复核。

颜色不得用 RGB 与 IR 简单合并统计。对于同一身份，应先按模态统计，再在允许的属性上融合。

### 5.2 加权概率

对身份 (p)、属性 (k) 和候选值 (v)，使用可靠性加权频率：

\[
P(v\mid p,k)=
\frac{\alpha+\sum_i r_{ik}\,\mathbb{1}(a_{ik}=v)}
{\alpha |V_k|+\sum_i r_{ik}}
\]

其中 \(\alpha\) 为平滑系数，初始可取 0.5；`unknown` 不进入分母。共识值为概率最大的候选值，共识强度为其概率。

只有同时满足以下条件时，才把某属性视为可靠共识：

- 有效支持样本不少于 5 条；
- 有效权重和不少于 3.0；
- 核心属性共识强度不低于 0.75；
- 非核心属性共识强度不低于 0.65。

不满足条件时该属性标记为“共识不足”，不据此处罚单条描述。

### 5.3 防止多数错误

采用两轮稳健估计：

1. 第一轮仅使用高质量 RGB 图像和高置信属性建立种子共识；
2. 根据种子共识计算初始异常分数；
3. 暂时移除高风险记录后重新计算共识；
4. 最多迭代两次，避免少数异常污染共识，也避免循环放大。

## 6. 异常分数

### 6.1 属性级冲突

对非 `unknown` 属性定义：

\[
d_{ik}=-\log\left(\max(P(a_{ik}\mid p,k),\epsilon)\right)
\]

相比简单的 (1-P)，负对数能更强地惩罚极低概率的核心冲突。建议取 \(\epsilon=0.05\)。

### 6.2 总分

\[
S_i=
\frac{\sum_{k\in K_i} w_k\,r_{ik}\,d_{ik}}
{\sum_{k\in K_i} w_k\,r_{ik}}
\]

其中 (K_i) 只包含“属性已识别且身份共识可靠”的项目。若有效属性少于 2 项，则不做自动高风险判定，只标记为 `insufficient_evidence`。

为便于阈值解释，可将分数映射到 0～1：

\[
S_i^{norm}=1-\exp(-S_i)
\]

### 6.3 硬冲突规则

满足以下任一条件时，可直接进入高风险候选，但仍保留人工或图文复核入口：

- 身份性别共识强度不低于 0.85，单条高置信描述与其冲突；
- RGB 上衣或下装主色与强共识冲突，且对应区域清晰可见；
- 上衣类型和下装类型两个核心属性同时冲突；
- 描述出现多个相互矛盾主体，或一句中包含两套完整人物描述；
- 文本为空、严重截断或包含已知异常生成模板。

瞬时物品缺失、IR 颜色、轻微颜色近邻差异不得触发硬冲突。

## 7. 三级判定与动作

初始阈值仅作为启动值，需用人工抽样校准：

| 等级 | 初始条件 | 建议动作 |
|---|---|---|
| 正常 | (S^{norm}<0.25)，且无硬冲突 | 直接保留 |
| 可疑 | (0.25\le S^{norm}<0.55)，或证据不足 | 训练降权、图文模型复核或人工抽查 |
| 高风险 | (S^{norm}\ge0.55)，或命中硬冲突 | 默认不进入训练，复核通过后恢复 |

降权训练时可使用：

\[
w_i^{train}=\max(0.2,1-S_i^{norm})
\]

第一版建议只自动排除“高风险且图文复核也不通过”的记录，其余记录保留或降权，避免清洗过度。

## 8. 图文复核阶段

统计筛选完成后，再对 `suspicious` 和 `high_risk` 使用图文相似度模型：

1. 计算原图与完整描述的相似度；
2. 将描述拆成属性短语，分别计算局部一致性；
3. 与同身份正常描述的相似度分布比较，而不是使用全局固定阈值；
4. RGB 用于颜色和细粒度属性复核，IR 只复核轮廓、衣物类型和大型携带物。

统计异常但图文高度一致的记录应进入人工抽查，而不是自动删除，因为它可能是身份内真实的少数变化。

## 9. 防误删规则

- 使用层级本体判断兼容性，例如 `hoodie` 与 `jacket` 可兼容；
- 邻近颜色可设置较低惩罚，如 `gray/black`、`navy/blue`；
- 外套开合、背包暂时不可见、手持物变化属于合理变化；
- 摄像头或时间跨度较大时，可先建立子簇共识，再判断是否存在真实换装；
- 低清晰度、背面或严重遮挡图像只降低证据权重，不作为其他描述的共识来源；
- 共识不足的身份不自动清洗，仅进入人工抽样。

## 10. 输出 JSONL 建议格式

```json
{
  "source_pid": "101",
  "modality": "rgb",
  "path": "vis/...jpg",
  "caption_original": "...",
  "attributes": {
    "gender": {"value": "female", "confidence": 0.91},
    "upper_type": {"value": ["jacket"], "confidence": 0.88}
  },
  "reliability": {"image": 0.83, "visibility": 0.76},
  "consensus_conflicts": ["gender"],
  "reason_codes": ["GENDER_STRONG_CONFLICT"],
  "anomaly_score": 0.72,
  "quality_level": "high_risk",
  "recommended_action": "vision_text_review"
}
```

建议同时产出：

- `caption_quality_scores.jsonl`：逐条完整审计结果；
- `caption_quality_summary.json`：整体与分模态统计；
- `identity_consensus.json`：身份属性共识；
- `accepted.jsonl`、`suspicious.jsonl`、`high_risk.jsonl`；
- `manual_review_sample.jsonl`：分层人工复核样本。

原始 selection JSONL 不覆盖，所有过滤结果通过新文件派生，并记录输入文件 SHA-256、配置、代码版本和随机种子。

## 11. 验证与阈值校准

首次运行后，至少人工标注 200 条分层样本：

- 正常、可疑、高风险各抽取一定数量；
- RGB、IR 分开抽样；
- 严格33与放宽207都要覆盖；
- 记录属性抽取是否正确、描述是否与图像一致、最终是否应保留。

优先优化以下指标：

- 高风险集合准确率不低于 90%；
- 正常集合误放率可控；
- 各模态和各身份的保留率无异常偏斜；
- 清洗后性别少数错误率与 RGB/IR 主属性冲突显著下降；
- 严格版与放宽版使用同一阈值时结果稳定。

最终通过 SYSU-only、LLCM 原始文本、LLCM 清洗文本三组训练比较 Rank-1、mAP 和 mINP，确认文本清洗带来的实际收益。

## 12. 推荐实施顺序

1. 固定属性本体、同义词表和 JSON Schema；
2. 对全量 LLCM 训练文本统一抽取属性；
3. 建立模态分层身份共识并执行两轮稳健估计；
4. 生成三级统计筛选结果；
5. 人工校准阈值和属性权重；
6. 对可疑/高风险记录执行图文复核；
7. 生成不覆盖原始数据的派生训练清单；
8. 将同一质量清单分别与严格、放宽 coverage 结果连接，开展受控跨域训练。
