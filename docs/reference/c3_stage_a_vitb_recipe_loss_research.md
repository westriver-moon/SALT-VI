# C3 StageA：ViT-B 训练配方与损失研究

更新：2026-08-30 12:56（北京时间）
状态：第一轮 13 个训练与 2 个诊断正在执行。本文区分已验证结果、运行中信号和待验证假设。

## 1. 结论与研究优先级

C3 当前并非原封不动复现 PMT，但核心训练逻辑仍继承自 PMT：灰度课程、PMT hard triplet、`MSEL=0.5`、`DCL=0.5` 和 24 epoch。它适合作为强基线，不宜直接作为论文方法。

StageA 应依次研究以下问题：

1. **跨模态对齐对象**：比较 MSEL/DCL 的点式或中心式约束，与只保持身份关系几何的 relation loss。
2. **困难样本挖掘**：比较 PMT mixed hard、双向 cross-modal hard 和 WRT；先在 cross-modal hard 内筛选 margin。
3. **课程调度**：分离灰度转真实 RGB、启用跨模态 mining、启用 auxiliary loss 三个时间点。
4. **训练预算**：仅对筛出的完整 recipe 比较 24/30/36 epoch，以及每身份正样本数 4/8。
5. **ViT 特有损失**：global relation 获得实证支持后，再研究浅层至深层的关系轨迹保持。

学习率、weight decay、cosine classifier scale 和 label smoothing 放在最后。它们可能改善数值，但无法回答 PMT 是否过对齐，也难以形成方法贡献。

## 2. C3 固定条件与 PMT 继承项

第一轮实验只改变 recipe/loss，以下条件固定：

| 项目 | C3 StageA 设定 |
|---|---|
| 数据与评测 | SYSU all-search、single-shot、10 gallery trials 聚合，IR query 检索 RGB gallery |
| 主干 | ViT-B/16，768维，12层，12 heads，patch 16，stride 12 |
| 输入 | 256×128 RGB/IR 经 SwinIR ×2 预处理为 512×256 |
| 训练模式 | `RGB_IR`、`image_only`，冻结 text，纯视觉特征评测 |
| 采样 | 每模态 batch 32，`num_pos=4`，camera-diverse，seed 0 |
| 分类器 | normalized cosine classifier，scale 30 |
| 优化器 | AdamW，LR `3e-4`，WD `1e-4`，3 epoch warm-up，cosine，24 epoch |
| PMT-derived baseline | epoch 0–5 灰度；随后真实 RGB；ID + PMT hard + `0.5*MSEL + 0.5*DCL` |

配置中的 `retrieval_backend=identity_text` 是历史协议标识。当前 query 和 gallery 均只读取视觉特征，不使用 caption、text 或 fusion feature。

PMT 官方论文和代码确认了 ViT-B/16、stride 12、6 epoch 灰度阶段、margin 0.1、`MSEL=DCL=0.5` 与 24 epoch。因此这些数值只能被标注为 inherited baseline。[PMT 论文](https://ojs.aaai.org/index.php/AAAI/article/download/25273/25045)；[PMT 代码](https://github.com/hulu88/PMT)

## 3. PMT loss 是否阻碍模态独有信息

### 3.1 机制判断

MSEL 比较跨模态同身份距离与模态内同身份距离：

\[
L_{MSEL}=\frac12\left[(\bar d^{v\rightarrow r}_{+}-\bar d^{v\rightarrow v}_{+})^2+
(\bar d^{r\rightarrow v}_{+}-\bar d^{r\rightarrow r}_{+})^2\right].
\]

DCL 以 RGB+IR 联合身份中心计算正样本距离，并以负样本距离归一化：

\[
L_{DCL}=\bar d_{+}(c_y,x)/\bar d_{-}(c_y,x).
\]

两者都把优化重心放在单一联合几何中。若 RGB 颜色/纹理残差或 IR 热响应同时携带身份信息，较强的 MSEL/DCL 可能压制这些线索。这是明确的过对齐风险，但公式本身不能证明模态特有信息一定被消除。

正确目标也不是最大化模态可分类性。MixER 表明 modality-erased 与 modality-related 表征可以互补；对于当前纯 IR→RGB 协议，残余模态信息降低既可能是有效对齐，也可能是身份相关差异受损。[MixER, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Alehdaghi_MixER_From_Cross-Modal_to_Mixed-Modal_Visible-Infrared_Re-Identification_WACV_2026_paper.pdf)

### 3.2 当前诊断

| 诊断 | 结果 | 可支持的结论 |
|---|---|---|
| 最后两层、2 batch 梯度余弦 | ID–MSEL `-0.0101`；ID–DCL `0.0758`；MSEL–DCL `0.0933` | MSEL 与 ID 在该局部样本上有轻微方向冲突；样本太小，不能推断全网络或统计显著性 |
| 6 batch、47 ID 表征诊断 | 配对中心 cosine `0.480→0.981`；中心距离 `1.016→0.193`；关系 Pearson `0.186→0.988` | PMT/C3 显著增强共享身份几何 |
| residual modality probe | 跨 ID 测试准确率 `1.000→0.608` | 可线性读出的模态残差降低，但仍高于二分类随机水平 |

现有证据支持“PMT/C3 对齐很强”，不支持“PMT 已经毁掉模态独有信息”。因果判断必须依赖 `l0/l1/l2/l4` 与 `s0/s1/s2`，并联合报告检索指标、center relation、residual modality probe 和 within-ID dispersion。

## 4. 文献证据如何约束 StageA 设计

### 4.1 直接、可复现的 ViT-B VI-ReID 证据

| 工作 | 对 StageA 的有效启示 | 使用边界 |
|---|---|---|
| [PMT, AAAI 2023](https://github.com/hulu88/PMT) | 灰度课程、overlap stride、ID+metric 强基线 | 当前研究要摆脱的 inherited recipe |
| [MIP, ICMR 2024 Oral](https://github.com/wurqjackey/MIP) | modality/instance prompt、更多同 ID 样本、更长训练可与 MSEL 共存 | 额外参数、80 epoch 和不同输入使其不能与 C3 直接比数值 |
| [SDCL, CVPR 2024](https://github.com/yangbincv/SDCL) | 浅层 patch 与深层 feature 都含跨模态信息 | 无监督设定；支持跨层建模，不是 C3 数值基线 |
| [TokenMatcher, AAAI 2025](https://github.com/liulekai123/TokenMatcher) | 单一 global CLS 可能遗漏局部对应，多 token/邻域关系有价值 | 无监督设定；不能照搬 token 结构后声称新颖 |

### 4.2 方法边界与反向证据

- [CMT, ECCV 2022](https://mlanthology.org/eccv/2022/jiang2022eccv-crossmodality/) 与 [MUN, ICCV 2023](https://arxiv.org/abs/2309.06262) 支持“shared-only 对齐可能抑制身份相关模态线索”的问题定义，但不能替代 C3 上的因果实验。
- [IDKL, CVPR 2024](https://github.com/1KK077/IDKL) 通过双流分离 shared/modality-specific feature。C3 若保持单 ViT，应以关系约束而非双流蒸馏作出区分。
- [GUR, ICCV 2023](https://github.com/yangbincv/GUR)、[BMIL, AAAI 2026](https://github.com/Terminator8758/BMIL) 与 [WSL-VIReID, ICCV 2025](https://github.com/KongLingqi2333/WSL-VIReID) 已研究 affinity、prototype 或 prediction relation。RPSA 不能笼统声称首次使用 relation consistency；其限定应是“有监督 paired-ID、单 ViT、无额外参数的身份几何约束”。
- [PCLHD, NeurIPS 2024](https://github.com/shijiangming1/PCLHD) 与 [IRD/DEN, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Enhancing_Diverse_Intra-Identity_Representation_for_Visible-Infrared_Person_Re-Identification.html) 表明 identity center 可能遗漏样本多样性。因此 relation loss 的结果必须同时报告 within-ID dispersion，不能默认表征越紧凑越好。
- [TOP-ReID, AAAI 2024](https://github.com/924973292/TOP-ReID) 已使用跨谱 token permutation 与 dense reconstruction。后续 ViT 方法应保持“关系轨迹”而非转向 token 重建。
- [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)、[PASS, ECCV 2022](https://github.com/CASIA-LMC-Lab/PASS-reID) 和 [Part-Aware Transformer, ICCV 2023](https://github.com/liyuke65535/Part-Aware-Transformer) 支持 person-domain 初始化、part token 和局部蒸馏。这些属于第二阶段结构/初始化研究，不应混入当前 loss 筛查。
- [TVI-LFM, NeurIPS 2024](https://github.com/WHU-HZY/TVI-LFM) 采用语义补偿而非抹除差异。它需要文本生成，应作为独立方向，不与纯视觉 StageA 共用主消融表。

## 5. 候选方法：RPSA

### 5.1 三阶段 recipe

| epoch | 训练目标 | 目的 |
|---|---|---|
| 0–3 | 灰度 RGB–IR；ID + 模态内 hard triplet | 降低初始颜色域差异，建立各模态身份结构 |
| 4–5 | 真实 RGB–IR；ID + 双向 cross-modal hard triplet | 先适应真实输入，并直接优化 IR↔RGB 检索边界 |
| 6–8 | relation 和 intra term 按 `1/3, 2/3, 1` ramp | 避免 RGB 切换与辅助约束同时突变 |
| 9–23 | 完整目标 | 稳定跨模态检索与模态内判别结构 |

令 \(c_y^v,c_y^r\) 为 batch 内身份 \(y\) 的归一化 RGB/IR 中心，\(G^m=C^m(C^m)^T\)。去掉对角线后：

\[
L_{rel}=\operatorname{Huber}\left(\operatorname{offdiag}(G^v),
\operatorname{offdiag}(G^r)\right).
\]

RPSA v0 的目标为：

\[
L=L_{id}+L_{x\text{-}tri}+a_t\left[
0.25(L^v_{intra\text{-}tri}+L^r_{intra\text{-}tri})+
\lambda_{rel}L_{rel}\right],
\quad \lambda_{rel}\in\{0.05,0.10\}.
\]

它不要求 RGB/IR feature 坐标重合，只要求身份之间的相对几何一致。当前实现不增加可训练参数；`PMTIdentityRelationLoss` 的 `state_dict` 为空。

### 5.2 候选升级：LRTP

只有 global relation 获得正向证据后，才研究 Layerwise Relation-Trajectory Preservation：

\[
L_{LRTP}=\lambda_sH(\bar G_v^s,\bar G_r^s)
+\lambda_dH(\bar G_v^d,\bar G_r^d)
+\lambda_\Delta H(\bar G_v^d-\bar G_v^s,\bar G_r^d-\bar G_r^s).
\]

该目标约束浅层到深层的身份关系演化，不要求不同模态的 token 坐标一致。实现前必须满足两个门槛：

1. `s1` 或 `s2` 相对 `s0` 至少不劣，并有实际检索收益；
2. frozen checkpoint 的 block-4/block-12 probe 显示跨层关系差异存在，且与 Rank-1 变化相关。

## 6. 第一轮实验与当前结果

### 6.1 13 个训练与 2 个诊断

| 研究问题 | 变体 | 严格比较对象 |
|---|---|---|
| gray/schedule | `r0/r1/r2` | PMT gray6 coupled；gray4 coupled；gray4 decoupled+ramp |
| auxiliary necessity | `l0/l1/l2/l4` | `0/0`、`.5/0`、`0/.5`、`.5/.1`，其余条件相同 |
| mining margin | `m1/m2/m3` | 双向 cross-modal hard，margin `.05/.10/.20` |
| relation causality | `s0/s1/s2` | relation `0/.05/.10`，其余条件完全相同 |
| 诊断 | `d0/d1` | 梯度冲突；模态与身份几何 |

`r0` 与 `l0` 不是单变量 loss ablation，因为 schedule 不同。relation 只能由 `s0/s1/s2` 归因。

### 6.2 选择规则

所有结果采用 SYSU all-search、single-shot、10 gallery trials aggregate。在每条训练的评测 epoch 中选择最高 Rank-1；完全相同则选较晚 epoch。mAP、mINP 用于解释，不参与当前 tie-break。完整曲线必须保留，不能用最终 epoch 替代最佳评测 epoch。

### 6.3 结果快照

截至 2026-08-30 12:56（北京时间），4/13 个训练完成，2 个运行，7 个排队；2/2 个诊断完成，无失败。

| 变体 | 状态 | 当前/最终最佳 epoch | Rank-1 | mAP | mINP | 解释边界 |
|---|---|---:|---:|---:|---:|---|
| `r0_current` | 完成 | 23 | 70.04 | 68.43 | 56.76 | PMT-derived C3 baseline；最终评测仍创新高，可能 under-trained |
| `r1_gray4_coupled` | 完成 | 19 | 68.19 | 67.00 | 55.50 | 单纯缩短灰度阶段未优于 r0 |
| `r2_gray4_decoupled_ramp3` | 完成 | 23 | 68.00 | 66.90 | 55.66 | gray4 下 decoupled+ramp 未优于 r0 |
| `l0_no_aux` | 完成 | 23 | 63.63 | 61.88 | 49.07 | gray4/decoupled 条件下完全删除 auxiliary 明显较弱 |
| `l1_msel_only` | 运行至 e9 | 9 | 62.53 | 59.59 | 45.52 | 尚未完成，不能用于最终选择 |
| `l2_dcl_only` | 运行至 e7 | 7 | 58.68 | 57.60 | 45.12 | 尚未完成，不能用于最终选择 |

当前只能确认：gray4 的两个 schedule 没有胜过 r0；完全删除 auxiliary 在对应控制条件下不足。MSEL、DCL、cross-modal mining 和 relation 的最终效果仍未确定。

## 7. 队列结束后的决策规则

| 问题 | 升级条件 | 下一步 |
|---|---|---|
| MSEL/DCL | `l1/l2/l4` 相对 `l0` 至少 `+0.5pp` Rank-1，且 mAP 不下降超过 `0.2pp` | 保留胜出的最小 auxiliary；否则不把中心约束作为方法主张 |
| cross-modal margin | 在 `m1/m2/m3` 中按 Rank-1 选择 | 若三者均不胜 r0，不扩展连续 margin 网格 |
| RPSA v0 | `s1/s2` 相对 `s0` 至少 `+0.5pp` Rank-1，且 mAP 不降 | 进入跨 schedule、时长、多 seed 与 RegDB 验证 |
| LRTP | RPSA 有正向信号，且跨层 probe 支持关系轨迹假设 | 再增加训练 hook 和跨层 loss |

若 relation 相对 `s0` 有效但仍低于 `r0`，先补两个交叉实验：gray6 + `s0`，以及 gray6 + relation winner。它们用于区分 relation 失效和 gray4 拖累，不与其他超参数同时改变。

若 relation 不通过门槛，第二轮优先补：

1. `MSEL=.25,DCL=.25`；
2. 相同 schedule/loss 下的 PMT-hard、WRT、cross-modal-hard；
3. 固定胜出的 loss/mining 后，再测试 gray=2/8，并为变化后的 gray 设置 matched control。

只有确定完整 loss/mining/schedule 后，才对 finalist 与 strongest control 成对测试 24/30/36 epoch、`num_pos=4/8`。所有投稿候选至少需要 SYSU 3 seeds 的均值±标准差，以及 RegDB 两方向/10 trials 的独立验证。

## 8. 投稿所需的最小证据链

| 命题 | 必须提供的证据 |
|---|---|
| relation 产生独立收益 | `s0` vs `s1/s2`，相同 schedule、mining、seed 和容量 |
| 收益不是训练更久 | candidate 与 strongest control 从头成对训练 24/30/36 epoch |
| 收益不是额外容量或文本 | 新增可训练参数为 0；无 caption/text；报告训练耗时 |
| 结果不是 seed 或数据集偶然 | SYSU、RegDB，多 seed 均值±标准差与逐 seed best epoch |
| 保留的是身份相关差异 | 检索指标 + center relation + residual modality probe + within-ID dispersion |

RPSA 当前是可反驳的候选，不是已成立的新方法。可投稿的核心命题应限定为：**在有监督 paired-ID、单 ViT 的纯视觉 VI-ReID 中，用身份关系保持替代强点式对齐，在不增加参数的条件下改善跨模态检索，并保留有用的模态内身份结构。**

## 9. 复现位置

- 工作树：`/home/lab929/ybj/autoresearch-v2/worktrees/c3-recipe-loss-research-20260829/w1`
- pipeline：`configs/pipelines/c3_recipe_loss_research_20260829.yaml`
- 配置：`configs/stage_a/recipe_research/`
- recipe/loss：`src/salt_vi/training/recipes.py`、`src/salt_vi/utils/loss.py`
- 诊断：`src/salt_vi/diagnostics/c3_recipe.py`
- 测试：`src/salt_vi/tests/test_c3_recipe_loss_research.py`（10 passed）
- 调度状态：`/home/lab929/ybj/experiments/stage_a/SALT-VI-c3-recipe-loss-research-20260829/scheduler/state.json`
- 结构化事件：上述实验目录下的 `events/*.jsonl`
