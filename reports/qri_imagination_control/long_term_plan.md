# QRI 文本级想象：长期实验计划（像素改写归档）

最后更新：2026-08-22
目标服务器：`lab929-3090`（物理 GPU0）
远端仓库：`/home/lab929/ybj/SALT-VI`
远端实验根目录：`/home/lab929/ybj/experiments/qri_imagination_control/`

## 一、长期目标

充分利用 VLM 对模糊视觉内容的推理与想象能力，在文本层面产生清晰、可归因、带概率
且可回退的细颗粒语义描述；保留原始/SwinIR 图片及其空间结构，不再让 VLM 想象指导
扩散模型生成或改写像素。

眼镜不是机制的最终对象，只是当前压力测试样本。实现必须使用通用的编辑程序、
空间控制图和源结构锁定接口，不能把算法写成眼镜专用模板。

自 2026-08-22 的跨样本验证起，主线范围收缩为**文本级 VLM 想象**：保留 VLM 对
模糊证据的抽象、世界知识推理、合理想象与概率世界采样，但不再让想象结果指导
SD/PASD 或任何图片生成。SwinIR 图像保持不变；VLM 只产生可区分来源、带概率且可
回退的文本假设，供后续 RGB-Text、IR-Text 或检索分数融合使用。此前像素改写阶段作为
否定结果和机制证据保留，不再是全数据集默认路线。

新阶段尤其关注两类能力：一是利用足够的抽象线索和世界知识识别熟悉对象或文化图案，
例如背包上的“樱桃小丸子”式人物特征；二是在像素不足时给出若干互斥且概率化的合理
解释，而不是强行选定一个事实。

## 二、固定测试样本

- LR：`/home/cgv841/datasets/SYSU-MM01/cam1/0001/0001.jpg`
- Swin 基线：`/home/lab929/ybj/experiments/archive/qri_v2/qri-v2-visual-demo/sources/cam1/0001/0001/swin_reference.png`
- 旧眼区软掩膜：`/home/lab929/ybj/experiments/archive/qri_v2/qri-v2-localized-eye-20260820/eye_soft_mask.png`
- 已有 SD1.5 Inpainting：`/home/lab929/ybj/models/qri-diffusion-bases/stable-diffusion-v1-5-inpainting/`
- 旧基线结果：`/home/lab929/ybj/experiments/qri_glasses/qri-glasses-gpu0-20260821/`

## 三、历史像素改写失败诊断（归档）

1. `strength=0.85` 并非天然错误；错误在于它作用于覆盖眉眼和皮肤的大块二值掩膜。
2. 文本提示只描述“生成什么”，没有明确目标组成、位置、连接关系和遮挡关系。
3. 高强度去噪时源图潜变量被丢失，模型为满足目标语义而重画眉眼、脸型和局部肤色。
4. 最终固定透明度融合同时削弱伪影和目标语义，无法兼得清晰改写与结构保持。
5. 全身 ReID 和“存在眼镜”VLM 判定均不能独立证明局部编辑质量。

## 四、历史像素改写机制（归档）

### 4.1 通用 VLM EditSpec

VLM 输出结构化编辑程序：

- `target_semantics`：需要明确补全的语义；
- `region`：目标 ROI；
- `layout`：组成、数量、相对位置、方向和连接；
- `depth_order`：与已有结构的遮挡关系；
- `appearance`：材质、颜色、粗细、清晰度；
- `creation_map`：高自由度生成区域；
- `transition_map`：边界、光照和遮挡协调区域；
- `preservation_map`：必须从源图锁定的结构；
- `style_reference`：需要匹配的局部/全局风格。

### 4.2 三值空间控制

- creation：高强度语义生成，允许 `strength≈0.75–0.95`；
- transition：中低强度协调；
- preservation：每个去噪步骤恢复源图潜变量，不允许目标语义传播。

### 4.3 源轨迹锁定

从 Swin 裁剪图获得源潜变量和对应噪声轨迹。每步去噪后按三值控制图混合：

`z_t = M_creation * z_edit + M_transition * z_blend + M_preservation * z_source`

### 4.4 空间语义条件

EditSpec 的 layout 转为通用边缘、分割、关键点或语义布局图。目标 token 的交叉注意力
只允许影响 creation/transition 区域；源图结构条件覆盖完整裁剪。

### 4.5 风格协调与 VLM 闭环

第一阶段生成明确内容；第二阶段只协调颜色、锐度、噪声、光照和边界。VLM 对源图、
EditSpec 和候选做放大审查，返回缺失组成、几何错误、结构破坏和风格不一致区域；最多
执行一次局部纠错，避免无界多轮生成。

## 五、历史像素阶段成功标准（归档）

必须同时满足，不允许只用单一指标宣布成功：

1. **语义明确**：VLM 确认目标组成完整、清晰可辨、无歧义；当前样本需确认双镜片、
   鼻梁桥和镜腿关系明确。
2. **空间精准**：目标位置、尺度、透视、连接和遮挡与 EditSpec 一致。
3. **结构保持**：目标之外的脸型、头发、眉眼、姿态、衣服和背景无可见漂移。
4. **风格一致**：目标区域的颜色、锐度、噪声和光照与 Swin 基线协调，无高清贴片感。
5. **局部指标**：区外变化、非目标脸部差异、边界能量和身份指标不过门槛。
6. **视觉审计**：服务器端 VLM 审计通过；最终人工/本模型目视比较需遵守远端结果下载
   许可，未获许可前不得将 PNG 拉到本地。

## 六、实验阶段

### Stage 0：冻结基线与评估协议

- [x] 复用 `strength=0.30–0.85` 扫描和旧 VLM/ReID 结果。
- [x] 确认远端 `diffusers==0.29.2` 支持 `callback_on_step_end` 返回新 latents。
- [x] 盘点现有权重：未发现可直接复用的 ControlNet/T2I-Adapter，首轮不下载新权重。
- [x] 增加非目标脸部、边界和风格一致性指标。

### Stage 1：三值控制图，不增加新模型

- [x] 实现通用 EditSpec JSON 和 creation/transition/preservation 图生成接口。
- [x] 当前样本由 EditSpec 提供目标布局；算法本身不含眼镜类别分支。
- [x] 以高 creation strength、低 transition strength和源图 preservation完成首轮消融。

### Stage 2：逐步源潜变量锁定

- [x] 基于 diffusers step callback 实现源轨迹混合。
- [x] 数值对照确认非目标脸部变化从 0.255 降至 0.170/0.099，ReID 从 0.9882 升至 0.9921/0.9991。
- [x] 视觉确认：恒定逐步锁定会明显抑制目标；延迟锁定虽更合理，但本样本仍不如
  “强语义提案 + 精确像素回写”。

### Stage 3：目标布局条件与局部注意力

- [x] 找到仓库既有 PASD ControlNet/UNet；无需下载新权重。
- [x] 直接把抽象布局喂给 PASD 会复制机械线稿，弱化控制又产生马赛克，因此否决。
- [x] 改为 SD1.5 高强度生成照片式语义提案，再由 PASD 以高条件强度做结构/风格精修。

### Stage 4：VLM 审查与一次纠错

- [x] 多模态目视返回结构化缺陷：框色不一致、鼻梁结块、高清贴片感。
- [x] 执行一次提示纠错和一次 PASD 精修；提示重采样失败，PASD 精修通过样本门槛。
- [ ] 全数据集运行前恢复服务器 Qwen 服务，将当前人工多模态审计自动化。

### Stage 10：范围收缩——文本级概率想象

本阶段停止像素改写，仅复用此前两个有效机制：

1. **概率世界采样**：VLM 为互斥文本假设分配概率，程序按概率在本地采样语义世界，
   不通过重复生成图片来表达不确定性；
2. **ROI 选择与二次观察**：沿用人体解析、姿态、SAM/框选区域和模糊度排序，将选定
   ROI 以 `A-tight / A-context / B-tight / B-context` 四格放大板重新输入 VLM，使其对
   小图案、配件、鞋、携带物和局部结构进行更细颗粒度观察。

VLM 输出分为四层：

- `observations`：只记录 LR/Swin 对照中可见的强弱像素事实；
- `world_knowledge`：明确标记使用了何种对象、设计或文化知识以及与像素的关系；
- `hypotheses`：3–5 个互斥文本解释，包含概率、证据来源、可见支持和未决部分；
- `unresolved`：像素和知识仍无法决定的属性。

禁止覆盖率代码自动插入“看似具体”的正向候选。若需要保留先验模板，必须另标
`coverage_prior`，默认不进入事实 caption，也不获得文本融合权重。

### Stage 11：thinking 开关受控消融

在相同模型、输入图片、ROI、提示、seed、temperature 与 token 上限下，仅比较：

- `no_thinking`：`enable_thinking=false`、`reasoning_effort=none`；
- `thinking`：`enable_thinking=true`、`reasoning_effort=high`。

固定测试三类区域：背包卡通图案的世界知识抽象、模糊眼区的眼镜结构推断、手边深色
携带物的互斥合理解释。记录每次请求耗时、token 使用、观察项数、世界知识项数、候选
长度、概率熵、未决概率、JSON/证据字段完整性，并对具体描述做人工/多模态复核。

实测完成：`no_thinking` 平均 79.11 秒/ROI，`thinking=high` 平均 269.10 秒/ROI（3.40×）。
前者已能以 0.75 概率识别“樱桃小丸子”式背包图案；后者主要改善模糊携带物的概率
校准和眼镜组成拆分，但没有稳定提高具体识别，并引入过度解释风险。因此生产默认固定为
`no_thinking`，只对高价值且首轮假设接近、未决质量高或证据冲突的少量 ROI 触发一次
thinking 复核。完整结果见 `reports/qri_text_imagination/thinking_ablation_gpu0_20260822.md`。

文本阶段成功标准：

1. 不生成、修改或回写任何图片；
2. 每个 ROI 返回可归因的观察、知识推断与未决信息；
3. 候选概率可归一化并支持确定性复现实验采样；
4. 不把世界知识或通用先验冒充像素事实；
5. thinking 对具体识别或合理性有可见收益时，报告其相对时间成本；若无收益则默认关闭；
6. 后续文本融合必须保留纯视觉回退，不能让单个想象世界成为硬标签。

## 七、首轮受控消融

固定 seed、prompt、crop、scheduler 与 steps，避免混入无关变量：

1. `baseline_broad_s085`：旧大掩膜、全局 0.85；
2. `trimap_pixel_comp_s085`：高强度 creation，最终像素三值合成；
3. `trimap_step_lock_s085`：高强度 creation，逐步源潜变量锁定；
4. `trimap_step_lock_s095`：验证更强想象是否仍能保持结构；
5. `trimap_step_lock_corrective`：VLM 一次纠错。

首轮只保留能解释失败模式的少量候选；不做无边界参数扫描。

## 八、代码与产物规范

- 通用代码：`scripts/experiments/qri_imagination_control/`
- 实验配置：`configs/experiments/qri_imagination_control/`
- 计划和结论：`reports/qri_imagination_control/`
- 大型产物：`/home/lab929/ybj/experiments/qri_imagination_control/`
- 模型：`/home/lab929/ybj/models/`，不得进入 Git 仓库。
- 不创建分支或 worktree，除非用户另行明确批准。
- 不将远端模型或结果下载到本地，除非用户另行明确批准。
- 新文本实验代码：`scripts/experiments/qri_text_imagination/`；
- 新文本实验配置：`configs/experiments/qri_text_imagination/`；
- 新文本实验报告：`reports/qri_text_imagination/`；
- 新文本大型产物：`/home/lab929/ybj/experiments/qri_text_imagination/`。

## 九、实验记录

| 日期 | 实验 | 结论 | 状态 |
|---|---|---|---|
| 2026-08-22 | 旧 SD1.5 strength 扫描 | 低强度减少破坏但语义不够明确；高强度目标清晰但重画脸部 | 已完成 |
| 2026-08-22 | 远端实现盘点 | 9 通道 Inpainting 不自动执行 4 通道管线的逐步源 latent 混合，但 step-end callback 可覆盖 latents；服务器无现成结构控制权重 | 已完成 |
| 2026-08-22 | Stage 1/2 v1 | 三值像素合成 creation change 16.829、preservation ROI 0.255、ReID 0.9882；step lock s0.85 为 9.818/0.170/0.9921；s0.95 为 4.423/0.099/0.9991。锁定有效但可能抑制目标清晰度 | 待视觉审计 |
| 2026-08-22 | Stage 1/2 v2 | 修短正向提示并加入 token 上限检查，消除 95>77 截断。pixel s0.85 为 16.909/0.280/0.9884；lock s0.85 为 9.773/0.170/0.9923；lock s0.95 为 4.496/0.100/0.9990。数值趋势稳定 | 待视觉审计 |
| 2026-08-22 | Stage 3 v3 | 放大脸部 ROI 后，高强度能生成完整双镜片，但整块眼带掩膜导致护目镜化和高清贴片；style energy ratio 达 1.83 | 否决 |
| 2026-08-22 | Stage 4 v4-v6 | 线稿回填暴露机械矩形；布局草图低强度不自然、高强度被擦除；一次提示纠错重采样未稳定生成完整对象 | 否决 |
| 2026-08-22 | Stage 7 | 发现仓库已有 PASD ControlNet/UNet。直接布局控制强时复制线稿，弱时马赛克，证明 PASD 条件分布是退化照片而非抽象边缘 | 否决直接布局 |
| 2026-08-22 | Stage 8 | 两阶段“SD1.5 语义提案 → PASD 照片式精修”成功。`semantic_c085_g6`：layout change 9.545、edit-region change 7.163、区外变化 0.0092、ReID 0.9751；双镜片/桥/镜腿明确，区外结构和整体风格保持 | 当前样本通过 |
| 2026-08-22 | Stage 9 | SAM 对薄透明对象分割覆盖 30.2% 眼睛保护区并产生块状写回，未优于 Stage 8 | 否决 |
| 2026-08-22 | Stage 10/11 文本级想象 | 3 类 ROI × thinking 开关共 6 次请求全部返回有效结构化文本；世界知识识别、互斥概率假设与未决回退可用；thinking 慢 3.40× 且收益不稳定 | 通过，默认 no thinking |

## 十、当前下一步

1. 在固定评估子集比较纯视觉、BLIP caption、Qwen 证据文本、Qwen 多世界想象文本。
2. 生产默认使用 `no_thinking`，仅对少量高价值不确定 ROI 做 thinking 复核。
3. 将事实观察与想象文本分通道接入既有 RGB-Text、IR-Text 融合，保留纯视觉回退。
4. 不再运行 SD/PASD 图片改写。
