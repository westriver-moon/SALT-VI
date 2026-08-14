# 加权语义想象的数学规范

本文档是 SALT Semantic Imagination 插件的权威语义规范。实现可以替换
VLM、扰动、文本嵌入或聚类算法，但不得改变下列随机变量、概率解释和不变量。

## 1. 问题与信息边界

给定一幅低质量行人图像

\[
x\in\mathcal X,
\]

目标不是输出唯一的“真实高清图像”，而是构造由当前观测支持的一组合理语义
世界，并将这些世界实例化为带权视觉结果。

基座 VLM 记为 \(\mathcal V_\phi\)。它首先提取在当前图像中能够稳定支持的
可见信息：

\[
c^{\mathrm{obs}}=\mathcal V_\phi^{\mathrm{obs}}(x).
\]

\(c^{\mathrm{obs}}\) 是所有后续假设和生成结果的共同约束。它不包含仅凭先验
猜测得到的细节。若某属性不能稳定确认，该属性必须进入假设部分，而不能进入
\(c^{\mathrm{obs}}\)。

## 2. 诱导语义分布

令 \(\xi\sim p_\Xi\) 为预先规定的语义保持扰动，且

\[
x_\xi=\mathcal T(x;\xi).
\]

\(\mathcal T\) 可以改变模型对局部模糊证据的感知，但不得改变行人主体、身份
或明确可见语义。给定固定 imagination instruction \(q\)，条件采样为

\[
h\sim p_\phi\!\left(h\mid x_\xi,c^{\mathrm{obs}},q\right),
\]

其中 \(h\) 只描述当前图像没有充分观测到、但与观测证据相容的一个原子潜在
细节。若一次生成包含多个可独立成立的属性，后端必须先拆为原子命题，或将该
输出标记为不合规后重新采样；不得用复合句桥接不同语义类。

扰动随机性和 VLM 解码随机性共同诱导分布

\[
\widetilde p_\phi(h\mid x,q)
=
\mathbb E_{\xi\sim p_\Xi}
\left[
p_\phi\!\left(h\mid\mathcal T(x;\xi),c^{\mathrm{obs}},q\right)
\right].
\]

插件对该诱导分布进行 \(M\) 次独立采样：

\[
h_m\sim
p_\phi\!\left(h\mid\mathcal T(x;\xi_m),c^{\mathrm{obs}},q\right),
\qquad m=1,\ldots,M.
\]

若要区分扰动敏感性与解码随机性，可以扩展为每个扰动采样 \(R\) 次的
\(h_{m,r}\)；基础实现只估计二者混合后的总经验分布。

## 3. 经验测度、语义簇与代表假设

采样结果定义经验测度

\[
\widehat\mu_M
=
\frac1M\sum_{m=1}^{M}\delta_{h_m}.
\]

令 \(\sim_{\mathrm{sem}}\) 表示任务规定的语义等价关系。聚类把每个样本恰好
映射到一个非空语义簇：

\[
\{h_m\}_{m=1}^{M}\longrightarrow\{C_1,\ldots,C_K\},
\qquad
\bigsqcup_{k=1}^{K}C_k=\{h_m\}_{m=1}^{M}.
\]

每个簇的代表是假设空间中的 medoid：

\[
\bar h_k
=
\arg\min_{h\in C_k}
\sum_{h'\in C_k}d(h,h').
\]

代表必须是该簇中真实采样到的成员，不能由插件另行总结或改写。

基础实现使用完全链接近似语义等价关系。若文本相似度为 \(s\)，阈值为
\(\tau\)，任意输出簇必须满足

\[
\min_{h,h'\in C_k}s(h,h')\ge\tau.
\]

不得仅因存在一条相似度链就把链两端合并。旧版单链接连通分量只能用于历史
结果复现，必须在 sampling contract 中明确记录，不能用于新实验。

对结构化原子假设，类别是硬边界，不同类别不得合并；`absent` 和
`no_additional_detail` 是与正向细节分离的硬状态。用于相似度的文本表示只应
编码原子 `value` 与 `location`，不得让重复的字段模板或 category 名称主导
句向量。

正式实现进一步使用每类别封闭的 canonical-state taxonomy。若样本具有合法
状态 (s(h))，语义等价类由

\[
h\sim_{\mathrm{sem}}h'
\iff
g(h)=g(h')\ \land\ s(h)=s(h')
\]

精确定义，不再由句向量阈值近似。自由文本 value/location 只用于选择簇内
medoid 与构造 caption。无法映射的状态必须显式记为 `invalid_output` 或重试，
不得丢弃后重新归一化。

## 4. 经验概率质量

第 \(k\) 个语义簇的权重为

\[
w_k=\frac{|C_k|}{M},
\qquad
w_k>0,
\qquad
\sum_{k=1}^{K}w_k=1.
\]

概率质量属于语义等价类 \(C_k\)，而不是恰好等于代表句 \(\bar h_k\) 的语言
字符串。严格解释为

\[
w_k
=
\widehat P_\phi
\left(
[H]_{\sim_{\mathrm{sem}}}=C_k
\mid \mathcal N(x),q,p_\Xi
\right).
\]

该权重是 VLM、instruction、扰动分布、解码参数和聚类规则共同诱导的经验质量。
它不是现实世界中假设为真的概率，也不是由 VLM 自行报告的置信度，更不自动
具备统计校准性质。

实现可以附带固定当前簇划分条件下的二项比例区间，用于表达有限 \(M\) 的抽样
误差。该区间不包含聚类误差、模型误差或现实真实性误差，不得解释为假设真实
概率的置信区间。

### 4.1 分层类别提议

当不同原子属性可以同时为真时，VLM 自行选择“谈论哪个类别”的频率不是这些
属性的真实性概率。实现可以预先规定类别 strata \(g\in\mathcal G\)，按设计
比例 \(\pi_g=M_g/M\) 指定每次采样的目标类别，并仅在类别内部估计

\[
\widehat p_{g,k}=\frac{|C_{g,k}|}{M_g}.
\]

用于兼容 flat PASD mixture 的联合权重为

\[
w_{g,k}=\pi_g\widehat p_{g,k}=\frac{|C_{g,k}|}{M}.
\]

\(\pi_g\) 是实验设计先验，不是模型估计的现实概率；manifest 必须同时记录
category weight 与 conditional weight，不能把二者混为一谈。

若启用校验，令 (V_g\le M_g) 为类别 (g) 的有效样本数，失败样本保留审计
记录但不进入任何语义簇。此时条件频率改为

\[
\widehat p_{g,k}=\frac{|C_{g,k}|}{V_g},
\]

并且必须同时报告 (V_g/M_g)。模型主动生成的 `no_additional_detail` 属于
(V_g)，校验耗尽不属于。PASD 对可用视图使用

\[
\widetilde w_{g,k}=\frac{\pi_g\widehat p_{g,k}}
{\sum_{a:V_a>0}\pi_a},
\]

从而满足下游概率单纯形约束；未归一化质量、未覆盖类别先验和校验失败率必须
保留在 record 诊断字段中。

## 5. Caption 组合

每个代表假设与共同观测组合为

\[
\bar c_k=c^{\mathrm{obs}}\oplus\bar h_k.
\]

组合操作 \(\oplus\) 只能进行语义拼接和去除表面重复，不得增加新的事实、改变
\(c^{\mathrm{obs}}\)，也不得把多个簇混合成一个新假设。最终得到

\[
\mathcal C_{\mathrm{imag}}(x)
=
\{(\bar c_k,w_k)\}_{k=1}^{K}.
\]

## 6. PASD 视觉实例化

令 PASD 条件生成分布为

\[
p_\theta(\hat x\mid x,\bar c_k).
\]

一次视觉采样为

\[
\hat x_k=\mathcal G_\theta(x,\bar c_k,z_k).
\]

语义分布向图像空间的传播是 mixture

\[
\widehat p(\hat x\mid x)
=
\sum_{k=1}^{K}
w_k\,p_\theta(\hat x\mid x,\bar c_k).
\]

若每个语义簇使用 \(J\) 个扩散噪声，则

\[
\hat x_{k,j}=\mathcal G_\theta(x,\bar c_k,z_{k,j}),
\qquad j=1,\ldots,J.
\]

\(k\) 表示语义不确定性，\(j\) 表示同一语义世界内部的生成随机性。基础
PASD records 中每个 hypothesis view 对应一个 \(k\)；增加 \(J\) 时不得把
扩散重复样本误计为新的语义质量。

## 7. 实现不变量

任何符合本规范的实现必须满足：

1. 每个源图像产生恰好 \(M\) 个有状态的采样任务；有效文本和校验耗尽必须可区分。
2. 每个有效 sample 恰好属于一个非空簇；失败 sample 不得获得 cluster id。
3. \(\sum_k|C_k|=V\le M\)；PASD 的 \(\widetilde w_k\) 权重和为1，失败率另行记录。
4. \(\bar h_k\in C_k\)，并按规定距离选择 medoid。
5. 所有 \(\bar c_k\) 共享完全相同的 \(c^{\mathrm{obs}}\)。
6. 未归一化经验质量必须原样保留；PASD 归一化权重不得改成均匀权重。
7. 旧 manifest 没有权重时可以按均匀分布读取，但不得把这种兼容行为描述为
   VLM 估计的经验质量。
8. 所有随机种子、模型标识、instruction、扰动与聚类参数必须进入可哈希的
   sampling contract，以便复现。
9. 新实验默认采用完全链接，且同簇任意样本对的相似度均不得低于阈值。
10. 每次有效采样只表达一个原子潜在细节；复合属性必须拆分或重采样。
11. 使用分层采样时必须记录 strata、设计比例和类别内条件权重；不同类别不得
    因文本模板相似而合并。

## 8. 本方法不声称的内容

本方法不声称 \(c^{\mathrm{obs}}\) 必然真实，不声称 \(w_k\) 是校准后验，不声称
PASD 输出保持身份，也不声称某个生成结果恢复了不可见的真实细节。这些性质
必须通过独立的事实一致性、身份保持和概率校准实验验证。

方法的数学实质固定为

\[
\boxed{
\text{Single Ambiguous Observation}
\rightarrow
\text{VLM-induced Empirical Mass over Semantic Equivalence Classes}
\rightarrow
\text{Weighted Mixture of Visual Realizations}
}.
\]
