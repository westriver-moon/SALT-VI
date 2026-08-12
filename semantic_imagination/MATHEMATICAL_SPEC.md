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

其中 \(h\) 只描述当前图像没有充分观测到、但与观测证据相容的潜在细节。

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

1. 每个源图像产生恰好 \(M\) 个非空 imagination samples。
2. 每个 sample 恰好属于一个非空簇，且 \(1\le K\le M\)。
3. \(\sum_k|C_k|=M\)，\(w_k=|C_k|/M\)，权重和为 1。
4. \(\bar h_k\in C_k\)，并按规定距离选择 medoid。
5. 所有 \(\bar c_k\) 共享完全相同的 \(c^{\mathrm{obs}}\)。
6. 权重必须原样传递到 PASD manifest 和 SALT sampler；不得在导出时改成均匀权重。
7. 旧 manifest 没有权重时可以按均匀分布读取，但不得把这种兼容行为描述为
   VLM 估计的经验质量。
8. 所有随机种子、模型标识、instruction、扰动与聚类参数必须进入可哈希的
   sampling contract，以便复现。

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
