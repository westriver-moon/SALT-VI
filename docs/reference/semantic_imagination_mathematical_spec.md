# QRI-v6 联合语义想象数学规范

本文档是 QRI-v6 的权威概率语义。实现可以替换 VLM、语义编码器和文本改写器，
但不得改变联合随机变量、经验质量与可复现边界。

## 1. 共同观测

给定低质量行人图像 $x$ 和目标区域集合

\[
\mathcal R=\{r_1,\ldots,r_L\},
\]

VLM 首先产生所有后续世界共享的可见观测

\[
c^{\mathrm{obs}},\quad
o_1,\ldots,o_L=\mathcal V^{\mathrm{obs}}_\phi(x,\mathcal R).
\]

$c^{\mathrm{obs}}$ 是全局 caption，$o_l$ 是区域稳定事实。不能稳定确认的内容
不得进入共同观测。

## 2. 联合世界抽样

一个语义世界不是若干区域边缘分布的事后笛卡尔组合，而是一次 VLM 请求产生的
联合随机变量：

\[
H=(h_1,\ldots,h_L)
\sim p_\phi(H\mid x,c^{\mathrm{obs}},o_{1:L},q).
\]

每个 $h_l$ 必须包含 `region/category/state/value/location`，且一次有效抽样
恰好为每个目标区域返回一个原子赋值。框架用互相独立、按命名空间派生的 seed
执行 $M$ 次直接联合抽样：

\[
H_m\sim p_\phi(H\mid x,c^{\mathrm{obs}},o_{1:L},q),
\qquad m=1,\ldots,M.
\]

不得先估计每个 $h_l$ 的边缘概率，再假设

\[
p(H)=\prod_l p(h_l).
\]

该乘积会制造模型从未联合提出的跨区域组合，QRI-v6 明确禁止这种实现。

请求重试耗尽的任务记为 `request_failed`，不进入聚类。令有效联合样本数为

\[
V\le M.
\]

失败质量 $1-V/M$ 必须保留，不能伪装为主动 abstention，也不能静默分配给
有效世界。

## 3. 联合语义等价簇

每个世界具有硬 state signature

\[
\sigma(H)=((r_l,g_l,s_l))_{l=1}^L.
\]

不同 signature 的世界不能合并。同一 signature 内，语义编码器对完整
value/location canonical text 产生向量 $e(H)$。完全链接阈值为 $\tau$，
每个输出簇 $C_k$ 必须满足

\[
\min_{H_i,H_j\in C_k}
\cos(e(H_i),e(H_j))\ge\tau.
\]

因此 canonical state 是硬边界，但不是最终等价类；同一 state 下语义不同的
颜色、对象或位置仍能被阈值拆开。single-link 和“只按 state 合并”仅属于旧
结果复现，不得标记为 QRI-v6。

簇代表必须是实际抽样成员的 medoid：

\[
\bar H_k=\arg\min_{H\in C_k}
\sum_{H'\in C_k}(1-\cos(e(H),e(H'))).
\]

## 4. 经验质量

每个簇保留两种不混淆的质量：

\[
\widehat\mu_k=\frac{|C_k|}{M},
\qquad
\widehat w_k=\frac{|C_k|}{V}.
\]

$\widehat\mu_k$ 是相对于全部计划任务的未归一化经验质量；
$\widehat w_k$ 是在有效联合抽样条件下的权重。若只保留 Top-K 簇，供下游
选择的权重为

\[
\widetilde w_k=\frac{|C_k|}
{\sum_{a\in\mathrm{retained}}|C_a|}.
\]

manifest 必须同时记录 retained empirical mass，不能把
$\widetilde w_k$ 误称为全部计划采样的概率。Wilson 区间只描述固定簇划分下
有限 $V$ 的频率误差，不包含模型、聚类或现实真实性误差。

QRI-v6 不从上述权重再次随机抽取“世界”。第二次 Monte Carlo 只会在已有经验
估计上增加噪声，并可能掩盖被截断质量。

## 5. 语义改写

每个保留簇调用一次改写接口：

\[
\bar c_k=\mathcal L_\psi
(c^{\mathrm{obs}},o_{1:L},\bar H_k).
\]

改写器只能组合共同观测与该代表世界，不得增加新事实、混入其他簇或删除共同
身份语义。最终输出为

\[
\mathcal C_{v6}(x)=
\{(\bar c_k,\widehat\mu_k,\widehat w_k,\widetilde w_k)\}.
\]

VLM 与 LLM 的具体实现不属于核心包；它们通过带 descriptor 的接口注入。

## 6. 可复现 contract

run signature 必须覆盖：

1. `qri-v6` schema、$M$、Top-K、$\tau$、complete-link、重试次数和基础 seed；
2. VLM、语义编码器、改写器的 backend/model ID 与 revision；
3. 三个后端的 prompt 版本、temperature、top-p、token 限制和其他采样参数。

source contract 另行记录图像 SHA-256、source key、模态、ROI 类别与坐标。所有
派生 seed 必须按 phase 与 sample index 分离，不能让联合抽样和文本改写共用同一
随机序列。

## 7. 实现不变量

1. 一个有效 sample 是一个完整联合世界，而不是独立区域抽样的乘积。
2. 每个有效 sample 恰好属于一个非空簇；失败 sample 不得获得 cluster ID。
3. \(\sum_k|C_k|=V\le M\)，失败质量和截断质量分别记录。
4. 不同 state signature 不合并；同一 signature 内阈值必须真实参与 complete-link。
5. 每个代表都是簇内真实成员；每个 caption 只对应一个代表世界。
6. 三类权重 `empirical_mass`、`valid_weight`、`selected_weight` 不得互相覆盖。
7. 实际后端调用次数、额外重试、耗时和 usage 必须按 phase 汇总。
8. 任何影响结果的算法、后端或 source contract 变化都必须改变对应哈希并使缓存失效。
9. 测试和 smoke 输出只能写入显式外部输出根或临时目录。

## 8. 解释边界

经验质量是当前模型、prompt、解码参数、输入和聚类规则共同诱导的分布；它不是
现实真值概率，也不是校准后验。QRI-v6 不声称 PASD 输出保持身份或恢复不可见
真值。真实 VLM/LLM 后端、概率校准、身份保持和 ReID 收益必须通过独立实验验证。
