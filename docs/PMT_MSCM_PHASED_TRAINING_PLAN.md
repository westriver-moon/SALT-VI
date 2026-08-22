# PMT 分阶段接入 MSCMNet 四路输入方案

> **历史设计说明（重要）**：本文记录的是早期“epoch 6 后切换、以 QCT 为主”的 recipe 设计，保留用于解释当时的动机和代码边界。它不是 2026-08-21/22 H1–H4 混合损失实验的结果说明。当前 H1–H4 的实际终端权重、渐进因子、加速设置和解析配置以 `configs/stage_a/plugins/hybrid_loss/common.yaml` 及各 H1–H4 YAML、归档中的 `configs.yaml` 为准；当前结果见 `reports/EXPERIMENT_STATUS_20260822.md`。

## 目标

保持原有 PMT Stage-A 前期训练不变；第 6 个 epoch 后切换为严格 MSCMNet 四路增强。超分图像必须先由数据源读取，再执行增强；超分不是增强算子。

## 训练阶段

### Epoch 0–5：原 PMTRecipe

输入：

```text
RGB 灰度增强图 + IR 图像
```

流程保持原实现：

```text
共享 PMT-ViT
→ RGB/IR 特征
→ ID Loss + 原 PMT Triplet Loss
```

此阶段：

- 使用原 `img_rgb_aug` 和 `img_ir`；
- `MSEL Loss = 0`；
- `DCL Loss = 0`；
- 不启用四路输入。

### Epoch ≥6：MSCMNet 四路输入

从同一批超分 RGB/IR 图像生成四路视图：

```text
RGB-1：RandomGrayscale → Pad → RandomCrop → Flip → RandomErasing
RGB-2：Pad → RandomCrop → Flip → RandomErasing → ChannelExchange
IR-1 ：Pad → RandomCrop → Flip → RandomErasing → ChannelAdapGray
IR-2 ：ColorJitter → Pad → RandomCrop → Flip → RandomErasing → ChannelT
```

四路输入顺序固定为：

```text
[RGB-1, RGB-2, IR-1, IR-2]
```

数据流：

```text
PASD 超分图像
→ 四种 MSCMNet 增强
→ 四个独立 Patch Embedding
→ 共享 ViT 主干
→ 四路分支分别计算 ID/Triplet
→ 四路联合计算 QCT
```

## 第 6 个 epoch 后的损失

MSEL 和 DCL 均取消。默认损失为：

\[
L = L_{ID} + L_{Triplet}^{phase} + \lambda_{QCT}^{phase}L_{QCT}
\]

其中：

- `ID Loss`：四个分支分别分类；RGB 两支取均值、IR 两支取均值后相加，保持 epoch 0–5 的两项 CE 尺度；
- `Triplet Loss`：epoch 6 从四路分支内 Triplet 开始，在 4 个 epoch 内平滑过渡到四个 RGB–IR 组合的 CrossTriplet；
- `QCT Loss`：在 L2 归一化空间计算，使用最近负身份中心和 `1.2` margin，并在相同过渡期从 0 增至 `0.1`。

四路特征不先取平均：所有分支先独立进入损失，QCT 直接接收 `[B,4,D]`。

## 优化器连续性

全程只使用同一个 AdamW，不在 epoch 6 重建优化器。epoch 0–5 训练共享模板 Patch Embedding；epoch 6 将模板权重及其 AdamW `step/exp_avg/exp_avg_sq` 状态复制到四个 Patch Embedding，然后继续使用原调度器。

## DCL 与 QCT

### DCL

对每个身份计算一个联合 RGB/IR 中心：

\[
c_k = \operatorname{mean}\{x_i \mid y_i=k\}
\]

计算中心到正样本和困难负样本的距离：

\[
d^+ = \operatorname{mean} d(c_k,x_i),\quad y_i=k
\]

\[
d^- = \operatorname{mean} d(c_k,x_j),\quad y_j\ne k
\]

实际实现选择距离小于负样本平均距离的困难负样本，最终：

\[
L_{DCL}=\frac{d^+}{d^-+\epsilon}
\]

DCL 使用一个联合身份中心，不显式区分四个输入分支。

### QCT

输入顺序为 `[RGB-1, RGB-2, IR-1, IR-2]`，分别计算：

```text
总体身份中心
RGB-1 身份中心
RGB-2 身份中心
IR-1 身份中心
IR-2 身份中心
```

QCT 包含两类约束：

1. 样本靠近对应的四路/双路身份中心；
2. 样本远离其他身份中心：

\[
L_{neg}=\operatorname{mean}[m-d(x_i,c_j)]_+,\quad y_i\ne j
\]

原 MSCMNet 代码中四路额外中心项的系数 `a=0`。新 recipe 将该系数配置化，默认设为 `0.25`；总 QCT 权重为 `0.1`，并渐进启用。

## 重合与取舍

DCL 和 QCT 都包含“样本靠近身份中心”的约束，存在重合，但不等价：

- DCL：联合中心、距离比值、困难负样本；
- QCT：四路中心、显式 margin、负身份中心排斥。

当前 recipe 不叠加 DCL、MSEL 与 QCT，只保留小权重 QCT。可进行以下消融：

```text
Triplet only
QCT
```

## 实施约束

1. Epoch 0–5 必须保持原 PMT 输入和损失路径不变。
2. Epoch ≥6 必须使用四种原 MSCMNet 算子。
3. 超分数据先采样，再增强；`ExactSize` 只做尺寸检查，不做增强。
4. 四路 Patch Embedding 参数独立，ViT Transformer 主干共享。
5. 四路特征不得先按模态平均，必须直接接受 branch-level loss。
6. 训练和测试的四路顺序必须始终为 `[RGB-1, RGB-2, IR-1, IR-2]`。

## Recipe 切换

原 recipe 保持默认：

```yaml
pmt_recipe_variant: original
```

新分阶段 recipe：

```yaml
pmt_recipe_variant: mscm_phased
quadruple_template_trainable: true
```
