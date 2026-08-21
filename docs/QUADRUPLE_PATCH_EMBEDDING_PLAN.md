# 四路独立 Patch Embedding + 共享 ViT 主干执行方案

## 1. 文档目的

本文定义 SALT-VI 将输入改造成“四个独立 Patch Embedding，同时保留 MSCMNet 浅层参数不共享、深层主干共享”思想的实施方案，供后续执行者直接据此改码、迁移权重和验收。

本阶段只完成以下边界内的工作：

1. 四路输入各自经过独立的 Patch Embedding。
2. 四路 token 在进入 Transformer 后合并到同一个共享 ViT 主干。
3. 保留现有 ViT-B/16 的三通道输入契约和预训练主干权重。
4. 预留后续 token-level ALB 的接口，但本阶段不把 ALB 与 Patch Embedding 改造混在一起。

本阶段不做：

- 将四张图拼成 12 通道后直接送入一个卷积；
- 复制四份完整 ViT；
- 为了适配而下载新的权重或数据；
- 把四路视图误当成四个独立身份；
- 在没有单元测试和基线对照的情况下直接接入 ALB。

## 2. 当前代码基线

重点阅读以下文件：

| 文件 | 当前职责 |
|---|---|
| `src/salt_vi/models/vision_transformer.py` | `PatchEmbedOverlap`、ViT token 准备、Transformer blocks、CLS/位置编码、最终归一化 |
| `src/salt_vi/models/vision_adapter.py` | 将视觉 ViT 封装成 SALT 可调用的视觉适配器 |
| `src/salt_vi/config/default.yaml` | `pmt_depth`、`pmt_embed_dim`、`pmt_patch_size`、`pmt_stride_size` 等配置 |

当前默认配置为：

| 项目 | 当前值 |
|---|---:|
| 输入通道 | 3 |
| 图像尺寸 | `288 x 144` |
| patch size | `16 x 16` |
| stride | `12 x 12` |
| embedding dim | 768 |
| Transformer depth | 12 |
| attention heads | 12 |

当前 `PatchEmbedOverlap` 本质上是一次带重叠步长的卷积式 Patch Embedding：

```python
self.proj = nn.Conv2d(
    in_chans,
    embed_dim,
    kernel_size=patch_size,
    stride=stride_size,
)
```

因此，四路独立 Patch Embedding 不是额外增加一种图像编码机制，而是将这个卷积式 Patch Embedding 复制成四套参数独立的浅层入口。

## 3. 目标结构

### 3.1 四路输入语义

第一版固定四路顺序，所有数据集、训练脚本、测试脚本和导出脚本都必须遵守同一顺序：

| branch id | 语义 | 说明 |
|---:|---|---|
| 0 | `visible_global` | 可见光原始/全局视图 |
| 1 | `visible_channel` | 可见光增强或通道特征视图 |
| 2 | `infrared_global` | 红外原始/全局视图 |
| 3 | `infrared_channel` | 红外增强或通道特征视图 |

如果后续数据管线的四路语义不同，必须修改一个集中式 branch-order 配置，而不能在各个数据集类中隐式改变顺序。

### 3.2 张量契约

建议主接口使用：

```text
views:  [B, 4, 3, H, W]
labels: [B]
```

四路图像仍然是四张独立的 3 通道图像。每一路先经过自己的 Patch Embedding：

```text
[B, 3, H, W] --patch_embed_0--> [B, N, D]
[B, 3, H, W] --patch_embed_1--> [B, N, D]
[B, 3, H, W] --patch_embed_2--> [B, N, D]
[B, 3, H, W] --patch_embed_3--> [B, N, D]
```

然后只在 batch 维度临时拼接：

```text
torch.cat([tokens_0, tokens_1, tokens_2, tokens_3], dim=0)
    -> [4B, N, D]
    -> shared CLS / position embedding
    -> shared Transformer blocks
    -> shared final norm / projection
    -> reshape [B, 4, N, D]
```

这里的 `4B` 表示一次共享主干前向中有四倍的图像样本，不表示图像尺寸变成四倍，也不表示通道数变成 12。四个样本属于同一个原始样本 ID，标签必须相应重复，而不能生成四个新身份。

### 3.3 共享与不共享的边界

| 模块 | 参数策略 | 原因 |
|---|---|---|
| 四套 Patch Embedding | 不共享 | 保留 MSCMNet 的浅层模态/视图专属特征提取能力 |
| CLS token | 共享 | 四路进入同一视觉语义空间 |
| 位置编码 | 共享 | 四路空间尺寸、patch 网格和 token 语义一致 |
| Transformer blocks | 共享 | 对应 MSCMNet 的深层共享主干 |
| 最终 norm / projection | 共享 | 保持 SALT 现有输出空间和下游接口 |
| 后续 ALB | 先不实现 | 避免与输入结构改造同时引入不可定位的变量 |

应明确：四套 Patch Embedding 即使初始权重来自同一个预训练卷积，也必须是四个独立的参数对象；“初始化相同”不等于“运行时共享”。

## 4. 不要误用现有的多分支 Patch 模块

代码中若存在 `MultiBranchPatchEmbedOverlap`，需要先区分它与本方案的含义。该模块是 Patch 级内部多分支后再融合的实现，包含自己的 branch/fuse 逻辑；它不等价于“四个有明确数据语义的输入流各自编码后再进入共享 ViT”。

本方案第一版要求四路输出在共享 Transformer 前保持可追踪：

```text
views[:, 0] -> patch_embeds[0]
views[:, 1] -> patch_embeds[1]
views[:, 2] -> patch_embeds[2]
views[:, 3] -> patch_embeds[3]
```

不要直接复用一个会在入口内部把四路融合掉的模块，否则后续无法验证每个 branch 的梯度、特征和 ALB 路径。

## 5. 推荐的代码接口

### 5.1 兼容现有单图接口

保留现有调用方式：

```python
visual(image)  # image: [B, 3, H, W]
```

这样可保证旧的单图推理、旧 checkpoint 加载和基础回归测试不被破坏。

### 5.2 新增四路接口

建议在视觉适配器中新增明确的四路入口，例如：

```python
visual.forward_quadruple(views)  # views: [B, 4, 3, H, W]
```

或者使用配置开关：

```yaml
visual_streams: 1       # 旧模式
visual_streams: 4       # 四路模式
branch_order:
  - visible_global
  - visible_channel
  - infrared_global
  - infrared_channel
```

默认值建议保持 `visual_streams: 1`，待四路模式通过完整测试后再切换默认训练配置。不得让调用方通过传入 12 通道张量来“猜测”四路模式。

### 5.3 新模块建议

可以新增独立模块，例如：

```text
src/salt_vi/models/quadruple_vision_adapter.py
```

模块职责仅包括：

1. 校验 `[B, 4, 3, H, W]` 输入；
2. 调用四个独立 `PatchEmbedOverlap`；
3. 对四路使用共享 CLS token 和位置编码；
4. 在 batch 维度合并后调用共享 Transformer；
5. 恢复 `[B, 4, N, D]`，提供分支特征和融合特征；
6. 不在模块内部偷偷改变 branch 顺序或标签。

伪代码如下：

```python
class QuadrupleVision(nn.Module):
    def __init__(self, base_vit):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            copy_patch_embed(base_vit.patch_embed),
            copy_patch_embed(base_vit.patch_embed),
            copy_patch_embed(base_vit.patch_embed),
            copy_patch_embed(base_vit.patch_embed),
        ])
        self.cls_token = base_vit.cls_token
        self.pos_embed = base_vit.pos_embed
        self.pos_drop = base_vit.pos_drop
        self.blocks = base_vit.blocks
        self.norm = base_vit.norm

    def forward_quadruple(self, views):
        # views: [B, 4, 3, H, W]
        branch_tokens = []
        for branch_id in range(4):
            x = self.patch_embeds[branch_id](views[:, branch_id])
            x = add_shared_cls_and_position(x, self.cls_token, self.pos_embed)
            branch_tokens.append(x)

        x = torch.cat(branch_tokens, dim=0)  # [4B, N, D]
        x = self.run_shared_blocks(x)
        x = self.norm(x)
        x = x.reshape(4, views.shape[0], x.shape[1], x.shape[2])
        return x.permute(1, 0, 2, 3).contiguous()  # [B, 4, N, D]
```

实际实现应复用现有 `prepare_tokens`、`run_blocks`、`finalize_tokens` 的逻辑，避免复制一份会逐渐漂移的 ViT 主干代码。上面的代码只描述结构，不要求逐字照抄。

第一阶段不做四路 token 之间的 attention。四路在 batch 维度合并只是复用 GPU 计算和共享参数的实现方式；它们在 Transformer 内仍然是四个彼此独立的序列。若未来需要跨模态交互，应另行设计显式 fusion/ALB 层，而不是误把 batch 拼接当成跨路注意力。

## 6. 预训练权重迁移

### 6.1 迁移原则

现有单 Patch Embedding 权重可以作为四个新 Patch Embedding 的初始值：

```text
old.patch_embed.proj.weight
    -> new.patch_embeds.0.proj.weight
    -> new.patch_embeds.1.proj.weight
    -> new.patch_embeds.2.proj.weight
    -> new.patch_embeds.3.proj.weight
```

对应 bias 也同样复制。复制后四个参数的数值可以相同，但 `data_ptr()` 必须不同，确保后续梯度更新不会共享存储。

### 6.2 迁移检查

权重加载脚本必须显式输出：

- 已加载的共享主干 key 数量；
- 四个 Patch Embedding key 是否均已初始化；
- missing keys；
- unexpected keys；
- 四套 Patch Embedding 的参数对象是否独立。

不允许用无日志的 `strict=False` 掩盖 key 错误。若必须非严格加载，必须将缺失和多余 key 写入加载报告，并让测试失败于未处理的关键 key。

### 6.3 兼容旧 checkpoint

旧 checkpoint 应继续支持单路模式。四路模式加载旧权重时，只新增四个 Patch Embedding 的复制步骤；CLS、位置编码、Transformer blocks、norm 和 projection 继续复用旧权重。

不得把 3 通道卷积权重平均、拼接或扩展成 12 通道权重作为默认迁移方案，因为那会改变第一层输入契约并破坏现有 ViT-B/16 权重的语义。

## 7. 位置编码与空间尺寸

四路输入在第一版必须使用相同的 `H/W`、patch size 和 stride，从而得到相同的 patch 网格 `N`。共享位置编码只需按现有逻辑 resize 一次，然后复用于四个分支。

如果未来允许四路空间尺寸不同，不能继续静默复用同一组位置编码；届时应明确选择：统一 resize、分支位置编码，或带 mask 的变长 token。该问题不属于本阶段。

## 8. 数据、标签和身份语义

### 8.1 数据加载器输出

数据加载器应尽量直接返回：

```python
views  # [B, 4, 3, H, W]
labels # [B]
meta   # 至少包含 branch_order 或可验证的分支信息
```

四路图像必须来自同一个样本 ID。若四路由不同增强或不同模态生成，应在数据层保证配对关系，而不是在模型层按 batch 顺序猜测配对。

### 8.2 临时 batch 展平时的标签

若损失函数需要对展平后的 `[4B, ...]` 计算，标签可以写为：

```python
flat_labels = labels.repeat(4)
```

但这只是计算布局，不是身份扩增。恢复到 `[B, 4, ...]` 后，仍然要能追溯到原始样本 ID。

### 8.3 防止分支错位

必须添加检查：

- 四路张量的 batch 维度一致；
- 四路 branch id 顺序固定；
- 同一 ID 的四路不会在 collate 时被分别 shuffle；
- 标签只复制，不重新编号；
- 导出或缓存特征时保存 branch id 和原始样本 ID。

## 9. 训练与推理输出

### 9.1 第一版训练策略

先使用共享身份监督验证结构本身：

1. 四路分别 Patch Embedding；
2. 共享 ViT 主干；
3. 输出四路 branch feature；
4. 先采用同一 ID 的监督目标；
5. 暂不加入复杂的跨路对比损失、ALB 损失或额外蒸馏损失。

这样可以把“输入结构改变带来的收益”与“新损失函数带来的收益”分开。

### 9.2 建议输出

视觉模块应提供至少两种输出：

```text
branch_features: [B, 4, D]
fused_features:  [B, D]
```

第一版融合建议采用可复现的固定策略：

```text
visible_feature = mean(branch_features[:, 0:2], dim=1)
infrared_feature = mean(branch_features[:, 2:4], dim=1)
fused_feature = normalize((visible_feature + infrared_feature) / 2)
```

也可以先使用四路等权平均作为基线。融合策略必须在配置中明确，不能由调用方临时写不同版本。

推理阶段应同时保留 branch feature，便于检查某一路是否失效；不能只保存最终融合向量而丢失诊断信息。

## 10. 与后续 ALB 的边界

MSCMNet 的 ALB 思想应在本结构稳定后再接入。第一阶段的边界是：

```text
四个独立 Patch Embedding
        -> 共享 ViT blocks
        -> branch-preserving features
        -> 固定融合
```

后续 token-level ALB 可以考虑在若干 block 位置插入，例如 block 3、6、9，但需要单独定义：

- 哪些层的 Q/K 使用浅层或模态专属表示；
- 哪些层的 V 使用深层共享表示；
- ALB 是在每个 branch 内执行，还是显式跨 branch 执行；
- residual、norm 和 shape 如何保持；
- 是否在训练和推理使用同一融合路径。

不能把 CNN 版 ALB 的 feature-map 操作直接复制到 ViT token 上。ViT 版 ALB 必须以 `[B, 4, N, D]` 或等价的 branch-preserving 表示为基础，并保持四路身份可追踪。

## 11. 推荐实施顺序

### Phase 0：接口和回归基线

1. 固定 branch order 和 `[B, 4, 3, H, W]` 输入契约。
2. 记录现有单路模型的参数量、显存、吞吐和验证指标。
3. 为现有单路路径补齐最小回归测试。

### Phase 1：四路独立 Patch Embedding

1. 新增四路视觉模块或在现有视觉适配器中增加清晰的四路入口。
2. 创建四个独立 `PatchEmbedOverlap`。
3. 复用共享 CLS、位置编码、Transformer blocks、norm 和 projection。
4. 先用旧 Patch Embedding 权重复制初始化四路入口。
5. 验证四路输出 shape、参数独立性和旧 checkpoint 加载。

### Phase 2：数据与训练基线

1. 让数据加载器输出配对的四路张量。
2. 验证同一 ID 的四路标签对齐。
3. 只使用最小身份监督完成一组可重复训练。
4. 与单路模型、四路共享入口模型进行对照。
5. 固定一条 canonical outcome，再决定是否进入下一阶段。

### Phase 3：ALB 原型

1. 先在 branch-preserving token 表示上实现一个最小 ALB。
2. 只增加一个控制变量，例如插入层位，不同时改变融合、损失和输入增强。
3. 对照无 ALB 的四路模型，记录收益、显存和速度。
4. 通过消融后再决定是否保留多个 ALB 版本。

## 12. 必须实现的测试

### 12.1 形状测试

- 输入 `[B, 4, 3, H, W]` 能正常前向；
- 四个 Patch Embedding 输出的 `N`、`D` 相同；
- shared trunk 输入为 `[4B, N, D]`；
- 输出可恢复为 `[B, 4, N, D]`；
- `branch_features` 和 `fused_features` 形状符合接口。

### 12.2 参数独立性测试

- `patch_embeds[0..3]` 是四个不同模块对象；
- 对应 weight/bias 的 `data_ptr()` 不相同；
- 修改一路权重不会改变另外三路；
- 四路共享 Transformer 参数对象保持一致。

### 12.3 单路等价测试

在四个 Patch Embedding 使用相同权重、四路输入复制同一张图时：

- 四路 Patch Embedding 输出应一致；
- 共享主干输出应一致；
- 四路融合结果应与单路基线在容许误差内一致。

这项测试用于证明 batch 维拼接没有改变单张图的空间尺寸或 token 计算逻辑。

### 12.4 梯度测试

- 四个 Patch Embedding 都能收到梯度；
- 共享 Transformer blocks 只保留一套梯度路径；
- 对某一路输入或参数做扰动时，可观察到对应 branch 的输出变化；
- 没有意外的 `.detach()`、跨路错位或标签错配。

### 12.5 权重迁移测试

- 旧 checkpoint 可以加载单路模式；
- 旧 checkpoint 可以初始化四路模式；
- missing/unexpected keys 有明确报告；
- 复制后的四套浅层参数数值一致但存储不共享。

### 12.6 运行资源测试

至少记录：

- 参数量；
- 单步显存；
- batch size；
- 吞吐；
- 单路与四路的训练/推理耗时；
- 四路特征导出所需存储。

## 13. 实验与文件管理要求

1. 不新建分支或 worktree；所有修改先在当前已存在的 SALT-VI 工作目录中进行。
2. 不把四路图像永久复制成四份数据集；优先在数据加载或 batch 内生成/组织视图。
3. 不下载新的远端数据、模型或权重；若确实需要远端文件，先请求许可。
4. 每个实验保存配置、随机种子、commit、branch order、权重来源和评价指标。
5. 同一个实验的导出副本、替代路径和重跑结果必须归并，不重复计数。
6. 每个 run 保留一个 baseline，只保留超过预设阈值的改进，并记录一个 canonical outcome。
7. 结构代码、实验配置、结果表和产物路径必须保持可追溯，不能把关键逻辑散落在临时脚本中。

## 14. 验收标准

只有同时满足以下条件，才可把四路模式视为可交付：

1. 旧的单路 `[B, 3, H, W]` 路径仍能正常工作。
2. 四路模式明确接收 `[B, 4, 3, H, W]`，而不是 12 通道输入。
3. 四个 Patch Embedding 参数独立，且共享同一套 ViT 深层主干。
4. CLS、位置编码、Transformer blocks、norm 和 projection 的共享策略已通过代码和测试验证。
5. 旧 ViT-B/16 权重可迁移，迁移日志中没有未处理的关键 key。
6. 同一 ID 的四路视图、标签和 branch order 始终对齐。
7. 可以同时取得四路特征和融合特征，便于诊断。
8. 单路等价、参数独立性、梯度、权重迁移和资源测试全部通过。
9. 没有把 ALB、额外损失和新的数据增强同时混入第一版基线。
10. 训练结果、配置和 canonical outcome 已进入总实验记录。

完成以上验收后，才进入 token-level ALB 的设计和消融；否则先修复结构、接口或数据对齐问题。
