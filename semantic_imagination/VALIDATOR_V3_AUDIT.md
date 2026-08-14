# 语义想象校验器 v3：误杀修正与验收审计

日期：2026-08-14

## 问题结论

旧校验器把 `value` 当作必须重复 `state` 的证据。例如
`footwear_detail | strap | black | foot` 中，`state=strap` 已经表达了新增语义，
但旧规则仍因 `value=black` 没有出现 `strap` 而拒绝。旧规则还只比较
`value` 与可见描述，因此会把“限定词已见、state 未见”的有效假设误判成重复观察。
重试阶段只更换随机种子，没有把失败原因返回给 VLM。

因此主要故障不是 InternVL2.5-8B 无法结构化输出，而是校验契约与字段语义不一致。

## v3 结构

- `validator/parser.py`：只负责解析，并修复可唯一确定的表面错误。
- `validator/semantic.py`：以 `state` 作为语义类别；`value` 只作为颜色、大小、材质或外观限定词。
- `validator/feedback.py`：把上次失败代码和定向修正提示加入下一次重试。
- `validation.py`：仅保留旧导入路径兼容层，旧单体严格实现已经删除。
- `sampling.py`：保存每次原始回答、提示、失败代码和修复记录，并统计 attempts、retries、repairs。

## 接受、修复与拒绝边界

接受：

- `graphic | red | chest`：state 本身提供新语义，value 不必重复 graphic。
- `strap | black | foot`：即使 black 已在观察中出现，strap 仍是新增语义。
- `watch | black band | wrist`、`backpack | white bag | back`：允许部件或上位词限定。

可逆修复：

- `no_additional_detail | none`：补齐 location，并规范成 value=`no_additional_detail`、location=`none`。
- sentinel 的常见同义词、遗漏位置、误放在 value 中的位置词。
- `no_additional_detail | white | right hand`：仅当 value 只有颜色/大小等泛化限定词时，修复为显式 abstention。

仍拒绝：

- 类别或受控 state 不存在。
- 正向 state 与 sentinel value 冲突。
- `cap | dark frame`、`bag_accessory | white bag` 等明确指向另一语义状态的 value。
- `other_* | white/small`：other 状态必须至少给出一个具体对象或细节名词。
- state、value、location 三者共同重复已观察事实。

## 验收结果

### 旧 N=512 部分结果离线重放

对停止实验时保存的同一批 512 个任务和全部原始 attempts 重新校验，不重新调用模型：

| 指标 | v2 | v3 |
|---|---:|---:|
| 有效任务 | 120/512 | 475/512 |
| 有效率 | 23.44% | 92.77% |
| 从 v2 失败中救回 | - | 355 |
| 仅第一次 attempt 即有效 | - | 332 |

这说明旧失败的主体是校验器误杀，不是模型无法输出结构。

### 自动化测试

语义想象单元测试、SALT PASD 多视图测试和离线 PASD 任务测试共 46 项，全部通过。

### InternVL2.5-8B 真实 smoke

同一张 SYSU-MM01 图像、seed=20260814、N=8：

| 指标 | v2 | v3 |
|---|---:|---:|
| 最终有效 | 2/8 | 8/8 |
| 最终失败 | 6 | 0 |
| 总 attempts / retries | 28 / 20 | 12 / 4 |
| 主动 abstention | 1 | 6 |
| PASD 权重检查 | 通过 | 通过 |

v3 的 6 个 abstention 表示模型没有提供可靠新增细节，不是校验失败回退；它们与
`validation_failed` 分开记账。N=8 只用于端到端验收，不足以估计正式概率分布。

## 保留限制

- 规则修正降低的是已知误杀，不能证明所有已接受假设都真实。
- `other_*` 的开放词汇仍需在更大样本中审计。
- 正式 N=512 必须由 v3 重新生成；旧 v2 权重不能与 v3 结果混用。
- 经验生成频率仍不是现实世界后验概率。
