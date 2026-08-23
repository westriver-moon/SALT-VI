# QRI 眼镜恢复：PASD 与 SD1.5 Inpainting 单图验证

日期：2026-08-22
目标服务器：3090，物理 GPU0
测试图：`SYSU-MM01/cam1/0001/0001.jpg`

## 结论

PASD 的保真设置基本保留了 Swin 参考中已有的模糊眼镜线索，但眼区平均变化只有
1.50–2.08，低于预设的有效恢复门槛 3.0；降低控制到 0.30 虽然产生更大变化，
却被 Qwen 判为马赛克或护目镜状伪影。因此 PASD 不适合承担这项细眼镜结构恢复。

专用 Stable Diffusion v1.5 Inpainting UNet 能生成清晰双镜片与鼻梁。未经后处理的
最佳候选为 `strength=0.85, guidance=7.5, seed=20260821`，Qwen 判定为
`valid_glasses`，但 ReID 余弦只有 0.9403。将该候选在 Swin 基线上做 0.25 透明度
融合后，同时满足局部变化、LR 一致性、身份余弦和 Qwen 视觉门槛，是本次推荐点。

## 模型与目录

- 4090 来源缓存：`/home/lab929/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/`
- 3090 目标：`/home/lab929/ybj/models/qri-diffusion-bases/stable-diffusion-v1-5-inpainting/`
- Hugging Face 仓库：`stable-diffusion-v1-5/stable-diffusion-inpainting`
- revision：`8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`
- UNet：`unet/diffusion_pytorch_model.bin`，3,438,412,325 字节
- 传输：4090 缓存普通复制到独立模型目录，再由 3090 通过内网 SSH/rsync 拉取
- 校验：4090 与 3090 字节数相同；未擅自计算 SHA-256
- 清理：传输校验后已删除 4090 的 3.44 GB 临时中转副本，保留既有来源缓存

模型未进入 SALT 仓库。实验代码位于：

`/home/lab929/ybj/SALT-VI/scripts/experiments/qri_glasses/`

实验产物位于：

`/home/lab929/ybj/experiments/qri_glasses/qri-glasses-gpu0-20260821/`

## PASD 对照

| 设置 | seed | 眼区变化 | ReID 余弦 | Qwen 判定 |
|---|---:|---:|---:|---|
| control 0.75 / guidance 7 | 20260821 | 1.512 | 0.9988 | valid，但变化不足 |
| control 0.75 / guidance 7 | 20260822 | 1.501 | 0.9989 | valid，但变化不足 |
| control 0.50 / guidance 9 | 20260821 | 1.945 | 0.9962 | ambiguous |
| control 0.50 / guidance 9 | 20260822 | 2.083 | 0.9975 | valid，但变化不足 |
| control 0.30 / guidance 10 | 20260821 | 6.817 | 0.9553 | artifact |
| control 0.30 / guidance 10 | 20260822 | 7.351 | 0.9618 | artifact |

Qwen 对高控制候选的 `valid` 只说明候选仍可见原有模糊眼镜，并不证明 PASD 完成了
恢复；有效恢复还要求眼区变化至少为 3.0。

## Inpainting 与后融合

未经后融合的四个初始候选中，仅 `preserve_s085_g75 / seed 20260821` 被 Qwen
判为自然眼镜；另三个分别为面部伪影或无眼镜。该有效候选的推理时间为 3.395 秒，
模型一次加载为 7.525 秒。

| 后融合透明度 | 眼区变化 | 掩膜外变化 | LR 循环误差 | ReID 余弦 | Qwen |
|---:|---:|---:|---:|---:|---|
| 0.25 | 4.346 | 0.00681 | 0.000034 | 0.9823 | valid, 0.95 |
| 0.40 | 6.894 | 0.00745 | 0.000084 | 0.9609 | valid, 0.95 |
| 0.55 | 9.477 | 0.00877 | 0.000155 | 0.9511 | valid, 0.95 |
| 0.70 | 12.011 | 0.00930 | 0.000249 | 0.9462 | valid, 0.95 |
| 0.85 | 14.534 | 0.00968 | 0.000365 | 0.9427 | valid, 0.92 |

推荐 `blend_t025`：眼区变化 ≥ 3、掩膜外变化 ≤ 0.1、LR 循环误差 ≤ 0.003、
ReID 余弦 ≥ 0.96，且 Qwen 看到了双镜片与鼻梁。

## 建议的简化流程

1. 仅当已有 Qwen 世界包含 `eyewear_present` 且未被像素证据否定时触发。
2. 从 Swin 结果取顶部 256×256 头部上下文，只对白色眼区掩膜执行 Inpainting。
3. 固定单种子、`strength=0.85`、`guidance=7.5`、30 步，生成一个候选。
4. 将候选按 0.25 透明度融合回 Swin；掩膜外直接保持 Swin。
5. 运行低成本门槛：眼区变化、掩膜外变化、LR 循环误差和 ReID 余弦。
6. 任一门槛失败即回退到原 Swin；不要在生产阶段做多种子或参数扫面。

## 全数据集可行性

SYSU-MM01 当前目录有 135,353 张 JPG。以热启动后约 2.2–2.6 秒/候选估算，对所有
帧无差别运行约需 83–98 GPU 小时，因此不可取。若只对 Qwen 预筛后的少量眼镜
不确定样本运行，以 2.5 秒/张估算：

- 1% 触发率：约 0.94 GPU 小时；
- 5% 触发率：约 4.70 GPU 小时；
- 10% 触发率：约 9.40 GPU 小时。

因此单张 GPU0 上可行的方案是“条件触发 + 单候选 + 0.25 后融合 + 自动回退”，
而不是全帧 diffusion。

## 审查状态

远端 Qwen 视觉审计、ReID 审计和数值门槛均已完成，足以完成本次服务器端模型
筛选。根据 skill 的本地下载限制，候选 PNG 未拉到本地；人工目视展示是可选后续，
只有在用户明确许可后才能执行。
