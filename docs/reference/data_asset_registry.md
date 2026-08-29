# 可复用数据资产索引

更新：2026-08-29。机器可读版本是
`reports/assets/person_assets_20260829.json`；本页只解释目录语义，不建立第二份资产表。

## 统一根目录

```text
/home/lab929/ybj/datasets/person-assets-512x256/
├── person_fit/
│   ├── {sysu,regdb,llcm}/images/
│   └── sysu/{pose,anatomy}/
├── resize/
│   └── sysu/{images,pose,anatomy}/
└── upstream/swinir-real-sr-x4-v1/
    ├── outputs/{sysu,regdb,llcm}/
    ├── manifests/
    ├── benchmarks/
    └── logs/
```

`person_fit` 是唯一受支持的防拉伸裁剪版本：V1 几何加审核后的 YOLO26
质量门，覆盖 99,752 张图，96,257 个 bbox pass、3,495 个完整画面 fallback。
SYSU 子集为 44,745 张，其中 44,031 pass、714 fallback。

`resize` 是严格对照：读取相同原生 SwinIR realSR ×4 图像后，直接 bicubic
缩放到 512×256，不应用 person-fit。当前只物化 SYSU 的完整 44,745 张。

## 姿态与 anatomy

两套 SYSU 512×256 输入都保存：

- `pose/`、`pose.jsonl`、`pose.contract.json`：COCO17 原始姿态及来源契约；
- `anatomy/`、`anatomy.jsonl`、`anatomy.contract.json`：CTI 使用的四部位
  token mask，patch 16×16、stride 12×12，对应 42×21 token grid；
- 每套均覆盖全部 44,745 source key。未检出姿态仍有显式记录，不会静默当作
  全人体；训练时人体候选不足的样本只跳过 CTI loss，仍参加原 C3 loss。

person-fit anatomy 中 43,780 张有完整人体姿态、965 张无检出；resize 分别为
43,593 与 1,152。人体部位可用计数及权重哈希以机器可读清单为准。

## 原生 SwinIR ×4

原生资产以原分辨率输入、FP32、整图推理、右/下对称补齐 8 像素窗口，不做
输入或输出 resize。权重是 realSR BSRGAN DFO SwinIR-M ×4 PSNR；IR 先按
BT.601 加权亮度转三通道，输出量化后再取三通道均值并复制。生产覆盖：
SYSU 44,745、RegDB 8,240、LLCM 46,767，共 99,752 个唯一源图。

生产代码、vendor 源码、配置和测试已进入
`person_preprocessing/swinir_x4/`；外部资产目录保留运行日志、benchmark、
manifest、output contract 与逐图进度。旧的 `/home/lab929/ybj/swinir_x4_pipeline`
仅为兼容符号链接，不再是第二份数据。

## 旧 C3 ×2 对照

旧 C3/SwinIR-x2 数据仍保留在
`/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1`（约 14G）。
它是 raw→256×128→classical SwinIR-M ×2→512×256，与当前 realSR ×4→
resize/person-fit 管线分布不同。比较时必须在总表中明确 `derived_data_root`，
不能把两者只描述为“是否裁剪”。
