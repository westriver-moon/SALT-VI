# PASD Offline Generator

`pasd_offline` 在 SALT 训练前生成固定尺寸或动态语义视图的数据集。它与 `src/salt_vi` 隔离，并使用 `vendor/pasd` 中固定版本的 PASD 实现。

## 当前正式数据链

单视图 PASD RGB 数据：

```text
/home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1
```

当前 Stage-A 主线使用的 RGB+IR geometry-matched 数据：

```text
/home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-ir-geomatched-512x256-1view-v1
```

其中 RGB 沿用 PASD 输出；IR 不进行语义生成，只以相同的人物比例、模糊背景和目标尺寸进行几何适配。组合数据共有 29,033 个 RGB 和 15,712 个 IR 视图，输出均为 256×512 RGB PNG，完整性报告当前为零错误。

扩展数据链（原文本 caption）：

```text
/home/lab929/datasets/derived/SYSU-MM01-pasd-ir-x4-blurpad-512x256-1view-v1
/home/lab929/datasets/derived/RegDB-pasd-rgb-ir-x4-blurpad-512x256-1view-v1
```

SYSU IR 使用 `SYSU-MM01/Text/Blip_IR/caption_dict_Blip_IR.json` 的原始 caption，按协议身份生成官方 15,712 个 IR 视图；RegDB RGB/IR 使用 `RegDB/Text/Blip_RGB/caption_dict_Blip_RGB.json` 与 `RegDB/Text/Blip_IR/caption_dict_Blip_IR.json` 的原始 caption，生成全部 4,120 RGB 与 4,120 IR 视图。RegDB records 同时写入 trial 1 的 train/test split，供后续训练直接消费。

派生数据必须放在 `/home/lab929/datasets/derived/`，不能写入源码仓库。

## 结构

```text
pasd_offline/
├── configs/                  # 生成配置
├── pasd_offline/             # records、geometry、调度、生成与校验
├── scripts/
│   ├── build_sysu_records.py
│   ├── build_regdb_records.py
│   ├── generate_dataset.py
│   ├── build_sysu_geomatched_dataset.py
│   └── validate_dataset.py
├── tests/
├── vendor/pasd/              # 固定上游源码
└── checkpoints/              # 本地权重，不进入 Git
```

## 环境与测试

```bash
conda create -n salt-pasd-offline python=3.10.19 -y
conda activate salt-pasd-offline
python -m pip install -r requirements-lock.txt
PYTHONPATH=. python -m pytest tests
```

标准模型目录为 `checkpoints/stable-diffusion-v1-5` 和 `checkpoints/pasd/checkpoint-100000`。

## Records 与生成

标准一视图 records：

```bash
python scripts/build_sysu_records.py \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --rgb-candidates /home/lab929/ybj/datasets/text_candidates/SYSU-MM01/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json \
  --views-per-source 1 \
  --output /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl \
  --pilot-output /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/pilot-records.jsonl
```

SYSU IR records：

```bash
python scripts/build_sysu_records.py \
  --dataset-root /home/cgv841/datasets/SYSU-MM01 \
  --rgb-candidates /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --ir-candidates /home/cgv841/datasets/SYSU-MM01/Text/Blip_IR/caption_dict_Blip_IR.json \
  --views-per-source 1 \
  --output /home/lab929/datasets/derived/SYSU-MM01-pasd-ir-x4-blurpad-512x256-1view-v1/source-records.jsonl
```

RegDB records：

```bash
python scripts/build_regdb_records.py \
  --dataset-root /home/cgv841/datasets/RegDB \
  --rgb-candidates /home/cgv841/datasets/RegDB/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --ir-candidates /home/cgv841/datasets/RegDB/Text/Blip_IR/caption_dict_Blip_IR.json \
  --output /home/lab929/datasets/derived/RegDB-pasd-rgb-ir-x4-blurpad-512x256-1view-v1/source-records.jsonl
```

生成：

```bash
python scripts/generate_dataset.py \
  --config configs/generate_sysu_rgb_single.yaml \
  --records /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl \
  --workers 2
```

SYSU IR 使用 `configs/generate_sysu_ir_single.yaml`，RegDB 使用 `configs/generate_regdb_rgb_ir_single.yaml`；两个配置的 `gpu_allowlist` 都是 `[1, 2, 3]`，调度器额外拒绝 GPU 0。

`--workers 1` 前台运行；2 或 3 使用配置中可用的物理 GPU。生成按 source 可恢复，并写出 PNG、source metadata、`build.json`、`manifest.jsonl` 和 `manifest.json`。build fingerprint 绑定配置和 records。

构建 geometry-matched RGB+IR 数据时使用 `scripts/build_sysu_geomatched_dataset.py`。构建器拒绝覆盖已存在的派生目录；IR 输出必须保持源宽高比且 `semantic_generation: false`。

## 动态加权视图

records 可以由 `semantic_imagination` 输出数据相关数量的假设视图。每个 source 的 `hypothesis_weight` 必须为正且总和为 1；该权重写入 manifest 并由 SALT sampler 原样使用。动态视图在生成配置中使用 `views_per_source: 0`，训练配置对应 `sysu_sr_views_per_image: 0`。

当前活跃训练仍使用单视图、权重 1；动态语义视图尚未成为当前实验变量。

## 校验与消费

```bash
python scripts/validate_dataset.py \
  --config configs/generate_sysu_rgb_single.yaml \
  --records /home/lab929/datasets/derived/SYSU-MM01-pasd-rgb-x4-blurpad-512x256-1view-v1/source-records.jsonl
```

SALT 配置必须同时设置 `sysu_sr_data_root`、`sysu_sr_view_manifest`、`sysu_sr_views_per_image`、`sysu_sr_modalities` 和 `sysu_sr_exact_size: true`。不要绕过 manifest 直接按文件名猜测来源或视图。
