# SYSU-MM01 SwinIR x2 PMT256 数据集完整生成配置

状态：当前有效数据集的恢复与审计记录
记录日期：2026-08-07
适用数据根：/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1

## 1. 适用范围

本文只描述当前 SALT-VI、Stage-A/Stage-B 和 B-SWINIR 实验实际引用的
SYSU-MM01-swinir-x2-pmt256-v1。它不是历史上已判定失效的
SYSU-MM01-swinir-x2-v1，也不是使用 288 x 144 输入的
SYSU-MM01-swinir-x2-v2。

当前数据集约占 14 GiB，manifest schema 为 v4，生成时间为 2026-07-19T20:06:27.379711+00:00。
训练输入统一为 256 x 128，SwinIR x2 输出为 512 x 256；RGB 和 IR 均离线生成。

## 2. 权威文件与完整性

| 对象 | 路径或标识 | SHA-256 |
| --- | --- | --- |
| 构建契约 | /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1/.build-contract.json | 9985960030521bac50cae7dce67b10e6f1ae0ed5f9cc817bbe9fbc944590daa7 |
| 数据 manifest | /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1/manifest.json | fc109c9eced1c6c11c0af56f786f3de2026caf6fe3b52e65457fd68321496c90 |
| 精确历史构建脚本 | vendor/legacy_code/source_core/tools/super_resolution/build_sysu_swinir_x2.py | e82833a9127d6694460df49bcc55adee046ed01492c8f19067e06ae647572fe9 |
| 当前 SALT 构建脚本 | src/salt_vi/utils/super_resolution/build_sysu_swinir_x2.py | 7f5ff978a905b170df11b2577cb3bc41eeb800c3e6e9d4560d821adbb57ca1b2 |
| 当前校验脚本 | src/salt_vi/utils/super_resolution/validate_sysu_swinir_x2.py | b2561ab572c34074719b503584d4cbf0bd49158adbaee8958cf586f35961e7c1 |
| SwinIR 权重 | /home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth | 2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac |
| SwinIR network_swinir.py | /home/cgv841/third_party/SwinIR-official-6545850-v2/models/network_swinir.py | 9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913 |

精确历史构建脚本的哈希与 build contract 中的 builder_sha256 完全一致。
当前 SALT 脚本仅因仓库迁移修改了 REPO_ROOT 计算和 provenance import 路径，
因此脚本 SHA 不同；恢复原始构建身份时，应以 vendor 中的精确历史脚本和本文件附录为准。

记录的算法仓库提交为 cdd173fbd3c7146f98d39338556a66d64be62f30。该 Git 对象目前不在服务器现存
仓库中，但对应的构建脚本内容已完整保存在 vendor/legacy_code 中。
SwinIR 上游提交仍存在且已核验为 6545850fbf8df298df73d81f3e8cba638787c8bd。

## 3. 原始生成命令

manifest 记录的原始 argv 如下：

~~~bash
tools/super_resolution/build_sysu_swinir_x2.py --source-root /home/cgv841/datasets/SYSU-MM01 --output-root /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1 --swinir-root /home/cgv841/third_party/SwinIR-official-6545850-v2 --model-path /home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --modalities rgb ir --device cuda:0 --batch-size 4 --source-size 256 128 --source-resampling bilinear --min-free-before-gb 40 --min-free-after-gb 20
~~~

关键运行参数：

| 参数 | 值 |
| --- | --- |
| modalities | rgb ir，必须同时构建 |
| device | cuda:0 |
| batch size | 4 |
| source size | 256 x 128 |
| output size | 512 x 256 |
| source resampling | bilinear |
| min free before | 40 GiB |
| min free after | 20 GiB |
| numeric policy | fp32-no-autocast-finite-before-uint8-v1 |
| output root | /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1 |

重新生成时不得覆盖当前目录，应先写到新的空目录。当前 SALT 等价调用形式为：

~~~bash
cd /home/cgv841/ybj/SALT-VI
PYTHONPATH=src python -m salt_vi.utils.super_resolution.build_sysu_swinir_x2 \
  --source-root /home/cgv841/datasets/SYSU-MM01 \
  --output-root /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1-rebuild \
  --swinir-root /home/cgv841/third_party/SwinIR-official-6545850-v2 \
  --model-path /home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
  --modalities rgb ir \
  --device cuda:0 \
  --batch-size 4 \
  --source-size 256 128 \
  --source-resampling bilinear \
  --min-free-before-gb 40 \
  --min-free-after-gb 20
~~~

该等价调用会生成新的 repository_revision 和 builder_sha256，不能期望 manifest
逐字节相同；图像内容是否逐字节相同必须使用第 7 节的输出哈希判断。

## 4. SwinIR 模型配置

算法：SwinIR-M classical SR x2，DF2K 权重。

| 网络参数 | 值 |
| --- | --- |
| upscale | 2 |
| in_chans | 3 |
| img_size | 64 |
| window_size | 8 |
| img_range | 1.0 |
| depths | [6, 6, 6, 6, 6, 6] |
| embed_dim | 180 |
| num_heads | [6, 6, 6, 6, 6, 6] |
| mlp_ratio | 2 |
| upsampler | pixelshuffle |
| resi_connection | 1conv |
| checkpoint state | 优先 params_ema，其次 params，最后直接使用 payload |
| load_state_dict | strict=True |
| inference | eval、torch.no_grad、FP32、autocast disabled |

记录的软件环境：

- PyTorch：1.12.1
- timm：0.8.6.dev0
- SwinIR revision：6545850fbf8df298df73d81f3e8cba638787c8bd
- numeric policy：fp32-no-autocast-finite-before-uint8-v1

原 manifest 未记录 GPU 型号、CUDA runtime、cuDNN 版本和确定性开关。因此配置和语义可恢复，
但跨环境重新运行不能先验保证逐字节无损；必须以输出 SHA-256 作为最终判据。

## 5. 输入数据契约

| 模态 | 训练数组 | 原始 shape | 图像数 | 数组 SHA-256 | 标签 SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| RGB | train_rgb_resized_img.npy | [22258, 384, 144, 3] | 22258 | f6b37a1bb7fdaba4892c983bd7d6082b5b969efd9d0265c5aff96236bbe332e1 | 29c214045cc9f11f4601286558bd14615dee4f52753951cdf2cda3159d23b2ce |
| IR | train_ir_resized_img.npy | [11909, 384, 144, 3] | 11909 | ec47b6fc0024053eb356501f2bed62edfe9f6aa4f47706c0fc118effe797aab6 | 0cad1d124e5c768766a11acfb1be0bab7d3cf97b299ddd062081757002591479 |

上述四个输入文件已于 2026-08-07 重新计算 SHA-256，均与 build contract 一致。

评估输入：

| 模态 | 相机 | 文件数 | 输入树 SHA-256 |
| --- | --- | ---: | --- |
| RGB | cam1、cam2、cam4、cam5 | 6775 | f1044bea50f5c30572ea776489957154e1ef9895fe26832f3525cc188ee935cd |
| IR | cam3、cam6 | 3803 | caf701503f5cdaf0d928010a42819ec9f68ab9f8eb6384ac8bd7fab67bc02a07 |

测试身份文件：/home/cgv841/datasets/SYSU-MM01/exp/test_id.txt
身份数：96
SHA-256：0571165836c315dc4d98a373e64c0b462585aa6ec1559f4cff25d32fe5b8c30e

## 6. 图像处理与数值策略

1. 原训练数组和评估 PNG 都先使用 PIL bilinear 统一到 256 x 128。
2. RGB 保持三通道。
3. IR 在超分前按 BT.601：Y = round(0.299R + 0.587G + 0.114B) 转为亮度并复制为三通道。
4. 输入从 uint8 NHWC 转为 float32 NCHW，并除以 255。
5. 推理强制 FP32，显式禁用 CUDA autocast。
6. 输出必须全部 finite，且空间尺寸必须严格等于输入的 2 倍。
7. 输出 clamp 到 [0, 1]，乘 255、round，再转为 uint8 NHWC。
8. IR 输出再次取三通道均值并复制，保证三个通道逐值相等。
9. 训练集保存为 NumPy .npy；评估集保留原相对文件名，以 lossless PNG、compress_level=3 原子写入。
10. 任何 NaN/Inf、全零图、常量图、动态范围过小、IR 通道不等或尺寸错误都会终止构建。

## 7. 输出文件与指纹

| 输出 | shape / 数量 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| train_rgb_swinir_x2_img.npy | [22258, 512, 256, 3] uint8 | 8752201856 | fa46068bcbf4bd435ffbc7e0e705b309a43bf80455b747ecd14a31bd9df4e604 |
| train_ir_swinir_x2_img.npy | [11909, 512, 256, 3] uint8 | 4682809472 | bdf58b9e0622d4177c68d3aa8bd511cae14c024b8c2a193221f47045c76d026f |
| RGB eval tree | 6775 PNG | 见文件树 | 5e99c07e9ea80f99715ff779f8974a62709bb2c2e5412827f95873ef0b2cb6d7 |
| IR eval tree | 3803 PNG | 见文件树 | 9ea4bd574958c8563d670d3895c1531393f1138f1de0d42634fdbe0f3c937ef7 |

两份训练输出数组已于 2026-08-07 重新计算 SHA-256，均与 manifest 一致。
评估目录共 10578 张 PNG；当前相机计数为
cam1=1374、cam2=1565、cam3=1883、cam4=1918、cam5=1918、cam6=1920。

内容门控摘要：

| 输出 | mean | std | all-zero | constant |
| --- | ---: | ---: | ---: | ---: |
| RGB train | 96.364951 | 57.339589 | 0 | 0 |
| IR train | 119.047477 | 50.553782 | 0 | 0 |
| RGB eval | 94.721119 | 57.206590 | 0 | 0 |
| IR eval | 113.448645 | 53.303785 | 0 | 0 |

真实权重 smoke test 的下采样源一致性 PSNR：
RGB=57.683845 dB，
IR=59.285107 dB。

## 8. 构建、断点续作和空间保护

- 运行前要求算法仓库和 SwinIR worktree 均为 clean。
- 固定校验 SwinIR revision、模型 SHA 和 network_swinir.py SHA。
- 完整生成前先运行每个模态的真实权重 FP32 smoke test。
- .build-contract.json 在写任何正式资产前建立，并绑定输入、标签、评估树、脚本、
  仓库提交、权重、网络文件、环境版本、batch、device、尺寸和数值策略。
- partial.npy 与 progress.json 必须成对存在；恢复时需与 build contract 完全匹配。
- 已完成 manifest 后禁止在同一输出目录再次生成。
- 构建前至少保留 40 GiB；预计输出完成后至少保留 20 GiB。
- 构建结束前再次核验算法代码、SwinIR 代码和全部身份输入未发生变化。
- .build-contract.json 与 manifest.json 最终权限为 0444。

## 9. 当前校验器限制

当前及 vendor 中保留的 validate_sysu_swinir_x2.py 都把输入/输出尺寸硬编码为
288 x 144 -> 576 x 288，因此不能不加修改地校验本 PMT256 数据集。
这是校验器适配缺口，不代表当前数据失效。

在新增 manifest-driven 尺寸参数前，恢复验证至少必须完成：

1. 重算第 5 节全部输入 SHA；
2. 重算第 7 节两份训练数组 SHA；
3. 按 manifest 的树哈希算法验证 10578 张评估 PNG；
4. 检查 shape、uint8、非退化内容、IR 通道相等；
5. 将 512 x 256 输出双三次下采样回 256 x 128，检查逐样本和分位数 PSNR。

## 10. 可恢复性结论

- 完整配置：已保留在本文，以及下方 build contract 和 manifest 原文中。
- 语义等价重建：在输入、SwinIR 权重、网络代码和数值策略保持一致时可行。
- 逐字节无损重建：不能仅凭配置承诺，因为原 manifest 缺少 GPU/CUDA/cuDNN 和确定性状态；
  重建后只有输出 SHA 全部匹配才能判定为无损。
- 当前数据若需删除，必须同时保留本文件、SwinIR 权重、原 SYSU 数据和 vendor 历史构建脚本。

## 附录 A：原始 build contract

~~~json
{
  "batch_size": 4,
  "builder_sha256": "e82833a9127d6694460df49bcc55adee046ed01492c8f19067e06ae647572fe9",
  "device": "cuda:0",
  "identity_type": "SYSU-MM01-SwinIR-x2-build",
  "modalities": [
    "rgb",
    "ir"
  ],
  "model_path": "/home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
  "model_sha256": "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac",
  "numeric_policy": "fp32-no-autocast-finite-before-uint8-v1",
  "output_root": "/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1",
  "output_size_hw": [
    512,
    256
  ],
  "repository_revision": "cdd173fbd3c7146f98d39338556a66d64be62f30",
  "schema_version": 4,
  "source_resampling": "bilinear",
  "source_root": "/home/cgv841/datasets/SYSU-MM01",
  "source_size_hw": [
    256,
    128
  ],
  "sources": {
    "ir": {
      "eval_tree": {
        "file_count": 3803,
        "sha256": "caf701503f5cdaf0d928010a42819ec9f68ab9f8eb6384ac8bd7fab67bc02a07"
      },
      "train_array": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_img.npy",
      "train_array_sha256": "ec47b6fc0024053eb356501f2bed62edfe9f6aa4f47706c0fc118effe797aab6",
      "train_label": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_label.npy",
      "train_label_sha256": "0cad1d124e5c768766a11acfb1be0bab7d3cf97b299ddd062081757002591479"
    },
    "rgb": {
      "eval_tree": {
        "file_count": 6775,
        "sha256": "f1044bea50f5c30572ea776489957154e1ef9895fe26832f3525cc188ee935cd"
      },
      "train_array": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_img.npy",
      "train_array_sha256": "f6b37a1bb7fdaba4892c983bd7d6082b5b969efd9d0265c5aff96236bbe332e1",
      "train_label": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_label.npy",
      "train_label_sha256": "29c214045cc9f11f4601286558bd14615dee4f52753951cdf2cda3159d23b2ce"
    }
  },
  "swinir_implementation": {
    "network_file": "/home/cgv841/third_party/SwinIR-official-6545850-v2/models/network_swinir.py",
    "network_sha256": "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913",
    "timm_version": "0.8.6.dev0",
    "torch_version": "1.12.1"
  },
  "swinir_revision": "6545850fbf8df298df73d81f3e8cba638787c8bd",
  "test_id_sha256": "0571165836c315dc4d98a373e64c0b462585aa6ec1559f4cff25d32fe5b8c30e"
}
~~~

## 附录 B：原始 manifest

~~~json
{
  "algorithm": "SwinIR-M-classical-SR-DF2K-x2",
  "build_contract_file_sha256": "9985960030521bac50cae7dce67b10e6f1ae0ed5f9cc817bbe9fbc944590daa7",
  "build_contract_path": "/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1/.build-contract.json",
  "build_identity": {
    "batch_size": 4,
    "builder_sha256": "e82833a9127d6694460df49bcc55adee046ed01492c8f19067e06ae647572fe9",
    "device": "cuda:0",
    "identity_type": "SYSU-MM01-SwinIR-x2-build",
    "modalities": [
      "rgb",
      "ir"
    ],
    "model_path": "/home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "model_sha256": "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac",
    "numeric_policy": "fp32-no-autocast-finite-before-uint8-v1",
    "output_root": "/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1",
    "output_size_hw": [
      512,
      256
    ],
    "repository_revision": "cdd173fbd3c7146f98d39338556a66d64be62f30",
    "schema_version": 4,
    "source_resampling": "bilinear",
    "source_root": "/home/cgv841/datasets/SYSU-MM01",
    "source_size_hw": [
      256,
      128
    ],
    "sources": {
      "ir": {
        "eval_tree": {
          "file_count": 3803,
          "sha256": "caf701503f5cdaf0d928010a42819ec9f68ab9f8eb6384ac8bd7fab67bc02a07"
        },
        "train_array": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_img.npy",
        "train_array_sha256": "ec47b6fc0024053eb356501f2bed62edfe9f6aa4f47706c0fc118effe797aab6",
        "train_label": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_label.npy",
        "train_label_sha256": "0cad1d124e5c768766a11acfb1be0bab7d3cf97b299ddd062081757002591479"
      },
      "rgb": {
        "eval_tree": {
          "file_count": 6775,
          "sha256": "f1044bea50f5c30572ea776489957154e1ef9895fe26832f3525cc188ee935cd"
        },
        "train_array": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_img.npy",
        "train_array_sha256": "f6b37a1bb7fdaba4892c983bd7d6082b5b969efd9d0265c5aff96236bbe332e1",
        "train_label": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_label.npy",
        "train_label_sha256": "29c214045cc9f11f4601286558bd14615dee4f52753951cdf2cda3159d23b2ce"
      }
    },
    "swinir_implementation": {
      "network_file": "/home/cgv841/third_party/SwinIR-official-6545850-v2/models/network_swinir.py",
      "network_sha256": "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913",
      "timm_version": "0.8.6.dev0",
      "torch_version": "1.12.1"
    },
    "swinir_revision": "6545850fbf8df298df73d81f3e8cba638787c8bd",
    "test_id_sha256": "0571165836c315dc4d98a373e64c0b462585aa6ec1559f4cff25d32fe5b8c30e"
  },
  "build_identity_sha256": "87352fda21f0ff349f8b4261eff89944364fe9c125adcb9abe48789f43fa7d48",
  "command": "tools/super_resolution/build_sysu_swinir_x2.py --source-root /home/cgv841/datasets/SYSU-MM01 --output-root /home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1 --swinir-root /home/cgv841/third_party/SwinIR-official-6545850-v2 --model-path /home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --modalities rgb ir --device cuda:0 --batch-size 4 --source-size 256 128 --source-resampling bilinear --min-free-before-gb 40 --min-free-after-gb 20",
  "created_at_unix": 1784491587.3797114,
  "dataset": {
    "sources": {
      "ir": {
        "eval_tree": {
          "file_count": 3803,
          "sha256": "caf701503f5cdaf0d928010a42819ec9f68ab9f8eb6384ac8bd7fab67bc02a07"
        },
        "train_array": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_img.npy",
        "train_array_sha256": "ec47b6fc0024053eb356501f2bed62edfe9f6aa4f47706c0fc118effe797aab6",
        "train_label": "/home/cgv841/datasets/SYSU-MM01/train_ir_resized_label.npy",
        "train_label_count": 11909,
        "train_label_sha256": "0cad1d124e5c768766a11acfb1be0bab7d3cf97b299ddd062081757002591479"
      },
      "rgb": {
        "eval_tree": {
          "file_count": 6775,
          "sha256": "f1044bea50f5c30572ea776489957154e1ef9895fe26832f3525cc188ee935cd"
        },
        "train_array": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_img.npy",
        "train_array_sha256": "f6b37a1bb7fdaba4892c983bd7d6082b5b969efd9d0265c5aff96236bbe332e1",
        "train_label": "/home/cgv841/datasets/SYSU-MM01/train_rgb_resized_label.npy",
        "train_label_count": 22258,
        "train_label_sha256": "29c214045cc9f11f4601286558bd14615dee4f52753951cdf2cda3159d23b2ce"
      }
    },
    "test_id_path": "/home/cgv841/datasets/SYSU-MM01/exp/test_id.txt",
    "test_id_sha256": "0571165836c315dc4d98a373e64c0b462585aa6ec1559f4cff25d32fe5b8c30e",
    "test_identity_count": 96
  },
  "eval_encoding": "lossless PNG bytes under original relative filenames",
  "ir_policy": "BT.601 luminance before SR; channel mean and replication after SR",
  "modalities": [
    "rgb",
    "ir"
  ],
  "numeric_policy": "fp32-no-autocast-finite-before-uint8-v1",
  "output_root": "/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1",
  "output_size_hw": [
    512,
    256
  ],
  "outputs": {
    "ir": {
      "eval_tree": {
        "content": {
          "all_zero_image_count": 0,
          "constant_image_count": 0,
          "image_count": 3803,
          "maximum": 255,
          "mean": 113.44864530761463,
          "minimum": 0,
          "nonzero_fraction": 0.9999990671395065,
          "nonzero_value_count": 1495399053,
          "std": 53.30378521716993,
          "sum": 169651155018,
          "sum_squares": 23495565312096,
          "value_count": 1495400448
        },
        "file_count": 3803,
        "sha256": "9ea4bd574958c8563d670d3895c1531393f1138f1de0d42634fdbe0f3c937ef7"
      },
      "train_array": {
        "content": {
          "all_zero_image_count": 0,
          "constant_image_count": 0,
          "image_count": 11909,
          "maximum": 255,
          "mean": 119.04747656452955,
          "minimum": 0,
          "nonzero_fraction": 0.999996921719647,
          "nonzero_value_count": 4682794929,
          "std": 50.55378153875169,
          "sum": 557476635636,
          "sum_squares": 78333971508408,
          "value_count": 4682809344
        },
        "dtype": "uint8",
        "sha256": "bdf58b9e0622d4177c68d3aa8bd511cae14c024b8c2a193221f47045c76d026f",
        "shape": [
          11909,
          512,
          256,
          3
        ]
      }
    },
    "rgb": {
      "eval_tree": {
        "content": {
          "all_zero_image_count": 0,
          "constant_image_count": 0,
          "image_count": 6775,
          "maximum": 255,
          "mean": 94.72111910173668,
          "minimum": 0,
          "nonzero_fraction": 0.9999562356158229,
          "nonzero_value_count": 2663921810,
          "std": 57.20658983574894,
          "sum": 252340698578,
          "sum_squares": 32620309236402,
          "value_count": 2664038400
        },
        "file_count": 6775,
        "sha256": "5e99c07e9ea80f99715ff779f8974a62709bb2c2e5412827f95873ef0b2cb6d7"
      },
      "train_array": {
        "content": {
          "all_zero_image_count": 0,
          "constant_image_count": 0,
          "image_count": 22258,
          "maximum": 255,
          "mean": 96.36495054436205,
          "minimum": 0,
          "nonzero_fraction": 0.9999454915443192,
          "nonzero_value_count": 8751724659,
          "std": 57.339589392321635,
          "sum": 843405486673,
          "sum_squares": 110050466393381,
          "value_count": 8752201728
        },
        "dtype": "uint8",
        "sha256": "fa46068bcbf4bd435ffbc7e0e705b309a43bf80455b747ecd14a31bd9df4e604",
        "shape": [
          22258,
          512,
          256,
          3
        ]
      }
    }
  },
  "schema_version": 4,
  "smoke_test": {
    "ir": {
      "all_zero_image_count": 0,
      "constant_image_count": 0,
      "downsampled_source_psnr": 59.2851066076247,
      "image_count": 1,
      "maximum": 203,
      "mean": 134.1271514892578,
      "minimum": 55,
      "nonzero_fraction": 1.0,
      "nonzero_value_count": 393216,
      "std": 33.73019493742006,
      "sum": 52740942,
      "sum_squares": 7521364404,
      "value_count": 393216
    },
    "rgb": {
      "all_zero_image_count": 0,
      "constant_image_count": 0,
      "downsampled_source_psnr": 57.68384455978581,
      "image_count": 1,
      "maximum": 182,
      "mean": 91.78369394938152,
      "minimum": 9,
      "nonzero_fraction": 1.0,
      "nonzero_value_count": 393216,
      "std": 32.82589140677421,
      "sum": 36090817,
      "sum_squares": 3736254135,
      "value_count": 393216
    }
  },
  "source_resampling": "bilinear",
  "source_root": "/home/cgv841/datasets/SYSU-MM01",
  "source_size_hw": [
    256,
    128
  ],
  "swinir": {
    "implementation": {
      "network_file": "/home/cgv841/third_party/SwinIR-official-6545850-v2/models/network_swinir.py",
      "network_sha256": "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913",
      "timm_version": "0.8.6.dev0",
      "torch_version": "1.12.1"
    },
    "model_path": "/home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "model_sha256": "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac",
    "revision": "6545850fbf8df298df73d81f3e8cba638787c8bd",
    "root": "/home/cgv841/third_party/SwinIR-official-6545850-v2"
  }
}
~~~
