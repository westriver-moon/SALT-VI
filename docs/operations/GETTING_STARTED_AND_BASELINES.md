# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: README

> Source document ID: `source_core:README.md`  
> Original SHA-256: `e4ca4b0ecced784be1036afcefbac3030d36cebbfdcd05e740e59a4eca05e05a`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

![GitHub](https://img.shields.io/badge/license-MIT-green)
![Python ==3.8.18](https://img.shields.io/badge/Python-3.8.18-blue.svg)
![PyTorch ==1.8.1+cu111](https://img.shields.io/badge/PyTorch-1.8.1%2Bcu111-yellow.svg)

# Empowering Visible-Infrared Person Re-Identification with Large Foundation Models

The *official* repository for [Empowering Visible-Infrared Person Re-Identification with Large Foundation Models](https://openreview.net/pdf?id=qQlmONeI5k). SALT-VI greatly mitigates the color information absence in the infrared modality by enriching the infrared representaitons with VLM generated texts, which is a **cross-modality retrieval task bolstered by heterogeneous text descriptions**.

**The overall framework**
![framework](figs/framework.png)

# Highlight

1. We design a Text-enhanced VI-ReID framework driven by Large Foundation Models (SALT-VI). It enriches infrared representations with generated textual descriptions, effectively mitigating the absence of critical information, e.g. color, in the infrared modality and significantly improving the performance of cross-modal retrieval.
2. We propose IFS that fine-tunes a pre-trained VLM to align generated texts with original images. It creates a fusion modality to learn complementary information from the infrared modality and jointly aligns features across all modalities. This ensures stable semantic consistency of text and fusion features with the visible modality during complementary information learning.
3. We propose Modality Ensemble Retrieval that leverages the complementary strengths of all query modalities to form ensemble queries, further improving the performance of cross-modality retrieval bolstered by heterogeneous text descriptions.
4. We introduce three extended VI-ReID datasets with VLM-generated textual descriptions for every image. Extensive experiments on these expanded datasets demonstrate the competitive performance of our SALT-VI framework, paving the way for utilizing large foundation models in downstream data-demanding multi-modal retrieval tasks.

# Prepare Datasets
* Put *SYSU-MM01*, *RegDB* and *LLCM* dataset into `datasets/sysu`, `data/regdb`, and `data/llcm`, as the following structure:
```
data_root
├── llcm
│   ├── idx
│   ├── nir
│   ├── test_nir
│   ├── test_vis
│   ├── Text
│   └── vis
├── regdb
│   ├── idx
│   ├── Text
│   ├── Thermal
│   └── Visible
└── sysu
    ├── cam1
    ├── cam2
    ├── cam3
    ├── cam4
    ├── cam5
    ├── cam6
    ├── exp
    ├── Text
    ├── train_ir_resized_img.npy
    ├── train_ir_resized_label.npy
    ├── train_rgb_resized_img.npy
    └── train_rgb_resized_label.npy
```


* then run `src/salt_vi/data/pre_data_processing.py` to process SYSU-MM01 dataset.
* There **have been** generated texts and LLM augmented texts for each image in the corresponding dataset directory `datasets/sysu/Text`, `datasets/regdb/Text` and `datasets/llcm/Text`.


# Open-Sources Zoos
We provide all texts of expanded datasets, weights and training logs of **pretrained base VI-ReID models** [[google drive]](https://drive.google.com/drive/folders/1DiyE1ySdWmAiNtWnG01FemZSbl2uR9aP?usp=drive_link) and **trained models** [[google drive]](https://drive.google.com/drive/folders/1CoUks3X7_ORui8Rj-Zxw317YhpKO2-yQ?usp=drive_link) for SALT-VI on three expanded datasets.

# Usage
We utilize a single Nvidia RTX 3090 GPU for training.

**Dependencies:**

First of all, **cd to the root directory of SALT-VI**:
```
cd path/to/SALT-VI
```
The formal server experiments use Python 3.8.18, PyTorch 1.8.1+cu111 and
torchvision 0.9.1+cu111. Create the recorded environment with:
```
conda env create -f environment-server.yml
conda activate clipreid
```
For an existing Python 3.8 environment, install the same direct runtime
dependencies with:
```
pip install -r requirements-server.txt
```

Install the canonical package so `python -m salt_vi...` commands resolve from any working directory:
```
pip install -e .
```
Install test tooling separately before running the regression suite:
```
pip install -r requirements-test.txt
```

`requirements-upstream-legacy.txt` preserves the original upstream export for
reference; it is not an installable server specification because it contains
machine-local build paths and a different PyTorch stack.

**parameters:**

* `--mode train` # train or test
* `--training_mode` # RGB_IR for base training and RGB_IR_Text for SALT-VI
* `--captioner_name` # the name of the captioner, here we use Blip.
* `--llm_aug` # whether to use LLM augmented texts
* `--joint_mode` # uni for MJL, ir_crossfusion for base training and baseline with text
* `--Feat_Filter` # Whether to use SFF
* `--lr_txt` # learning rate for text
* `--text_weight_decay` # weight decay for text
* `--text_weight_decay_bias` # weight decay for text bias
* `--fusion_way` # the way to fuse the features, 'add' as default
* `--Fix_Visual` # first we train a base model, then we fix it and fine-tune the text encoder (CLIP)
* `--training_weight_init` # the path to the base model weights
* `--output_path` # the output path of logs and weights
* `--dataset` # sysu, regdb or llcm
* `--loss_names` # choose the type of loss functions, here we use id and wrt loss
* `--test_modality`  # Fusion or IR (->RGB)
* `--eval_start_epoch` # the epoch to start evaluation
* `--CUDA_VISIBLE_DEVICES` # visible GPU
* `--gpu_id` # chosen GPU
* `--trial` # the split number of RegDB
* `--CAT_EVAL` # whether to use ensemble retrieval
* `--LOG4TEST` # whether to log the test results


**Examples:**

Tri-SYSU-MM01:

Step 1. Download the **pretrained base VI-ReID models** [[google drive]](https://drive.google.com/drive/folders/1DiyE1ySdWmAiNtWnG01FemZSbl2uR9aP?usp=drive_link) and put it to `base_model/sysu/`,

or,

train the base model by yourself:
```shell
bash scripts/base_training/sysu_base_run.sh
```
The trained ckpt of base models and logs can be found in `logs/raw/source_core/logs/sysu/base`.
Then we need to put the base model weights into `base_model/sysu/`.

Step 2. Incrementally train the SALT-VI **w/o MER**:(need to input the path to the base model weights)
```shell
bash scripts/training/sysu_train_run.sh
```
We can find the log files and weights save in `logs/raw/source_core/logs/sysu/`. 

Step 3. Test trained models **with MER** module: (need to input the path to the **trained models** [[google drive]](https://drive.google.com/drive/folders/1CoUks3X7_ORui8Rj-Zxw317YhpKO2-yQ?usp=drive_link))
```shell
bash scripts/testing/sysu_test.sh
```
Through MER we can get further improved retrieval performance.


**Note**: The results may slightly vary due to the different types of GPUs and different versions of CUDA. Don't mind the `GIT`, it doesn't participate in training process, please keep it there to keep the randomness of the training process of SALT-VI.

# Generators
We provide the weights and inference code of the LLM rephraser [Vicuna-7b v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) and IR & RGB modality-specific text generators [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) in `./generators/`. The code is based on Huggingface's Transformers library.

Generator dependencies are optional and intentionally kept out of the training runtime:
```
pip install -r requirements-generators.txt
```

# Acknowledgments

Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [DEEN](https://github.com/ZYK100/LLCM), [UNIReID](https://github.com/ccq195/UNIReID) and [CAJ](https://github.com/zesenwu23/caj) and [Documents from hugginface](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning). We sincerely appreciate for their contributions.


# Citation
If you find this code useful for your research, please cite our paper.

```
@inproceedings{2024SALT-VI,
    title={Empowering Visible-Infrared Person Re-Identification with Large Foundation Models},
    author={Zhangyi Hu and Bin Yang and Mang Ye},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
}
```

# Contact
zhangyi_hu@whu.edu.cn; yangbin_cv@whu.edu.cn; yemang@whu.edu.cn.

## Current Stage A Main Config

The selected Stage A main config is:

```text
configs/stage_a/vision_text_encoder_stage_a_current_best.yaml
```

It corresponds to `PMT_VIT + PMT recipe + 288x144 + 768 no-projection +
identity_auto_replace + PK=8x4 + pmt_cross_modal_hard`.

The historical PMT recipe baseline is retained for comparison at:

```text
configs/stage_a/vision_text_encoder_stage_a_vision_text_recipe_288x144_768.yaml
```

## Current research extensions

The maintained research line extends the upstream SALT-VI implementation with reproducible Stage A, Stage B, A3-E4 HPT, and bidirectional token-cycle configurations. Stage B configurations live under `configs/stage_b/`; formal experiment launchers and analysis tools live under `scripts/metric_boost/`, `scripts/token_cycle/`, and `scripts/analysis/`.

Compact completed-experiment archives are stored under `reports/`. They retain selection decisions, same-epoch metrics, source/config/data provenance, checkpoint identities, and compact analysis summaries. Large runtime logs, caches, repeated metadata, and full per-trial feature exports are intentionally excluded from Git. See `docs/protocols/experiment_governance.md` and validate changes with:

```shell
python scripts/validate_experiment_archives.py
pytest -q
```


---

## Migrated source: README

> Source document ID: `source_baseline:README.md`  
> Original SHA-256: `b2841b517cf8e6895c30cd62078416f537bce67bb9bbe63b0653d5fdca074339`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Independent PMT SYSU-MM01 Baseline

This directory is an independent reproduction baseline for:

Learning Progressive Modality-shared Transformers for Effective Visible-Infrared Person Re-identification, AAAI 2023.

Official references:

- Paper: https://arxiv.org/abs/2212.00226
- Official PMT repository: https://github.com/hulu88/PMT
- Official PyTorch code: https://github.com/hulu88/PMT/tree/main/Pytorch-PMT-VI-ReID
- SYSU-MM01 page: https://isee.sysu.edu.cn/project/RGBIRReID.htm

This implementation is intentionally separate from `SALT-VI`. It does not use CLIP, text descriptions, LASTViT, VCM, RegDB pretraining, or SALT-VI fusion code.

## Data

Default SYSU root:

```bash
/home/cgv841/datasets/SYSU-MM01
```

Required files:

```text
train_rgb_resized_img.npy
train_rgb_resized_label.npy
train_ir_resized_img.npy
train_ir_resized_label.npy
exp/train_id.txt
exp/val_id.txt
exp/test_id.txt
```

If the `.npy` files are absent, refer to the official PMT preprocessing script:

https://github.com/hulu88/PMT/blob/main/Pytorch-PMT-VI-ReID/process_sysu.py

Do not reprocess SYSU when the existing cache is already valid.

## Weights

ImageNet ViT-B/16:

```bash
python -m salt_vi.baselines.vision_text.tools.download_weights --imagenet
```

Official PMT SYSU checkpoint:

```bash
python -m pip install gdown
python -m salt_vi.baselines.vision_text.tools.download_weights --official
```

Manual URLs:

- ImageNet ViT-B/16: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
- Official PMT SYSU checkpoint: https://drive.google.com/file/d/1S7Upn_8dWHNN5R3woazpocFU6J8hvCIe/view?usp=share_link

Weights and datasets must not be committed.

## Preflight

```bash
python -m salt_vi.baselines.vision_text.tools.preflight \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

For pipeline-only checks without ImageNet weights:

```bash
python -m salt_vi.baselines.vision_text.tools.preflight \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --allow-missing-pretrained \
  --device cuda:0
```

## Train

```bash
python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/official_reproduction \
  --device cuda:0
```

One-batch smoke test:

```bash
python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/smoke \
  --device cuda:0 \
  --smoke-batches 1
```

Resume:

```bash
python -m salt_vi.baselines.vision_text.train \
  --config configs/vision_text/sysu_baseline.yaml \
  --resume logs/raw/source_baseline/outputs/vision_text/official_reproduction/checkpoints/latest.pth \
  --device cuda:0
```

## PMT-MBPatch Variant

`configs/sysu_vision_text_mbpatch.yaml` keeps the PMT SYSU data, losses, training schedule, and evaluation protocol unchanged, but replaces the single overlapping patch embedding with a two-branch patch embedding:

```text
[16,16] stride [12,12]
[16,8]  stride [12,6]
```

The first branch is the anchor branch, so the token count remains compatible with the original PMT transformer and losses. ImageNet single-branch patch weights are copied into the anchor branch and resized for the added branch; the 1x1 fusion starts as an anchor-branch identity.

Startup smoke command:

```bash
python -m salt_vi.baselines.vision_text.train \
  --config configs/sysu_vision_text_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output logs/raw/source_baseline/outputs/vision_text/mbpatch_smoke \
  --device cuda:0 \
  --smoke-batches 1
```

## Test

Self-trained best checkpoint:

```bash
python -m salt_vi.baselines.vision_text.test \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights logs/raw/source_baseline/outputs/vision_text/official_reproduction/checkpoints/best.pth \
  --mode all \
  --gallery-mode single \
  --trials 10 \
  --device cuda:0
```

Official PMT checkpoint:

```bash
python -m salt_vi.baselines.vision_text.test \
  --config configs/vision_text/sysu_baseline.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights pretrained/pmt_sysu_vit_official.pth \
  --mode all \
  --gallery-mode single \
  --trials 10 \
  --device cuda:0
```

Expected official SYSU all-search single-shot reference is approximately Rank-1 67.53%, mAP 64.98%, mINP 51.86%. Reasonable variance is expected; the metric must not be hard-coded.

## Output

```text
logs/raw/source_baseline/outputs/vision_text/<run_name>/
├── config_resolved.yaml
├── train.log
├── metrics.jsonl
├── metrics.csv
├── checkpoints/
│   ├── latest.pth
│   ├── best.pth
│   └── epoch_XX.pth
└── evaluation/
    ├── trial_00.json
    └── average.json
```

## Compatibility Changes

Compared with the official PMT code, this independent version only changes engineering compatibility:

- paths are YAML/CLI driven;
- `.cuda()` is replaced with `.to(device)`;
- `torch.load` uses `map_location`;
- obsolete `Variable` usage is removed;
- old PyTorch `addmm_` calls use the modern signature;
- checkpoint save/resume records optimizer, scaler, best mAP, config, and random states;
- runtime assertions check PMT batch layout and finite losses.

The PMT model, two-stage training schedule, PK sampling layout, MSEL/DCL logic, and SYSU evaluation protocol are preserved.
