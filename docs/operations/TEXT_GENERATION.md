# SALT-VI canonical document

This document consolidates related legacy material. All configuration, code, data and output references below have been rewritten to the SALT-VI layout.


---

## Migrated source: README

> Source document ID: `source_core:generators/README.md`  
> Original SHA-256: `3d909553738c75fdb85a5c92c135f4a3b674a4bcafa6bea067444d9a37c9b489`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Generators

### Preparing

Install the optional generator dependencies from the project root:
```shell
pip install -r requirements-generators.txt
```

Then download the well-trained [IR-CAPTION.tar.gz](https://drive.google.com/file/d/17nOBeGHf4r4MHSeuFut4Pf-4m8ZSPPjC/view?usp=drive_link) and [RGB-CAPTION.tar.gz](https://drive.google.com/file/d/1_w751YFyBLnnVBcnnLK4gFRvbrng1t1E/view?usp=drive_link), put them in the `generators/weights/IR_CAPTION` directory and `generators/weights/RGB_CAPTION` directory, respectively. Then decompress them.

Now we have the file tree like:

```
weights
├── IR_CAPTION
│   ├── checkpointxxxx/
│   ├── IR-CAPTION.tar.gz
├── RGB_CAPTION
│   ├── checkpointxxxx/
│   ├── RGB-CAPTION.tar.gz
```

### Guidance


`demo/generator_demo.ipynb`: The textual expanding code for IR/RGB person images. (We can get text descriptions for infrared images without color and those for visible images with color.)

`demo/llm_rephrase.py`: The code for rephrasing generated texts with LLM.
```
python generators/code/llm_rephrase.py --input 'Her black shoes are a matching sky blue red' --gpus 0
```

`demo/color_mover.ipynb` is a tool aims to remove the color in the rephrased infrared image texts caused by LLM hallucination. (**Note**: the BLIP generated infrared image texts do not contain any color, to mitigate this hallucination we can manually add prompts against color representation while LLM rephrasing infrared image texts.)


---

## Migrated source: README QWEN CAPTION AUGMENTATION

> Source document ID: `source_core:generators/README_QWEN_CAPTION_AUGMENTATION.md`  
> Original SHA-256: `bbbb6c65778152fbc461b04604f757322f369b9e00d619847cbd644b46394937`  
> Canonical runtime: `/home/cgv841/ybj/SALT-VI/src/salt_vi/` and `/home/cgv841/ybj/SALT-VI/scripts/`  
> This section is rewritten for the SALT-VI layout; it is not an active compatibility layer.

# Qwen caption augmentation

This offline pipeline turns every original RGB BLIP caption into exactly four
faithful paraphrases with `Qwen/Qwen3-14B-AWQ`. Runtime data and model weights
remain outside Git.

The canonical server outputs are:

- `/home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ/`
- `/home/cgv841/datasets/RegDB/Text/Blip_RGB_Qwen3_14B_AWQ/`

Generation is journaled per shard and can be resumed by rerunning the same
command. Do not delete `paraphrases.shard-*.jsonl` until the final merged JSON
has been verified.

## Environment

```bash
/home/cgv841/anaconda3/envs/diffsensei/bin/python -m venv \
  --system-site-packages /home/cgv841/.venvs/qwen-caption
/home/cgv841/.venvs/qwen-caption/bin/pip install --no-deps autoawq==0.2.9
/home/cgv841/.venvs/qwen-caption/bin/pip install \
  transformers==4.53.3 accelerate==1.8.1 datasets==3.6.0 zstandard==0.25.0
```

## Dry run

```bash
python generators/qwen_caption_augmentation.py \
  --input datasets/sysu/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --output-dir /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ \
  --dry-run
```

## Single-GPU generation

On a dedicated 24 GB RTX 3090, `batch-size=32` uses about 16 GB in the tested
environment. The command below writes one append-only journal and can be
rerun unchanged after interruption:

```bash
CUDA_VISIBLE_DEVICES=1 /home/cgv841/.venvs/qwen-caption/bin/python \
  generators/qwen_caption_augmentation.py \
  --input datasets/sysu/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --output-dir /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ \
  --model /home/cgv841/models/Qwen3-14B-AWQ \
  --device cuda:0 --batch-size 32 --checkpoint-every 256
```

For RegDB, replace the input with
`datasets/regdb/Text/Blip_RGB/caption_dict_Blip_RGB.json` and the output root
with `/home/cgv841/datasets/RegDB/Text/Blip_RGB_Qwen3_14B_AWQ`.

## Resume and format failures

Keep `shard-id` and `num-shards` unchanged when resuming. Successful records
are read from the journal and are not generated again. A nonzero exit after
the first pass means only the keys recorded in `failures.shard-*.json` remain.
Rerun with a different `--seed` to avoid repeating a deterministic invalid
answer. For a stubborn length failure, request a shorter answer while keeping
the original validation contract, for example:

```bash
--max-words 45 --prompt-max-words 35
```

`prompt-max-words` changes only the length requested from Qwen;
`max-words` remains the hard acceptance limit. A successful resume atomically
clears the shard failure file to `{}`. Always run the merger before treating
the canonical JSON as complete.

## Four-GPU sharding

Run one process per completely free GPU. Use the same `--num-shards` value and
a distinct `--shard-id` for every process. Example for GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 /home/cgv841/.venvs/qwen-caption/bin/python \
  generators/qwen_caption_augmentation.py \
  --input datasets/sysu/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --output-dir /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ \
  --model /home/cgv841/models/Qwen3-14B-AWQ \
  --shard-id 0 --num-shards 4 --batch-size 32 --checkpoint-every 256
```

After every shard reports complete, merge and verify:

```bash
python generators/merge_qwen_caption_shards.py \
  --input datasets/sysu/Text/Blip_RGB/caption_dict_Blip_RGB.json \
  --shard-dir /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ \
  --output /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json
```
