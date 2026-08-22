#!/usr/bin/env bash
set -euo pipefail

export SALT_QRI_DATA_ROOT="${SALT_QRI_DATA_ROOT:-/home/cgv841/datasets/SYSU-MM01}"
export SALT_QRI_MODEL_ROOT="${SALT_QRI_MODEL_ROOT:-/home/lab929/ybj/models}"
export SALT_QRI_RUNTIME_ROOT="${SALT_QRI_RUNTIME_ROOT:-/home/lab929/ybj/models/qri-v1}"
export SALT_SWINIR_ROOT="${SALT_SWINIR_ROOT:-/home/cgv841/third_party/SwinIR-official-6545850-v2}"
export SALT_SWINIR_MODEL="${SALT_SWINIR_MODEL:-/home/cgv841/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth}"
export SALT_PRECOMPUTED_SWINIR_ROOT="${SALT_PRECOMPUTED_SWINIR_ROOT:-/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1}"
export QRI_EXACT_TEXT_ANNOTATION_OUTPUT_ROOT="${QRI_EXACT_TEXT_ANNOTATION_OUTPUT_ROOT:-/home/lab929/ybj/experiments/qri_text_annotations/sysu_exact_fast_v1}"

repository=/home/lab929/ybj/SALT-VI
plugin_root="${repository}/plugins/qwen_imagination"
export PYTHONPATH="${repository}:${repository}/src:${plugin_root}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia_root=/home/lab929/ybj/.conda-envs/salt-vi-flash/lib/python3.9/site-packages/nvidia
cuda_libraries="${nvidia_root}/cublas/lib:${nvidia_root}/cuda_nvrtc/lib"
cuda_libraries="${cuda_libraries}:${nvidia_root}/cudnn/lib:${nvidia_root}/nvtx/lib"
cuda_libraries="${cuda_libraries}:${nvidia_root}/nccl/lib:${nvidia_root}/cusparse/lib"
cuda_libraries="${cuda_libraries}:${nvidia_root}/cuda_runtime/lib:${nvidia_root}/cufft/lib"
cuda_libraries="${cuda_libraries}:${nvidia_root}/cuda_cupti/lib:${nvidia_root}/nvjitlink/lib"
cuda_libraries="${cuda_libraries}:${nvidia_root}/curand/lib:${nvidia_root}/cusolver/lib"
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python=/home/lab929/ybj/.venvs/qri-v1/bin/python
launcher="${repository}/scripts/experiments/qri_text_annotations/run_exact_multi_gpu.py"
config="${repository}/plugins/qwen_imagination/configs/text_annotation_sysu_exact_v1.yaml"

exec "${python}" "${launcher}" --config "${config}" "$@"
