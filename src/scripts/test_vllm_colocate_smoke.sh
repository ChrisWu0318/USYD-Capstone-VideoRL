#!/bin/bash
# vLLM Colocate 冒烟测试
# 用法: bash test_vllm_colocate_smoke.sh /path/to/model

set -euo pipefail

MODEL_PATH="${1:-/workspace/Video-R1/Qwen2.5-VL-7B-COT-SFT}"

echo "=========================================="
echo "vLLM Colocate Smoke Test"
echo "Model: ${MODEL_PATH}"
echo "TP: 4, gpu_mem_util: 0.3"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    "${SCRIPT_DIR}/test_vllm_colocate_smoke.py"
