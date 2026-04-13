#!/bin/bash
# =============================================================================
# vLLM Colocate Mode for Video-R1 / T-GRPO on 4x A100-SXM (80GB each)
# =============================================================================
# Key difference from server-mode: NO dedicated vLLM GPU needed.
# Training (ZeRO-3 DP=4) and generation (vLLM TP=4) share the same 4 GPUs.
# Memory handoff is managed by vLLM sleep/wake API (requires vllm >= 0.8.0).
# =============================================================================

set -euo pipefail

cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./vllm_colocate_run.txt"

QWEN_PATH='/workspace/Video-R1/Qwen2.5-VL-7B-COT-SFT'
HF_DATASET="./Video-R1-data/Video-R1-500-subset.json"
OUTPUT_DIR="./log/Qwen2.5-VL-7B-Video-GRPO-Colocate"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Qwen2.5-VL-7B-Video-GRPO-Colocate"
DS_CONFIG="local_scripts/zero3.json"

# =============================================================================
# Colocate-specific settings:
#   --use_vllm true               : Enable vLLM generation (not HF generate)
#   --vllm_tensor_parallel_size 4 : TP=4 to use all 4 GPUs for generation
#   --vllm_gpu_memory_utilization 0.3 : Reserve 30% GPU VRAM for vLLM KV cache;
#                                     70% left for ZeRO-3 training states
#   --nproc_per_node 4            : DP=4 (same 4 GPUs, no 5th GPU needed)
#   NOTE: Do NOT set --vllm_device in colocate mode; vLLM shares training GPUs
# =============================================================================

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model false \
    --temporal true \
    --len_control true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 4 \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.3 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
