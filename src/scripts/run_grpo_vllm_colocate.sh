#!/bin/bash
# ============================================================================
# run_grpo_vllm_colocate.sh
# vLLM Colocate Mode: All 4 GPUs used for BOTH training (ZeRO-3 DP) and
# generation (vLLM TP=4). No 5th GPU needed. Memory is managed via
# vLLM sleep()/wake_up() to alternate between training and generation.
#
# Requirements: vllm >= 0.8.0, trl == 0.16.0
# ============================================================================

cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./vllm_colocate_log.txt"

QWEN_PATH='SFT Model Path'
HF_DATASET="./Video-R1-data/Video-R1-260k.json"
OUTPUT_DIR="./log/Qwen2.5-VL-7B-GRPO-vLLM-Colocate"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Qwen2.5-VL-7B-GRPO-vLLM-Colocate"
DS_CONFIG="local_scripts/zero3.json"

# Key difference from server mode: NO extra GPU needed.
# All 4 GPUs participate in both training (DeepSpeed ZeRO-3) and generation (vLLM TP=4).
# nproc_per_node=4 matches the number of visible GPUs.

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --deepspeed ${DS_CONFIG} \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
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
    --beta 0.04 \
    --max_grad_norm 5 \
    --num_generations 4 \
    --use_vllm true \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.3 \
    --report_to wandb \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
