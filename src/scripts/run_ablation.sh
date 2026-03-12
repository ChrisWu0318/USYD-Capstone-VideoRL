#!/bin/bash
# Ablation Study Launch Script
# Usage:
#   RUN_NAME=ablation_baseline MULTI_CORRUPTION=false MARGINAL_REWARD=false bash src/scripts/run_ablation.sh
#
# Four ablation configs:
#   1. Baseline:    RUN_NAME=ablation_baseline     MULTI_CORRUPTION=false MARGINAL_REWARD=false
#   2. Task A only: RUN_NAME=ablation_taskA_only   MULTI_CORRUPTION=true  MARGINAL_REWARD=false
#   3. Task B only: RUN_NAME=ablation_taskB_only   MULTI_CORRUPTION=false MARGINAL_REWARD=true
#   4. Task A + B:  RUN_NAME=ablation_taskA_taskB   MULTI_CORRUPTION=true  MARGINAL_REWARD=true

cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_${RUN_NAME}.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir "./log/${RUN_NAME}" \
    --model_name_or_path 'SFT Model Path' \
    --dataset_name "./Video-R1-data/Video-R1-260k.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --multi_corruption ${MULTI_CORRUPTION} \
    --marginal_reward ${MARGINAL_REWARD} \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 3 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8
