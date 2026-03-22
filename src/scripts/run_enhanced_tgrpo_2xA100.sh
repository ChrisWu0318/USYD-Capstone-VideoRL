cd src/r1-v
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_enhanced_tgrpo.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir "./log/Enhanced-TGRPO-Phase1-continued" \
    --model_name_or_path './log/Enhanced-TGRPO-Phase1/checkpoint-50' \
    --dataset_name "./Video-R1-data/Video-R1-500-subset.json" \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 8192 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --corruption_strength 1.0 \
    --curriculum_learning true \
    --margin_scale 0.5 \
    --reward_threshold 0.1 \
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --max_steps 150 \
    --run_name Enhanced-TGRPO-Phase1-continued \
    --save_steps 50 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 4 \
    --report_to wandb
