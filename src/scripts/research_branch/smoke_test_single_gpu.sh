#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

config_path="$(resolve_existing_path "src/r1-v/configs/research_branch/smoke_d1_d2_d3.yaml")" || fail "Smoke config not found."
ds_config="$(resolve_existing_path "${RESEARCH_DEEPSPEED_CONFIG:-local_scripts/zero3.json}")" || fail "DeepSpeed config not found."
model_ref="$(resolve_local_or_remote_ref "RESEARCH_MODEL_NAME_OR_PATH" "${RESEARCH_MODEL_NAME_OR_PATH:-}")"
dataset_ref="$(resolve_local_or_remote_ref "RESEARCH_DATASET_NAME" "${RESEARCH_DATASET_NAME:-}")"

git_branch="$(git -C "${repo_root}" branch --show-current)"
git_commit="$(git -C "${repo_root}" rev-parse --short HEAD)"
timestamp="$(date +%Y%m%d-%H%M%S)"
visible_devices="${RESEARCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
output_dir="${r1_root}/log/research_branch/smoke/smoke-single-gpu-${git_commit}-${timestamp}"
run_name="smoke-single-gpu-${git_commit}-${timestamp}"

mkdir -p "${output_dir}"

export DEBUG_MODE="${RESEARCH_DEBUG_MODE:-true}"
export LOG_PATH="${output_dir}/debug.log"

echo "[research_branch] branch=${git_branch}"
echo "[research_branch] commit=${git_commit}"
echo "[research_branch] stage=smoke-single-gpu"
echo "[research_branch] config=${config_path}"
echo "[research_branch] output_dir=${output_dir}"
echo "[research_branch] model=${model_ref}"
echo "[research_branch] dataset=${dataset_ref}"
echo "[research_branch] deepspeed=${ds_config}"
echo "[research_branch] mode=HF non-colocate"
echo "[research_branch] resolved_nproc_per_node=1"
echo "[research_branch] cuda_visible_devices=${visible_devices}"

cmd=(
  deepspeed
  --num_gpus=1
  src/open_r1/grpo.py
  --output_dir "${output_dir}"
  --model_name_or_path "${model_ref}"
  --dataset_name "${dataset_ref}"
  --deepspeed "${ds_config}"
  --max_prompt_length "${RESEARCH_MAX_PROMPT_LENGTH:-4096}"
  --max_completion_length "${RESEARCH_MAX_COMPLETION_LENGTH:-192}"
  --per_device_train_batch_size "${RESEARCH_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${RESEARCH_GRADIENT_ACCUMULATION_STEPS:-1}"
  --learning_rate "${RESEARCH_LEARNING_RATE:-1e-6}"
  --lr_scheduler_type "${RESEARCH_LR_SCHEDULER_TYPE:-cosine}"
  --weight_decay "${RESEARCH_WEIGHT_DECAY:-0.01}"
  --bf16 true
  --logging_steps "${RESEARCH_LOGGING_STEPS:-1}"
  --gradient_checkpointing true
  --attn_implementation "${RESEARCH_ATTN_IMPLEMENTATION:-sdpa}"
  --min_pixels "${RESEARCH_MIN_PIXELS:-3136}"
  --max_pixels "${RESEARCH_MAX_PIXELS:-501760}"
  --num_train_epochs "${RESEARCH_NUM_TRAIN_EPOCHS:-1}"
  --max_steps "${RESEARCH_MAX_STEPS:-3}"
  --run_name "${run_name}"
  --save_steps "${RESEARCH_SAVE_STEPS:-50}"
  --save_only_model false
  --temporal "${RESEARCH_TEMPORAL:-true}"
  --len_control "${RESEARCH_LEN_CONTROL:-true}"
  --report_to "${RESEARCH_REPORT_TO:-wandb}"
  --beta "${RESEARCH_BETA:-0.04}"
  --max_grad_norm "${RESEARCH_MAX_GRAD_NORM:-5}"
  --num_generations "${RESEARCH_NUM_GENERATIONS:-2}"
  --experiment_config "${config_path}"
)

echo "[research_branch] command:"
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "${RESEARCH_DRY_RUN:-false}" == "true" ]]; then
  echo "[research_branch] dry run requested; not launching training."
  exit 0
fi

cd "${r1_root}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${cmd[@]}" 2>&1 | tee "${output_dir}/training_log.txt"
