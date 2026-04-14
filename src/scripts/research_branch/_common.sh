#!/usr/bin/env bash

research_branch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${research_branch_dir}/../../.." && pwd)"
r1_root="${repo_root}/src/r1-v"

fail() {
  echo "[research_branch] ERROR: $*" >&2
  exit 1
}

resolve_existing_path() {
  local raw_path="$1"

  if [[ -z "${raw_path}" ]]; then
    return 1
  fi
  if [[ -e "${raw_path}" ]]; then
    local dir_name=""
    local base_name=""
    dir_name="$(cd "$(dirname "${raw_path}")" && pwd)"
    base_name="$(basename "${raw_path}")"
    printf '%s/%s\n' "${dir_name}" "${base_name}"
    return 0
  fi
  if [[ -e "${repo_root}/${raw_path}" ]]; then
    printf '%s\n' "${repo_root}/${raw_path}"
    return 0
  fi
  if [[ -e "${r1_root}/${raw_path}" ]]; then
    printf '%s\n' "${r1_root}/${raw_path}"
    return 0
  fi
  return 1
}

resolve_local_or_remote_ref() {
  local label="$1"
  local raw_value="$2"
  local resolved_path=""

  [[ -n "${raw_value}" ]] || fail "${label} is required."

  if resolved_path="$(resolve_existing_path "${raw_value}" 2>/dev/null)"; then
    printf '%s\n' "${resolved_path}"
    return 0
  fi

  if [[ "${raw_value}" == /* || "${raw_value}" == ./* || "${raw_value}" == ../* || "${raw_value}" == *.json || "${raw_value}" == *.jsonl ]]; then
    fail "${label} looks like a local path but does not exist: ${raw_value}"
  fi

  printf '%s\n' "${raw_value}"
}

infer_gpu_count_from_visible_devices() {
  local visible_devices="$1"
  local compact_devices=""

  compact_devices="${visible_devices// /}"
  compact_devices="${compact_devices#,}"
  compact_devices="${compact_devices%,}"

  if [[ -z "${compact_devices}" ]]; then
    fail "Cannot infer GPU count from an empty RESEARCH_CUDA_VISIBLE_DEVICES value."
  fi

  IFS=',' read -r -a device_array <<< "${compact_devices}"
  printf '%s\n' "${#device_array[@]}"
}

resolve_nproc_per_node() {
  if [[ -n "${RESEARCH_NPROC_PER_NODE:-}" ]]; then
    printf '%s\n' "${RESEARCH_NPROC_PER_NODE}"
    return 0
  fi

  if [[ -n "${RESEARCH_CUDA_VISIBLE_DEVICES:-}" ]]; then
    infer_gpu_count_from_visible_devices "${RESEARCH_CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  printf '1\n'
}

run_research_training() {
  local stage_name="$1"
  local config_ref="$2"
  local experiment_label="$3"
  shift 3

  local config_path=""
  local ds_config=""
  local model_ref=""
  local dataset_ref=""
  local git_branch=""
  local git_commit=""
  local timestamp=""
  local output_dir=""
  local run_name=""
  local visible_devices=""
  local resolved_nproc_per_node=""
  local report_to=""
  local max_steps=""
  local num_train_epochs=""

  config_path="$(resolve_existing_path "${config_ref}")" || fail "Experiment config not found: ${config_ref}"
  ds_config="$(resolve_existing_path "${RESEARCH_DEEPSPEED_CONFIG:-local_scripts/zero3.json}")" || fail "DeepSpeed config not found."
  model_ref="$(resolve_local_or_remote_ref "RESEARCH_MODEL_NAME_OR_PATH" "${RESEARCH_MODEL_NAME_OR_PATH:-}")"
  dataset_ref="$(resolve_local_or_remote_ref "RESEARCH_DATASET_NAME" "${RESEARCH_DATASET_NAME:-}")"

  git_branch="$(git -C "${repo_root}" branch --show-current)"
  git_commit="$(git -C "${repo_root}" rev-parse --short HEAD)"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  output_dir="${r1_root}/log/research_branch/${stage_name}/${experiment_label}-${git_commit}-${timestamp}"
  run_name="${experiment_label}-${git_commit}-${timestamp}"
  visible_devices="${RESEARCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
  resolved_nproc_per_node="$(resolve_nproc_per_node)"
  report_to="${RESEARCH_REPORT_TO:-wandb}"
  max_steps="${RESEARCH_MAX_STEPS:-300}"
  num_train_epochs="${RESEARCH_NUM_TRAIN_EPOCHS:-1}"

  mkdir -p "${output_dir}"

  export DEBUG_MODE="${RESEARCH_DEBUG_MODE:-false}"
  export LOG_PATH="${output_dir}/debug.log"

  echo "[research_branch] branch=${git_branch}"
  echo "[research_branch] commit=${git_commit}"
  echo "[research_branch] stage=${stage_name}"
  echo "[research_branch] config=${config_path}"
  echo "[research_branch] output_dir=${output_dir}"
  echo "[research_branch] model=${model_ref}"
  echo "[research_branch] dataset=${dataset_ref}"
  echo "[research_branch] deepspeed=${ds_config}"
  echo "[research_branch] mode=$([[ "${RESEARCH_USE_VLLM:-false}" == "true" ]] && echo 'vLLM colocate' || echo 'HF non-colocate')"
  echo "[research_branch] resolved_nproc_per_node=${resolved_nproc_per_node}"
  echo "[research_branch] max_prompt_length=${RESEARCH_MAX_PROMPT_LENGTH:-16384}"
  echo "[research_branch] max_completion_length=${RESEARCH_MAX_COMPLETION_LENGTH:-768}"
  echo "[research_branch] num_generations=${RESEARCH_NUM_GENERATIONS:-4}"
  echo "[research_branch] max_steps=${max_steps}"
  echo "[research_branch] save_steps=${RESEARCH_SAVE_STEPS:-100}"
  echo "[research_branch] logging_steps=${RESEARCH_LOGGING_STEPS:-1}"
  echo "[research_branch] temporal=${RESEARCH_TEMPORAL:-true}"
  echo "[research_branch] len_control=${RESEARCH_LEN_CONTROL:-true}"
  echo "[research_branch] cuda_visible_devices=${visible_devices}"

  local torchrun_cmd=(
    torchrun
    --nproc_per_node "${resolved_nproc_per_node}"
    --nnodes "${RESEARCH_NNODES:-1}"
    --node_rank "${RESEARCH_NODE_RANK:-0}"
    --master_addr "${RESEARCH_MASTER_ADDR:-127.0.0.1}"
    --master_port "${RESEARCH_MASTER_PORT:-12365}"
    src/open_r1/grpo.py
    --output_dir "${output_dir}"
    --model_name_or_path "${model_ref}"
    --dataset_name "${dataset_ref}"
    --deepspeed "${ds_config}"
    --max_prompt_length "${RESEARCH_MAX_PROMPT_LENGTH:-16384}"
    --max_completion_length "${RESEARCH_MAX_COMPLETION_LENGTH:-768}"
    --per_device_train_batch_size "${RESEARCH_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${RESEARCH_GRADIENT_ACCUMULATION_STEPS:-1}"
    --learning_rate "${RESEARCH_LEARNING_RATE:-1e-6}"
    --lr_scheduler_type "${RESEARCH_LR_SCHEDULER_TYPE:-cosine}"
    --weight_decay "${RESEARCH_WEIGHT_DECAY:-0.01}"
    --bf16 true
    --logging_steps "${RESEARCH_LOGGING_STEPS:-1}"
    --gradient_checkpointing true
    --attn_implementation "${RESEARCH_ATTN_IMPLEMENTATION:-flash_attention_2}"
    --min_pixels "${RESEARCH_MIN_PIXELS:-3136}"
    --max_pixels "${RESEARCH_MAX_PIXELS:-501760}"
    --num_train_epochs "${num_train_epochs}"
    --max_steps "${max_steps}"
    --run_name "${run_name}"
    --save_steps "${RESEARCH_SAVE_STEPS:-100}"
    --save_only_model false
    --temporal "${RESEARCH_TEMPORAL:-true}"
    --len_control "${RESEARCH_LEN_CONTROL:-true}"
    --report_to "${report_to}"
    --beta "${RESEARCH_BETA:-0.04}"
    --max_grad_norm "${RESEARCH_MAX_GRAD_NORM:-5}"
    --num_generations "${RESEARCH_NUM_GENERATIONS:-4}"
    --experiment_config "${config_path}"
  )

  if [[ "${RESEARCH_USE_VLLM:-false}" == "true" ]]; then
    torchrun_cmd+=(
      --use_vllm true
      --vllm_tensor_parallel_size "${RESEARCH_VLLM_TENSOR_PARALLEL_SIZE:-4}"
      --vllm_gpu_memory_utilization "${RESEARCH_VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
    )
  fi

  torchrun_cmd+=("$@")

  echo "[research_branch] command:"
  printf '  %q' "${torchrun_cmd[@]}"
  printf '\n'

  if [[ "${RESEARCH_DRY_RUN:-false}" == "true" ]]; then
    echo "[research_branch] dry run requested; not launching training."
    return 0
  fi

  cd "${r1_root}"
  CUDA_VISIBLE_DEVICES="${visible_devices}" "${torchrun_cmd[@]}" 2>&1 | tee "${output_dir}/training_log.txt"
}
