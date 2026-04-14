#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

export RESEARCH_MAX_PROMPT_LENGTH="${RESEARCH_MAX_PROMPT_LENGTH:-4096}"
export RESEARCH_MAX_COMPLETION_LENGTH="${RESEARCH_MAX_COMPLETION_LENGTH:-192}"
export RESEARCH_NUM_GENERATIONS="${RESEARCH_NUM_GENERATIONS:-2}"
export RESEARCH_MAX_STEPS="${RESEARCH_MAX_STEPS:-3}"
export RESEARCH_SAVE_STEPS="${RESEARCH_SAVE_STEPS:-50}"
export RESEARCH_LOGGING_STEPS="${RESEARCH_LOGGING_STEPS:-1}"
export RESEARCH_DEBUG_MODE="${RESEARCH_DEBUG_MODE:-true}"

run_research_training \
  "smoke" \
  "src/r1-v/configs/research_branch/smoke_d1_d2_d3.yaml" \
  "smoke-d1-d2-d3"
