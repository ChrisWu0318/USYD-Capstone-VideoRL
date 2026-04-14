#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

export RESEARCH_MAX_PROMPT_LENGTH="${RESEARCH_MAX_PROMPT_LENGTH:-8192}"
export RESEARCH_MAX_COMPLETION_LENGTH="${RESEARCH_MAX_COMPLETION_LENGTH:-256}"
export RESEARCH_NUM_GENERATIONS="${RESEARCH_NUM_GENERATIONS:-4}"
export RESEARCH_MAX_STEPS="${RESEARCH_MAX_STEPS:-20}"
export RESEARCH_SAVE_STEPS="${RESEARCH_SAVE_STEPS:-100}"
export RESEARCH_LOGGING_STEPS="${RESEARCH_LOGGING_STEPS:-1}"

run_research_training \
  "budget_probe" \
  "src/r1-v/configs/research_branch/budget_probe_d1_d2_d3.yaml" \
  "budget-probe-d1-d2-d3"
