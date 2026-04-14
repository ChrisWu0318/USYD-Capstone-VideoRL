#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

# Adjust RESEARCH_MAX_STEPS, RESEARCH_SAVE_STEPS, or RESEARCH_USE_VLLM on Runpod
# only when you intentionally want a different budget or colocate launch mode.
export RESEARCH_MAX_STEPS="${RESEARCH_MAX_STEPS:-1200}"
export RESEARCH_SAVE_STEPS="${RESEARCH_SAVE_STEPS:-100}"
export RESEARCH_LOGGING_STEPS="${RESEARCH_LOGGING_STEPS:-1}"

run_research_training \
  "formal" \
  "src/r1-v/configs/research_branch/formal_d1_d2_d3.yaml" \
  "formal-d1-d2-d3"
