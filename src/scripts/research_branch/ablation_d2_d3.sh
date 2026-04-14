#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

run_research_training \
  "ablation" \
  "src/r1-v/configs/research_branch/ablation_d2_d3.yaml" \
  "ablation-d2-d3"
