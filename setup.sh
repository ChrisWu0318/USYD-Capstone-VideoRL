#!/usr/bin/env bash
# ============================================================================
# Blackwell (RTX Pro 6000 / sm_120) compatible install script for AutoDL pods.
#
# Strategy: pin torch to 2.9.1+cu128 so we can use the prebuilt flash-attn wheel
# (torch 2.10 has no matching flash-attn release as of 2026-04). vLLM is skipped
# here — install manually if needed (see docs/experiments/autodl_pro6000_setup.md).
#
# Full step-by-step guide with rationale, troubleshooting, and verification:
#   docs/experiments/autodl_pro6000_setup.md
# ============================================================================
set -euo pipefail

# AutoDL cross-device link guard: if pip cache is on data disk, TMPDIR must be
# on the same filesystem or flash-attn wheel install will fail mid-copy.
if [[ "${PIP_CACHE_DIR:-}" == /root/autodl-tmp/* && -z "${TMPDIR:-}" ]]; then
  export TMPDIR=/root/autodl-tmp/tmp
  mkdir -p "$TMPDIR"
  echo "[setup] TMPDIR set to $TMPDIR (same filesystem as PIP_CACHE_DIR)"
fi

echo "[setup] Verifying NVIDIA driver and CUDA toolkit..."
if ! nvidia-smi > /dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed"
  exit 1
fi
nvidia-smi 2>&1 | head -n 5 || true
nvcc --version 2>&1 | tail -n 3 || echo "WARN: nvcc not found"

# 1) PyTorch 2.9.1 + cu128 — has Blackwell sm_120 kernels and matches the
#    prebuilt flash-attn wheel we install below.
echo "[setup] Installing torch 2.9.1 + cu128..."
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) Core repo (editable). Core deps only — skip [dev]/[eval] (lighteval has
#    legacy PEP 508 syntax and we don't use it for training).
echo "[setup] Installing r1-v core deps..."
cd src/r1-v
pip install -e .
cd -

# 3) Logging + eval extras.
pip install wandb==0.19.1 tensorboardx qwen_vl_utils nltk rouge_score

# 4) FlashAttention 2.8.3 — use the prebuilt wheel matching our torch / cu /
#    Python / ABI. Avoids 10-15 min source compile and avoids the "guess wheel
#    URL then 404" failure mode when torch version has no published wheel.
echo "[setup] Installing flash-attn 2.8.3 (prebuilt wheel for torch 2.9)..."
pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# 5) DeepSpeed 0.16.5+ — has sm_120 kernels.
pip install --upgrade "deepspeed>=0.16.5" --no-build-isolation

# 6) bitsandbytes 0.45+ — sm_120 kernels.
pip install --upgrade "bitsandbytes>=0.45.0"

# NOTE: vLLM intentionally NOT installed. Latest vLLM pulls torch 2.10 which
# breaks flash-attn ABI. Training defaults to HF generation (RESEARCH_USE_VLLM=false).
# If you need vLLM, see docs/experiments/autodl_pro6000_setup.md Step 10.

echo ""
echo "[setup] Done. Verify Blackwell stack:"
echo "  python -c \"import torch, flash_attn, deepspeed; print(torch.__version__, torch.cuda.get_device_capability(0), flash_attn.__version__, deepspeed.__version__)\""
echo "  # Expected: 2.9.1+cu128 (12, 0) 2.8.3 <deepspeed-version>"
