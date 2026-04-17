#!/usr/bin/env bash
# ============================================================================
# Blackwell (RTX Pro 6000 / sm_120) compatible install script.
# Order matters: torch first, then anything that builds CUDA kernels against it.
# Tested assumption: pod ships with CUDA 12.8 runtime.
# ============================================================================
set -euo pipefail

echo "[setup] Verifying NVIDIA driver and CUDA toolkit..."
nvidia-smi | head -5 || { echo "ERROR: nvidia-smi failed"; exit 1; }
nvcc --version | tail -3 || echo "WARN: nvcc not found, flash-attn may fail to build"

# 1) PyTorch 2.7+ with CUDA 12.8 — has sm_120 kernels for Blackwell.
echo "[setup] Installing torch 2.7 + cu128..."
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Core repo (editable). setup.py now requires torch>=2.7, deepspeed>=0.16.5, vllm>=0.8.5.
echo "[setup] Installing r1-v..."
cd src/r1-v
pip install -e ".[dev]"
cd -

# 3) Logging and eval extras.
pip install wandb==0.19.1
pip install tensorboardx
pip install qwen_vl_utils
pip install nltk
pip install rouge_score

# 4) FlashAttention 2.7.4+ — first version with Blackwell kernels. Must build
#    against the torch installed above, so do it after torch.
echo "[setup] Installing flash-attn (this compiles, ~5-10 min on 16 vCPU)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 5) vLLM 0.8.5+ — Blackwell stability fixes. Reinstall forces wheel over
#    whatever setup.py resolved.
pip install --upgrade "vllm>=0.8.5"

# 6) DeepSpeed 0.16.5+ — has sm_120 kernels. --no-build-isolation avoids
#    rebuild-with-old-torch issues.
pip install --upgrade "deepspeed>=0.16.5" --no-build-isolation

# 7) bitsandbytes 0.45+ — sm_120 kernels.
pip install --upgrade "bitsandbytes>=0.45.0"

echo ""
echo "[setup] Done. Verify Blackwell is visible:"
echo "  python -c \"import torch; print(torch.cuda.get_device_capability(0))\""
echo "  # Expected: (12, 0) for RTX Pro 6000 Blackwell"
echo ""
echo "[setup] Then sanity check kernels compile/run:"
echo "  python -c \"import torch; x = torch.randn(4, 4, device='cuda', dtype=torch.bfloat16); print((x @ x.T).shape)\""
