# Blackwell (RTX Pro 6000) Compatibility Runbook

Branch: `blackwell_rtx6000_compat` (derived from `D4_algorithm_oom_fix_local`)

All 4D algorithm logic and OOM fixes from the parent branch are preserved.
Only the dependency stack and install order are changed.

## Dependency deltas vs parent branch

| Package | Parent | This branch | Why |
|---|---|---|---|
| torch | >=2.5.1 | >=2.7.0 (cu128) | sm_120 kernels |
| deepspeed | ==0.15.4 | >=0.16.5 | Blackwell CUDA kernels |
| vllm | >=0.8.0 | >=0.8.5 | Blackwell stability |
| flash-attn | unpinned | ==2.7.4.post1 | First Blackwell-capable release |
| bitsandbytes | >=0.43.0 | >=0.45.0 | sm_120 kernels |
| liger_kernel | ==0.5.2 | >=0.5.6 | Compat with torch 2.7 |
| trl | ==0.16.0 | ==0.16.0 (unchanged) | GRPOTrainer modified; must not upgrade |

## Pre-flight on a fresh pod

1. Verify CUDA toolkit and driver:
   ```bash
   nvidia-smi                  # Driver must be >= 570 for CUDA 12.8
   nvcc --version              # Toolkit should be 12.8.x
   ```
2. Run `bash setup.sh` (takes 15-25 min, flash-attn build dominates).
3. Sanity check the card is detected as sm_120:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_capability(0))"
   # Expected: (12, 0)
   ```
4. Sanity check bf16 matmul works:
   ```bash
   python -c "import torch; x=torch.randn(4,4,device='cuda',dtype=torch.bfloat16); print((x@x.T).shape)"
   ```
5. Sanity check flash-attn imports:
   ```bash
   python -c "import flash_attn; print(flash_attn.__version__)"
   ```
6. Sanity check deepspeed ZeRO-3 config loads:
   ```bash
   python -c "from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled; print('ok')"
   ```

## Known risks (most likely failure points)

### Risk 1: `_hf_deepspeed_config_weak_ref` internal API removed
The ref-model-outside-ZeRO-3 fix (commit `694cf82`) uses
`transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref` to
temporarily suppress ZeRO-3 context during `from_pretrained`. This is a
private attribute.

**Mitigation:** If upgrading transformers breaks this attribute, the smoke
test will crash with `AttributeError` at ref model load. Fallback: pin
transformers to the version shipped with `trl==0.16.0`
(`transformers==4.49.0` or the version trl 0.16.0 requires).

### Risk 2: `trl==0.16.0` incompatible with torch 2.7
`trl==0.16.0` was released around Feb 2025 and may require an older
transformers. If pip resolves a transformers that lacks torch 2.7 support,
either:
- Install `transformers==4.49.0` explicitly after the main `pip install -e`, or
- Unpin trl (last resort; will require verifying GRPOTrainer signature
  against our modifications).

### Risk 3: vLLM on Blackwell is still rough
vLLM 0.8.5 claims Blackwell support but kernel registration for sm_120 has
been flaky. If `RESEARCH_USE_VLLM=true` crashes, fall back to HF generation
(default `RESEARCH_USE_VLLM=false`).

### Risk 4: flash-attn build fails
If `pip install flash-attn==2.7.4.post1 --no-build-isolation` fails:
- Check `nvcc --version` matches torch's CUDA (`python -c "import torch; print(torch.version.cuda)"`)
- Set `MAX_JOBS=4` to limit parallel build (OOM on small-RAM pods)
- Try prebuilt wheel from https://github.com/Dao-AILab/flash-attention/releases

## What to test first on Blackwell pod

Same smoke test as the parent branch, just to verify the algorithm still
produces the same metrics under the new stack:

```bash
RESEARCH_CUDA_VISIBLE_DEVICES=0,1 \
RESEARCH_MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/path/to/smoke_test_10.json \
bash src/scripts/research_branch/smoke_test.sh
```

**Required success criteria (must match parent branch within tolerance):**
- kl ~= 0.0006 (order of magnitude)
- causal_reward_mean numerically sane
- kl_truncation_ratio >= 0
- length_penalty / welford stats populated
- All 3 smoke steps complete without OOM (96GB per card should handle step 2
  easily, unlike 2×A100 80GB which OOM'd on video encoder attention)

If any metric drifts substantially from the parent branch, the dep upgrade
introduced silent numerical changes and needs investigation before running
ablation.

## GPU count guidance for Pro 6000

- **1× Pro 6000 (96GB)**: may work for smoke test, but no room for ZeRO-3
  shard benefit; ref model CPU-staging is required
- **2× Pro 6000**: recommended minimum; matches 2× A100 baseline and 96GB
  gives breathing room for video encoder activations
- **4× Pro 6000**: ideal; matches recommended A100 setup

## Fallback

If dep upgrades cause unresolvable regressions: `git checkout
D4_algorithm_oom_fix_local` and use 4× A100 80GB on RunPod as planned.
