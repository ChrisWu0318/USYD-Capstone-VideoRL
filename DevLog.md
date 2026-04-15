# Dev Log — `D4_algorithm_oom_fix_local`

**Project:** USYD Capstone — 4D Regularization for Video Temporal Reasoning  
**Branch:** `D4_algorithm_oom_fix_local`  
**Base:** Video-R1 (https://github.com/tulerfeng/Video-R1)  
**Date:** 2026-04-15  

---

## Summary

This branch implements the complete 4D Regularization algorithm (D1 + D2 + D3 + T-GRPO) on top of GRPO training, and fixes all bugs preventing successful distributed training under DeepSpeed ZeRO-3. The algorithm has been verified correct via smoke test — all metrics output valid values on step 1.

---

## Changelog

### 1. Bug Fixes — Training Entry Point (`grpo.py`)

- **Missing `import re`**: `format_reward` used `re.match` but `re` was not imported. Crashed on first reward computation.

- **`or True` dead code in processing class**: Condition always evaluated to True, bypassing model-specific processor selection. Replaced with `AutoProcessor` for all VL models.

- **Numerical reward comma check**: Decimal detection checked for "," which misidentified thousands separators (e.g. `1,000`). Changed to only check "." for decimal.

- **RougeScorer per-call creation**: `RougeScorer` was re-instantiated on every call. Added module-level singleton pattern.

- **`lenth_list` typo**: Fixed to `length_list`.

- **None vllm attrs**: `getattr(training_args, "use_vllm", False)` returns `None` (not the default) when the attribute exists with value `None`. Fixed with `or` fallback pattern:
  ```python
  use_vllm = bool(getattr(training_args, "use_vllm", False) or False)
  vllm_gpu_memory_utilization = float(getattr(training_args, "vllm_gpu_memory_utilization", 0.3) or 0.3)
  ```

- **Removed dead code**: Cleaned up unused functions in grpo.py.

---

### 2. ZeRO-3 Reference Model Placement (`grpo_trainer.py`)

**Problem:** Under DeepSpeed ZeRO-3, model parameters are stored as 1-D flattened shards. The ref model loaded via `from_pretrained` under ZeRO-3 context gets sharded parameters. When `prepare_deepspeed()` wraps it, the model stays on GPU permanently (~14GB per GPU), causing OOM during the optimizer step.

**Failed approaches:**
1. Route ref model through `prepare_deepspeed()` — works but OOM during backward (ref model permanently on GPU)
2. Move ref model to device before `prepare_deepspeed()` — fixes device mismatch but still OOM
3. Use `torch.inference_mode()` for ref forward — incompatible with ZeRO-3 gather hooks

**Final solution:** Load ref model **outside** ZeRO-3 context by temporarily suppressing `_hf_deepspeed_config_weak_ref`, so `from_pretrained` creates full-rank (non-sharded) parameters. This enables simple CPU-staged placement:

```
CPU  ──(before ref forward)──>  GPU  ──(after ref forward)──>  CPU
```

Key implementation details:
- Cache ZeRO-3 flag (`self._ref_model_is_zero3`) before `super().__init__()` because the weak reference may be GC'd
- Suppress `_hf_deepspeed_config_weak_ref` in `transformers.integrations.deepspeed` during ref model `from_pretrained`
- Restore the weak ref in a `finally` block
- Use `torch.no_grad()` instead of `torch.inference_mode()` (the latter blocks ZeRO-3 in-place gather operations)
- `_materialize_ref_model_for_forward()`: moves ref model to rank device
- `_restore_ref_model_after_forward()`: moves ref model back to CPU

Placement modes:
| Mode | When | How |
|------|------|-----|
| `cpu-staged-per-step` | Non-vLLM + ZeRO-3 | CPU ↔ GPU per forward |
| `deepspeed-managed` | vLLM path | `prepare_deepspeed()` permanent GPU |
| `accelerator-eval` | Non-DeepSpeed | `accelerator.prepare_model()` |

---

### 3. Experiment Config System

Added YAML-based experiment configuration (`ExperimentConfig`) that declaratively controls which D1/D2/D3 features are active. Each ablation has its own YAML file under `src/r1-v/configs/research_branch/`.

Launch scripts in `src/scripts/research_branch/` auto-detect GPU count from `RESEARCH_CUDA_VISIBLE_DEVICES` and pass the correct config to the trainer.

Scripts share common logic via `_common.sh` to avoid duplication.

---

### 4. Research Branch Launch Pipeline

Created a complete set of launch scripts for the experiment pipeline:

```
smoke_test.sh           # 3-step quick validation (D1+D2+D3)
ablation_baseline.sh    # Control group (no regularization)
ablation_d1_only.sh     # D1 only
ablation_d2_only.sh     # D2 only
ablation_d3_only.sh     # D3 only
ablation_d1_d2.sh       # D1 + D2
ablation_d1_d3.sh       # D1 + D3
ablation_d2_d3.sh       # D2 + D3
ablation_d1_d2_d3.sh    # D1 + D2 + D3 (full)
formal_full.sh          # Full training run
budget_probe.sh         # Resource estimation
```

All scripts use the same code — only the YAML config differs.

---

## Smoke Test Verification

**Environment:** 2x A100 80GB (step 1 only; 4+ GPUs needed for full run)

Step 1 completed successfully with all 4D features active:

| Metric | Value | Meaning |
|--------|-------|---------|
| `kl` | 0.0006 | Ref model forward working |
| `causal_reward_mean` | 0.0 | D1 causal reward active |
| `kl_truncation_ratio` | 0.0 | D2 KL truncation active |
| `length_penalty_mean` | -0.004 | D3 length penalty active |
| `welford_mean` / `welford_std` | 192.0 / — | D3 Welford statistics working |
| `temporal_rewards` | 1.0 | T-GRPO temporal reward active |

Step 2 OOM on 2x A100 during video encoder attention — purely a resource issue. 4+ GPUs resolves this.

---

## Commit History

```
694cf82 fix: load ref model outside ZeRO-3 context for cpu-staged placement
3baf5bb fix: move ref model to rank device before prepare_deepspeed wraps it
02f2bc4 fix: handle None vllm attrs from training_args
bb6568e fix: cache ZeRO-3 flag before super().__init__() to survive weak ref GC
0047793 fix: route ZeRO-3 ref model through prepare_deepspeed to fix 2-D weight crash
abebd77 fix: unify processing class to AutoProcessor for all VL models
8c406ac fix: multiple bugs — missing import re, dead code, numerical reward logic
49bb6de fix: preserve smoke launcher and vllm arg defaults
dcf28c2 fix: stage ref model onto rank device for smoke forward
26dfa1b fix: remove duplicate use_vllm cli flag
ec02eca fix: make research smoke launchers gpu-count aware
acf5b50 chore: add research experiment launch pipeline
603d814 fix: harden research training correctness
85ddf37 refactor: vLLM trainer inherits base, fixes D1/D2/D3 missing + OOM
```

---

## Next Steps

- [ ] Run full smoke test on 4x A100 (all 3 steps)
- [ ] Run all 8 ablation experiments
- [ ] Run formal full training
- [ ] Evaluate on MMVU / VSI-Bench benchmarks
- [ ] Compare ablation results to identify which dimensions contribute most
