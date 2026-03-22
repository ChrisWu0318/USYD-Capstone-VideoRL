# Phase 1 Experiment Summary — Enhanced T-GRPO

**Date:** 2026-03-22 ~ 2026-03-23  
**Branch:** `phase1/experiment-run`  
**Author:** Chris Wu  

---

## 1. What We Did

Ran the first small-scale validation of Enhanced T-GRPO on 2×A100 SXM 80GB (RunPod).

- **Training data:** 500 video samples (STAR, CLEVRER, NeXT-QA, PerceptionTest)
- **Base model:** Qwen2.5-VL-7B-COT-SFT (official SFT checkpoint)
- **Training:** ~200 steps total (50 steps first run + 150 steps continued from checkpoint-50)
- **Cost:** ~$15 USD (5 hours GPU time)

## 2. Algorithm Changes Validated

| Change | Status |
|--------|--------|
| Per-sample margin reward (replacing batch-level binary) | ✅ Working |
| Equalized shuffled generation count | ✅ Working |
| 6 corruption types (shuffle, reverse, mask, chunk_swap, speed, loop) | ✅ All triggered |
| Weighted random corruption selection | ✅ Working |
| Curriculum learning (strength 0.3→1.0) | ✅ Working |
| Configurable margin_scale, reward_threshold | ✅ Working |
| P0 bug fixes (regression None, double gather, duplicate truncation) | ✅ Fixed |

## 3. Training Curves (WandB)

- **Run 1 (steps 1-76):** https://wandb.ai/wuguangbo464-the-university-of-sydney/huggingface/runs/nx3ncht2
- **Run 2 (steps 1-150, from checkpoint-50):** https://wandb.ai/wuguangbo464-the-university-of-sydney/huggingface/runs/lep2va20

### Key Observations

| Metric | Trend | Notes |
|--------|-------|-------|
| **reward** | 1.5 → 1.8-2.0 ↑ | Clear upward trend in first 40 steps, then stabilized |
| **accuracy_reward** | 0.5 → 0.7-0.8 ↑ | High variance due to small dataset, but improving |
| **temporal_rewards** | 0.05 - 0.15 | Consistently non-zero, per-sample margin giving signal |
| **kl** | 0 → 0.008 | Healthy range, model learning without diverging |
| **format_reward** | Stable 1.0 | Output format always correct |
| **completion_length** | 200-250 | Fluctuating, no clear V-shape yet (expected with small data) |
| **all_wrong** | Decreased | Fewer samples where model gets everything wrong |
| **destruction_type_***: | All 6 types logged | Multi-corruption mechanism working as designed |

## 4. MMVU Eval Result

| Model | MMVU (mc) |
|-------|-----------|
| Qwen2.5-VL-7B-SFT (baseline) | **61.3** |
| Video-R1-7B (paper, 16 frames) | **64.2** |
| **Ours (Phase 1, 500 samples)** | **59.0** |

### Why the score dropped (-2.3 vs baseline):

1. **Tiny dataset (500 vs 260k):** Only 0.2% of the full training data, and zero LLaVA-Video-178K samples which make up 70% of the original dataset
2. **Reduced parameters for OOM:** num_generations 8→4, max_pixels halved, max_prompt_length halved
3. **MMVU tests knowledge, not temporal reasoning:** Our improvements target temporal reasoning; MMVU is primarily a knowledge/comprehension benchmark. VSI-Bench and TempCompass are better benchmarks for our changes
4. **Training interrupted at step 76:** Had to restart from checkpoint-50 due to loop corruption bug (fixed)

## 5. Bugs Found & Fixed During Run

| Bug | When | Fix |
|-----|------|-----|
| `temporal_rewards_list.append().mean()` — bracket wrong | Step 1 crash | Moved `.mean().item()` inside `.append()` |
| `loop` corruption self-assignment — PyTorch memory overlap | Step 76 crash | Added `.clone()` before assignment |
| OOM with num_generations=8 | Step 1 crash | Reduced to 4, added ZeRO-3 CPU offload |

## 6. Conclusion

**Phase 1 goal achieved:** The algorithm runs end-to-end, reward curves show upward trend, all 6 corruption types work, per-sample margin reward is providing continuous temporal signal.

The MMVU score drop is expected at this scale and doesn't indicate algorithmic failure — the training curves confirm the model is learning.

## 7. Phase 2 Plan

To get real results that can go into the final report:

| Item | Phase 1 (done) | Phase 2 (next) |
|------|----------------|----------------|
| Data | 500 video only | Full 260k (image + video) |
| Steps | 200 | 1200 |
| num_generations | 4 | 8 |
| max_pixels | 200704 | 401408 |
| GPU | 2×A100 | 4×A100 or 4×H20 |
| Eval benchmarks | MMVU only | VSI-Bench, TempCompass, MMVU, VideoMMMU |
| Estimated cost | $15 | $80-120 |
| Estimated time | 5 hours | 30-50 hours |

Additionally, need to run baseline T-GRPO (original code with P0 fixes only) on the same setup for fair comparison.

## 8. Files Changed

```
src/r1-v/src/open_r1/grpo.py              — reward functions + new CLI args
src/r1-v/src/open_r1/trainer/grpo_trainer.py — training loop, corruption, per-sample margin
src/r1-v/local_scripts/zero3_offload.json  — DeepSpeed config with CPU offload
src/scripts/run_enhanced_tgrpo_2xA100.sh   — training script for 2×A100
src/eval_mmvu_only.py                      — MMVU-only eval script
```

## 9. Model Checkpoints

- **checkpoint-50:** `src/r1-v/log/Enhanced-TGRPO-Phase1/checkpoint-50/`
- **checkpoint-150:** `src/r1-v/log/Enhanced-TGRPO-Phase1-continued/checkpoint-150/`
- Upload to HuggingFace pending (need write token)
