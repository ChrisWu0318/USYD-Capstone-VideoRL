# Video-R1: 4D Regularization for Video Temporal Reasoning
### USYD IT Capstone Project (Master of Computer Science)

## Project Overview

This repository implements **4D Regularization** on top of the [Video-R1](https://github.com/tulerfeng/Video-R1) framework, enhancing video temporal reasoning through four complementary regularization dimensions applied during GRPO (Group Relative Policy Optimization) training.

**Base Model:** Qwen2.5-VL-7B-COT-SFT (Chain-of-Thought supervised fine-tuned)

### 4D Regularization Components

| Dimension | Name | Description |
|-----------|------|-------------|
| **D1** | Causal Reward | Temporal frame masking to evaluate causal understanding |
| **D2** | KL Truncation | Token-level KL divergence truncation against reference model |
| **D3** | Length Penalty | Task-aware length penalty using Welford online statistics |
| **T-GRPO** | Temporal GRPO | Frame-shuffled contrastive generation for temporal grounding |

## Repository Structure

```
src/
  r1-v/                         # Core training framework
    src/open_r1/
      grpo.py                   # Training entry point
      trainer/grpo_trainer.py   # GRPO trainer with 4D regularization
    configs/research_branch/    # Experiment YAML configs (per-ablation)
  scripts/research_branch/      # Launch scripts (smoke, ablation, formal)
  qwen-vl-utils/                # Qwen VL utilities (local fork)
  eval_bench.py                 # Evaluation benchmark script
transformers-main.zip           # Modified transformers (local fork)
```

## Experiment Pipeline

All experiments use the same codebase — only the launch script (and its YAML config) changes.

```
smoke_test.sh  -->  ablation_*.sh  -->  formal_full.sh
  (3 steps)        (8 ablations)        (full training)
```

### Available Scripts (`src/scripts/research_branch/`)

| Script | Config | What it runs |
|--------|--------|-------------|
| `smoke_test.sh` | `smoke_d1_d2_d3.yaml` | 3-step quick validation |
| `ablation_baseline.sh` | `ablation_baseline.yaml` | No D1/D2/D3 (control) |
| `ablation_d1_only.sh` | `ablation_d1.yaml` | D1 only |
| `ablation_d2_only.sh` | `ablation_d2.yaml` | D2 only |
| `ablation_d3_only.sh` | `ablation_d3.yaml` | D3 only |
| `ablation_d1_d2.sh` | `ablation_d1_d2.yaml` | D1 + D2 |
| `ablation_d1_d3.sh` | `ablation_d1_d3.yaml` | D1 + D3 |
| `ablation_d2_d3.sh` | `ablation_d2_d3.yaml` | D2 + D3 |
| `ablation_d1_d2_d3.sh` | `ablation_d1_d2_d3.yaml` | D1 + D2 + D3 (full) |
| `formal_full.sh` | `formal_d1_d2_d3.yaml` | Full training run |

### Running an Experiment

```bash
RESEARCH_CUDA_VISIBLE_DEVICES=0,1,2,3 \
RESEARCH_MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/path/to/dataset.json \
bash src/scripts/research_branch/smoke_test.sh
```

GPU count is auto-detected from `RESEARCH_CUDA_VISIBLE_DEVICES`.

## Technical Highlights

- **DeepSpeed ZeRO-3** for multi-GPU distributed training
- **HF generate** path (no vLLM / colocate) for simplicity and stability
- **CPU-staged reference model**: ref model loaded outside ZeRO-3 context with full-rank parameters, staged CPU -> GPU only during forward pass to save ~14GB per GPU
- **ExperimentConfig YAML system**: each ablation is a declarative config toggling D1/D2/D3 flags

## Hardware Requirements

| Experiment | Minimum GPUs | Recommended |
|-----------|-------------|-------------|
| Smoke test | 4x A100 80GB / 4x RTX Pro 6000 96GB | same |
| Ablation | 4x A100 80GB / 4x RTX Pro 6000 | 4–8x A100 80GB |
| Formal training | 8x A100 80GB / 4x RTX Pro 6000 | 8x A100 80GB |

> **Pro 6000 note:** 2-GPU runs OOM when the ref model materializes on top of
> step-1 activations (~67 GB + 14 GB ref > 96 GB). ZeRO-3 with 4 shards is the
> confirmed path; vLLM colocate may further reduce rollout memory but sm_120
> support is still being verified — see `docs/experiments/pro6000_archive.md`.

## Getting Started

### A100 path
1. Checkout branch `D4_algorithm_oom_fix_local`
2. Install deps (see setup guide)
3. Download model + dataset
4. Run `smoke_test.sh` to validate
5. Proceed to ablation / formal training

### RTX Pro 6000 (Blackwell, sm_120) path
1. Checkout branch `blackwell_rtx6000_compat`
2. Follow `docs/experiments/autodl_pro6000_setup.md` (14-step guide) — pins
   torch 2.9.1+cu128, flash-attn 2.8.3 prebuilt wheel, deepspeed ≥0.16.5
3. See `docs/experiments/pro6000_archive.md` for full rationale, pitfalls,
   dataset extraction commands, and verification scripts

## Status & Next Steps (2026-04-18, `blackwell_rtx6000_compat`)

**Validated on 2× RTX Pro 6000 (Blackwell, sm_120, CUDA 12.8):**
- ✅ Full dependency stack installs clean (torch 2.9.1, flash-attn 2.8.3,
  deepspeed 0.18.9, bitsandbytes ≥0.45, transformers 4.51.3)
- ✅ Model loads (Qwen2.5-VL-7B-COT-SFT, 4 shards)
- ✅ Dataset loader + CLEVRER video path resolution work
- ✅ **Smoke step 1 metrics match parent branch within tolerance:**
  `kl=0.000539`, `welford_mean=192.0`, `length_penalty_mean=-0.004`,
  `causal_reward_mean=0.0` — algorithm is numerically consistent on Blackwell
- ⚠️ **2-GPU OOMs at step 2** (ref model materialize on top of ~67 GB
  step-1 activations); need 4-GPU ZeRO-3 to shard model state/grad/optim

**Next:**
1. Re-run smoke + `budget_probe` on 4× Pro 6000 (ZeRO-3 should shrink
   per-GPU model state from ~28 GB → ~7 GB and halve ref materialize cost)
2. Investigate vLLM 0.8.5+ Blackwell (sm_120) compatibility in isolation
   — if rollout works, switch to colocate mode for 3-5× faster generation
3. Run the 8 ablations + formal_full once either path is stable

## Tracking Docs

| Doc | Purpose |
|-----|---------|
| `docs/experiments/autodl_pro6000_setup.md` | Step-by-step AutoDL reproduction (14 steps) |
| `docs/experiments/blackwell_runbook.md` | Dependency delta matrix + risk table |
| `docs/experiments/pro6000_archive.md` | **Full archive** — what we did, why, and all pitfalls |

## Acknowledgement

This project is a derivative work of **[Video-R1](https://github.com/tulerfeng/Video-R1)**. We acknowledge the original authors for their foundational contributions.
