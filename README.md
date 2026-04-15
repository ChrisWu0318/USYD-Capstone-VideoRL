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
| Smoke test | 4x A100 80GB | 4x A100 80GB |
| Ablation | 4x A100 80GB | 4-8x A100 80GB |
| Formal training | 8x A100 80GB | 8x A100 80GB |

## Getting Started

See the [Smoke Test Guide](https://github.com/ChrisWu0318/USYD-Capstone-VideoRL/wiki) or ask your team lead for the one-click setup document.

1. Clone repo and checkout branch `D4_algorithm_oom_fix_local`
2. Install dependencies (conda + pip, see setup guide)
3. Download model weights and dataset
4. Run `smoke_test.sh` to validate
5. Proceed to ablation / formal training

## Acknowledgement

This project is a derivative work of **[Video-R1](https://github.com/tulerfeng/Video-R1)**. We acknowledge the original authors for their foundational contributions.
