# Dev Log — CS41 Enhanced T-GRPO for Video Temporal Reasoning

**Project:** CS41 Enhanced T-GRPO for Video Temporal Reasoning  
**Repository:** https://github.com/ChrisWu0318/USYD-Capstone-VideoRL  
**Base Code:** Video-R1 (https://github.com/tulerfeng/Video-R1)  
**Team:** Chris Wu + teammates  

---

## Branch Overview

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Original Video-R1 code (untouched) | Stable |
| `feature/enhanced-tgrpo` | All algorithm changes (pre-experiment) | Complete |
| `phase1/experiment-phase1` | Phase 1 experiment code + results + runtime fixes | Complete |

---

## Timeline

### 2026-03-17 — Code Review & Algorithm Design

- Completed full code review of `grpo.py` and `grpo_trainer.py`
- Identified 6 categories, 20+ optimization points (documented in `OPTIMIZATION_PROPOSALS.docx`)
- Prioritized into P0 (bugs), P1 (core algorithm), P2 (enhancements), P3 (cleanup)

### 2026-03-17 ~ 2026-03-18 — Code Implementation

All changes made on local Mac, pushed to `feature/enhanced-tgrpo` branch.

**Commit 1: P0 Bug Fixes**
- Fixed regression reward None fallthrough (`grpo.py` ~line 150)
- Removed duplicate prompt truncation (`grpo_trainer.py` ~line 452-454)
- Fixed temporal_rewards double gather (`grpo_trainer.py` ~line 748-749)

**Commit 2: Per-Sample Margin Reward + Equalized Generation Count**
- Changed `shuffled_num_generations = num_generations // 2` → `= num_generations`
- Replaced batch-level margin with per-sample margin calculation
- Each generation now gets its own boost based on individual temporal dependency

**Commit 3: Configurable Corruption Strength + Curriculum Learning**
- Added `--corruption_strength` parameter (float 0.0-1.0)
- Added `--curriculum_learning` flag (strength ramps from 0.3→1.0 over training)
- All corruption types now respect strength via `num_to_corrupt = int(num_frames * strength)`

**Commit 4: New Corruption Types + Weighted Selection + Logging**
- Added chunk_swap, speed, loop corruption types (total 6 types now)
- Replaced `random.choice()` with weighted `random.choices()`
- Weights: shuffle=0.25, reverse=0.20, mask=0.15, chunk_swap=0.20, speed=0.10, loop=0.10
- Added per-step destruction type logging in metrics

**Commit 5: Configurable Margin Scale and Reward Threshold**
- Added `--margin_scale` (default 0.5, was hardcoded)
- Added `--reward_threshold` (default 0.1, was hardcoded)

**Commit 6: Reward Function Fixes**
- rouge_scorer: moved from per-call instantiation to module-level singleton
- numerical comparison: removed comma/decimal pre-check that misidentified thousands separators
- format_reward: changed `re.fullmatch` to `re.match` with `^\s*...\s*$` for whitespace tolerance

**Commit 7: Code Cleanup**
- Replaced all `print()` with `logger.debug()` / `logger.warning()`
- Fixed typo: `lenth_list` → `length_list`
- Removed dummy generation for non-video data (was wasting compute)
- Fixed all_wrong/all_correct metrics: now uses accuracy reward column directly instead of hardcoded total reward thresholds

### 2026-03-22 — Phase 1 Experiment Run

#### Server Setup

**Platform:** RunPod  
**GPU:** 2× A100 SXM 80GB ($2.98/hr on-demand)  
**RAM:** 234 GB  
**Disk:** 1000 GB Volume Disk  
**Container:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel`  

#### Environment

```
Python: 3.11
PyTorch: 2.5.1+cu124
trl: 0.16.0
vllm: 0.7.2
flash_attn: 2.7.0.post2
deepspeed: latest
transformers: Video-R1 provided version
av: 12.0.0 (downgraded from latest to fix AVError attribute issue)
```

#### Data Preparation

- Downloaded Video-R1-data via `GIT_LFS_SKIP_SMUDGE=1 git clone` (skip large files)
- Pulled only needed subsets: CLEVRER, STAR, NeXT-QA, PerceptionTest (skipped LLaVA-Video-178K to save disk)
- Pulled image data: Chart, General, Knowledge, Math, OCR, Spatial
- Generated 500-sample video-only subset from the 4 available video sources
- Verified all 500 video files exist before training

**Data distribution in subset:**
```
STAR: ~149 samples
CLEVRER: ~123 samples  
NeXT-QA: ~113 samples
PerceptionTest: ~95 samples
```

#### Training Configuration

**DeepSpeed:** ZeRO-3 with CPU offload (`zero3_offload.json`)
- `offload_optimizer.device: "cpu"` — Required to fit 8B model on 2×A100
- `offload_param.device: "cpu"` — Further reduces GPU memory pressure
- Trade-off: ~30% slower per step due to CPU↔GPU communication

**Training script:** `src/scripts/run_enhanced_tgrpo_2xA100.sh`

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node="2" \
    src/open_r1/grpo.py \
    --model_name_or_path './Qwen2.5-VL-7B-COT-SFT' \
    --dataset_name "./Video-R1-data/Video-R1-500-subset.json" \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 8192 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --corruption_strength 1.0 \
    --curriculum_learning true \
    --margin_scale 0.5 \
    --reward_threshold 0.1 \
    --attn_implementation flash_attention_2 \
    --max_pixels 200704 \
    --max_steps 200 \
    --num_generations 4 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_steps 50 \
    --save_only_model true \
    --report_to wandb
```

**Key parameter choices and reasons:**

| Parameter | Original (4-5 GPU) | Ours (2×A100) | Why |
|-----------|-------------------|---------------|-----|
| `nproc_per_node` | 4 | 2 | Only 2 GPUs |
| `num_generations` | 8 | 4 | OOM with 8 (equalized shuffled doubles rollout) |
| `max_prompt_length` | 16384 | 8192 | Save memory |
| `max_pixels` | 401408 | 200704 | Save memory (half resolution per frame) |
| `gradient_accumulation_steps` | 1 | 2 | Compensate for fewer GPUs |
| `deepspeed` | zero3.json | zero3_offload.json | CPU offload for optimizer states |
| `max_steps` | 1200 | 200 | Phase 1 validation only |
| `dataset` | 260k full | 500 subset | Phase 1 validation only |

#### Runtime Bugs Found & Fixed

**Bug 1: `temporal_rewards` bracket error (step 1)**
```python
# WRONG — .append() returns None, then .mean() crashes
self._metrics["temporal_rewards"].append(temporal_rewards_list).mean().item()

# FIXED
self._metrics["temporal_rewards"].append(temporal_rewards_list.mean().item())
```
Root cause: bracket was in wrong position, `.mean().item()` needs to be inside `.append()`

**Bug 2: Loop corruption memory overlap (step 76)**
```python
# WRONG — PyTorch disallows self-referencing tensor assignment
shuffled_video[-loop_len:] = shuffled_video[:loop_len]

# FIXED — .clone() creates independent copy
shuffled_video[-loop_len:] = shuffled_video[:loop_len].clone()
```
Root cause: source and destination slices can overlap in memory, PyTorch raises RuntimeError

**Bug 3: OOM with num_generations=8 (step 1)**
- Equalized shuffled generation (8+8=16 rollouts) exceeded 160GB VRAM
- Solution: reduced num_generations to 4, added ZeRO-3 CPU offload
- Original zero3.json had `offload_optimizer.device: "none"`

**Bug 4: PyAV version incompatibility**
```
AttributeError: module 'av' has no attribute 'AVError'
```
- Fix: `pip install av==12.0.0`
- Root cause: newer PyAV removed `av.AVError`, torchvision's video reader still references it

#### Training Timeline

| Time | Event |
|------|-------|
| 11:09 | First run start |
| 11:10 | Crash — temporal_rewards bracket bug |
| 11:11 | Fix applied, restart |
| 11:13 | Crash — OOM with num_generations=8 |
| 11:14 | Reduced to 4, added zero3_offload.json, restart |
| 11:16 | Still OOM (zero3 without offload was cached) |
| 11:17 | Created zero3_offload.json with CPU offload, restart |
| 11:19 | Training started successfully |
| ~12:30 | Step 50 checkpoint saved |
| 13:15 | Crash at step 76 — loop corruption clone bug |
| 13:20 | Fix applied |
| 13:26 | Attempted resume from checkpoint-50, failed (save_only_model=true has no optimizer state) |
| 13:30 | Restarted from checkpoint-50 as new base model, 150 more steps |
| ~17:00 | Training complete (step 150 of second run) |
| 17:11 | MMVU eval started |
| 17:22 | MMVU eval complete |

**Total training time:** ~5 hours  
**Cost:** ~$15 USD  
**Speed:** ~82 seconds per step

#### Training Results (WandB)

**WandB Links:**
- Run 1 (steps 1-76, crashed): https://wandb.ai/wuguangbo464-the-university-of-sydney/huggingface/runs/nx3ncht2
- Run 2 (steps 1-150, from checkpoint-50): https://wandb.ai/wuguangbo464-the-university-of-sydney/huggingface/runs/lep2va20

**Key Metrics Summary:**

| Metric | Start | End (step 150) | Trend |
|--------|-------|----------------|-------|
| reward | 1.5 | 1.7-2.0 | ↑ Upward, stabilized after step 40 |
| accuracy_reward | 0.5 | 0.6-0.9 | ↑ High variance but improving |
| temporal_rewards | 0.0 | 0.05-0.15 | ✅ Consistently non-zero |
| kl | 0.0 | 0.008-0.01 | ↑ Healthy, well below 0.1 danger zone |
| format_reward | 1.0 | 1.0 | — Stable |
| completion_length | 210 | 200-260 | ~ Fluctuating |
| grad_norm | 2-3 | 2-4 | — Stable (one spike to 14 at step 35, recovered) |
| all_wrong | 0.5 | 0.0-0.25 | ↓ Decreasing |
| all_correct | 0.0 | 0.25-1.0 | ↑ Increasing |

**Observations:**
1. Reward curve shows clear upward trend in first 40 steps, then stabilizes around 1.8 — expected with only 500 samples (model sees all data multiple times)
2. temporal_rewards consistently 0.05-0.15, confirming per-sample margin mechanism is providing temporal signal
3. All 6 corruption types logged (shuffle, reverse, mask, chunk_swap, speed, loop)
4. KL divergence healthy throughout, never exceeded 0.01
5. No V-shaped completion_length pattern (expected with small dataset and few steps)

#### Evaluation Results

| Model | MMVU (mc) | Notes |
|-------|-----------|-------|
| Qwen2.5-VL-7B (COT) | 59.2 | Base model, no SFT/RL |
| **Qwen2.5-VL-7B-SFT** | **61.3** | **Our training starting point** |
| Video-R1-7B (paper) | 64.2 | Full 260k data, 1200 steps, 4-5 GPUs |
| **Ours (Phase 1)** | **59.0** | 500 samples, 200 steps, 2 GPUs |

**Gap analysis (-2.3 points vs SFT baseline):**

1. **Data coverage:** Used only 500 video samples from 4 subsets. Missing LLaVA-Video-178K (70% of original data) and all image data. MMVU covers chemistry, biology, engineering etc. — our training data has zero overlap with these domains
2. **Reduced parameters:** num_generations 8→4 (higher variance), max_pixels halved (lower visual quality), max_prompt_length halved
3. **Wrong benchmark for our changes:** MMVU tests domain knowledge comprehension. Our improvements target temporal reasoning. VSI-Bench and TempCompass are the right benchmarks to show improvement
4. **Training interrupted:** Lost steps 50-76 due to loop bug, had to restart from checkpoint-50

### 2026-03-23 — Model & Code Backup

- Model checkpoint-150 uploaded to HuggingFace (private repo)
- Code pushed to `phase1/experiment-phase1` branch
- Eval results committed to repository
- WandB logs preserved in cloud (permanent links)
- Pod stopped to save costs

---

## Files Changed (vs Original Video-R1)

### Core Algorithm

**`src/r1-v/src/open_r1/grpo.py`**
- Added CLI parameters: `corruption_strength`, `curriculum_learning`, `margin_scale`, `reward_threshold`
- Fixed regression reward None fallthrough (P0)
- Simplified numerical comparison (removed comma pre-check)
- Made rouge_scorer a module-level singleton
- Relaxed format_reward from `re.fullmatch` to `re.match` with whitespace tolerance

**`src/r1-v/src/open_r1/trainer/grpo_trainer.py`**
- Equalized shuffled_num_generations = num_generations (was // 2)
- Per-sample margin reward replacing batch-level binary
- 6 corruption types with weighted random selection
- Corruption strength controls num_to_corrupt
- Curriculum learning: strength ramps 0.3→1.0
- Loop corruption .clone() fix
- Temporal_rewards bracket fix
- Removed duplicate prompt truncation
- Fixed double gather on temporal_rewards
- Removed dummy generation for non-video data
- Fixed all_wrong/all_correct to use accuracy column
- Replaced print() with logging
- Fixed lenth_list typo
- Per-step destruction type logging

### Infrastructure

**`src/r1-v/local_scripts/zero3_offload.json`** (NEW)
- DeepSpeed ZeRO-3 config with CPU offload for optimizer and parameters
- Required for 2×A100 80GB (original zero3.json causes OOM)

**`src/scripts/run_enhanced_tgrpo_2xA100.sh`** (NEW)
- Training script configured for 2×A100
- All new CLI parameters included
- Points to 500-sample subset

**`src/eval_mmvu_only.py`** (NEW)
- Copy of eval_bench.py modified to only run MMVU benchmark

**`.gitignore`** (NEW)
- Excludes model weights, video files, training logs, checkpoints

---

## New CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--corruption_strength` | float | 1.0 | Fraction of frames to corrupt (0.0-1.0) |
| `--curriculum_learning` | bool | False | Ramp corruption strength 0.3→1.0 over training |
| `--margin_scale` | float | 0.5 | Scaling factor for per-sample marginal boost |
| `--reward_threshold` | float | 0.1 | Min accuracy score to qualify for temporal boost |

---

## Phase 2 Plan

| Item | Phase 1 (done) | Phase 2 (next) |
|------|----------------|----------------|
| Data | 500 video only | Full 260k (image + video) |
| Steps | 200 | 1200 |
| num_generations | 4 | 8 |
| max_pixels | 200704 | 401408 |
| GPU | 2×A100 (ZeRO-3 offload) | 4×A100 or 4×H20 (ZeRO-3 no offload) |
| Eval | MMVU only | VSI-Bench, TempCompass, MMVU, VideoMMMU |
| Baseline comparison | None | Original T-GRPO with P0 fixes |
| Ablation | None | curriculum on/off, corruption subsets, margin_scale values |
| Estimated cost | $15 | $80-120 |
| Estimated time | 5 hours | 30-50 hours |

**Key actions for Phase 2:**
1. Run baseline T-GRPO (original algorithm with only P0 bugfixes) on same data for fair comparison
2. Use full 260k dataset including LLaVA-Video-178K
3. Prioritize VSI-Bench and TempCompass evaluation — these test temporal reasoning directly
4. Run ablation: Enhanced T-GRPO without curriculum learning, to isolate contribution of each change
5. Consider increasing num_generations back to 8 (need 4×A100 to avoid OOM)
