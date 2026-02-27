# USYD Capstone: Video-R1 T-GRPO Upgrade DevLog

## Core Objective
Enhance the Video-R1 baseline T-GRPO training pipeline for **video temporal reasoning** by:
1) introducing **multi-corruption temporal destruction** (Task A), and  
2) replacing the **binary/threshold temporal reward** with a **marginal reward signal** (Task B),  
with clear reproducibility notes and evaluation plans.

---

## Project Timeline & TODO List

### Phase 1: Environment Setup & Baseline (Almost Complete)
- [x] Configure AutoDL server environment (Python 3.11, Conda `video-r1`).
- [x] Download base weights: Qwen2.5-VL-7B-COT-SFT.
- [x] Initialize private GitHub repo and configure `.gitignore`.
- [ ] (Optional but recommended) Run a baseline smoke-train/eval to obtain reference logs/metrics.

---

### Phase 2: Core Algorithm Modification — `grpo_trainer.py` (In Progress)

#### 2.1 Code Navigation & Baseline Understanding
- [x] Located **temporal corruption / shuffled video** pipeline around ~line 330.
- [x] Located **temporal reward logic** (original binary/threshold gating) around ~line 530.
- [x] Confirmed the trainer computes reward for **normal video vs corrupted video** to enforce temporal reasoning behavior.

#### 2.2 Task A — Multi-Corruption Temporal Destruction (Done, minimal runnable)
- [x] Implemented **multi-corruption selection** for temporal destruction:
  - `shuffle`: random frame permutation  
  - `reverse`: reverse time order  
  - `mask`: frame masking (baseline implementation; may revise to temporal-drop/chunk-mask later)
- [x] **Fixed a critical bug**: removed unintended baseline shuffle overwrite so that the selected corruption is actually used.
- [x] Ensured corrupted-video prompts are constructed correctly via `processing_class(..., videos=shuffled_video_inputs)`.

**Notes / Known limitations**
- Current `mask` is a strong visual corruption (risk of shortcut learning). Planned improvement: temporal dropout / chunk mask / jitter.

#### 2.3 Task B — Marginal Reward (Done, minimal runnable; not per-sample yet)
- [x] Replaced binary/threshold temporal reward with a **margin-based reward boost** computed from the performance gap between normal and corrupted videos.
- [x] Kept reward change minimal to preserve training stability and enable end-to-end pipeline execution first.

**Notes / Known limitations**
- Current marginal reward uses **batch-level mean margin** (not per-sample). Planned next step: per-sample margin + analysis by corruption type.

---

### Phase 3: Smoke Test, Training, and Evaluation (Next)
- [ ] Run **syntax + pipeline smoke tests**:
  - `python -m py_compile src/r1-v/src/open_r1/trainer/grpo_trainer.py`
  - One short RL run (few steps) to confirm no runtime errors in temporal+video path.
- [ ] Full RL training on multi-GPU node(s).
- [ ] Evaluation on provided benchmark scripts / datasets.
- [ ] Ablation study plan:
  - baseline (single shuffle + binary reward)
  - Task A only
  - Task B only
  - Task A + Task B
- [ ] Produce final deliverables:
  - report, reproducible commands, and recorded knowledge transfer session.

---

## Current Branch & Commits
- Working branch: `feat/marginal-reward-upgrade`
- Next commit(s) plan:
  1) “Task A pipeline fix + multi-corruption”
  2) “Task B marginal reward (batch-level)”
  3) (Later) “Task B per-sample margin + logging by corruption type”

---

## Quick Verification Checklist
- [ ] No `NameError` for `shuffled_prompt_inputs/shuffled_prompt_ids/shuffled_prompt_mask`
- [ ] Selected corruption type is not overwritten by baseline shuffle
- [ ] Training step runs for at least N steps without crashing