# USYD Capstone: Video-R1 T-GRPO Upgrade DevLog

## Core Objective
Enhance the Video-R1 baseline T-GRPO training pipeline for video temporal reasoning by:
1) introducing multi-corruption temporal destruction (Task A), and
2) replacing the binary/threshold temporal reward with a marginal reward signal (Task B),
with clear reproducibility notes and evaluation plans.

---

## Project Timeline & TODO List

### Phase 1: Environment Setup & Baseline (Complete)
- [x] Configure AutoDL server environment (Python 3.11, Conda video-r1).
- [x] Download base weights: Qwen2.5-VL-7B-COT-SFT.
- [x] Initialize private GitHub repo and configure .gitignore.
- [x] **Infrastructure Troubleshooting & Optimization (New)**

---

### Phase 1.1: Infrastructure Lessons Learned (Troubleshooting)

#### 1.1.1 Compilation Bottlenecks & CPU Utilization
* **Issue**: Observed extreme compilation times (6+ hours) for FlashAttention-2.
* **Root Cause**: The compilation command was restricted to `--threads 4`, utilizing only 25% of the available 16 vCPU Xeon Gold 6430 resources.
* **Fix**: For future setups, match parallelism to hardware by setting `export MAX_JOBS=14` to fully leverage the 16-core architecture while leaving overhead for system I/O.

#### 1.1.2 C++ ABI & Undefined Symbol Errors
* **Issue**: Encountered `ImportError: undefined symbol` when importing `flash_attn` post-compilation.
* **Root Cause**: Version mismatch between the Python 3.11 build environment and the system's default Python 3.12/PyTorch 2.5 symbols, causing C++ ABI incompatibility in the generated `.so` files.
* **Fix**: Prioritize using pre-built `.whl` files (e.g., matching CUDA 12.4 + Torch 2.5) to avoid unstable source builds and ensure correct symbol linking.

#### 1.1.3 Network Acceleration
* **Issue**: Extremely slow GitHub clone/download speeds on internal server networks.
* **Fix**: Always initialize the terminal with `source /etc/network_turbo` to bypass bandwidth restrictions when fetching remote repositories or large model weights.

---

### Phase 2: Core Algorithm Modification — grpo_trainer.py (In Progress)

#### 2.1 Code Navigation & Baseline Understanding
- [x] Located temporal corruption / shuffled video pipeline around ~line 330.
- [x] Located temporal reward logic (original binary/threshold gating) around ~line 530.
- [x] Confirmed the trainer computes reward for normal video vs corrupted video to enforce temporal reasoning behavior.

#### 2.2 Task A — Multi-Corruption Temporal Destruction (Done, minimal runnable)
- [x] Implemented multi-corruption selection for temporal destruction:
  - shuffle: random frame permutation
  - reverse: reverse time order
  - mask: frame masking (baseline implementation; may revise to temporal-drop/chunk-mask later)
- [x] Fixed a critical bug: removed unintended baseline shuffle overwrite so that the selected corruption is actually used.
- [x] Ensured corrupted-video prompts are constructed correctly via processing_class(..., videos=shuffled_video_inputs).

#### 2.3 Task B — Marginal Reward (Done, minimal runnable; not per-sample yet)
- [x] Replaced binary/threshold temporal reward with a margin-based reward boost computed from the performance gap between normal and corrupted videos.
- [x] Kept reward change minimal to preserve training stability and enable end-to-end pipeline execution first.

---

### Phase 3: Smoke Test, Training, and Evaluation (Next)
- [ ] Run syntax + pipeline smoke tests.
- [ ] Full RL training on multi-GPU node(s) with DeepSpeed 0.18.6 and FlashAttention-2.
- [ ] Ablation study plan:
  - baseline (single shuffle + binary reward)
  - Task A only
  - Task B only
  - Task A + Task B
- [ ] Produce final deliverables: report, reproducible commands, and recorded knowledge transfer session.

---

## Current Branch & Commits
- Working branch: feat/marginal-reward-upgrade
- Next commit(s) plan:
  1) “Task A pipeline fix + multi-corruption”
  2) “Task B marginal reward (batch-level)”
  3) “Task B per-sample margin + logging by corruption type”