# USYD Capstone: Video-R1 T-GRPO Upgrade DevLog

## Core Objective

Enhance the Video-R1 baseline T-GRPO training pipeline for video temporal reasoning by:

1. introducing multi-corruption temporal destruction (Task A), and
2. replacing the binary/threshold temporal reward with a marginal reward signal (Task B),
   with clear reproducibility notes and evaluation plans.

---

## Project Timeline & TODO List

### Phase 1: Environment Setup & Baseline (Complete)

* [x] Configure AutoDL server environment (Python 3.11, Conda `video-r1`).
* [x] Download base weights: `Qwen2.5-VL-7B-COT-SFT`.
* [x] Initialize private GitHub repo and configure `.gitignore`.
* [x] **Infrastructure troubleshooting and dependency stabilization**

---

### Phase 1.1: Infrastructure Lessons Learned (Troubleshooting)

#### 1.1.1 FlashAttention Build Instability and Long Compile Time

* **Issue**: FlashAttention-2 source installation took several hours and was unstable across repeated attempts.
* **Observation**: Increasing compilation parallelism (`MAX_JOBS=14`) did speed up worker scheduling, but aggressive parallel build settings caused the CUDA compilation process to be killed during kernel compilation.
* **Root Cause**: Source compilation of FlashAttention is expensive and memory-intensive. On the AutoDL node, high parallelism improved CPU usage but exceeded practical memory headroom, leading to `Killed` errors during `nvcc` execution.
* **Fix**:

  * Treat source compilation as a fallback option only.
  * Prefer a **prebuilt wheel** matched to the active environment.
  * If source build is unavoidable, reduce compile parallelism (e.g. `MAX_JOBS=2~4`) instead of maximizing CPU workers.

#### 1.1.2 C++ ABI / Binary Compatibility Mismatch

* **Issue**: Importing `flash_attn` failed with an `undefined symbol` error after installation.
* **Root Cause**: The installed FlashAttention binary was not aligned with the active PyTorch/CUDA ABI combination in the `video-r1` environment. The environment used:

  * Python 3.11
  * PyTorch `2.5.1+cu124`
  * `GLIBCXX_USE_CXX11_ABI = False`

  This meant that an incompatible prebuilt artifact or stale compiled `.so` file had been loaded.
* **Fix**:

  * Verified the active PyTorch ABI using:

    * `torch.__version__`
    * `torch.version.cuda`
    * `torch._C._GLIBCXX_USE_CXX11_ABI`
  * Switched to the **official FlashAttention wheel** matching:

    * CUDA 12
    * Torch 2.5
    * CPython 3.11
    * `cxx11abiFALSE`
  * Installed the wheel directly with `pip install --no-deps <wheel_path>` to avoid rebuilding and eliminate ABI mismatch.

#### 1.1.3 Correct Environment Isolation

* **Issue**: Package installation risked landing in the wrong Conda environment during repeated debugging.
* **Root Cause**: AutoDL shells can default to `base`, while project runtime depended on the `video-r1` Conda environment.
* **Fix**:

  * Explicitly activated `conda activate video-r1` before all dependency operations.
  * Verified the active interpreter path using:

    * `which python`
    * `python -c "import sys; print(sys.executable)"`
  * Performed all FlashAttention / DeepSpeed / Torch checks only inside `video-r1`.

#### 1.1.4 Network and Artifact Download Strategy

* **Issue**: Downloading large binary dependencies directly from GitHub Releases was extremely slow on the training node.
* **Root Cause**: PyPI mirror acceleration does not speed up GitHub Release asset downloads.
* **Fix**:

  * Used local/offline artifact transfer as a more reliable path for large wheels.
  * Downloaded the required FlashAttention wheel externally and uploaded it to the AutoDL workspace.
  * Installed from the local file path to avoid repeated failed or throttled remote downloads.

#### 1.1.5 Final Dependency Resolution Outcome

* **Resolved Strategy**:

  * **Do not rely on repeated FlashAttention source compilation**
  * **Use a version-matched prebuilt wheel whenever possible**
  * Keep source build only as a backup path, with conservative parallelism and strict environment checks

---

### Phase 2: Core Algorithm Modification — `grpo_trainer.py` (In Progress)

#### 2.1 Code Navigation & Baseline Understanding

* [x] Located temporal corruption / shuffled video pipeline around ~line 330.
* [x] Located temporal reward logic (original binary/threshold gating) around ~line 530.
* [x] Confirmed the trainer computes reward for normal video vs corrupted video to enforce temporal reasoning behavior.

#### 2.2 Task A — Multi-Corruption Temporal Destruction (Done, minimal runnable)

* [x] Implemented multi-corruption selection for temporal destruction:

  * shuffle: random frame permutation
  * reverse: reverse time order
  * mask: frame masking (baseline implementation; may revise to temporal-drop/chunk-mask later)
* [x] Fixed a critical bug: removed unintended baseline shuffle overwrite so that the selected corruption is actually used.
* [x] Ensured corrupted-video prompts are constructed correctly via `processing_class(..., videos=shuffled_video_inputs)`.

#### 2.3 Task B — Marginal Reward (Done; per-sample margin implemented)

* [x] Replaced binary/threshold temporal reward with a margin-based reward boost computed from the performance gap between normal and corrupted videos.
* [x] Kept reward change minimal to preserve training stability and enable end-to-end pipeline execution first.
* [x] Upgraded from batch-level scalar margin to **per-sample marginal reward**: each sample's margin is computed independently using its own G normal-video generations vs its shuffled_G corrupted-video generations.
* [x] Added **corruption type logging** to training metrics (`corruption/shuffle`, `corruption/reverse`, `corruption/mask`), enabling per-step tracking of which destruction method was applied.

---

### Phase 3: Smoke Test, Training, and Evaluation (Next)

* [x] Run syntax + dependency smoke tests (syntax verified: `grpo_trainer.py` passes `ast.parse`).
* [ ] Validate FlashAttention / DeepSpeed runtime integration on the target multi-GPU node.
* [ ] Full RL training with DeepSpeed and stable environment lockfile / package notes.
* [ ] Ablation study plan:

  * baseline (single shuffle + binary reward)
  * Task A only
  * Task B only
  * Task A + Task B
* [ ] Produce final deliverables: report, reproducible commands, and recorded knowledge transfer session.

---

## Reproducibility Notes

### Confirmed Runtime Environment

* Python: 3.11
* PyTorch: `2.5.1+cu124`
* CUDA runtime in PyTorch: 12.4
* GPU: NVIDIA GeForce RTX 4090
* FlashAttention installation strategy: **prebuilt wheel matched to Torch/CUDA ABI**
* Conda environment: `video-r1`

### Practical Setup Lessons

* Always verify the active Conda environment before installing performance-critical libraries.
* For FlashAttention, **binary compatibility matters more than raw compile speed**.
* High compile parallelism can reduce wall-clock time but may fail under limited memory.
* Local wheel installation is often more reliable than repeated source rebuilds on cloud GPU nodes.

---

## Current Branch & Commits

* Working branch: `feat/marginal-reward-upgrade`

### Next commit(s) plan

1. `Task A pipeline fix + multi-corruption` ✓
2. `Task B marginal reward (batch-level)` ✓
3. `Environment stabilization notes + FlashAttention reproducibility` ✓
4. `Task B per-sample margin + logging by corruption type` ✓
