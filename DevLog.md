# Dev Log — `feature/enhanced-tgrpo`

**Project:** CS41 Enhanced T-GRPO for Video Temporal Reasoning  
**Branch:** `feature/enhanced-tgrpo`  
**Base:** Video-R1 (https://github.com/tulerfeng/Video-R1)  
**Date:** 2026-03-17  

---

## Summary

This branch implements the Enhanced T-GRPO algorithm, upgrading the original binary temporal contrast signal to a multi-corruption, margin-based reward system. Changes span two core files: `grpo.py` (reward functions + training args) and `grpo_trainer.py` (training loop + temporal reward logic).

---

## Commit Log

### 1. P0 Bug Fixes

**Files:** `grpo.py`, `grpo_trainer.py`

- **Regression reward None fallthrough** (`grpo.py`):  
  The `regression` question type checked `if gt_number is None` and assigned `reward = 0.0`, but lacked an `else` branch. Execution continued to `rel_diff = abs(out_number - gt_number)`, causing a `TypeError` when either value was `None`. Fixed by adding `else` to guard the calculation.

- **Duplicate prompt truncation** (`grpo_trainer.py`):  
  `max_prompt_length` truncation was applied twice — first on `prompt_inputs["input_ids"]`, then again on `prompt_ids` (which references the same tensor). Removed the redundant second truncation.

- **Double gather on temporal_rewards** (`grpo_trainer.py`):  
  `gather_for_metrics` was called twice — once to create `temporal_rewards_list`, and again when appending to metrics. In multi-GPU settings this duplicates data and corrupts the reported mean. Fixed by removing the second gather call.

---

### 2. Per-Sample Margin Reward + Equalized Generation Count

**Files:** `grpo_trainer.py`

- **Equalized shuffled generation count**:  
  Original code set `shuffled_num_generations = num_generations // 2` (e.g., 8 normal vs 4 corrupted). This made statistical comparison unstable and blocked per-sample pairing. Changed to `shuffled_num_generations = num_generations` for fair 1:1 comparison.

- **Per-sample margin reward** (Task B core):  
  Original logic computed a single batch-level margin (`acc_mean - shuffled_acc_mean`) and applied a uniform boost to all samples. Replaced with per-generation margin calculation:
  ```
  per_sample_margin = (normal_acc - shuffled_acc) / (normal_acc + 1e-6)
  per_sample_boost = clamp(per_sample_margin, min=0) * margin_scale
  ```
  Each generation now receives a boost proportional to how much it individually benefits from correct temporal order.

---

### 3. Configurable Corruption Strength + Curriculum Learning

**Files:** `grpo.py`, `grpo_trainer.py`

- Added `--corruption_strength` (float, 0.0–1.0) to control what fraction of frames are disrupted. Previously all corruption types operated at full intensity.

- Added `--curriculum_learning` (bool) flag. When enabled, corruption strength ramps from 0.3 → 1.0 over the course of training:
  ```
  strength = 0.3 + 0.7 * (global_step / max_steps)
  ```
  This allows the model to learn from easy (weak corruption) to hard (strong corruption).

- All three existing corruption types (shuffle, reverse, mask) now respect the strength parameter via `num_to_corrupt = int(num_frames * strength)`.

---

### 4. New Corruption Types + Weighted Selection + Logging

**Files:** `grpo_trainer.py`

- **New corruption types** (Task A):
  - `chunk_swap`: Splits video into N chunks and shuffles their order. Number of chunks scales with strength.
  - `speed`: Drops frames to simulate fast-forward, repeats last frame to maintain length.
  - `loop`: Copies early frames over late frames, simulating a temporal loop.

- **Weighted random selection**: Replaced `random.choice()` with `random.choices()` using configurable probability weights:
  ```
  types:   [shuffle, reverse, mask, chunk_swap, speed, loop]
  weights: [0.25,    0.20,   0.15, 0.20,       0.10,  0.10]
  ```

- **Destruction type logging**: Each step records which corruption type was used in metrics (`destruction_type_<name>`), enabling analysis of per-type effectiveness in TensorBoard.

---

### 5. Configurable Margin Scale and Reward Threshold

**Files:** `grpo.py`, `grpo_trainer.py`

- Added `--margin_scale` (float, default 0.5): Scaling factor for the marginal boost. Previously hardcoded as `* 0.5`.

- Added `--reward_threshold` (float, default 0.1): Minimum accuracy score to qualify for temporal boost. Previously hardcoded as `> 0.1`.

---

### 6. Reward Function Fixes

**Files:** `grpo.py`

- **rouge_scorer singleton**: `RougeScorer` was re-instantiated on every call to `compute_rouge_score`. Moved to a module-level singleton `_rouge_scorer_instance`.

- **Numerical comparison**: Removed the comma/decimal pre-check that misidentified thousands separators (e.g., `1,000`) as decimals. Now relies directly on `normalize_number()` float comparison.

- **format_reward whitespace tolerance**: Changed from `re.fullmatch` to `re.match` with `^\s*...\s*$` pattern. Model outputs with trailing whitespace or newlines after `</answer>` no longer receive a 0 format reward.

---

### 7. Code Cleanup

**Files:** `grpo_trainer.py`

- **print → logging**: Replaced all `print()` calls with `logger.debug()` / `logger.warning()` using Python's `logging` module. Removed two bare `print(rewards)` / `print(completion_mask.sum(1))` debug statements.

- **Typo fix**: `lenth_list` → `length_list`.

- **Removed dummy generation**: When temporal mode is on but input is an image (no video), the code previously ran a dummy `generate()` call producing 1 token that was never used. Replaced with `pass`.

- **Fixed all_wrong/all_correct metrics**: Previously used total reward sum with hardcoded thresholds (`<= 1` for wrong, `>= 2` for correct) that depended on reward function count. Now uses accuracy reward column (index 0) directly: `== 0` for wrong, `== 1` for correct.

---

## New CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--corruption_strength` | float | 1.0 | Fraction of frames to corrupt (0.0–1.0) |
| `--curriculum_learning` | bool | False | Ramp corruption strength over training |
| `--margin_scale` | float | 0.5 | Scaling factor for marginal boost |
| `--reward_threshold` | float | 0.1 | Min accuracy to receive temporal boost |

---

## Next Steps

- [ ] Prepare 500-sample video subset from Video-R1-260k for validation
- [ ] Adapt training script for 2×A100 SXM (DeepSpeed ZeRO-3)
- [ ] Run first validation: 200 steps, monitor reward curves
- [ ] Evaluate on MMVU / VSI-Bench, compare against baseline
- [ ] Ablation: curriculum learning on/off, corruption type subsets
