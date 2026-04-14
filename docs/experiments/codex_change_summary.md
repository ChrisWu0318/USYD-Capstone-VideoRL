# Codex Change Summary

## Files Changed
- Core RL logic:
  - `src/r1-v/src/open_r1/grpo.py`
  - `src/r1-v/src/open_r1/experiment_config.py`
  - `src/r1-v/src/open_r1/trainer/grpo_trainer.py`
  - `src/r1-v/src/open_r1/welford.py`
  - `src/r1-v/src/open_r1/research_logic.py`
- Canonical research launchers:
  - `src/scripts/research_branch/_common.sh`
  - `src/scripts/research_branch/smoke_test.sh`
  - `src/scripts/research_branch/budget_probe.sh`
  - `src/scripts/research_branch/ablation_*.sh`
  - `src/scripts/research_branch/formal_full.sh`
- Legacy top-level RL launchers tightened to require an explicit config:
  - `src/scripts/run_grpo_video.sh`
  - `src/scripts/run_grpo_vllm_qwen25vl.sh`
  - `src/scripts/run_grpo_vllm_colocate.sh`
- Research configs:
  - `src/r1-v/configs/research_branch/*.yaml`
- Lightweight validation:
  - `src/r1-v/tests/test_research_logic.py`
  - `src/r1-v/tests/test_welford_state.py`
- Documentation:
  - `docs/experiments/research_branch_runbook.md`
  - `docs/experiments/codex_change_summary.md`

## What Was Fixed
- Experiment config wiring:
  - Canonical research runs now use explicit wrapper scripts that always pass `--experiment_config`.
  - Direct config-less `grpo.py` launches remain compatible, but they print a loud legacy warning at startup.
  - Trainer startup now prints effective runtime behavior, including config path, HF vs colocate mode, D1 lazy-eval policy, D2 clamp, and D3 mode.
- Reward correctness:
  - `accuracy_reward` no longer reuses the first sample’s `problem_type` across a mixed batch.
  - Existing dataset labels are preserved exactly: `multiple choice`, `numerical`, `OCR`, `free-form`, `regression`.
- D1 causal lazy evaluation:
  - `causal_lazy_eval` is now respected explicitly.
  - Exact-match tasks gate on `1.0`.
  - Continuous tasks gate on the configured threshold, default `0.95`.
- D3 online updates:
  - Silent index clamping was removed.
  - Generation-expanded metadata is validated explicitly.
  - Welford now keeps task-aware online stats through the existing internal mcq/open-ended mapping while preserving backward-compatible load behavior for older flat checkpoints.
- Silent corruption risks:
  - Multimodal `per_token_logps` and `ref_per_token_logps` no longer retry without vision inputs.
  - Corrupted or missing media is no longer replaced with unrelated files.
  - Invalid media now causes an explicit skipped batch with loud logging instead.
- Operator usability:
  - Added smoke, budget probe, 8 ablation, and formal wrappers with strict shell behavior, config printing, git metadata, and explicit output naming.
  - Added a short runbook describing stage order, YAML mappings, and the logs to inspect.

## Assumptions Made
- Canonical research runs on this branch will use the new `src/scripts/research_branch/` wrappers.
- The conservative continuous-task D1 lazy threshold should be `0.95`.
- The existing internal D3 bucket mapping remains the right minimal change: `multiple choice` maps to the mcq bounds, and all other existing `problem_type` labels map to the open-ended bounds.
- Non-colocate HF generation remains the default research mode for this branch. Colocate stays opt-in.

## Validation Performed
- `python -m py_compile` on the touched Python files and new lightweight tests
- `bash -n` on the updated legacy launchers and all new research wrappers
- YAML parse validation for all files under `src/r1-v/configs/research_branch`
- Direct Python execution of the new lightweight tests:
  - mixed-batch reward typing and D1 lazy gating
  - task-aware Welford save/load and legacy compatibility

## Remaining Runpod Validation
- End-to-end smoke execution with real model weights and dataset paths
- Budget probe confirmation for actual `step_compute_time_sec` and `cuda_max_memory_allocated_gb` behavior on the target GPU shape
- Verification that skipped invalid-media batches behave acceptably under the real training dataloader
- Resume-from-checkpoint validation with a real D3-enabled run on Runpod
- Formal run confirmation for checkpoint cadence, WandB logging, and any colocate-mode overrides if deliberately enabled

## Notes
- `pytest` is not installed in this local environment, so the new test files were executed directly with Python instead of `python -m pytest`.
