# Research Branch Runbook

## Purpose
This branch is the research-ready GRPO training branch for `D4_algorithm_oom_fix`. It keeps the current D1/D2/D3 algorithm intent, hardens the active RL training path, and provides explicit launchers for smoke tests, budget probes, ablations, and the formal run.

## Run Order
Run the stages in this order:

1. `bash /Users/chris/projects/capstone-d4-algorithm-oom-fix/src/scripts/research_branch/smoke_test.sh`
2. `bash /Users/chris/projects/capstone-d4-algorithm-oom-fix/src/scripts/research_branch/budget_probe.sh`
3. Run the ablation wrappers under `/Users/chris/projects/capstone-d4-algorithm-oom-fix/src/scripts/research_branch/ablation_*.sh`
4. `bash /Users/chris/projects/capstone-d4-algorithm-oom-fix/src/scripts/research_branch/formal_full.sh`

Every canonical research launcher requires:

- `RESEARCH_MODEL_NAME_OR_PATH`
- `RESEARCH_DATASET_NAME`

Optional environment overrides include `RESEARCH_MAX_STEPS`, `RESEARCH_SAVE_STEPS`, `RESEARCH_USE_VLLM`, `RESEARCH_CUDA_VISIBLE_DEVICES`, and `RESEARCH_DEEPSPEED_CONFIG`.

## Stage Meanings
- Smoke test: tiny run to verify YAML wiring, reward-path execution, and D1/D2/D3 visibility in logs.
- Budget probe: short run to inspect `step_compute_time_sec`, `cuda_max_memory_allocated_gb`, reward logging, and checkpoint cadence before spending real budget.
- Ablations: fixed launch surface where only the YAML changes D1/D2/D3 behavior. By default the shared launcher uses `max_steps=300`, `num_generations=4`, and non-colocate HF mode for every ablation wrapper.
- Formal run: full training entrypoint for the main research run on this branch. By default it uses `max_steps=1200`, `save_steps=100`, and non-colocate HF mode unless you explicitly override it.

## YAML and Script Mapping
- `smoke_test.sh` -> `src/r1-v/configs/research_branch/smoke_d1_d2_d3.yaml`
- `budget_probe.sh` -> `src/r1-v/configs/research_branch/budget_probe_d1_d2_d3.yaml`
- `ablation_baseline.sh` -> `src/r1-v/configs/research_branch/ablation_baseline.yaml`
- `ablation_d1_only.sh` -> `src/r1-v/configs/research_branch/ablation_d1.yaml`
- `ablation_d2_only.sh` -> `src/r1-v/configs/research_branch/ablation_d2.yaml`
- `ablation_d3_only.sh` -> `src/r1-v/configs/research_branch/ablation_d3.yaml`
- `ablation_d1_d2.sh` -> `src/r1-v/configs/research_branch/ablation_d1_d2.yaml`
- `ablation_d1_d3.sh` -> `src/r1-v/configs/research_branch/ablation_d1_d3.yaml`
- `ablation_d2_d3.sh` -> `src/r1-v/configs/research_branch/ablation_d2_d3.yaml`
- `ablation_d1_d2_d3.sh` -> `src/r1-v/configs/research_branch/ablation_d1_d2_d3.yaml`
- `formal_full.sh` -> `src/r1-v/configs/research_branch/formal_d1_d2_d3.yaml`

## What to Inspect in Logs
At startup, confirm the trainer prints:

- the resolved config path or a loud `LEGACY / NO EXPERIMENT CONFIG` warning
- generation mode: `HF generate / non-colocate` or `vLLM colocate`
- D1 effective behavior, including whether lazy evaluation is on and the `0.95` continuous threshold
- D2 clamp behavior and `kl_d_max`
- D3 effective behavior, including task-aware online bounds and the resolved `len_control` value

During training, inspect:

- `rewards/accuracy_reward`
- `causal_eval_ratio`
- `causal_reward_mean`
- `length_penalty_mean`
- `welford_mean`
- `welford_std`
- `kl`
- `kl_truncation_ratio` when D2 is on
- `step_compute_time_sec`
- `cuda_max_memory_allocated_gb`
- `skipped_batches` if corrupted media is encountered

## Caveats
- Canonical research runs should use the new `src/scripts/research_branch/` wrappers. Direct config-less `grpo.py` runs are still possible for compatibility, but they now log a loud legacy warning.
- Corrupted media is no longer replaced with unrelated files. Invalid samples are skipped loudly instead.
- Multimodal logprob failures now fail the step instead of silently retrying without vision inputs.
- Non-colocate HF mode remains the default research path. Set `RESEARCH_USE_VLLM=true` only when you intentionally want colocate mode.
