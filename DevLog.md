# USYD Capstone: Video-R1 T-GRPO Algorithm Upgrade Log

## Core Objective
Based on the Video-R1 baseline, upgrade the original "Binary Reward" to a "Multiple Destruction" mechanism and a "Marginal Reward" system to enhance the model's temporal understanding capabilities.

## Project Timeline & TODO List
### Phase 1: Environment Setup & Baseline (Almost Complete)
- [x] Configure AutoDL server environment (Python 3.11, Conda video-r1).
- [x] Download Qwen2.5-VL-7B-COT-SFT model weights.
- [x] Initialize private GitHub repository and configure .gitignore.

### Phase 2: Core Algorithm Modification - grpo_trainer.py (In Progress)
- [x] Locate Reward Logic: Identified hardcoded binary reward logic at approx. line 530.
- [x] Locate Data Processing: Identified single-direction destruction at approx. line 330.
- [ ] Task A: Modify line 330 to introduce "Multiple Destruction" mechanism.
- [ ] Task B: Modify line 530 to replace binary logic with "Marginal Reward".

### Phase 3: Compute Scaling & Full Training (Pending)
- [ ] Write README.md and rent multi-GPU nodes for full RL training.
