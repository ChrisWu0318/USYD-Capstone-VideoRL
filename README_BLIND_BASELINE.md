# Blind Baseline Experiment: Quantifying Language Priors in Video QA

## Overview

This document reports the results of our **blind baseline experiment**, designed to validate the motivation behind the **T-GRPO** temporal reward design. Specifically, we investigate how much of a video QA model's accuracy is attributable to:

1. **Language priors** — patterns in question text alone (no video)
2. **Static visual content** — having frames available, regardless of order
3. **Temporal structure** — the correct ordering of frames

We run three inference conditions on the same model (`Qwen2.5-VL-7B-SFT`) across three benchmark datasets.

---

## Experimental Setup

### Model
- **Qwen2.5-VL-7B-SFT**: A 7B vision-language model fine-tuned with supervised learning on video QA data. This serves as the pre-GRPO baseline (no reinforcement learning applied).

### Three Inference Modes

| Mode | Visual Input | What It Measures |
|------|-------------|-----------------|
| `normal` | Full video, correct frame order | Upper-bound accuracy |
| `shuffled` | Full video, randomised frame order | Sensitivity to temporal structure |
| `blind` | No video — text only | Pure language-prior baseline |

The key insight: if `blind` accuracy is significantly above random chance (25% for 4-choice MCQ), it means the dataset leaks substantial signal through question text alone, independent of any visual reasoning.

### Script
```bash
python src/blind_baseline_test.py \
    --model_path /path/to/Qwen2.5-VL-7B-SFT \
    --data_path src/r1-v/Evaluation/eval_{dataset}.json \
    --output_dir ./blind_baseline_results \
    --modes normal,shuffled,blind \
    --sample_size 500 \
    --bsz 32
```

---

## Datasets

### 1. MMVU (Massive Multi-discipline Video Understanding)
A large-scale benchmark covering diverse academic and professional domains (science, engineering, medicine, law, etc.). Questions require multi-discipline knowledge combined with video understanding. All questions are **4-choice multiple choice**. The dataset emphasises knowledge-intensive reasoning over purely perceptual temporal tasks.

### 2. MVBench
A comprehensive video understanding benchmark with 20 subtasks spanning action recognition, scene understanding, object tracking, and temporal reasoning. Videos are sourced from 11 different datasets (SSomething-Something v2, Kinetics, etc.), providing broad coverage of real-world video content. All questions are **4-choice multiple choice**.

### 3. TempCompass
A benchmark specifically designed to evaluate **temporal understanding** in video models. Questions probe fine-grained temporal properties: action speed, direction, order, and multi-event relationships. Unlike general benchmarks, TempCompass is constructed so that correct temporal reasoning is theoretically necessary to answer correctly — making it the most demanding for our shuffled vs. normal comparison.

---

## Results

### Raw Accuracy Table

| Dataset | Mode | N | Overall Acc | MC Acc |
|---------|------|---|-------------|--------|
| MMVU | normal | 500 | 57.4% | 57.4% |
| MMVU | shuffled | 500 | 60.4% | 60.4% |
| MMVU | **blind** | 500 | **47.4%** | 47.4% |
| MVBench | normal | 500 | 45.0% | 45.0% |
| MVBench | shuffled | 500 | 44.4% | 44.4% |
| MVBench | **blind** | 500 | **33.8%** | 33.8% |
| TempCompass | normal | 500 | 84.2% | 84.2% |
| TempCompass | shuffled | 500 | 79.8% | 79.8% |
| TempCompass | **blind** | 500 | **68.2%** | 68.2% |

### Key Metrics

| Dataset | Temporal Boost (normal−shuffled) | Language Prior (blind) | True Visual Gain (shuffled−blind) |
|---------|----------------------------------|----------------------|----------------------------------|
| MMVU | **−3.0pp** ⚠️ | 47.4% | +13.0pp |
| MVBench | +0.6pp | 33.8% | +10.6pp |
| TempCompass | **+4.4pp** | 68.2% | +11.6pp |

---

## Analysis

### Finding 1: Language Priors Are Pervasive and Large

Across all three datasets, the model answers correctly on a large fraction of questions **without seeing any video**:

- MMVU: **47.4%** (random = 25%, gap = +22.4pp)
- MVBench: **33.8%** (random = 25%, gap = +8.8pp)
- TempCompass: **68.2%** (random = 25%, gap = +43.2pp)

TempCompass is the most striking case: **68.2% of questions can be answered from text alone**, despite the benchmark being explicitly designed for temporal video understanding. This reveals that the dataset's question phrasing contains strong distributional cues — certain answer options are statistically more plausible given the question text, regardless of video content.

This finding directly motivates T-GRPO's causal reward design: if reward signals are computed only from normal-mode outputs, they are substantially contaminated by language prior accuracy. The model may receive high rewards for exploiting textual patterns rather than developing genuine visual/temporal understanding.

### Finding 2: MMVU Shows Negative Temporal Boost (Anomaly)

On MMVU, the shuffled condition **outperforms** normal by 3.0pp (60.4% vs 57.4%). This counterintuitive result indicates that MMVU questions **do not require temporal reasoning**. Frame order provides no useful signal, and the shuffled condition may even help by exposing the model to a broader range of individual frames without the misleading expectation of sequential structure.

This has a direct implication for T-GRPO: the temporal reward fires when `p_normal ≥ p_shuffled`. On MMVU, this condition rarely holds at the dataset level, meaning T-GRPO's temporal reward signal would be largely suppressed on this benchmark. This is appropriate — MMVU is not a temporal reasoning benchmark — but it highlights that T-GRPO's reward is data-dependent.

### Finding 3: True Visual Gain Is Consistent but Modest

The gap between shuffled and blind accuracy (i.e., the gain from having visual frames, regardless of order) is remarkably consistent across datasets:

- MMVU: +13.0pp
- MVBench: +10.6pp
- TempCompass: +11.6pp

This ~11–13pp gain represents the contribution of **static visual content** (individual frame features) independent of temporal structure. Importantly, this is larger than the temporal boost (0.6–4.4pp) on every dataset, suggesting that **what the model sees matters more than the order in which it sees it**.

### Finding 4: TempCompass Is the Most Temporally Sensitive — But Still Affected by Language Priors

TempCompass shows the largest temporal boost (+4.4pp), consistent with its design as a temporal reasoning benchmark. However, even here, the language prior (68.2%) dwarfs the temporal boost, meaning the majority of the model's performance on TempCompass can be explained without reference to temporal video understanding.

### Implication for T-GRPO

These results collectively establish two points:

1. **The temporal reward is necessary**: Without explicitly contrasting normal vs. shuffled performance, standard GRPO rewards conflate language prior accuracy with genuine video understanding. The 33–68% blind accuracy means a substantial fraction of rewards in vanilla GRPO have nothing to do with video reasoning.

2. **The temporal reward is limited by benchmark design**: The temporal boost is small (0.6–4.4pp) relative to the language prior gap. This means T-GRPO's temporal reward signal is a weak but real signal buried under substantial noise. Stronger gains would require benchmarks with lower language prior leakage.

---

## Summary

```
=== Blind Baseline Summary ===

                  MMVU     MVBench   TempCompass
                --------  ---------  -----------
normal           57.4%     45.0%       84.2%
shuffled         60.4%     44.4%       79.8%
blind            47.4%     33.8%       68.2%
random (4-MCQ)   25.0%     25.0%       25.0%

Temporal boost   -3.0pp    +0.6pp      +4.4pp
Language prior   47.4%     33.8%       68.2%
True visual gain +13.0pp  +10.6pp     +11.6pp
```

The experiment confirms that **language priors are a significant confound** in video QA benchmarks, and that temporal structure contributes a modest but measurable signal above static visual content. These findings provide empirical grounding for T-GRPO's design of a causal temporal reward.

---

## Reproduction

```bash
# Environment
conda activate video-r1

# Run all three modes on TempCompass
python src/blind_baseline_test.py \
    --model_path /path/to/Qwen2.5-VL-7B-SFT \
    --data_path src/r1-v/Evaluation/eval_tempcompass.json \
    --output_dir ./blind_baseline_results \
    --modes normal,shuffled,blind \
    --sample_size 500 --bsz 32

# Run on MVBench (use filtered JSON to exclude unavailable videos)
python src/blind_baseline_test.py \
    --model_path /path/to/Qwen2.5-VL-7B-SFT \
    --data_path src/r1-v/Evaluation/eval_mvbench_filtered.json \
    --output_dir ./blind_baseline_results \
    --modes normal,shuffled,blind \
    --sample_size 500 --bsz 32

# Run on MMVU
python src/blind_baseline_test.py \
    --model_path /path/to/Qwen2.5-VL-7B-SFT \
    --data_path src/r1-v/Evaluation/eval_mmvu.json \
    --output_dir ./blind_baseline_results \
    --modes normal,shuffled,blind \
    --sample_size 500 --bsz 32
```

Results are saved to `./blind_baseline_results/results_{mode}_eval_{dataset}.json`.
