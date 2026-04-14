from pathlib import Path
import math
import sys
import types


try:
    from rouge_score import rouge_scorer as _rouge_scorer  # noqa: F401
except ModuleNotFoundError:
    rouge_score_module = types.ModuleType("rouge_score")
    rouge_scorer_module = types.ModuleType("rouge_scorer")

    class _FakeRougeScore:
        def __init__(self, fmeasure):
            self.fmeasure = fmeasure

    class _FakeRougeScorer:
        def __init__(self, metrics, use_stemmer=True):
            self.metrics = metrics

        def score(self, reference, hypothesis):
            ref_tokens = set(reference.split())
            hyp_tokens = set(hypothesis.split())
            overlap = len(ref_tokens & hyp_tokens)
            denom = max(len(ref_tokens | hyp_tokens), 1)
            fmeasure = overlap / denom
            return {metric: _FakeRougeScore(fmeasure) for metric in self.metrics}

    rouge_scorer_module.RougeScorer = _FakeRougeScorer
    rouge_score_module.rouge_scorer = rouge_scorer_module
    sys.modules["rouge_score"] = rouge_score_module
    sys.modules["rouge_score.rouge_scorer"] = rouge_scorer_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "open_r1"))

from research_logic import compute_accuracy_rewards, should_evaluate_causal_reward


def test_compute_accuracy_rewards_uses_per_sample_problem_types():
    completions = [
        [{"role": "assistant", "content": "<think>x</think><answer>A</answer>"}],
        [{"role": "assistant", "content": "<think>x</think><answer>cat mat</answer>"}],
        [{"role": "assistant", "content": "<think>x</think><answer>9</answer>"}],
        [{"role": "assistant", "content": "<think>x</think><answer>3.14</answer>"}],
        [{"role": "assistant", "content": "<think>x</think><answer>red car</answer>"}],
    ]
    solutions = [
        "<think>y</think><answer>A</answer>",
        "<think>y</think><answer>cat sat</answer>",
        "<think>y</think><answer>10</answer>",
        "<think>y</think><answer>3.14</answer>",
        "<think>y</think><answer>blue car</answer>",
    ]
    problem_types = [
        "multiple choice",
        "OCR",
        "regression",
        "numerical",
        "free-form",
    ]

    rewards = compute_accuracy_rewards(
        [completion[0]["content"] for completion in completions],
        solutions,
        problem_types,
    )

    assert rewards[0] == 1.0
    assert rewards[1] == 0.5
    assert math.isclose(rewards[2], 0.9, rel_tol=1e-6)
    assert rewards[3] == 1.0
    assert 0.0 < rewards[4] < 1.0


def test_should_evaluate_causal_reward_respects_task_specific_thresholds():
    assert should_evaluate_causal_reward("multiple choice", 1.0, True, 0.95)
    assert not should_evaluate_causal_reward("multiple choice", 0.99, True, 0.95)
    assert should_evaluate_causal_reward("numerical", 1.0, True, 0.95)
    assert should_evaluate_causal_reward("OCR", 0.95, True, 0.95)
    assert not should_evaluate_causal_reward("OCR", 0.94, True, 0.95)
    assert should_evaluate_causal_reward("free-form", 0.20, False, 0.95)
