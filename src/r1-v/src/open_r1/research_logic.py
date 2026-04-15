import re

from rouge_score import rouge_scorer


EXACT_MATCH_PROBLEM_TYPES = {"multiple choice", "numerical"}
CONTINUOUS_PROBLEM_TYPES = {"OCR", "free-form", "regression"}


def extract_answer(text: str) -> str:
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_number(num_str: str):
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    distances = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        distances[i][0] = i
    for j in range(n + 1):
        distances[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(
                    distances[i - 1][j],
                    distances[i][j - 1],
                    distances[i - 1][j - 1],
                )
    return distances[m][n] / max(1, m)


_rouge_scorer_instance = None


def _get_rouge_scorer(use_stemmer: bool = True):
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        _rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    return _rouge_scorer_instance


def compute_rouge_score(reference: str, hypothesis: str, use_stemmer: bool = True) -> float:
    scorer = _get_rouge_scorer(use_stemmer)
    scores = scorer.score(reference, hypothesis)
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3


def compute_accuracy_reward_for_sample(content: str, solution: str, problem_type: str) -> float:
    output_ans = extract_answer(content)
    gt_ans = extract_answer(solution)

    if problem_type == "multiple choice":
        return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

    if problem_type == "numerical":
        gt_has_decimal = "." in gt_ans
        out_has_decimal = "." in output_ans
        if gt_has_decimal != out_has_decimal:
            return 0.0
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0

    if problem_type == "OCR":
        reward = 1 - word_error_rate(gt_ans, output_ans)
        return max(0.0, min(1.0, reward))

    if problem_type == "free-form":
        reward = compute_rouge_score(gt_ans, output_ans)
        return max(0.0, min(1.0, reward))

    if problem_type == "regression":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
        rel_diff = min(1.0, max(0.0, rel_diff))
        return 1 - rel_diff

    return 0.0


def compute_accuracy_rewards(contents, solutions, problem_types):
    rewards = []
    for content, solution, problem_type in zip(contents, solutions, problem_types):
        rewards.append(compute_accuracy_reward_for_sample(content, solution, problem_type))
    return rewards


def should_evaluate_causal_reward(
    problem_type: str,
    accuracy_reward: float,
    causal_lazy_eval: bool,
    continuous_threshold: float,
) -> bool:
    if not causal_lazy_eval:
        return True
    if problem_type in EXACT_MATCH_PROBLEM_TYPES:
        return accuracy_reward >= 1.0
    return accuracy_reward >= continuous_threshold


def describe_causal_lazy_eval(causal_lazy_eval: bool, continuous_threshold: float) -> str:
    if not causal_lazy_eval:
        return "evaluate all samples"
    return (
        "lazy eval: exact-match tasks require 1.0; "
        f"continuous tasks require >= {continuous_threshold:.2f}"
    )
