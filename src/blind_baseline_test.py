"""
blind_baseline_test.py

Runs three inference conditions on a video QA model to quantify how much
accuracy comes from language priors vs. actual visual/temporal understanding.

Modes:
  normal   - full video, correct frame order (upper bound)
  shuffled - full video, random frame order (temporal sensitivity)
  blind    - no video at all, text only (pure language-prior baseline)

Usage:
  python src/blind_baseline_test.py \
      --model_path /path/to/checkpoint \
      --data_path src/r1-v/Evaluation/eval_mvbench.json \
      --output_dir ./blind_baseline_results \
      --modes normal,shuffled,blind \
      --sample_size 500 \
      --bsz 32

  # Quick smoke test (blind only, 50 samples)
  python src/blind_baseline_test.py \
      --model_path /path/to/checkpoint \
      --data_path src/r1-v/Evaluation/eval_mvbench.json \
      --output_dir ./blind_baseline_results \
      --modes blind \
      --sample_size 50
"""

import os
import json
import re
import copy
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Prompt templates (identical to eval_bench.py)
# ---------------------------------------------------------------------------

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}


# ---------------------------------------------------------------------------
# Helper functions (verbatim from eval_bench.py)
# ---------------------------------------------------------------------------

def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_number(num_str):
    try:
        return float(num_str.replace(',', ''))
    except Exception:
        return None


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    return conditions.float().mean().item()


def reward_fn(sample, model_output, question_type):
    """Identical to eval_bench.py's reward_fn."""
    try:
        output_ans = extract_answer(model_output)
        if output_ans == '':
            output_ans = model_output
        gt_ans = extract_answer(sample.get("solution", ""))
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return mean_relative_accuracy(out_number, gt_number)
        else:
            return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_question_text(x):
    if x["problem_type"] == "multiple choice":
        question = x["problem"] + "Options:\n"
        for op in x["options"]:
            question += op + "\n"
    else:
        question = x["problem"]
    return question


def build_messages_normal(data, eval_dir):
    """Build messages with full video/image (same as eval_bench.py)."""
    messages = []
    for x in data:
        question = build_question_text(x)
        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x["data_type"],
                    x["data_type"]: eval_dir + x["path"][1:],
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x["problem_type"]],
                },
            ],
        }]
        messages.append(msg)
    return messages


def build_messages_blind(data):
    """Build text-only messages (no video/image block at all)."""
    messages = []
    for x in data:
        question = build_question_text(x)
        msg = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[x["problem_type"]],
                },
            ],
        }]
        messages.append(msg)
    return messages


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_mode(mode, data, messages, llm, processor, sampling_params, bsz, output_path):
    """
    Run inference for one mode over all samples.

    For 'shuffled', messages must be the normal (video-containing) messages —
    frames are permuted after process_vision_info.
    For 'blind', messages must be text-only messages.
    """
    final_output = []
    start_idx = 0

    # Resumability: if output file exists, skip already-processed samples
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"[{mode}] Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"[{mode}] Error reading existing output: {e}")

    acc_by_type = defaultdict(list)

    for i in tqdm(range(start_idx, len(messages), bsz), desc=f"[{mode}]"):
        batch_messages = messages[i:i + bsz]
        batch_data = data[i:i + bsz]

        prompts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        try:
            llm_inputs = []

            if mode == "blind":
                # No visual input at all
                for prompt in prompts:
                    llm_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {},
                        "mm_processor_kwargs": {},
                    })
            else:
                # normal or shuffled: process vision info
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    batch_messages, return_video_kwargs=True
                )

                if mode == "shuffled":
                    # Permute frame dimension (dim 0) for each video tensor
                    for vi in range(len(video_inputs)):
                        n_frames = video_inputs[vi].shape[0]
                        if n_frames > 1:
                            idx = torch.randperm(n_frames)
                            video_inputs[vi] = video_inputs[vi][idx]

                image_idx = 0
                video_idx = 0
                for idx, prompt in enumerate(prompts):
                    mm_type = batch_messages[idx][0]["content"][0]["type"]
                    sample_mm_data = {}
                    sample_video_kw = {}
                    if mm_type == "image":
                        sample_mm_data["image"] = image_inputs[image_idx]
                        image_idx += 1
                    elif mm_type == "video":
                        sample_mm_data["video"] = video_inputs[video_idx]
                        for key, value in video_kwargs.items():
                            sample_video_kw[key] = value[video_idx]
                        video_idx += 1
                    llm_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": sample_mm_data,
                        "mm_processor_kwargs": sample_video_kw,
                    })

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]

        except Exception as e:
            path_info = batch_data[0].get("path", "unknown") if batch_data else "unknown"
            print(f"[{mode}] Error at batch starting idx {i}, path: {path_info}")
            print(f"[{mode}] Exception: {e}")
            batch_output_text = ["<answer>error</answer>"] * len(batch_data)

        for sample, model_output in zip(batch_data, batch_output_text):
            sample = copy.deepcopy(sample)
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, q_type)
            sample["correct"] = sample["reward"] == 1.0
            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
            acc_by_type[q_type].append(sample["reward"])

        # Save checkpoint after each batch
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[{mode}] Error writing checkpoint: {e}")

    # Compute per-type stats
    by_type_stats = {}
    all_acc = []
    all_mra = []
    for q_type, rewards in acc_by_type.items():
        n = len(rewards)
        mean = sum(rewards) / n if n > 0 else 0.0
        by_type_stats[q_type] = {"n": n, "acc": round(mean, 4)}
        if q_type == "regression":
            all_mra.extend(rewards)
        else:
            all_acc.extend(rewards)

    overall_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    overall_mra = sum(all_mra) / len(all_mra) if all_mra else 0.0

    # Final save with accuracy summary
    summary = {
        "mode": mode,
        "overall_acc": round(overall_acc, 4),
        "overall_mra": round(overall_mra, 4),
        "by_type": by_type_stats,
        "results": final_output,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[{mode}] Saved to {output_path}")
    except Exception as e:
        print(f"[{mode}] Error writing final output: {e}")

    return overall_acc, by_type_stats


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(dataset_name, n_samples, mode_results):
    """Print a comparison table across all modes."""
    all_types = set()
    for _, by_type in mode_results.values():
        all_types.update(by_type.keys())
    all_types = sorted(all_types)

    header = f"{'Mode':<12}" + f"{'overall':>9}" + "".join(f"{t:>18}" for t in all_types)
    sep = "-" * len(header)

    print(f"\n{'='*60}")
    print(f"=== Blind Baseline Summary ===")
    print(f"Dataset: {dataset_name}  (N={n_samples})")
    print(sep)
    print(header)
    print(sep)
    for mode in ["normal", "shuffled", "blind"]:
        if mode not in mode_results:
            continue
        overall_acc, by_type = mode_results[mode]
        row = f"{mode:<12}" + f"{overall_acc:>8.1%}"
        for t in all_types:
            acc = by_type.get(t, {}).get("acc", float("nan"))
            row += f"{acc:>17.1%}"
        print(row)
    print(sep)

    # Derived metrics (if all three modes ran)
    if all(m in mode_results for m in ["normal", "shuffled", "blind"]):
        normal_acc = mode_results["normal"][0]
        shuffled_acc = mode_results["shuffled"][0]
        blind_acc = mode_results["blind"][0]
        print(f"\nTemporal boost   (normal  - shuffled): {(normal_acc - shuffled_acc):+.1%}")
        print(f"Language prior   (blind):               {blind_acc:.1%}")
        print(f"True visual gain (shuffled - blind):    {(shuffled_acc - blind_acc):+.1%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Blind baseline experiment for video QA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to eval JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, default="./blind_baseline_results", help="Directory for output JSONs")
    parser.add_argument("--modes", type=str, default="normal,shuffled,blind",
                        help="Comma-separated modes to run: normal,shuffled,blind")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Limit to N samples (default: all)")
    parser.add_argument("--bsz", type=int, default=32, help="Batch size for VLLM inference")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    for m in modes:
        if m not in ("normal", "shuffled", "blind"):
            raise ValueError(f"Unknown mode: {m}. Choose from: normal, shuffled, blind")

    os.makedirs(args.output_dir, exist_ok=True)

    # Derive dataset name from data_path for output filenames
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    # eval_dir is the directory that contains the media files referenced by x['path']
    eval_dir = os.path.dirname(os.path.abspath(args.data_path))

    # Load data
    data = []
    if args.data_path.endswith(".jsonl"):
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif args.data_path.endswith(".json"):
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("data_path must be .json or .jsonl")

    if args.sample_size is not None:
        data = data[: args.sample_size]
    print(f"Loaded {len(data)} samples from {args.data_path}")

    # Initialize VLLM (once, shared across all modes)
    print(f"Loading model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=8192 * 2,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    # Pre-build messages for each mode (blind uses a different structure)
    normal_messages = build_messages_normal(data, eval_dir)
    blind_messages = build_messages_blind(data)

    mode_results = {}

    for mode in modes:
        output_path = os.path.join(args.output_dir, f"results_{mode}_{dataset_name}.json")
        print(f"\n{'='*60}")
        print(f"Running mode: {mode}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        if mode == "blind":
            messages = blind_messages
        else:
            # normal and shuffled both use the full-video messages
            messages = normal_messages

        overall_acc, by_type = run_mode(
            mode=mode,
            data=data,
            messages=messages,
            llm=llm,
            processor=processor,
            sampling_params=sampling_params,
            bsz=args.bsz,
            output_path=output_path,
        )
        mode_results[mode] = (overall_acc, by_type)
        print(f"[{mode}] overall_acc={overall_acc:.1%}  by_type={by_type}")

    print_summary(dataset_name, len(data), mode_results)


if __name__ == "__main__":
    main()
