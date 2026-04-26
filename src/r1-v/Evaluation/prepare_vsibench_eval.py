#!/usr/bin/env python3
"""
Convert VSI-Bench test.jsonl to eval_bench.py format.

Usage:
    python prepare_vsibench_eval.py

Input:  Downloads test.jsonl from HuggingFace (or reads local copy)
Output: eval_vsibench.json in the same directory
"""

import json
import os
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_URL = "https://huggingface.co/datasets/nyu-visionx/VSI-Bench/raw/main/test.jsonl"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "eval_vsibench.json")

# -- Download or read ------------------------------------------------------------------
jsonl_path = os.path.join(SCRIPT_DIR, "test.jsonl")
if not os.path.exists(jsonl_path):
    print(f"Downloading {JSONL_URL} ...")
    urllib.request.urlretrieve(JSONL_URL, jsonl_path)
    print(f"Saved to {jsonl_path}")

with open(jsonl_path) as f:
    raw = [json.loads(line.strip()) for line in f if line.strip()]
print(f"Loaded {len(raw)} entries from test.jsonl")

# -- Convert ---------------------------------------------------------------------------
data = []
for item in raw:
    has_options = item.get("options") and len(item["options"]) > 0

    if has_options:
        problem_type = "multiple choice"
    else:
        try:
            float(item["ground_truth"].replace(",", ""))
            problem_type = "numerical"
        except ValueError:
            problem_type = "free-form"

    data.append({
        "data_type": "video",
        "path": f"/VSI-Bench/{item['dataset']}/{item['scene_name']}.mp4",
        "problem_type": problem_type,
        "problem": item["question"],
        "options": item["options"] if item["options"] else [],
        "solution": f"<answer>{item['ground_truth']}</answer>",
        "id": f"vsibench_{item['id']}",
        "question_type": item["question_type"],
        "dataset": item["dataset"],
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Stats
mc = sum(1 for d in data if d["problem_type"] == "multiple choice")
num = sum(1 for d in data if d["problem_type"] == "numerical")
free = sum(1 for d in data if d["problem_type"] == "free-form")
n_videos = len(set(d["path"] for d in data))
print(f"Wrote {len(data)} entries → {OUTPUT_FILE}")
print(f"  Multiple choice: {mc}")
print(f"  Numerical:       {num}")
print(f"  Free-form:       {free}")
print(f"  Unique videos:   {n_videos}")
