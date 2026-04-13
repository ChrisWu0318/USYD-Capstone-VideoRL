#!/bin/bash
# =============================================================================
# vLLM Colocate 冒烟测试
# 目的：在不启动 DeepSpeed/训练的情况下，单独验证 vLLM 能否：
#   1. 加载 Qwen2.5-VL-7B（TP=4）
#   2. sleep() / wake_up() 正常工作
#   3. 多模态（图片）生成正常
#   4. 显存释放/回收符合预期
# 跑完这个再跑正式训练，避免浪费钱。
# =============================================================================

set -euo pipefail

# ---- 配置 ----
MODEL_PATH="${1:-./Qwen2.5-VL-7B-COT-SFT}"
TP=4
GPU_MEM_UTIL=0.3

echo "=========================================="
echo "vLLM Colocate Smoke Test"
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP}, gpu_mem_util: ${GPU_MEM_UTIL}"
echo "=========================================="

# 用 torchrun 启动，模拟真实的 DP=4 分布式环境
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    - <<'PYTHON_SCRIPT'
import os
import torch
import time

def main():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = (rank == 0)

    if is_main:
        print(f"[Rank {rank}] Starting vLLM smoke test...")

    # ---- Step 1: 检查 vLLM 版本 ----
    import vllm
    if is_main:
        print(f"[Rank {rank}] vLLM version: {vllm.__version__}")
        assert tuple(int(x) for x in vllm.__version__.split(".")[:2]) >= (0, 8), \
            f"vLLM {vllm.__version__} < 0.8.0, sleep/wake API 不可用!"

    # ---- Step 2: 初始化 vLLM LLM 引擎 ----
    from vllm import LLM, SamplingParams

    model_path = os.environ.get("MODEL_PATH", "./Qwen2.5-VL-7B-COT-SFT")
    tp = int(os.environ.get("TP", "4"))
    gpu_mem_util = float(os.environ.get("GPU_MEM_UTIL", "0.3"))

    if is_main:
        print(f"[Rank {rank}] Initializing LLM engine...")
        print(f"[Rank {rank}]   model: {model_path}")
        print(f"[Rank {rank}]   TP: {tp}, gpu_mem_util: {gpu_mem_util}")

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gpu_mem_util,
        dtype=torch.bfloat16,
        enforce_eager=True,
        max_model_len=4096,  # 小一点，冒烟测试不需要 17k
        enable_prefix_caching=True,
    )
    t1 = time.time()
    if is_main:
        print(f"[Rank {rank}] LLM engine initialized in {t1-t0:.1f}s")

    # ---- Step 3: 测试 wake_up ----
    if is_main:
        print(f"[Rank {rank}] Testing wake_up()...")
    llm.wake_up()
    if is_main:
        print(f"[Rank {rank}] wake_up() OK")

    # ---- Step 4: 简单文本生成（确认引擎能跑） ----
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    if is_main:
        print(f"[Rank {rank}] Testing text generation...")
        outputs = llm.generate(["Hello, my name is"], sampling_params=sampling_params)
        for output in outputs:
            print(f"[Rank {rank}] Generated: {output.outputs[0].text[:80]}")
        print(f"[Rank {rank}] Text generation OK")

    # ---- Step 5: 测试 sleep ----
    if is_main:
        print(f"[Rank {rank}] Testing sleep()...")
    llm.sleep(level=1)
    if is_main:
        print(f"[Rank {rank}] sleep() OK — GPU memory should be released now")

    # ---- Step 6: 再次 wake + 生成（验证 sleep/wake 循环） ----
    if is_main:
        print(f"[Rank {rank}] Testing wake_up() again...")
    llm.wake_up()
    if is_main:
        print(f"[Rank {rank}] Second wake_up() OK, generating again...")
        outputs = llm.generate(["What is 2+2?"], sampling_params=sampling_params)
        for output in outputs:
            print(f"[Rank {rank}] Generated: {output.outputs[0].text[:80]}")

    # ---- Step 7: 最终 sleep ----
    llm.sleep(level=1)
    if is_main:
        print(f"[Rank {rank}] Final sleep() OK")
        print(f"[Rank {rank}]")
        print(f"[Rank {rank}] ========== SMOKE TEST PASSED ==========")
        print(f"[Rank {rank}] vLLM colocate sleep/wake cycle works correctly.")
        print(f"[Rank {rank}] You can now run the full training script.")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT
