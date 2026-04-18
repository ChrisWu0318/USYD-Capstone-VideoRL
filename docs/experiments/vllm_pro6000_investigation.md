# vLLM on RTX Pro 6000 (Blackwell / sm_120) — 兼容性调研

> **调研时间**：2026-04-18
> **目的**：在 `blackwell_rtx6000_compat` 分支上，评估是否将 rollout
> 从 HF `generate` 切到 vLLM colocate 模式。
> **上下文**：2 卡 Pro 6000 因 ref-model materialize 撞 96 GB 上限而 OOM；
> vLLM colocate 理论上可以压缩 rollout 峰值显存并提速 3-5×。

---

## 1. 核心结论

| 项 | 结论 |
|---|---|
| **bf16 Qwen2.5-VL 非量化路径** | 有较大概率可用（sm_120 上主要报告的 bug 都在 FP8/NVFP4/MoE/MLA 通路） |
| **vLLM 官方 sm_120 状态** | 0.8.5（我们当前 pin）= 早期支持；0.15.1（2026-02）= 完整支持；NVIDIA release 25.09 正式认证 Pro 6000 Server Edition |
| **阻塞风险（已发生在社区）** | 主要集中在 FP8 量化模型、MoE kernels、MLA attention —— 我们都不用 |
| **兼容性风险（对我们）** | 中。不是技术不可行，而是依赖版本对齐（vLLM × torch × flash-attn × cu 版本 matrix） |
| **推荐下一步** | 不升级 vLLM，先用 `pip install vllm==0.8.5 --no-deps` 跑一次**独立** Qwen2.5-VL offline inference smoke；通过了再考虑接 colocate |

---

## 2. 社区证据总览

### 已知能跑的场景

- **vLLM Release 25.09**（NVIDIA 官方 docker）正式标注 Pro 6000 Server
  Edition 功能性支持。
- **vLLM 0.15.1+**（2026 年 2 月）添加了完整 SM120 支持与 H200 优化；
  被 VRLA Tech 评为目前在 Pro 6000 上最稳的版本。
- 社区基准测试已能在 Pro 6000 上跑 Qwen3 系列（dense + bf16）。

### 已知会炸的场景（我们不走这些路径）

| 场景 | 相关 Issue | 我们受影响吗 |
|---|---|---|
| Qwen-2.5-VL-**FP8**-Dynamic 72B 在 Pro 6000 上 `cutlass_scaled_mm` 报错 | [vllm#20221](https://github.com/vllm-project/vllm/issues/20221) | ❌ 我们用 bf16，不是 FP8 量化 |
| MLA 模型（GLM-5、DeepSeek-V3.2）在 sm_120 上不能跑 | [vllm#37113](https://github.com/vllm-project/vllm/issues/37113) | ❌ Qwen2.5-VL 不是 MLA |
| NVFP4 MoE kernels 在 sm_120 + CPU offload 下输出乱码 | [vllm#38718](https://github.com/vllm-project/vllm/issues/38718) | ❌ Qwen2.5-VL 7B 不是 MoE |
| SM120 NVFP4 MoE 走 Marlin fallback 而非 native | [vllm#31085](https://github.com/vllm-project/vllm/issues/31085) | ❌ 同上 |
| SM120 + CUDA 13 pip install 连续 5 次失败 | [vllm#37714](https://github.com/vllm-project/vllm/issues/37714) | ⚠️ 我们用 CUDA 12.8，不是 13；但提醒安装流程可能曲折 |

### 模糊地带

- **sm_120 kernel 注册**：早期（2025-Q2）报告过注册不全导致 TP>1 时崩溃，
  后续版本陆续修复。0.8.5 时代的 bug 可能仍然存在。
- **vLLM colocate + DeepSpeed ZeRO-3**：这个组合在 sm_120 上未见公开成功
  报告（大多数 Pro 6000 上 vLLM 报告都是**纯 inference**场景）。

---

## 3. 我们场景下的具体风险

### 3.1 版本 matrix 风险（**主要阻塞**）

当前安装栈：
```
torch             2.9.1+cu128
flash-attn        2.8.3 (prebuilt wheel, CXX11_ABI=True, torch 2.9 binding)
deepspeed         0.18.9
transformers      4.51.3
trl               0.16.0
```

- vLLM 0.8.5：当时 pin 的 torch ≤ 2.7；硬装到 torch 2.9 上**大概率装不起**，
  即使装上，内部 CUDA kernel ABI 可能 mismatch。
- vLLM 0.9.x：开始适配 torch 2.8；仍然不是 2.9。
- vLLM 0.15.1：要 torch 2.10；**会破坏 flash-attn 2.8.3**（没有 2.10 的
  prebuilt wheel）。

**简言之**：装 vLLM 会倒推改 torch → 连锁影响 flash-attn。

### 3.2 颗粒度风险

- **Qwen2.5-VL vision encoder + vLLM multimodal 通路**在 sm_120 上未经
  大规模验证。大部分 Pro 6000 benchmark 跑的是纯文本 LLM（Qwen3-dense、
  LLaMA 3），多模态 VL 在这卡上的 PagedAttention 通路相对小众。
- **colocate 模式**比 TP-serve 模式复杂：需要和 ZeRO-3 训练进程共用
  GPU 显存，`vllm_gpu_memory_utilization` 调参错了会两边都崩。

### 3.3 可忍受的风险

- bf16 dense 7B 走非量化路径，vLLM 的"危险区"（FP8/MoE/MLA）都不沾。
- Pro 6000 96 GB 对 colocate 显存分配比 A100 80 GB 宽松。

---

## 4. 建议的三步验证流程（不搞就不知道）

### Step A — 独立 vLLM offline inference（不接训练）

目的：**隔离验证 vLLM 能否在 sm_120 上跑起 Qwen2.5-VL bf16。**

```bash
# 创建独立 venv，避免污染训练环境
python -m venv /root/autodl-tmp/vllm_test
source /root/autodl-tmp/vllm_test/bin/activate

# 试最低代价版本（0.8.5 原 pin）——大概率装失败或跑崩
pip install "vllm==0.8.5"

# 如果失败，升到 0.9.x
# pip install "vllm==0.9.2"

python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(
    model="/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT",
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    max_model_len=4096,
    trust_remote_code=True,
)
out = llm.generate(["Hello"], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)
EOF
```

**通过标准**：不 crash，能输出合理文本。

### Step B — 如果 Step A 过了：colocate 最小串联

只改 `RESEARCH_USE_VLLM=true`，用 smoke_test_10.json 跑 3 步：

```bash
RESEARCH_USE_VLLM=true \
RESEARCH_VLLM_TENSOR_PARALLEL_SIZE=1 \
RESEARCH_VLLM_GPU_MEMORY_UTILIZATION=0.3 \
... bash src/scripts/research_branch/smoke_test.sh
```

**通过标准**：step-1 指标和 HF 路径一致（kl≈0.00054, welford_mean=192）。

### Step C — 如果 Step B 过了：上 budget_probe

同 HF 路径，只是 `RESEARCH_USE_VLLM=true`，跑 20 步看稳定性。

---

## 5. 决策建议

| 场景 | 建议 |
|---|---|
| **4 卡 Pro 6000 ZeRO-3 跑通 HF 路径** | ✅ 优先走这条；vLLM 当作加速路径 |
| **Step A 失败（0.8.5 和 0.9.x 都装不上或跑不起）** | 放弃 vLLM colocate，4 卡 + HF 路径即可完成所有实验 |
| **Step A 成功但 Step B 指标偏差 > 10%** | 回滚，走 HF 路径；vLLM 在 sm_120 上未必与训练 logprob 完全一致 |
| **Step A+B 都通** | 上 budget_probe，估算收益是否值得部署到 8 ablations |

---

## 6. 时间预算

| 步骤 | 估计耗时 |
|---|---|
| 4 卡 Pro 6000 smoke（HF 路径） | 5–10 min |
| 4 卡 Pro 6000 budget_probe（20 步） | 20–30 min |
| vLLM Step A（独立 offline） | 30–60 min（大部分在装依赖） |
| vLLM Step B + C | +20 min（如果 Step A 过） |

**结论**：先在 HF 路径上做完 budget_probe；vLLM 作为**可选加速路径**，
在 ablation 开始前单独花 1 小时验证，不做也不影响实验推进。

---

## 参考

- [vLLM forum: Support for RTX 6000 Blackwell 96GB card](https://discuss.vllm.ai/t/support-for-rtx-6000-blackwell-96gb-card/1707)
- [vllm#20221 — Qwen-2.5-VL-72B-FP8 on RTX 6000 Blackwell](https://github.com/vllm-project/vllm/issues/20221)
- [vllm#31085 — SM120 NVFP4 MoE kernels](https://github.com/vllm-project/vllm/issues/31085)
- [vllm#37113 — MLA on SM 120](https://github.com/vllm-project/vllm/issues/37113)
- [vllm#37714 — SM120 + CUDA 13 install failures](https://github.com/vllm-project/vllm/issues/37714)
- [vllm#38718 — NVFP4 MoE garbage output on SM120](https://github.com/vllm-project/vllm/issues/38718)
- [NVIDIA vLLM Release 25.09](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html)
- [VRLA Tech — RTX PRO 6000 Blackwell for LLMs](https://vrlatech.com/rtx-pro-6000-blackwell-for-llms-why-96gb-changes-everything/)
