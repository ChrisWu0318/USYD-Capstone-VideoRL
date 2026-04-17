# AutoDL RTX Pro 6000 (Blackwell) 完整配置指南

> 本文档汇总了所有踩过的坑，给出一次跑通的配置流程。
> 目标：在 AutoDL 的 RTX Pro 6000 (Blackwell, sm_120) pod 上完整配置
> `blackwell_rtx6000_compat` 分支的 VideoRL 训练环境。

## 关键决策（为什么这么装）

| 组件 | 版本 | 原因 |
|---|---|---|
| torch | **2.9.1+cu128** | Blackwell sm_120 从 2.7 开始支持；2.10 没有 flash-attn 预编译 wheel；2.9 是最新稳定且有 wheel 的版本 |
| flash-attn | **2.8.3** (prebuilt wheel) | 官方 release 有 `torch2.9+cu12+cxx11abiTRUE+cp312` 的 wheel，免编译 |
| vllm | **跳过**（可选） | vllm 新版本会把 torch 拉到 2.10 打破 flash-attn 兼容；训练默认用 HF 生成，设 `RESEARCH_USE_VLLM=false` |
| deepspeed | >=0.16.5 | Blackwell CUDA kernels |
| bitsandbytes | >=0.45.0 | sm_120 kernels |
| Python | 3.12 (pod 自带) | 不用 conda，AutoDL 自带的 miniconda3 Python 3.12 直接用 |

## Pod 存储布局（重要）

AutoDL 容器有两块独立文件系统：
- `/root` — 系统盘，约 30 GB，**装满会挂**
- `/root/autodl-tmp` — 数据盘，约 550 GB，放模型/数据集/pip 缓存

所有大文件必须放 `/root/autodl-tmp`。

## Step 0 — Pod 准备

开 pod 时选：
- **镜像**：PyTorch 镜像（任意版本都可，我们自己覆盖）
- **GPU 数量**：≥2 卡（2 卡可跑 smoke test，4 卡跑 ablation）
- **数据盘**：默认 50 GB 就够 smoke test，正式跑需扩到 200+ GB

## Step 1 — 环境变量和缓存路径

**一次性配置 `.bashrc`，把所有缓存/临时目录重定向到数据盘**：

```bash
cat >> ~/.bashrc <<'EOF'

# === USYD VideoRL AutoDL config ===
# 所有大缓存/临时文件放数据盘
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
export TORCH_HOME=/root/autodl-tmp/torch_cache
export WANDB_DIR=/root/autodl-tmp/wandb
EOF

source ~/.bashrc
mkdir -p /root/autodl-tmp/{hf_cache,pip_cache,tmp,torch_cache,wandb}
```

> **为什么要设 TMPDIR**：PIP_CACHE_DIR 在数据盘 (`md0` 文件系统)，但默认 TMPDIR 在 `/tmp`（系统盘 overlay 文件系统）。pip 装完把临时 wheel 从 TMPDIR 移到 cache 时会触发 `Invalid cross-device link`（flash-attn 尤其容易撞）。把两者放同一盘就解决。

## Step 2 — 开启 AutoDL 学术加速

访问 GitHub / HuggingFace 必开：

```bash
source /etc/network_turbo
```

> 每次新开 shell 都要 source 一次。如果下载卡住或 connection refused，优先检查这条。

## Step 3 — 拉代码

```bash
cd /root/autodl-tmp
git clone -b blackwell_rtx6000_compat https://github.com/ChrisWu0318/USYD-Capstone-VideoRL USYD-Capstone-VideoRL
cd USYD-Capstone-VideoRL
```

## Step 4 — 验证 GPU / CUDA 工具链

```bash
nvidia-smi               # 驱动 >=570 支持 CUDA 12.8
nvcc --version           # Toolkit 应该是 12.8.x
```

pod 应该出 `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`。

## Step 5 — 装 PyTorch 2.9.1 + cu128（**顺序很重要**）

```bash
pip install --upgrade pip


unset http_proxy https_proxy all_proxy
  pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
  
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
# 期望: 2.9.1+cu128 12.8 (12, 0)
```

如果 `(12, 0)` 出来了说明 torch 认到了 Blackwell。

## Step 6 — 装 r1-v core（不带 dev/eval extras）

```bash
cd /root/autodl-tmp/USYD-Capstone-VideoRL/src/r1-v
pip install -e .
cd -
```

> 不要加 `[dev]` 或 `[eval]` — 里面的 lighteval 依赖有 PEP 508 兼容问题，并且训练用不到。

## Step 7 — 装训练周边依赖

```bash
pip install wandb==0.19.1
pip install tensorboardx
pip install qwen_vl_utils
pip install nltk rouge_score
```

## Step 8 — 装 flash-attn（**关键：用预编译 wheel，不要编译**）

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> 这个 URL 是 torch 2.9 + cu12 + CXX11_ABI=True + Python 3.12 + x86_64 的精确匹配。
> 如果 network_turbo 没开会失败，先 `source /etc/network_turbo`。
>
> **绝对不要**裸跑 `pip install flash-attn`，它会尝试猜 wheel URL 然后 404 / 触发源码编译（15+ 分钟）。

验证：
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
# 期望: 2.8.3
```

## Step 9 — 装 DeepSpeed + bitsandbytes

```bash
pip install --upgrade "deepspeed>=0.16.5" --no-build-isolation
pip install --upgrade "bitsandbytes>=0.45.0"
```

## Step 10 — （可选）vLLM

**建议先跳过**。我们的 smoke test / ablation 默认用 HF 生成（`RESEARCH_USE_VLLM=false`）。

如果后面确实需要 vLLM 加速推理：

```bash
# ⚠️ 这会试图升 torch，装之前先记下 torch 版本
pip install "vllm==0.17.0"   # 0.17 系列最后一个支持 torch 2.9 的

# 装完立刻验证 torch 没被偷偷升到 2.10
python -c "import torch; print(torch.__version__)"
# 必须还是 2.9.1+cu128；如果升了就用 pip install torch==2.9.1 强制降回
```

## Step 11 — 验证整个栈

一次性全跑：

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("SM:", torch.cuda.get_device_capability(0))
print("CXX11_ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)

import flash_attn
print("flash_attn:", flash_attn.__version__)

import deepspeed
print("deepspeed:", deepspeed.__version__)

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)

# bf16 matmul on Blackwell
x = torch.randn(4, 4, device='cuda', dtype=torch.bfloat16)
print("bf16 matmul shape:", (x @ x.T).shape)

# flash-attn actual call
from flash_attn import flash_attn_func
q = k = v = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print("flash_attn_func out shape:", out.shape)

print("\nALL OK")
EOF
```

**期望输出全部正常、最后打印 `ALL OK`**。任何一项报错都停下来修。

## Step 12 — HuggingFace / wandb 登录

```bash
# wandb
wandb login

# HuggingFace（训练脚本里读 model / dataset）
# 新版 CLI 叫 hf，huggingface-cli 已废弃
hf auth login
```

## Step 13 — 下模型和数据集

```bash
# 模型
cd /root/autodl-tmp
mkdir -p models && cd models
# 1) 关掉代理 — hf-mirror 国内直连就是最快的，不需要走 turbo
unset http_proxy https_proxy all_proxy

# 2) 确认 hf_transfer 装了
pip show hf_transfer >/dev/null 2>&1 || pip install hf_transfer

# 3) 显式再走一遍，不依赖 .bashrc
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Video-R1/Qwen2.5-VL-7B-COT-SFT \
  --local-dir ./Qwen2.5-VL-7B-COT-SFT

# 数据集（Smoke test 用 10 条的子集即可）
cd /root/autodl-tmp
mkdir -p datasets && cd datasets
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Video-R1/Video-R1-data --repo-type dataset \
  --local-dir ./Video-R1-data
  
cd /root/autodl-tmp/datasets
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download Video-R1/Video-R1-data --repo-type dataset \
    --local-dir ./Video-R1-data
```

> 如果 hf-mirror 当前很慢，备选走 modelscope（阿里镜像）：
> ```bash
> pip install modelscope
> modelscope download --model Video-R1/Qwen2.5-VL-7B-COT-SFT \
>   --local_dir /root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT
> ```

## Step 14 — Smoke test

```bash
cd /root/autodl-tmp/USYD-Capstone-VideoRL

RESEARCH_CUDA_VISIBLE_DEVICES=0,1 \
RESEARCH_MODEL_NAME_OR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/root/autodl-tmp/datasets/Video-R1-data/smoke_test_10.json \
bash src/scripts/research_branch/smoke_test.sh 2>&1 | tee /root/autodl-tmp/smoke_test.log
```

**成功标准**（必须和 parent branch A100 跑出来的数值一致）：
- `kl ~= 0.0006`（数量级）
- `causal_reward_mean` 数值合理
- `kl_truncation_ratio >= 0`
- `welford_mean ~= 192.0`
- 3 个 smoke step 全部完成，无 OOM

如果任何指标漂移严重 → 依赖升级引入了数值变化，停下来查。

## 踩过的坑（避雷参考）

| 坑 | 症状 | 原因 | 防御 |
|---|---|---|---|
| conda 装的 Python 很卡 | `conda create` 卡十几分钟 | AutoDL 容器 disk IO 瓶颈 | 直接用 pod 自带 miniconda3 Python 3.12，别再 create env |
| flash-attn 编译 `Invalid cross-device link` | pip 装到最后崩 | TMPDIR 在系统盘，PIP_CACHE 在数据盘，两个文件系统 | 把 TMPDIR 也设到 autodl-tmp |
| `nvidia-smi: exit 1` 但其实正常 | setup.sh 开头的 preflight 崩 | `nvidia-smi \| head` 在 `set -o pipefail` 下 SIGPIPE 当失败 | 分开检查 exit code 和输出 |
| lighteval `invalid-egg-fragment` | `pip install -e .[dev]` 崩 | lighteval 用 `#egg=name[extra]` 老语法，新 pip 拒了 | 改成 PEP 508 + 不装 `[dev]` |
| flash-attn `undefined symbol` | import 报 ABI 不匹配 | vllm 0.19 把 torch 从 2.7 升到 2.10，flash-attn wheel 对不上 | 锁 torch 2.9.1，跳过 vllm |
| flash-attn 下 wheel 404 | 编译脚本猜的 URL 不存在 | torch 2.10 没人上传预编译 wheel | 用 torch 2.9，直接指定 wheel URL |
| GitHub / HF 下载 reset | `wget` / `pip` 连接被掐 | AutoDL 默认不走代理 | `source /etc/network_turbo` |

## 故障排查快速索引

- 任何下载失败 → `source /etc/network_turbo`
- 任何 import ABI 报错 → `python -c "import torch; print(torch.__version__)"` 确认还是 2.9.1
- flash-attn import 报错 → 对照 Step 11 重装对应 wheel
- OOM → 首先看是不是算法 bug（步骤是否成功过），再考虑加卡（参考 `blackwell_runbook.md` 的 GPU count 指南）

## Fallback

如果 Blackwell 栈搞不定：`git checkout D4_algorithm_oom_fix_local`，在 RunPod 上用 4× A100 80GB（parent branch 已验证可跑）。
