# RTX Pro 6000 (Blackwell) 适配归档

> **用途**：这是 `blackwell_rtx6000_compat` 分支的完整归档文档。记录我们为适配
> AutoDL 的 RTX Pro 6000 GPU 做的所有工作、踩过的所有坑、以及一次性跑通的配置流程。
>
> **配套文档**：
> - `docs/experiments/autodl_pro6000_setup.md` — 纯复现流程（Step 0–14）
> - `docs/experiments/blackwell_runbook.md` — 依赖 delta + 风险表 + GPU 数量指南
> - 本文档（归档）— 上面两个的 WHY 版本 + 代码改动清单 + 坑表

---

## 1. 背景

### 1.1 为什么要做这个适配

父分支 `D4_algorithm_oom_fix_local` 在 2× A100 80GB 上跑 smoke test，step 1
通过（算法 4D 正则全部 OK），但 step 2 在 video encoder attention 处 OOM。
结论：**算法正确，资源不够**。需要更大显存或更多卡。

RunPod 的 4× A100 80GB 方案可行但成本高。国内 AutoDL 有 **RTX Pro 6000
Blackwell Workstation Edition (96GB GDDR7)** 实例，单卡显存比 A100 80GB 还大，
性价比明显更好。代价是：Blackwell 架构（sm_120）是 2025 年的新硬件，
**PyTorch 2.7 才开始加 sm_120 kernels**，flash-attn / deepspeed / bitsandbytes
都要升级到对应版本。

### 1.2 硬件对比

| 指标 | A100 80GB (父分支目标) | RTX Pro 6000 (本分支) |
|---|---|---|
| 架构 | Ampere (sm_80) | **Blackwell (sm_120)** |
| 显存 | 80 GB HBM2e | **96 GB GDDR7** |
| CUDA cap | 12.x OK | 需要 **CUDA 12.8+** |
| 驱动 | 525+ | **570+** |
| 推荐 torch | 2.5.1 | **2.7+ (我们用 2.9.1)** |

### 1.3 分支关系

```
main
 └── D4_algorithm_oom_fix_local   ← 算法本体 + 所有 4D 正则修复（已验证）
      └── blackwell_rtx6000_compat ← 本分支：仅改依赖栈和安装顺序
```

**关键原则**：算法代码（`grpo.py`、`grpo_trainer.py`、`temporal_mask.py` 等）
**全部不改**，只调依赖版本和文档。如果 Blackwell 搞不定，一键回退到父分支。

---

## 2. 我们针对 Pro 6000 做了什么

### 2.1 依赖版本调整（`src/r1-v/setup.py`）

仅改 pin，不加新依赖：

| Package | 父分支 | 本分支 | 为什么 |
|---|---|---|---|
| torch | >=2.5.1 | **>=2.7.0** | Blackwell sm_120 从 2.7 开始支持 |
| deepspeed | ==0.15.4 | **>=0.16.5** | 0.16.5 开始有 Blackwell CUDA kernels |
| vllm | >=0.8.0 | **>=0.8.5** | 0.8.5 开始自称 Blackwell 兼容（实测仍需跳过，见 §5.6）|
| bitsandbytes | >=0.43.0 | **>=0.45.0** | sm_120 kernels |
| liger_kernel | ==0.5.2 | **>=0.5.6** | 兼容 torch 2.7+ |
| trl | ==0.16.0 | ==0.16.0（**不动**）| 我们改过 GRPOTrainer，升级会打烂 |
| lighteval | 老 egg 语法 | **PEP 508 语法** | 新 pip 拒 `#egg=name[extra]` |

> trl 不升级是硬约束：`src/r1-v/src/open_r1/trainer/grpo_trainer.py` 相对 trl
> 0.16.0 有 ~947 行改动（ZeRO-3 ref model CPU-staging、KL 截断、长度惩罚、
> T-GRPO 等）。升级 trl 等于推倒重来。

### 2.2 `setup.sh` 改动

父分支 setup.sh 基本是裸 `pip install -e .`。本分支做了 6 件事：

1. **固定 torch 版本到 2.9.1+cu128**
   - 不是最新（2.10 才是最新），但 2.10 **没有 flash-attn 预编译 wheel**。
   - 2.9.1 是「有 sm_120 kernels 且有匹配 flash-attn wheel」的最新版本。

2. **直接装 flash-attn 预编译 wheel（不走源码编译）**
   ```bash
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
   ```
   - 这个精确匹配 `torch 2.9 + cu12 + CXX11_ABI=True + cp312 + x86_64`。
   - 避开了源码编译的 15 分钟 + 猜 URL 失败的 404（见 §5.4）。

3. **完全跳过 vllm 安装**
   - vllm 新版本会把 torch 静默升到 2.10，**打破 flash-attn ABI**（见 §5.6）。
   - 训练默认用 HF 生成（`RESEARCH_USE_VLLM=false`），不依赖 vllm。
   - 如果后期确实需要 vllm，用 `pip install "vllm==0.17.0"` 装最后一个兼容 torch 2.9 的版本，装完立即验证 torch 没被升。

4. **TMPDIR 守护（跨文件系统链接保护）**
   ```bash
   if [[ "${PIP_CACHE_DIR:-}" == /root/autodl-tmp/* && -z "${TMPDIR:-}" ]]; then
     export TMPDIR=/root/autodl-tmp/tmp
     mkdir -p "$TMPDIR"
   fi
   ```
   - AutoDL pod 有两块独立文件系统：`/root`（系统盘，overlay fs）和
     `/root/autodl-tmp`（数据盘，md0 fs）。
   - pip 装完会把临时 wheel 从 TMPDIR 移动到 PIP_CACHE_DIR。如果两者不在
     同一 fs，触发 `Invalid cross-device link`（flash-attn 必撞）。

5. **nvidia-smi preflight 防 SIGPIPE 误报**
   - 原本写法 `nvidia-smi | head -n 5` 在 `set -o pipefail` 下，`head` 关闭
     pipe 会让 `nvidia-smi` 收到 SIGPIPE，脚本以为 GPU 异常退出。
   - 改成独立的 `nvidia-smi 2>&1 | head -n 5 || true`。

6. **跳过 `[dev]` / `[eval]` extras**
   - `[eval]` 里的 lighteval 即使修了 PEP 508 语法，也会拖下来一堆评测工具，
     训练阶段用不到。

### 2.3 文档

| 文档 | 作用 |
|---|---|
| `docs/experiments/blackwell_runbook.md` | 依赖 delta 表 + 已知风险 + GPU 数量指南 |
| `docs/experiments/autodl_pro6000_setup.md` | 从零开 pod 到跑 smoke test 的 14 步复现流程 |
| `docs/experiments/pro6000_archive.md`（本文档）| 归档：背景 + 代码改动 + 坑表 |

---

## 3. 关键决策矩阵

所有「为什么选这个版本」一次讲清楚：

| 决策 | 选择 | 拒绝的其他方案 | 理由 |
|---|---|---|---|
| CUDA 工具链 | **cu128** | cu126 / cu121 | Pro 6000 驱动要 12.8+ |
| torch 版本 | **2.9.1** | 2.7.0 / 2.10.0 / nightly | 2.7 早，2.10 没 flash-attn wheel；2.9 是「有 sm_120 kernels + 有 flash-attn wheel」的最新 |
| flash-attn 安装方式 | **预编译 wheel URL** | `pip install flash-attn --no-build-isolation` | 后者会猜 URL 404 或触发 15 分钟源码编译 |
| flash-attn 版本 | **2.8.3** | 2.7.4.post1（初版）| 2.7.4 是最早支持 Blackwell 的；2.8.3 是最新稳定且有 torch2.9 wheel |
| vllm | **跳过**（默认不装）| 装 0.19 / 0.8.5 | 0.19 会偷偷升 torch 到 2.10；0.8.5 在 sm_120 kernel 注册不稳定 |
| Python 版本 | **3.12（pod 自带）** | 用 conda 重建 3.10 / 3.11 | AutoDL 容器 disk IO 瓶颈让 `conda create` 卡很久；3.12 和所有新依赖兼容 |
| GPU 数量 | **≥ 2** | 1 卡 | 1 卡 ZeRO-3 没分片收益；2 卡 96GB 刚够（父分支 2× A100 80GB OOM 的是 step 2，Pro 6000 96GB 有余量） |
| 数据存放 | `/root/autodl-tmp/` | `/root/` | 系统盘仅 30GB，模型+数据集就 100GB+ |
| pip/HF 缓存 | 数据盘 | 默认位置 | 同上；并且 TMPDIR 必须跟 PIP_CACHE_DIR 同 fs |
| 下载镜像 | **hf-mirror.com + hf_transfer** | 官方 HF / AutoDL turbo | 国内直连 hf-mirror 最快；turbo 对 HF 反而更慢 |
| HF CLI | **新版 `hf`** | `huggingface-cli` | 后者 2026 年已彻底废弃 |

---

## 4. 复现流程简述

完整步骤见 `docs/experiments/autodl_pro6000_setup.md`。这里只列骨架：

```
Step 0  开 pod（PyTorch 镜像任意版本，≥2 卡，数据盘 ≥50GB 用于 smoke，≥200GB 用于正式训练）
Step 1  配 .bashrc — HF_HOME / PIP_CACHE_DIR / TMPDIR 全部指向 /root/autodl-tmp
Step 2  source /etc/network_turbo
Step 3  git clone -b blackwell_rtx6000_compat ...
Step 4  nvidia-smi / nvcc --version 确认驱动 570+ / CUDA 12.8
Step 5  pip install torch==2.9.1 torchvision --index-url .../cu128
Step 6  pip install -e src/r1-v （不带 [dev]/[eval]）
Step 7  pip install wandb==0.19.1 tensorboardx qwen_vl_utils nltk rouge_score
Step 8  pip install <flash-attn 2.8.3 wheel URL>
Step 9  pip install "deepspeed>=0.16.5" --no-build-isolation && pip install "bitsandbytes>=0.45.0"
Step 10 （可选，默认跳过）vllm
Step 11 跑 §6 的验证脚本，必须打印 ALL OK
Step 12 wandb login + hf auth login
Step 13 下模型（Video-R1/Qwen2.5-VL-7B-COT-SFT）和数据集（Video-R1/Video-R1-data）
Step 14 跑 smoke test（2 卡就够，3 个 step 必须全部过，指标和父分支对齐）
```

---

## 5. 踩过的坑（每条都是一次 commit）

### 5.1 `conda create` 在 AutoDL 上卡死
**症状**：`conda create -n videorl python=3.11` 卡十几分钟无响应。
**根因**：AutoDL 容器 disk IO 有瓶颈，conda 的 solver 又特别 IO 密集。
**对策**：**不用 conda**。直接用 pod 自带的 miniconda3 Python 3.12，所有依赖往 base 里装。
**Commit**：（文档约定，无代码改动）

### 5.2 lighteval 装不上（老 egg 语法）
**症状**：`pip install -e .[dev]` 报 `invalid-egg-fragment`。
**根因**：原 setup.py 用 `lighteval[math] @ git+...#egg=lighteval[math]` 老语法，新 pip（≥24.0）拒收。
**对策**：改成 PEP 508 语法：
```python
"lighteval[math] @ git+https://github.com/huggingface/lighteval.git@<hash>"
```
并且**训练时不装 `[dev]`**，只装 core：`pip install -e .`。
**Commit**：`8827aaa fix: lighteval PEP 508 syntax + drop dev extras from training install`

### 5.3 `nvidia-smi` preflight 误报失败
**症状**：`setup.sh` 开头 preflight 报 `nvidia-smi failed`，但手动跑完全正常。
**根因**：`nvidia-smi | head -n 5` 在 `set -o pipefail` 下，`head` 关 pipe 触发 nvidia-smi SIGPIPE，退出码被 pipefail 捕捉当失败。
**对策**：`nvidia-smi > /dev/null 2>&1` 先独立检查成功，再 `nvidia-smi 2>&1 | head -n 5 || true` 取信息。
**Commit**：`0eec6ee fix: nvidia-smi SIGPIPE false-positive in setup.sh preflight`

### 5.4 flash-attn 编译 `Invalid cross-device link`
**症状**：`pip install flash-attn --no-build-isolation` 编译末尾崩，报跨设备链接。
**根因**：PIP_CACHE_DIR 在 `/root/autodl-tmp`（md0 fs），TMPDIR 默认在 `/tmp`（overlay fs），pip 收尾 `mv` wheel 时两个 fs 不能硬链接。
**对策**：`setup.sh` 检测到 PIP_CACHE_DIR 在数据盘时，自动把 TMPDIR 也设到数据盘。
**Commit**：`7832f0d fix: pin TMPDIR to data disk on AutoDL-style pods`

### 5.5 flash-attn wheel URL 404
**症状**：`pip install flash-attn` 走 `--no-build-isolation`，日志里打印 `Guessing wheel URL: .../flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`，然后 404。
**根因**：当前 torch 是 2.10，但 flash-attn release 里**只有 torch 2.4–2.9 的 wheel**，没人上传 2.10。
**对策**：**把 torch 锁到 2.9.1**，然后直接给 pip 完整的 wheel URL，不让它自己猜。
**Commit**：`482fc3f docs+fix: clean Blackwell Pro 6000 setup — pin torch 2.9.1, skip vllm`

### 5.6 flash-attn import `undefined symbol`
**症状**：装完 flash-attn 能 import torch，但 `import flash_attn` 报 `undefined symbol: _ZN3c104impl...`（C++ ABI 不匹配）。
**根因**：装 vllm 时它偷偷把 torch 2.7/2.9 升到 2.10，flash-attn wheel 的 C++ ABI 对不上新 torch。
**对策**：
1. 默认**完全跳过 vllm**。
2. 训练脚本里设 `RESEARCH_USE_VLLM=false`，走 HF 生成。
3. 如真要装 vllm，用 `vllm==0.17.0`（最后一个兼容 torch 2.9 的），装完立即 `python -c "import torch; print(torch.__version__)"` 核对。
**Commit**：`482fc3f`

### 5.7 GitHub / HuggingFace 下载反复 reset
**症状**：`git clone` 或 `pip install` 从 github 拉东西时连接被掐。
**根因**：AutoDL 默认出口不通 github/HF，需要走学术加速。
**对策**：每次新开 shell `source /etc/network_turbo`。
**注意**：**HF 下载时反而要关 turbo**，直接走 hf-mirror 更快：
```bash
unset http_proxy https_proxy all_proxy
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 hf download ...
```

### 5.8 PyTorch 下载 14 kB/s
**症状**：`pip install torch --index-url .../cu128` 预计 14 小时。
**根因**：turbo 代理对 PyTorch CDN 不友好。
**对策**：装 torch 时 `unset http_proxy https_proxy all_proxy` 走直连；如果仍慢换 SJTU 镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/`。

### 5.9 `huggingface-cli` 报 deprecated
**症状**：`huggingface-cli login` 打印 `deprecated and no longer works. Use hf instead.`
**对策**：全部换成新 CLI：
- `hf auth login`（原 `huggingface-cli login`）
- `hf download <repo> --local-dir <dir>`（原 `huggingface-cli download`，注意新 CLI **不再有 `--local-dir-use-symlinks`** 参数）

### 5.10 Pod ID 隔会儿变一次
**症状**：shell prompt hash 从 `afd3cc83` 变成 `7a5c3feb`，以为断了。
**根因**：AutoDL 重连/重启容器会换 hostname，**但数据盘内容保留**。
**对策**：新 shell 需要重新 `source /etc/network_turbo` 和 `source ~/.bashrc`，**不需要重装依赖或重下数据**。

### 5.11 Video-R1-data 数据集 120GB
**症状**：hf download 下了 2 小时还在下，怀疑是不是下错了东西。
**根因**：Video-R1 官方数据集就是 120GB，包含 260k RL 样本 + 165k SFT 样本 + 全量视频。
**对策**：这就是完整数据集，下完不需要再下其他的；smoke test 只用里面的 `smoke_test_10.json`。

---

## 6. 下载 + 解压 + 形成可训练数据集

### 6.1 下载模型

```bash
cd /root/autodl-tmp
mkdir -p models && cd models

unset http_proxy https_proxy all_proxy   # hf-mirror 直连最快
pip show hf_transfer >/dev/null 2>&1 || pip install hf_transfer

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Video-R1/Qwen2.5-VL-7B-COT-SFT \
  --local-dir ./Qwen2.5-VL-7B-COT-SFT
```

### 6.2 下载数据集（完整 120GB）

```bash
cd /root/autodl-tmp
mkdir -p datasets && cd datasets

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Video-R1/Video-R1-data --repo-type dataset \
  --local-dir ./Video-R1-data
```

下完后 `Video-R1-data/` 目录里的结构（大致）：

```
Video-R1-data/
├── Video-R1-260k.json          # RL 训练索引（26 万条）
├── Video-R1-COT-165k.json      # SFT 冷启动（我们不用）
├── smoke_test_10.json          # 10 条 smoke 子集
├── <类别>/*.zip                # 视频文件被按类别打包成 zip
└── ...
```

### 6.3 解压视频 zip（**训练前必须做**）

仓库自带 `src/unzip.py`，会**递归**遍历 `Video-R1-data/` 下所有子目录，把每个
`.zip` 就地解压到它所在的目录（不会移动文件）。

**关键**：这个脚本里**硬编码了相对路径 `./src/r1-v/Video-R1-data`**，所以
**必须从仓库根目录跑**，并且数据集要在 `src/r1-v/Video-R1-data/` 位置。

我们的数据实际放在 `/root/autodl-tmp/datasets/Video-R1-data/`，有两种处理方式：

**方式 A（推荐）：软链到仓库里**

```bash
cd /root/autodl-tmp/USYD-Capstone-VideoRL
ln -s /root/autodl-tmp/datasets/Video-R1-data src/r1-v/Video-R1-data
python ./src/unzip.py
```

> 软链不占盘，解压结果在数据盘上，不会把系统盘撑爆。

**方式 B：直接改脚本跑**

```bash
cd /root/autodl-tmp/USYD-Capstone-VideoRL
python - <<'EOF'
import os, zipfile
root = "/root/autodl-tmp/datasets/Video-R1-data"
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(".zip"):
            p = os.path.join(dp, f)
            print("Extracting:", p)
            try:
                with zipfile.ZipFile(p) as z:
                    z.extractall(dp)
                print("OK:", p)
            except Exception as e:
                print("FAIL:", p, e)
EOF
```

### 6.4 解压后的磁盘占用

- 下载态：~120 GB（zip 压缩态）
- 解压后：**再增加 ~100–150 GB**（视频文件是主要体积）
- **所以数据盘建议 ≥ 300 GB**（110 模型 + 120 zip + 150 解压 = 380GB 峰值；解压后可删 zip 回到 ~260GB）

**解压完可选清理 zip 腾空间**（会失去「重解压」能力，谨慎）：

```bash
find /root/autodl-tmp/datasets/Video-R1-data -name "*.zip" -delete
```

### 6.5 训练脚本如何引用数据集

smoke / budget / ablation / formal 全部都只需要指两个路径：

```bash
RESEARCH_MODEL_NAME_OR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT
RESEARCH_DATASET_NAME=/root/autodl-tmp/datasets/Video-R1-data/<索引.json>
```

| 阶段 | 用的 JSON |
|---|---|
| Smoke | `smoke_test_10.json` |
| Budget probe | `Video-R1-260k.json`（从里面采样 20 step） |
| Ablation | `Video-R1-260k.json` |
| Formal full | `Video-R1-260k.json` |

索引 JSON 里的视频路径是**相对** `Video-R1-data/` 的，所以解压步骤完成后，
训练代码走 `dirname(dataset_name)` + 相对路径就能找到视频。

### 6.6 下载失败怎么办

1. hf-mirror 当前拥堵 → 换 modelscope：
   ```bash
   pip install modelscope
   modelscope download --model Video-R1/Qwen2.5-VL-7B-COT-SFT \
     --local_dir /root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT
   ```
2. 断线重连 → `hf download` 会自动续传，直接重跑同一条命令即可。
3. 只想要 smoke 能跑 → 用 `--include` 过滤：
   ```bash
   hf download Video-R1/Video-R1-data --repo-type dataset \
     --local-dir ./Video-R1-data \
     --include "smoke_test_10.json" "<smoke 用到的 10 个视频 zip>"
   ```
   但这要求你已经看过 `smoke_test_10.json` 里引用的视频清单，一般不推荐。

---

## 7. 一次性验证脚本

装完所有依赖后，跑这段确认栈完整：

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

# Blackwell bf16 matmul
x = torch.randn(4, 4, device='cuda', dtype=torch.bfloat16)
print("bf16 matmul shape:", (x @ x.T).shape)

# flash-attn actual call (抓 ABI 不匹配)
from flash_attn import flash_attn_func
q = k = v = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print("flash_attn_func out shape:", out.shape)

print("\nALL OK")
EOF
```

**期望输出**：
```
torch: 2.9.1+cu128
cuda: 12.8
SM: (12, 0)
CXX11_ABI: True
flash_attn: 2.8.3
deepspeed: 0.16.x
bitsandbytes: 0.45.x
bf16 matmul shape: torch.Size([4, 4])
flash_attn_func out shape: torch.Size([1, 16, 8, 64])

ALL OK
```

任何一项出错都停下修，不要带着伤跑训练。

---

## 8. Smoke test 成功标准

smoke test 必须跟父分支 A100 跑出来的指标对齐（数量级一致即可）：

| 指标 | 期望值 | 含义 |
|---|---|---|
| `kl` | ~0.0006 | ref model forward 正常 |
| `causal_reward_mean` | 非 NaN，数值合理 | D1 因果奖励 OK |
| `kl_truncation_ratio` | ≥ 0 | D2 KL 截断 OK |
| `welford_mean` | ~192.0 | D3 Welford 统计 OK |
| `length_penalty_mean` | ~-0.004 | D3 长度惩罚 OK |
| `temporal_rewards` | 1.0 | T-GRPO OK |
| OOM | 无 | Pro 6000 96GB 必须够（父分支 2× A100 80GB 在 step 2 video encoder OOM，这里有 32GB 余量） |
| 3 个 step | 全部完成 | 算法没回归 |

如果任何指标漂移一个数量级以上 → **停下**，依赖升级引入了数值变化，要回头查。

---

## 9. 实验路线图（2026-04-18 预算优化版）

**硬件前提**：3× RTX Pro 6000（96 GB, sm_120, CUDA 12.8），AutoDL 计费
5.98 RMB/卡·小时 → 整机 17.94 RMB/h。

**验证基线**（2026-04-18 budget_probe 实测，3 卡 ZeRO-3，filtered 8220 样本）：
- step 时间 ~70 s（含 I/O）/ ~63 s（纯 compute），稳定无漂移
- 峰值显存 84.7 GB（< 96 GB，留 11 GB headroom）
- KL 0.0005 → 0.0013 单调增长，无爆炸
- D1/D2/D3 三路都在按设计开/关，reward 上行，grad_norm 2–5 稳定

### 阶段 1：Baseline gate（必跑）

| 项 | 值 |
|---|---|
| 脚本 | `ablation_baseline.sh`（`ablation_baseline.yaml`，D1/D2/D3 全关） |
| 步数 | `RESEARCH_MAX_STEPS=200` |
| 耗时 | ~3.9 h（200 × 70s） |
| 成本 | ~70 RMB |
| 验收 | loss 下降、reward 上行、显存稳、无 NaN/OOM；200 步末尾 checkpoint 可评测 |
| 放行条件 | baseline 正常 → 进入阶段 2；异常 → 回查再跑，不动正则项 |

### 阶段 2：4 组 ablation 串行（gate 过后再跑）

| 组 | 脚本 | YAML | 预计耗时 | 成本 |
|---|---|---|---|---|
| D1 only | `ablation_d1_only.sh` | `ablation_d1.yaml` | ~3.9 h | ~70 RMB |
| D2 only | `ablation_d2_only.sh` | `ablation_d2.yaml` | ~3.9 h | ~70 RMB |
| D3 only | `ablation_d3_only.sh` | `ablation_d3.yaml` | ~3.9 h | ~70 RMB |
| D1+D2+D3 | `ablation_d1_d2_d3.sh` | `ablation_d1_d2_d3.yaml` | ~3.9 h | ~70 RMB |

**均统一** `RESEARCH_MAX_STEPS=200`，`RESEARCH_SAVE_STEPS=100`，`RESEARCH_REPORT_TO=wandb`。

### 阶段 3（可选，仅为论文补图）：pair-wise × 3

`ablation_d1_d2` / `ablation_d1_d3` / `ablation_d2_d3`，每组 ~3.9 h / ~70 RMB。
若阶段 2 四条曲线已能支撑核心结论（additive / best component），**跳过**。

### 总预算

| 方案 | 总时长 | 总成本 |
|---|---|---|
| 最小闭环（阶段 1 + 2 = 5 runs × 200 步） | ~19.5 h | **~350 RMB** |
| 含 pair-wise（+3 runs） | ~31 h | ~560 RMB |
| 原计划（8 × 300 + formal 1200） | ~77 h | ~1380 RMB |

**正式 full run 暂不列入** — 200 步 ablation 结果收敛后再决定是否补 1200 步
formal（单次 ~23 h / ~415 RMB）。

### 运行示例

```bash
# 阶段 1：baseline
RESEARCH_CUDA_VISIBLE_DEVICES=0,1,2 \
RESEARCH_MAX_STEPS=200 \
RESEARCH_SAVE_STEPS=100 \
RESEARCH_REPORT_TO=wandb \
RESEARCH_MODEL_NAME_OR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.filtered.json \
bash src/scripts/research_branch/ablation_baseline.sh 2>&1 | tee /root/autodl-tmp/logs/baseline.log
```

GPU 数量由 `RESEARCH_CUDA_VISIBLE_DEVICES` 自动推导（`_common.sh` 里
`resolve_nproc_per_node`），2 卡 / 3 卡 / 4 卡不用改脚本。

---

## 10. Fallback

如果 Blackwell 栈出现无法解决的回归（比如 smoke test 指标严重漂移）：

```bash
git checkout D4_algorithm_oom_fix_local
# 在 RunPod 上开 4× A100 80GB，按父分支 README 的流程跑
```

父分支算法已经验证通过，只是需要更多卡。这是稳妥退路。

---

## 11. Commits 清单（本分支相对父分支）

```
482fc3f docs+fix: clean Blackwell Pro 6000 setup — pin torch 2.9.1, skip vllm
7832f0d fix: pin TMPDIR to data disk on AutoDL-style pods
8827aaa fix: lighteval PEP 508 syntax + drop dev extras from training install
0eec6ee fix: nvidia-smi SIGPIPE false-positive in setup.sh preflight
0aaa287 feat: Blackwell (RTX Pro 6000, sm_120) compat — bump deps + install order
```

文件改动范围（5 个 commit 总计）：

```
docs/experiments/autodl_pro6000_setup.md    （新增，复现流程）
docs/experiments/blackwell_runbook.md       （新增，依赖 delta + 风险）
docs/experiments/pro6000_archive.md         （新增，本文档）
setup.sh                                    （改装机脚本）
src/r1-v/setup.py                           （改依赖 pin）
```

算法代码（`grpo.py` / `grpo_trainer.py` / `temporal_mask.py` / `research_logic.py`
/ `experiment_config.py` 等）**全部未改动**，与父分支完全一致。

---

## 12. AutoDL Pod 迁移 & budget_probe 期间的新踩坑（2026-04-18）

从 2× → 3× Pro 6000 换 pod 后新 surface 的问题，按排查顺序记录。每条都含
根因和修复动作，方便下次迁机直接照做。

### 12.1 DeepSpeed 0.18.9 在 sm_120 上 muon kernel 编译失败
**症状**：`setup.sh` 装完后 import deepspeed 段错误；trace 指向 muon CUDA 扩展。
**根因**：0.18.x 把 muon optimizer 的 CUDA 源码标为必编译，对 sm_120
arch flag 处理有 bug。
**修复**：固定到 `deepspeed==0.16.9`（仍有 sm_120 kernels，还没引入 muon 必编路径）：
```bash
pip install "deepspeed==0.16.9" --no-build-isolation --force-reinstall --no-deps
```
**后续**：`setup.sh` 的 `"deepspeed>=0.16.5"` 上限需要在下一个 commit 改成
`>=0.16.5,<0.17`，避免 pip 自动拉到 0.18。

### 12.2 `pip install --force-reinstall` 不带 `--no-deps` 会把 torch 升到 2.11
**症状**：重装某个包后，`python -c "import torch; print(torch.__version__)"`
显示 2.11 —— flash-attn 2.8.3 wheel 直接 ABI mismatch。
**根因**：很多上游包（deepspeed、bitsandbytes）声明 `torch`（无上限）。
`--force-reinstall` 会重新解依赖把 torch 也升了。
**修复**：任何单包 reinstall 必须带 `--no-deps`。需要真的补依赖时，先用
`pip check` 看谁缺，再定点装。

### 12.3 `WANDB_DISABLED=true` 触发 WandbCallback 的坏路径
**症状**：设 `WANDB_DISABLED=true` 启动后仍然报 wandb 相关 TypeError。
**根因**：transformers 的 WandbCallback 读取 `WANDB_DISABLED` 的逻辑在
4.51.3 上有 bug，即使禁用仍然尝试 import/初始化。
**修复**：改用 `WANDB_MODE=disabled`（wandb 原生支持的变量，走 wandb 自己的
noop 后端，不经过 WandbCallback 的禁用分支）。

### 12.4 transformers 4.51.3 把 `Trainer._save_checkpoint(metrics=...)` 删了
**症状**：`budget_probe` 跑到 `save_steps` 时 `TypeError: _save_checkpoint()
got an unexpected keyword argument 'metrics'`。
**根因**：父分支按旧签名调 `super()._save_checkpoint(model, trial, metrics=metrics)`；
4.51.3 签名变成 `(self, model, trial)`。
**修复**：`src/r1-v/src/open_r1/trainer/grpo_trainer.py:538` 去掉 `metrics` kwarg：
```python
super()._save_checkpoint(model, trial)
```
已提交（`7068af3`）。

### 12.5 数据根目录是硬编码的，symlink 必须放在 `src/r1-v/Video-R1-data`
**症状**：数据集 JSON 路径正确、文件也能打开，但 `Dataset.__getitem__`
里解析出的视频路径 FileNotFound。
**根因**：`grpo_trainer.py:586` 用
`os.path.join(os.getcwd(), "Video-R1-data", sample["path"].lstrip("/"))`
拼路径，而 `_common.sh:206` 在启动前 `cd src/r1-v`。所以数据根**必须**
出现在 `src/r1-v/Video-R1-data/`，仓库根放 symlink 没用。
**修复**：
```bash
ln -sfn /root/autodl-tmp/datasets/Video-R1-data \
        /root/autodl-tmp/USYD-Capstone-VideoRL/src/r1-v/Video-R1-data
```

### 12.6 Video-R1-260k 97% 媒体缺失 → ZeRO-3 grad size mismatch
**症状**：symlink 对了以后立刻报
`RuntimeError: 0 != 1505280 Cannot reduce scatter gradients whose size is
not same as the params`，发生在 step 0 的 reduce_scatter。
**根因**：per_device_batch_size=1，某些 rank 抽到的样本视频文件 0 字节或缺失，
dataloader 返回空 tensor → forward 出 0-sized grad → ZeRO-3 和其他 rank 的
参数 size mismatch。我们下载的 260k 数据集只补了 CLEVRER，其余
（Math / NeXT-QA / Knowledge / ArxivQA / LLaVA）大部分子目录是空的。
**修复**：启动前**预过滤** JSON，只保留媒体文件存在且非空的样本：
```python
import json, os
src = "/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.json"
dst = "/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.filtered.json"
root = "/root/autodl-tmp/datasets/Video-R1-data"
with open(src) as f: data = json.load(f)
kept = []
for item in data:
    p = item.get("path", "").lstrip("./")
    full = os.path.join(root, p)
    if os.path.isfile(full) and os.path.getsize(full) > 0:
        kept.append(item)
with open(dst, "w") as f: json.dump(kept, f)
print(f"kept: {len(kept)} / {len(data)}")
```
结果：8220 / 263071（3.1%）。启动时 `RESEARCH_DATASET_NAME=...filtered.json`。

### 12.7 wandb API key 新格式（`wandb_v1_...`, 86 字符）需要 wandb ≥ 0.20
**症状**：`wandb login` 报 `API key must be 40 characters long, yours was 86`。
**根因**：2026 年 wandb 切到新 token 格式，前缀 `wandb_v1_`，总长 86。
老 wandb 0.19.1 仍然按旧 40 字符校验。
**修复**：
```bash
pip install --upgrade "wandb>=0.20.0"
# 或直接设环境变量绕开 login 校验
export WANDB_API_KEY=wandb_v1_xxxxxxxxxxxx...
```

### 12.8 AutoDL pod 迁移会清空 `/root/autodl-tmp/` 的大文件
**症状**：从旧 pod 克隆 image 到新 pod 后，模型权重和 wandb 缓存不见。
**根因**：AutoDL 只保留容器的 image 层，`autodl-tmp` 是数据盘，换 pod
就是新盘。
**修复**：迁 pod checklist：
1. 重新下载模型权重（走 modelscope / 本地冷备）
2. 重新 `wandb login`
3. 重建数据 symlink（见 12.5）
4. 跑 smoke 之前先 `python -c "import torch, flash_attn, deepspeed; ..."`
   确认依赖还在

### 12.9 budget_probe 验证通过的基线数字（供后续对比）

3× Pro 6000，ZeRO-3，filtered 8220 样本，20 步实测：

| 指标 | 值 | 备注 |
|---|---|---|
| step 时间（end-to-end） | ~70 s | 含 dataloader I/O |
| step 时间（compute-only） | ~63 s | trainer 内部计时 |
| 峰值显存 | 84.7 GB / 96 GB | 11 GB headroom |
| KL | 0.0005 → 0.0013 | 单调，无爆炸 |
| welford_mean | 197 → 225 | 在线统计收敛 |
| reward_mean | 缓慢上行 | 正常 |
| grad_norm | 2–5 | 稳定 |
| D1/D2/D3 | 全部按配置 fire | 算法分支正确 |
