# Pod 迁移一键复现指南（懒人版）

> 目标：在一个新 AutoDL Pod 上从零开始，按此文档从上到下执行，**一次过**跑起
> baseline ablation，不再踩之前踩过的所有坑。
>
> 踩坑全集见 `pro6000_archive.md §12`，这里只列操作。
>
> 适用：`blackwell_rtx6000_compat` 分支，3× 或 4× RTX Pro 6000（Blackwell
> sm_120, CUDA 12.8）。

---

## 0. 开 Pod 时的选项

| 字段 | 选什么 |
|---|---|
| GPU | RTX Pro 6000 × 3（最低）/ × 4（推荐） |
| 镜像 | PyTorch ≥ 2.1 / CUDA ≥ 12.1 的任何官方镜像（我们会重装 torch 2.9.1） |
| 数据盘 | ≥ 200 GB（模型 ~16 GB + 数据集 ~80 GB + checkpoint ~30 GB） |

---

## 1. 依赖安装（一次过，~20 min）

```bash
cd /root/autodl-tmp
git clone <repo-url> USYD-Capstone-VideoRL
cd USYD-Capstone-VideoRL
git checkout blackwell_rtx6000_compat

# 主安装脚本
bash setup.sh

# ⚠️ setup.sh 里的 deepspeed pin 太宽，会拉到 0.18.9（sm_120 muon kernel 编译崩）
# 必须手动降到 0.16.9，注意三个关键 flag：
pip install "deepspeed==0.16.9" \
    --no-build-isolation --no-deps --force-reinstall

# ⚠️ wandb 新 API key（wandb_v1_ 开头，86 字符）需要 0.20+ 才能识别
pip install --upgrade "wandb>=0.20.0"
```

**立刻验证**（任何一项不对就停下）：
```bash
python -c "
import torch, flash_attn, deepspeed, wandb, transformers
print('torch          ', torch.__version__)            # 2.9.1+cu128
print('device cap     ', torch.cuda.get_device_capability(0))  # (12, 0)
print('flash_attn     ', flash_attn.__version__)       # 2.8.3
print('deepspeed      ', deepspeed.__version__)        # 0.16.9
print('wandb          ', wandb.__version__)            # 0.20.x+
print('transformers   ', transformers.__version__)     # 4.51.3
"
```

---

## 2. 模型权重

```bash
mkdir -p /root/autodl-tmp/models
# Qwen2.5-VL-7B-COT-SFT（~16 GB），走 modelscope / 本地冷备都行
# 目标路径：/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT
```

---

## 3. 数据集（**坑点最多的一步**）

Video-R1-260k 官方数据集只有 CLEVRER 子集可免费下载完整，其它
（Math / NeXT-QA / Knowledge / ArxivQA / LLaVA）大部分子目录是空的。
**直接用原 JSON 启动 → ZeRO-3 grad size mismatch 必报错**。

### 3.1 下载 CLEVRER 子集

```bash
mkdir -p /root/autodl-tmp/datasets/Video-R1-data
cd /root/autodl-tmp/datasets/Video-R1-data
# 下载 CLEVRER_part1.zip / CLEVRER_part2.zip / Video-R1-260k.json
# 走 modelscope 或本地冷备
unzip CLEVRER_part1.zip && unzip CLEVRER_part2.zip
```

### 3.2 预过滤 JSON（只保留文件存在且非空的样本）

```bash
python - <<'EOF'
import json, os
src  = "/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.json"
dst  = "/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.filtered.json"
root = "/root/autodl-tmp/datasets/Video-R1-data"
data = json.load(open(src))
def ok(item):
    p = item.get("path", "").lstrip("./")
    full = os.path.join(root, p)
    return os.path.isfile(full) and os.path.getsize(full) > 0
kept = [x for x in data if ok(x)]
json.dump(kept, open(dst, "w"))
print(f"kept: {len(kept)} / {len(data)}")
EOF
# 只装了 CLEVRER 的情况下：kept: ~8200 / 263071
```

**后续用 `...filtered.json` 这个**，不是原版。

---

## 4. 数据 symlink（**最容易漏的一步**）

`grpo_trainer.py:586` 用 `os.getcwd() + "/Video-R1-data"` 拼数据路径，而
`_common.sh:206` 会 `cd src/r1-v/`。所以 symlink **必须**放在 `src/r1-v/` 下：

```bash
ln -sfn /root/autodl-tmp/datasets/Video-R1-data \
        /root/autodl-tmp/USYD-Capstone-VideoRL/src/r1-v/Video-R1-data

# 验证
ls /root/autodl-tmp/USYD-Capstone-VideoRL/src/r1-v/Video-R1-data/CLEVRER/
# 应该能看到 CLEVRER_part1.zip / train_videos/ 等
```

放在仓库根、放在 `/root/autodl-tmp/` 下都**没用**。

---

## 5. wandb 登录

```bash
# 从 wandb.ai/authorize 复制新格式 key（wandb_v1_... 86 字符）
export WANDB_API_KEY=wandb_v1_xxxxxxxxxxxxxxxxxxxxxxxx...
# 或
wandb login  # 走 0.20 的交互式登录
```

**不要**用 `WANDB_DISABLED=true` 关闭 wandb，transformers 4.51.3 的
WandbCallback 在这条路径上有 bug。需要关就用 `WANDB_MODE=disabled`。

---

## 6. Smoke 验证（5 min）

```bash
cd /root/autodl-tmp/USYD-Capstone-VideoRL
mkdir -p /root/autodl-tmp/logs

RESEARCH_CUDA_VISIBLE_DEVICES=0,1,2 \
RESEARCH_MODEL_NAME_OR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.filtered.json \
bash src/scripts/research_branch/smoke_test.sh 2>&1 | tee /root/autodl-tmp/logs/smoke.log
```

**通过标准**（见 `pro6000_archive.md §8`）：
- step 1 `kl` ≈ 5e-4
- step 1 `welford_mean` ≈ 192
- 3 步全部完成，无 OOM

---

## 7. 正式跑 baseline ablation（200 步，~3.9 h，~70 RMB）

```bash
RESEARCH_CUDA_VISIBLE_DEVICES=0,1,2 \
RESEARCH_MAX_STEPS=200 \
RESEARCH_SAVE_STEPS=100 \
RESEARCH_REPORT_TO=wandb \
RESEARCH_MODEL_NAME_OR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-7B-COT-SFT \
RESEARCH_DATASET_NAME=/root/autodl-tmp/datasets/Video-R1-data/Video-R1-260k.filtered.json \
bash src/scripts/research_branch/ablation_baseline.sh 2>&1 | tee /root/autodl-tmp/logs/baseline.log
```

**健康指标**（wandb 上看）：
- `cuda_max_memory_allocated_gb`：~80-85（< 96 GB 即 OK）
- `step_compute_time_sec`：~65 s
- `reward`：上行（1.5 → 1.9+）
- `kl`：缓慢上涨，不爆炸（baseline 无 D2 截断，到 step 200 估计 0.02-0.04）

## 8. 续跑（如果 200 步不够）

```bash
# 把 max_steps 改成新的总步数（不是增量）；checkpoint 路径指老目录
RESEARCH_MAX_STEPS=400 \
... 同上环境变量 ... \
bash src/scripts/research_branch/ablation_baseline.sh \
  --resume_from_checkpoint \
  /root/autodl-tmp/USYD-Capstone-VideoRL/src/r1-v/log/research_branch/ablation/ablation_baseline-<commit>-<timestamp>/checkpoint-200
```

Welford state / optimizer / scheduler / RNG 全部会恢复，scheduler 会按新的
max_steps 重新归一化 cosine 曲线。

---

## 故障速查表

| 症状 | 根因 | 修 |
|---|---|---|
| step 0 `0 != N Cannot reduce scatter gradients...` | 数据文件缺失 / symlink 位置错 | §3 过滤 + §4 symlink |
| `_save_checkpoint() got unexpected keyword 'metrics'` | transformers 4.51.3 API drift | 已在分支 patch（commit `7068af3`） |
| `API key must be 40 characters, yours was 86` | wandb 0.19 认不了新 key | `pip install -U "wandb>=0.20"` |
| `nvidia-smi 失败` / deepspeed import 段错误 | deepspeed 0.18.9 的 sm_120 muon bug | §1 降到 0.16.9 |
| 装完别的包后 torch 变 2.11 / 2.10 | 上游依赖没 pin torch 上限 | 所有 `--force-reinstall` 必带 `--no-deps` |
| `WANDB_DISABLED=true` 还是 TypeError | transformers WandbCallback bug | 改用 `WANDB_MODE=disabled` |
| 换 pod 后模型 / 数据没了 | AutoDL `autodl-tmp` 是数据盘，换 pod = 新盘 | 重跑 §1-§5 |

---

## TL;DR — 能背下来的 6 条

1. `deepspeed==0.16.9`，**不要** 0.18.x
2. `wandb>=0.20`，新 key 86 字符
3. 数据集必须预过滤（~8000 / 263000 可用）
4. symlink 必须在 `src/r1-v/Video-R1-data`
5. `--force-reinstall` 必带 `--no-deps`
6. 关 wandb 用 `WANDB_MODE=disabled`，不用 `WANDB_DISABLED=true`
