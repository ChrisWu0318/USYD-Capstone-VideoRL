# OOM 修复与实验规划报告

**项目**: CS41 Enhanced T-GRPO for Video Temporal Reasoning  
**日期**: 2026-04-13  
**分支**: D4_algorithm (commit 93b9c15), D4_algorithm_oom_fix (commit 57eaf1c)  

---

## 第一部分: OOM 根因分析与修复

### 1.1 OOM 根本原因

OOM 不是某个 bug, 而是**结构性的内存赤字**.

GRPO 一次 training step 需要的 forward pass:

```
(1) Policy model generate        -> KV cache + 生成激活
(2) Policy model forward          -> per_token_logps (带梯度的完整激活)
(3) Ref model forward            -> ref_per_token_logps (又一整套完整激活)
(4) D1 masked forward (如果开启)  -> 又一整套完整激活, mask 掉一半帧再跑
(5) Shuffled temporal generation -> 额外的一整套 generate + reward
```

在 7B bf16 模型 + 4x A100-80GB (320GB 总 VRAM) 下:

| 阶段 | 峰值占用 (单 GPU) | 说明 |
|------|-------------------|------|
| 模型分片 (ZeRO-3) | ~4GB | 按需 gather, 非常驻 |
| ref_model 常驻 | ~14GB | ZeRO-3 不管 ref_model, 加载时单 GPU 吃满 |
| CUDA context + NCCL | ~2GB | 框架开销 |
| Policy forward 激活 | ~8-12GB | 正比于序列长度 |
| Ref forward 激活 | ~14-20GB | 和 policy 同等规模 |
| D1 masked 激活 | ~8-12GB | 和 policy forward 规模相当 |
| KV cache (4 gen) | ~4-6GB | 生成阶段 |
| 碎片化开销 | ~5-17GB | 大 tensor 分配不可避免 |
| **总峰值** | **~60-87GB** | 超过 80GB 就 OOM |

原始论文声明需要 **5x A100-80GB (400GB) 或 4x H20-96GB (384GB)**.  
我们只有 4x A100-80GB (320GB), 且多了 D1 masked forward, 压力更大.

### 1.2 OOM 触发点

**触发点 1: ref_model 初始化 (Step 0)**

`from_pretrained(model_id, **model_init_kwargs)` 会把完整的 7B 模型
(~14GB bf16) 先加载到默认 GPU 上, DeepSpeed ZeRO-3 的分片还没开始.
这一瞬间某张 GPU 峰值 +14GB, 直接触发 OOM.

**触发点 2: D1 masked forward (Step N, N>=1)**

D1 需要 mask 掉一半帧再跑一次 forward. 此时 policy 模型的激活
(per_token_logps) 还占着内存, 又要额外跑一次 forward, 峰值翻倍.

**触发点 3: KL 计算阶段**

`per_token_logps` 和 `ref_per_token_logps` 两个大 tensor 同时驻留.
如果不及时释放 ref 侧的 tensor, 后续 D1 计算和 backward 的内存
空间不够.

### 1.3 修复方案

#### Fix 1: ref_model CPU offload (最关键)

**之前 (错误):**
```python
self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, **model_init_kwargs
)
```

**D4_algorithm_oom_fix 分支 (也有问题):**
```python
ref_init_kwargs["device_map"] = "cpu"  # ZeRO-3 下会报 ValueError!
```

**现在 (正确):**
```python
ref_init_kwargs = dict(model_init_kwargs)
ref_init_kwargs["low_cpu_mem_usage"] = False
ref_init_kwargs.pop("device_map", None)   # ZeRO-3 不兼容 device_map
ref_init_kwargs.pop("use_cache", None)
self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, **ref_init_kwargs
)
self.ref_model = self.ref_model.cpu()  # 手动搬回 CPU
```

**节省: ~14GB 常驻 GPU 内存**

关键点:
- `device_map="cpu"` 在 ZeRO-3 下会抛 `ValueError: DeepSpeed Zero-3
  is not compatible with passing a device_map.`
- 正确做法是 `low_cpu_mem_usage=False` + 加载后手动 `.cpu()`
- `prepare_deepspeed()` 之后会正确地分片管理 ref_model

#### Fix 2: D1 masked forward 后清理

```python
finally:
    # [OOM-FIX] Free masked activation memory before moving on
    del masked_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
```

**节省: ~8-12GB 峰值** (D1 的 masked 激活在 ref forward 之前释放)

#### Fix 3: KL 计算后清理

```python
per_token_kl = torch.exp(x_clamped) - x_clamped - 1

# [OOM-FIX] Free large intermediate tensors after KL is computed.
# Do NOT delete per_token_logps — it is needed later for:
#   1. D1 causal reward: log_P1 = (per_token_logps[i] * completion_mask[i]).sum()
#   2. Policy loss: per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
del ref_per_token_logps, x_clamped
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**节省: ~5-8GB 峰值**

关键点: `per_token_logps` 绝对不能删! D1 和 policy loss 都需要它.
D4_algorithm_oom_fix 的旧代码错误地删了 `per_token_logps`, 会导致
D1 计算时 crash 或 loss 计算报错.

### 1.4 之前 OOM fix 的两个 Bug

| Bug | 分支 | 影响 |
|-----|------|------|
| `device_map="cpu"` | D4_algorithm_oom_fix | ZeRO-3 下抛 ValueError, 训练根本跑不起来 |
| `del per_token_logps` (KL 后) | D4_algorithm_oom_fix | D1 causal reward 和 policy loss 计算缺 tensor, crash |

两个 bug 都已在本次修复中纠正.

---

## 第二部分: 消融实验 len_control Double-Penalty 问题

### 2.1 问题描述

原始 T-GRPO 的 `len_control=True` 时, 在 compute_loss 里硬编码了:
```python
if 320 <= lenth_list[idx] <= 512:
    rewards[idx] += 0.2
```

我们的 D3 (Task-Aware Length Penalty) 用 Bayesian Welford 动态计算
长度边界, 并对超长/过短的 completion 施加惩罚:
```python
if length > l_max:
    length_penalties[i] = -alpha * (length - l_max)
elif length < l_min:
    length_penalties[i] = -beta * (l_min - length)
```

如果两个同时开启, 一个给 320-512 范围内的 +0.2 bonus, 另一个又
按动态边界惩罚, 就产生了 **double-penalty**:

- MCQ 任务: D3 的 l_max=256, 原始 len_control 的 bonus 范围是 320-512,
  两者边界冲突, 256-320 区间的样本同时被 D3 惩罚和 len_control 奖励
- Open-ended 任务: D3 的 l_min=200, len_control 的 l_min=320,
  200-320 区间的行为不可预测

### 2.2 修复方案

在 `ExperimentConfig` 里新增 `override_len_control` 字段:

```python
override_len_control: Optional[bool] = None
# None = 不覆盖, 遵从命令行 --len_control 的值
# True/False = 强制覆盖
```

在 trainer 里加载 experiment config 后检查:
```python
if self.exp_config.override_len_control is not None:
    if self.len_control != self.exp_config.override_len_control:
        print(f"[FIX-9] Experiment config overrides --len_control ...")
        self.len_control = self.exp_config.override_len_control
```

在 D3 开启的 YAML 配置 (ablation_d2d3, ablation_full) 中设置:
```yaml
override_len_control: false
```

在 baseline 和 D2-only 配置中不设置 (默认 None, 保持 CLI 原值).

---

## 第三部分: 正式实验规划

### 3.1 训练参数 (与原论文对齐)

| 参数 | 原论文 Video-R1 | 我们的值 |
|------|-----------------|----------|
| 基模型 | Qwen2.5-VL-7B-COT-SFT | 同 |
| num_generations | 4 | 4 |
| max_frames | 16 | 16 |
| per_device_train_batch_size | 1 | 1 |
| RL 训练步数 | ~1200 | 1200+ |
| DeepSpeed | ZeRO-3 | ZeRO-3 |
| 硬件 | 5x A100-80GB 或 4x H20-96GB | 4x A100-80GB (需 OOM fix) |

**为什么 num_generations=4 就够了:**

1. 原论文自己也是用 4, 不是 8
2. GRPO 的 advantage 估计只需要组内有方差, 4 个样本足够
3. num_generations=8 会把每步 GPU 内存需求翻倍, 4x A100 跑不动
4. 在同等条件下 (都是 4) 比原论文好, 结论更有说服力

**关键原则: 公平对比, 清晰归因.** baseline 和改进组用完全相同的
硬件、数据、步数、num_generations, 唯一变量是 D1/D2/D3.

### 3.2 四组消融实验

| 实验 | 配置文件 | D1 | D2 | D3 | len_control | 说明 |
|------|----------|----|----|----|----|------|
| A: Baseline | baseline_tgrpo.yaml | OFF | OFF | OFF | True (CLI默认) | 复现原始 T-GRPO |
| B: +D2 | ablation_d2_kl.yaml | OFF | ON | OFF | True (CLI默认) | 只加 KL clip |
| C: +D2+D3 | ablation_d2d3.yaml | OFF | ON | ON | False (override) | 加动态长度惩罚, 关闭原始 len_control |
| D: Full | ablation_full.yaml | ON | ON | ON | False (override) | 全量三维正则化 |

### 3.3 评估 Benchmark

与原论文 Table 1 对齐:

| Benchmark | 评估能力 | 评估帧数 |
|-----------|----------|----------|
| VSI-Bench | 视频空间推理 | 16, 64 |
| VideoMMMU | 多学科推理 | 16 |
| MVBench | 通用视频理解 | 16 |
| VideoMME | 综合视频理解 | 16, 64 |

额外建议:
- 帧数缩放实验: 16 vs 64 帧, 验证 D1 causal reward 在更多帧时效果更明显
- Temporal usage rate: 统计模型推理中使用时序信息的比例

### 3.4 训练曲线监控 (WandB)

必须记录和对比的指标:
- `rewards/accuracy`: 准确率奖励曲线
- `rewards/total`: 总奖励曲线
- `completion_length`: 输出长度变化
- `kl`: KL 散度 (D2 开启前后对比)
- `temporal_rewards`: 时序奖励
- `exp/raw_kl` (D2): clip 前的原始 KL, 对比 D2 的截断效果
- `length_penalties` (D3): 动态长度惩罚值
- `causal_rewards` (D1): 因果奖励值

### 3.5 分析指标 (论文必需)

1. **D2 效果**: clip 前后的 KL 分布对比图, 展示单 token KL 爆炸被消除
2. **D3 效果**: MCQ/open-ended 的长度分布直方图, Welford 动态边界变化图
3. **D1 效果**: temporal reward 对比 (原始 0.8 硬阈值 vs softplus)
4. **Temporal usage rate**: 原论文 75% vs 60.2%, 我们也应该统计
5. **帧数缩放**: 16 vs 64 帧 accuracy 变化, 验证 D1 在多帧下的优势

### 3.6 OOM fix 对消融公平性的影响

OOM fix (ref_model CPU offload + tensor cleanup) 影响所有实验组:
- Baseline 和 D2-only 不需要 D1 masked forward, 内存压力更小
- D3 和 Full 组内存压力相同 (D3 不增加 forward pass)
- Full 组因 D1 多一次 forward, 峰值最高

OOM fix 是对训练效率的优化, 不改变算法逻辑, 对所有组公平.

### 3.7 硬件建议

- **4x A100-80GB**: baseline / D2 / D3 安全; Full (D1) 需配合 OOM fix
  和可能的 num_generations=2 或 max_pixels 限制
- **5x A100-80GB**: 所有配置安全, 和原论文硬件对齐, 推荐
- **4x H20-96GB**: 原论文推荐配置, 也安全

如果 AutoDL 可以加第 5 张 A100, 强烈建议加. 320GB -> 400GB 的
差距是 ref_model 那一层的安全余量.

### 3.8 执行顺序

1. 跑 baseline 200 步, 确认不 OOM + reward 曲线合理
2. 跑 full (ablation_full.yaml) 200 步, 确认 D1 OOM fix 生效
3. 两组都通过后, 正式跑全部 4 组各 1200+ 步
4. 训练完成后做 benchmark 评估 (VSI-Bench, VideoMMMU, MVBench, VideoMME)
5. 收集分析指标, 写论文

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `trainer/grpo_trainer.py` | Fix 1: ref_model CPU offload (low_cpu_mem_usage=False + .cpu()) |
| `trainer/grpo_trainer.py` | Fix 2: D1 masked forward 后 del + empty_cache |
| `trainer/grpo_trainer.py` | Fix 3: KL 后 del ref_per_token_logps + x_clamped + empty_cache |
| `trainer/grpo_trainer.py` | Fix 9: len_control override 逻辑 |
| `experiment_config.py` | 新增 override_len_control 字段 + summary 展示 |
| `configs/ablation_d2d3.yaml` | 新增 override_len_control: false |
| `configs/ablation_full.yaml` | 新增 override_len_control: false |
