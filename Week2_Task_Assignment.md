# 第二周任务分配计划（8 人）

## Context

- 项目周期：12 周，当前第 2 周
- 第 1 周：团队接触项目、初步阅读代码
- 第 2 周核心目标：**跑通冒烟测试**，确保代码能在 AutoDL 服务器上正常运行
- 代码修改已完成：新增 `--multi_corruption` 和 `--marginal_reward` 独立控制标志
- 核心成员已熟悉代码，其他成员仍在熟悉中

---

## 建议角色划分（8 人）

| 角色 | 人数 | 职责范围 |
|------|------|---------|
| A - 算法核心 | 2 人 | 负责核心训练代码（grpo_trainer.py, grpo.py），理解 Task A/B 实现 |
| B - 工程环境 | 2 人 | 负责 AutoDL 服务器环境、DeepSpeed、FlashAttention、多 GPU 配置 |
| C - 数据与评估 | 2 人 | 负责数据集准备、评估脚本、benchmark 验证 |
| D - 测试与文档 | 2 人 | 负责冒烟测试脚本编写、测试报告、DevLog 更新、项目管理 |

---

## 详细任务分配

### 成员 1（算法核心 A1）—— 训练器代码审查与验证

**目标**：确保 `grpo_trainer.py` 的 Task A/B 分支逻辑正确无误

**具体任务**：

1. **代码审查**（Day 1-2）
   - 逐行阅读 `src/r1-v/src/open_r1/trainer/grpo_trainer.py` 全文（~830 行）
   - 重点审查 Task A 分支（第 458-480 行）：`self.multi_corruption` 为 `true/false` 时的两条路径
   - 重点审查 Task B 分支（第 633-652 行）：`self.marginal_reward` 为 `true/false` 时的两条路径
   - 检查 baseline 二值奖励恢复逻辑是否与原始注释代码一致
   - 确认 `device` 变量在所有分支中正确使用（不能硬编码 `'cuda'`）

2. **逻辑验证**（Day 2-3）
   - 确认四种配置组合的代码路径：
     - `multi_corruption=false, marginal_reward=false` → 单一 shuffle + 二值奖励
     - `multi_corruption=true, marginal_reward=false` → 多破坏 + 二值奖励
     - `multi_corruption=false, marginal_reward=true` → 单一 shuffle + 边际奖励
     - `multi_corruption=true, marginal_reward=true` → 多破坏 + 边际奖励
   - 追踪 `shuffled_rewards_per_func` 变量的数据流，确认在所有配置下都正确初始化

3. **输出物**
   - 代码审查报告（发现的问题 / 确认无误）
   - 如有 bug，提交修复 PR

---

### 成员 2（算法核心 A2）—— 奖励函数与训练入口审查

**目标**：确保奖励计算、数据加载、训练入口的端到端正确性

**具体任务**：

1. **奖励函数审查**（Day 1-2）
   - 阅读 `src/r1-v/src/open_r1/grpo.py` 中的 `accuracy_reward`（第 74-173 行）
   - 逐个验证 5 种题型的奖励计算：multiple choice、numerical、OCR、free-form、regression
   - 阅读 `format_reward`（第 176-181 行），确认正则模式正确
   - 确认 `GRPOScriptArguments` 中新增的 `multi_corruption` 和 `marginal_reward` 参数定义正确

2. **数据加载路径验证**（Day 2-3）
   - 追踪数据集加载流程：`grpo.py` 的 `main()` → 数据加载 → 传入 Trainer
   - 确认 `Video-R1-data/` 路径在 AutoDL 服务器上的实际位置
   - 检查视频文件路径拼接逻辑（`grpo.py` 中 `os.getcwd() + "/Video-R1-data" + path`）
   - 抽检 2-3 个视频样本，确认路径可访问、视频可读取

3. **新参数传递验证**（Day 3）
   - 追踪 `multi_corruption` 和 `marginal_reward` 从命令行 → `GRPOScriptArguments` → `Qwen2VLGRPOTrainer.__init__` → `compute_loss` 的完整传递链
   - 确认参数名称、类型在每一层一致

4. **输出物**
   - 奖励函数审查报告
   - 数据路径确认清单
   - 如有 bug，提交修复 PR

---

### 成员 3（工程环境 B1）—— AutoDL 服务器环境搭建与验证

**目标**：在 AutoDL 上搭建可运行的训练环境

**具体任务**：

1. **环境搭建**（Day 1-2）
   - 创建 / 验证 Conda 环境 `video-r1`（Python 3.11）
   - 安装核心依赖并记录版本：
     ```bash
     conda activate video-r1
     pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
     pip install transformers accelerate datasets trl==0.16.0 deepspeed==0.15.4
     pip install wandb nltk rouge_score einops pillow
     ```
   - 安装 FlashAttention 预编译 wheel（参考 DevLog 1.1.2 的 ABI 匹配策略）
   - 安装 qwen-vl-utils（项目内 `src/qwen-vl-utils/`）

2. **环境验证**（Day 2-3）
   - 运行依赖导入验证脚本：
     ```bash
     python -c "
     import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
     import deepspeed; print(f'deepspeed: {deepspeed.__version__}')
     import torch; print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}')
     import transformers; print(f'transformers: {transformers.__version__}')
     import trl; print(f'trl: {trl.__version__}')
     import qwen_vl_utils; print('qwen_vl_utils: OK')
     "
     ```
   - 验证 GPU 可见性：`nvidia-smi`，确认 4 张 RTX 4090
   - 验证 PyTorch CUDA 连通：`python -c "import torch; print(torch.cuda.device_count())"`
   - 运行 `ds_report` 确认 DeepSpeed 检测到所有 GPU

3. **模型权重准备**（Day 3）
   - 确认 `Qwen2.5-VL-7B-COT-SFT` 权重已下载到服务器
   - 记录权重实际路径，通知全组更新 `run_ablation.sh` 中的 `'SFT Model Path'`

4. **输出物**
   - 环境搭建文档（精确的安装命令 + 版本号）
   - 环境验证截图（所有依赖导入成功、GPU 检测正确）
   - 模型权重路径

---

### 成员 4（工程环境 B2）—— DeepSpeed 多 GPU 配置与验证

**目标**：确保多 GPU 分布式训练配置可用

**具体任务**：

1. **DeepSpeed 配置审查**（Day 1-2）
   - 阅读并理解 4 个 DeepSpeed 配置文件：
     - `src/r1-v/local_scripts/zero2.json`
     - `src/r1-v/local_scripts/zero3.json`（主要使用）
     - `src/r1-v/local_scripts/zero3_offload.json`（OOM 备用）
     - `src/r1-v/local_scripts/zero1_no_optimizer.json`
   - 确认 `zero3.json` 中的 `"auto"` 字段能被 HuggingFace Trainer 正确解析
   - 检查 bf16/fp16 配置是否与训练脚本一致

2. **多 GPU 通信测试**（Day 2-3，依赖成员 3 环境搭建完成）
   - 运行 NCCL 通信测试：
     ```bash
     python -c "
     import torch.distributed as dist
     import torch
     dist.init_process_group('nccl')
     print(f'Rank {dist.get_rank()}/{dist.get_world_size()}')
     "
     ```
   - 测试 `torchrun` 多进程启动：
     ```bash
     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -c "import torch; print(torch.cuda.current_device())"
     ```
   - 如遇 NCCL 问题，调试 `NCCL_DEBUG=INFO` 并记录解决方案

3. **DeepSpeed 冒烟测试**（Day 3-4，依赖成员 5 提供迷你数据集）
   - 用迷你数据集运行 4 GPU + ZeRO-3 的 2 步训练：
     ```bash
     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
         --nnodes=1 --node_rank=0 \
         --master_addr="127.0.0.1" --master_port="12345" \
         src/open_r1/grpo.py \
         --deepspeed local_scripts/zero3.json \
         --max_steps 2 --num_generations 4 \
         [其他参数参考 run_ablation.sh]
     ```
   - 验证 4 个 rank 都正常初始化、2 步完成、无 NCCL timeout
   - 用 `nvidia-smi` 确认 4 张 GPU 都有利用率

4. **输出物**
   - DeepSpeed 配置说明文档
   - 多 GPU 测试通过截图
   - 遇到的问题和解决方案记录

---

### 成员 5（数据与评估 C1）—— 数据集准备与验证

**目标**：确保训练和评估数据完整可用

**具体任务**：

1. **训练数据检查**（Day 1-2）
   - 检查 `Video-R1-data/Video-R1-260k.json` 是否存在且完整
   - 统计数据分布：
     ```bash
     python -c "
     import json
     with open('./Video-R1-data/Video-R1-260k.json') as f:
         data = json.load(f)
     print(f'Total samples: {len(data)}')
     types = {}
     for s in data:
         t = s.get('data_type', 'unknown')
         types[t] = types.get(t, 0) + 1
     print(f'Data types: {types}')
     ptypes = {}
     for s in data:
         t = s.get('problem_type', 'unknown')
         ptypes[t] = ptypes.get(t, 0) + 1
     print(f'Problem types: {ptypes}')
     "
     ```
   - 抽检 5 个视频样本路径，确认视频文件存在且可读取
   - 抽检 5 个图片样本路径，确认图片文件存在

2. **创建迷你测试数据集**（Day 2）
   - 创建 `smoke_test_2.json`（2 个视频样本，供冒烟测试用）：
     ```bash
     python -c "
     import json
     with open('./Video-R1-data/Video-R1-260k.json') as f:
         data = json.load(f)
     video_samples = [s for s in data if s.get('data_type') == 'video'][:2]
     with open('./Video-R1-data/smoke_test_2.json', 'w') as f:
         json.dump(video_samples, f, indent=2)
     print(f'Wrote {len(video_samples)} video samples')
     "
     ```
   - 手动验证这 2 个样本的视频文件确实存在且可打开

3. **评估数据集检查**（Day 3-4）
   - 检查评估数据集是否存在：
     ```bash
     ls ./src/r1-v/Evaluation/eval_*.json
     ```
   - 需要的 6 个 benchmark：mvbench、tempcompass、videomme、videommmu、vsibench、mmvu
   - 如有缺失，查找下载方式并记录到文档
   - 统计每个评估数据集的样本数量

4. **输出物**
   - 数据集完整性报告（训练集 + 6 个评估集的统计）
   - `smoke_test_2.json` 迷你测试数据集
   - 缺失数据集的下载指南（如有）

---

### 成员 6（数据与评估 C2）—— 评估流程验证

**目标**：确保评估脚本 `eval_bench.py` 可用，为后续评估做准备

**具体任务**：

1. **评估脚本阅读**（Day 1-2）
   - 通读 `src/eval_bench.py`，理解评估流程：
     - 模型加载方式（vLLM）
     - 输入数据格式
     - 推理参数（batch_size=64, greedy decoding）
     - 输出结果格式和保存路径
   - 记录评估脚本的命令行参数（`--model_path`, `--file_name` 等）

2. **评估指标理解**（Day 2-3）
   - 理解各 benchmark 的评估指标：
     - BLEU score 计算方式
     - ROUGE-1/2/L 计算方式
     - mean_acc 和 mean_mra 的含义
   - 明确哪些 benchmark 与时序理解最相关（tempcompass, mvbench）

3. **评估脚本语法检查**（Day 3）
   - 运行语法检查：
     ```bash
     python -c "import py_compile; py_compile.compile('src/eval_bench.py', doraise=True); print('OK')"
     ```
   - 检查 vLLM 依赖是否安装：
     ```bash
     python -c "import vllm; print(f'vllm: {vllm.__version__}')"
     ```

4. **结果汇总脚本编写**（Day 4）
   - 编写一个 Python 脚本，自动从 6 个 benchmark 的输出 JSON 中提取 mean_acc，生成对比表
   - 参考计划中 Step 4.3 的汇总脚本模板

5. **输出物**
   - 评估流程说明文档（命令、参数、输出格式）
   - 结果汇总脚本
   - 评估所需依赖清单

---

### 成员 7（测试与文档 D1）—— 冒烟测试执行

**目标**：执行端到端的冒烟测试，验证四种配置都能跑通

**具体任务**：

1. **语法检查**（Day 1，依赖成员 3 环境就绪）
   - 运行语法检查：
     ```bash
     python -c "import py_compile; py_compile.compile('src/open_r1/trainer/grpo_trainer.py', doraise=True); print('OK')"
     python -c "import py_compile; py_compile.compile('src/open_r1/grpo.py', doraise=True); print('OK')"
     ```

2. **单 GPU 单步冒烟测试**（Day 2-3，依赖成员 5 的迷你数据集）
   - 对四种配置分别执行 1 步训练：
     ```bash
     # Config 1: Baseline（单一 shuffle + 二值奖励）
     CUDA_VISIBLE_DEVICES=0 python src/open_r1/grpo.py \
         --output_dir "./log/smoke_baseline" \
         --model_name_or_path '/path/to/Qwen2.5-VL-7B-COT-SFT' \
         --dataset_name "../../Video-R1-data/smoke_test_2.json" \
         --max_prompt_length 4096 \
         --max_completion_length 128 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 1 \
         --learning_rate 1e-6 \
         --bf16 \
         --logging_steps 1 \
         --gradient_checkpointing true \
         --temporal true \
         --multi_corruption false \
         --marginal_reward false \
         --len_control false \
         --attn_implementation flash_attention_2 \
         --max_pixels 401408 \
         --max_steps 1 \
         --num_generations 2 \
         --beta 0.04 \
         --save_steps 999999 \
         --run_name smoke_baseline

     # Config 2: Task A only（多破坏 + 二值奖励）
     # 改变: --multi_corruption true --marginal_reward false

     # Config 3: Task B only（单一 shuffle + 边际奖励）
     # 改变: --multi_corruption false --marginal_reward true

     # Config 4: Task A + B（多破坏 + 边际奖励）
     # 改变: --multi_corruption true --marginal_reward true
     ```
   - 每个配置验证：1 步训练完成、无报错、日志中能看到 `rewards` 输出

3. **错误排查与记录**（Day 3-4）
   - 记录每次测试的完整日志
   - 遇到错误时记录：错误信息、原因分析、解决方案
   - 常见问题参考：
     - `FlashAttention not found` → 联系成员 3 重新安装 wheel
     - `CUDA OOM` → 降低 `--max_pixels` 到 200704
     - `FileNotFoundError` → 联系成员 5 确认数据路径

4. **输出物**
   - 冒烟测试报告：四种配置的通过/失败状态
   - 错误排查记录
   - 测试日志文件

---

### 成员 8（测试与文档 D2）—— 项目文档与 DevLog 维护

**目标**：维护项目文档，记录第二周进展，确保知识共享

**具体任务**：

1. **DevLog 更新**（持续整周）
   - 在 `DevLog.md` 中更新 Phase 3 进度：
     - 标记已完成的步骤
     - 记录环境搭建中的问题和解决方案
     - 记录冒烟测试结果
   - 保持 DevLog 格式与现有内容一致

2. **第二周工作记录**（Day 1-4）
   - 收集每个成员的工作进展和产出
   - 记录遇到的阻塞问题和解决方案
   - 维护一份共享的进度跟踪表

3. **代码库 README 更新**（Day 2-3）
   - 在 `README.md` 中补充：
     - 新增的 `--multi_corruption` 和 `--marginal_reward` 参数说明
     - `run_ablation.sh` 的使用方法
     - 四种消融配置的简要说明

4. **环境搭建指南整理**（Day 3-4）
   - 基于成员 3 的搭建过程，整理一份可复现的环境安装手册
   - 包含：依赖版本、安装顺序、FlashAttention wheel 安装方法、常见问题

5. **输出物**
   - 更新后的 DevLog.md
   - 更新后的 README.md
   - 环境搭建指南文档
   - 第二周周报

---

## 任务依赖关系与时间线

```
Day 1-2:
  成员 1,2: 代码审查（独立进行）
  成员 3:   环境搭建（前置任务，优先级最高）
  成员 5:   数据集检查 + 创建迷你测试集
  成员 6:   评估脚本阅读
  成员 8:   开始 DevLog 更新

Day 2-3:
  成员 3:   环境验证 + 模型权重确认
  成员 4:   DeepSpeed 配置审查 + 多 GPU 通信测试（依赖成员 3 环境就绪）
  成员 7:   语法检查（依赖成员 3 环境就绪）

Day 3-4:
  成员 4:   DeepSpeed 冒烟测试（依赖成员 5 迷你数据集）
  成员 7:   四种配置冒烟测试（依赖成员 3 环境 + 成员 5 数据集）
  成员 6:   评估脚本验证 + 汇总脚本编写
  成员 8:   收集成果、整理文档

Day 4-5:
  全组：    汇总结果，确认冒烟测试全部通过
  成员 8:   产出周报
```

---

## 关键里程碑

| 里程碑 | 负责人 | 预期完成时间 |
|--------|--------|------------|
| AutoDL 环境搭建完成 | 成员 3 | Day 2 |
| 迷你测试数据集就绪 | 成员 5 | Day 2 |
| 代码审查完成（无遗留 bug） | 成员 1, 2 | Day 3 |
| 单 GPU 冒烟测试通过（4 种配置） | 成员 7 | Day 3 |
| 多 GPU DeepSpeed 测试通过 | 成员 4 | Day 4 |
| 评估流程文档就绪 | 成员 6 | Day 4 |
| 第二周周报和 DevLog 更新 | 成员 8 | Day 5 |

---

## 沟通建议

- 成员 3（环境搭建）是关键路径，优先级最高，其他人的测试都依赖环境就绪
- 成员 1, 2（代码审查）可与环境搭建并行进行
- 建议每天一次简短站会（10 分钟），同步进度和阻塞问题
- 用共享文档（飞书/Notion/GitHub Issues）跟踪任务状态
