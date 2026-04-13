"""
experiment_config.py
====================
四维正则化消融实验的统一配置管理。

用法:
    # 从 YAML 加载
    config = ExperimentConfig.from_yaml("configs/ablation_d2_kl.yaml")

    # 命令行覆盖 (与 GRPOScriptArguments 配合)
    config = ExperimentConfig.from_yaml(script_args.experiment_config_path)

    # 直接构造 (默认 = baseline, 所有 flag 关闭)
    config = ExperimentConfig()
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple


@dataclass
class ExperimentConfig:
    """
    统一管理所有实验变体的配置。
    每个维度有独立的 enable flag + 对应参数，
    flag=False 时行为与原始 T-GRPO 完全一致。
    """

    # ================================================================
    # 实验元信息
    # ================================================================
    experiment_name: str = "baseline_tgrpo"
    experiment_tag: str = "v0.0"
    description: str = ""
    seed: int = 42

    # ================================================================
    # 维度二: Token-Clipped KL Penalty
    # ================================================================
    enable_token_clipped_kl: bool = False
    kl_d_max: float = 5.0       # 每个 token 的 KL 上界 D_max

    # ================================================================
    # 维度三: Task-Aware Length Penalty + Welford
    # ================================================================
    enable_length_penalty: bool = False
    # MCQ 任务
    length_min_mcq: int = 50
    length_max_mcq: int = 256
    # Open-ended 任务
    length_min_open: int = 200
    length_max_open: int = 600
    # 惩罚系数
    length_penalty_alpha: float = 0.001     # 过长惩罚
    length_penalty_beta: float = 0.0005     # 过短惩罚
    # Welford Bayesian warmup (先验从 Phase 1 统计得到)
    welford_sigma_prior: float = 100.0      # σ_prior
    welford_k_prior: int = 50               # k_prior (等效先验样本数)
    welford_warmup_steps: int = 50          # 前 N 步用纯静态边界

    # ================================================================
    # 维度一: Soft-Truncated Counterfactual Causal Reward
    # ================================================================
    enable_causal_reward: bool = False
    softplus_beta: float = 2.0              # β_s (Softplus 硬度)
    causal_mask_ratio: float = 0.5          # Mask 掉多少比例的帧
    causal_lazy_eval: bool = True           # 只对 R_acc=1 的样本计算
    causal_reward_clip: float = 10.0        # R_causal 硬性上界 (防 Advantage 极化)
    causal_reward_scale: float = 0.5        # 叠加到 reward 时的缩放因子

    # ================================================================
    # 方法
    # ================================================================

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载配置, 未指定的字段使用默认值"""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml --break-system-packages")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 只取 ExperimentConfig 认识的字段, 忽略 YAML 中的多余 key
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """导出为 plain dict"""
        return asdict(self)

    def to_wandb_config(self) -> Dict[str, Any]:
        """导出为 WandB config dict (加前缀避免与训练参数冲突)"""
        return {f"exp/{k}": v for k, v in asdict(self).items()}

    def get_length_bounds(self, task_type: str) -> Tuple[int, int]:
        """
        根据任务类型返回静态 (L_min, L_max)

        Args:
            task_type: "mcq" | "open_ended"
        """
        if task_type == "mcq":
            return (self.length_min_mcq, self.length_max_mcq)
        else:
            return (self.length_min_open, self.length_max_open)

    def summary(self) -> str:
        """打印实验配置摘要, 方便训练开始时 log"""
        lines = [
            f"=== Experiment: {self.experiment_name} ({self.experiment_tag}) ===",
            f"  Description: {self.description or '(none)'}",
            f"  Seed: {self.seed}",
            f"  [D2] Token-Clipped KL: {'ON' if self.enable_token_clipped_kl else 'OFF'}"
            + (f" (D_max={self.kl_d_max})" if self.enable_token_clipped_kl else ""),
            f"  [D3] Length Penalty:    {'ON' if self.enable_length_penalty else 'OFF'}"
            + (f" (α={self.length_penalty_alpha}, β={self.length_penalty_beta})" if self.enable_length_penalty else ""),
            f"  [D1] Causal Reward:     {'ON' if self.enable_causal_reward else 'OFF'}"
            + (f" (β_s={self.softplus_beta}, clip={self.causal_reward_clip})" if self.enable_causal_reward else ""),
        ]
        return "\n".join(lines)


# ====================================================================
# 任务类型检测 (供 Length Penalty 使用)
# ====================================================================

def detect_task_type(prompt_text: str) -> str:
    """
    从 prompt 文本判断任务类型。

    策略:
    - 包含 "A." / "B." / "(A)" 等选项标记 → "mcq"
    - 否则 → "open_ended"

    注意: 如果数据集自带 problem_type 字段, 优先使用那个,
    这个函数只是 fallback。
    """
    mcq_patterns = [
        r"\bA\.\s",           # A. xxx
        r"\bB\.\s",           # B. xxx
        r"\(A\)",             # (A)
        r"Options?\s*:",      # Options: / Option:
    ]
    for pattern in mcq_patterns:
        if re.search(pattern, prompt_text):
            return "mcq"
    return "open_ended"


def task_type_from_problem_type(problem_type: str) -> str:
    """
    从数据集的 problem_type 字段映射到我们的二分类。
    Video-R1 数据集的 problem_type 有:
      multiple choice, numerical, OCR, free-form, regression
    """
    if problem_type == "multiple choice":
        return "mcq"
    return "open_ended"
