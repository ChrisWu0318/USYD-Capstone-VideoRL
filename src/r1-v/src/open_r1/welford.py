"""
welford.py
==========
Welford 在线算法 + Bayesian Prior warmup。

用于实时追踪 completion length 的全局统计量,
作为 Task-Aware Length Penalty 的动态基线。

数学:
    σ²_slow = (S_k + σ²_prior) / (k + k_prior)

用法:
    welford = BayesianWelford(sigma_prior=100.0, k_prior=50)
    welford.update_batch([320, 450, 128, 512])
    print(welford.mean, welford.std)
"""

import math
from typing import List, Tuple


class BayesianWelford:
    """
    O(1) 空间、数值稳定的在线均值/方差估计器。
    Bayesian prior 解决训练初期 (k < 50) 的冷启动问题。
    """

    def __init__(self, sigma_prior: float = 100.0, k_prior: int = 50):
        """
        Args:
            sigma_prior: 先验标准差 σ_prior (从 Phase 1 SFT 统计得到)
            k_prior:     等效先验样本数 (控制先验的强度)
        """
        self.sigma_prior_sq = sigma_prior ** 2
        self.k_prior = k_prior

        # Welford 内部状态
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0       # 偏差平方和 S_k

    def update(self, value: float) -> None:
        """单个观测值的在线更新 (Welford 算法核心)"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    def update_batch(self, values: List[float]) -> None:
        """批量更新, 每个 training step 结束时调用"""
        for v in values:
            self.update(v)

    @property
    def variance(self) -> float:
        """
        Bayesian 后验方差:
            σ²_slow = (S_k + σ²_prior) / (k + k_prior)

        训练初期 (k 小): σ²_prior 主导 → 避免 NaN / 极端值
        训练后期 (k → ∞): 先验消失 → 退化为标准 Welford
        """
        return (self._m2 + self.sigma_prior_sq) / (self.count + self.k_prior)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def get_dynamic_bounds(
        self,
        task_type: str,
        config: "ExperimentConfig",
    ) -> Tuple[int, float]:
        """
        结合 Welford 统计量和任务类型, 返回动态 (L_min, L_max)。

        策略:
        - warmup 阶段 (count < welford_warmup_steps): 用纯静态值
        - 之后: L_max = max(静态值, mean + 2σ), L_min 保持静态

        Args:
            task_type: "mcq" | "open_ended"
            config:    ExperimentConfig 实例

        Returns:
            (l_min, l_max) 元组
        """
        base_min, base_max = config.get_length_bounds(task_type)

        # warmup 期间用静态值
        if self.count < config.welford_warmup_steps:
            return (base_min, base_max)

        # 动态上界: 取 config 静态值和统计值的较大者
        dynamic_max = max(base_max, self.mean + 2.0 * self.std)
        return (base_min, dynamic_max)

    def state_dict(self) -> dict:
        """导出状态, 用于 checkpoint 保存"""
        return {
            "count": self.count,
            "mean": self.mean,
            "_m2": self._m2,
            "sigma_prior_sq": self.sigma_prior_sq,
            "k_prior": self.k_prior,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 checkpoint 恢复状态"""
        self.count = state["count"]
        self.mean = state["mean"]
        self._m2 = state["_m2"]
        self.sigma_prior_sq = state["sigma_prior_sq"]
        self.k_prior = state["k_prior"]

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "M2": self._m2}

    def load_state_dict(self, state):
        self.count = state["count"]
        self.mean = state["mean"]
        self._m2 = state["M2"]
