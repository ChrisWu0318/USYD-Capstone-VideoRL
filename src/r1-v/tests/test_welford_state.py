from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "open_r1"))

from experiment_config import ExperimentConfig
from welford import BayesianWelford


def test_task_aware_welford_bounds_and_state_round_trip():
    config = ExperimentConfig(
        enable_length_penalty=True,
        welford_warmup_steps=1,
        length_max_mcq=50,
        length_max_open=80,
    )
    welford = BayesianWelford(sigma_prior=10.0, k_prior=1)
    welford.update_batch([10, 12, 14], task_type="mcq")
    welford.update_batch([120, 130, 140], task_type="open_ended")

    mcq_bounds = welford.get_dynamic_bounds("mcq", config)
    open_bounds = welford.get_dynamic_bounds("open_ended", config)

    assert mcq_bounds[0] == config.length_min_mcq
    assert open_bounds[0] == config.length_min_open
    assert open_bounds[1] > mcq_bounds[1]

    restored = BayesianWelford(sigma_prior=1.0, k_prior=1)
    restored.load_state_dict(welford.state_dict())

    assert restored.count == welford.count
    assert restored.task_stats["mcq"].count == 3
    assert restored.task_stats["open_ended"].count == 3


def test_legacy_welford_state_load_stays_backward_compatible():
    legacy_state = {
        "count": 2,
        "mean": 15.0,
        "M2": 8.0,
        "sigma_prior_sq": 100.0,
        "k_prior": 50,
    }

    restored = BayesianWelford(sigma_prior=1.0, k_prior=1)
    restored.load_state_dict(legacy_state)

    assert restored.count == 2
    assert restored.mean == 15.0
    assert restored.task_stats == {}
