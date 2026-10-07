import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinance_ml.sanity import naive_point_baseline, naive_quantile_pinball, point_sanity_check, quantile_sanity_check


def _frame(values):
    return pd.DataFrame({"r_1": values})


def test_naive_point_baseline_is_mean_abs_and_majority_class_accuracy():
    train = _frame([0.01, 0.02, -0.01])  # majority sign: up
    evals = _frame([0.02, -0.02, 0.04, 0.02])
    mae_, acc = naive_point_baseline(train, evals, [1])
    assert mae_ == np.mean([0.02, 0.02, 0.04, 0.02])
    assert acc == 0.75  # 3 of 4 eval rows are up


def test_naive_quantile_pinball_zero_when_constant_matches_distribution():
    train = _frame([1.0] * 100)
    evals = _frame([1.0] * 10)
    assert naive_quantile_pinball(train, evals, [1], (0.1, 0.9)) == 0.0


def test_point_sanity_passes_when_at_least_naive():
    passed, reason = point_sanity_check(0.0100, 0.52, 0.0101, 0.50, max_mae_ratio=1.02, min_dir_acc_edge=0.0)
    assert passed and reason == "ok"


def test_point_sanity_fails_when_mae_clearly_worse_than_naive():
    passed, reason = point_sanity_check(0.0120, 0.52, 0.0100, 0.50, max_mae_ratio=1.02, min_dir_acc_edge=0.0)
    assert not passed and reason.startswith("mae_worse_than_naive")


def test_point_sanity_tolerates_mae_within_ratio():
    passed, _ = point_sanity_check(0.01015, 0.52, 0.0100, 0.50, max_mae_ratio=1.02, min_dir_acc_edge=0.0)
    assert passed


def test_point_sanity_fails_when_direction_below_base_rate():
    passed, reason = point_sanity_check(0.0100, 0.48, 0.0100, 0.50, max_mae_ratio=1.02, min_dir_acc_edge=0.0)
    assert not passed and reason.startswith("dir_acc_below_naive")


def test_point_sanity_nan_never_passes():
    passed, reason = point_sanity_check(float("nan"), 0.52, 0.0100, 0.50, 1.02, 0.0)
    assert not passed and reason == "nan_metric"


def test_quantile_sanity_requires_calibration_and_naive_beating():
    assert quantile_sanity_check(0.001, 0.0011, True, 1.02) == (True, "ok")
    assert quantile_sanity_check(0.001, 0.0011, False, 1.02) == (False, "coverage_not_calibrated")
    passed, reason = quantile_sanity_check(0.0013, 0.0010, True, 1.02)
    assert not passed and reason.startswith("pinball_worse_than_naive")
    assert quantile_sanity_check(float("nan"), 0.001, True, 1.02) == (False, "nan_metric")
