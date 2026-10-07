import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_models import _feature_baseline  # noqa: E402
from pinance_ml.config import DRIFT_BASELINE_N_BINS, DRIFT_FEATURE_COLUMNS  # noqa: E402


def _dataset_with(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base = {col: np.zeros(n) for col in DRIFT_FEATURE_COLUMNS}
    base.update(columns)
    return pd.DataFrame(base)


def test_feature_baseline_covers_every_drift_column():
    rng = np.random.default_rng(0)
    dataset = _dataset_with(**{col: rng.normal(size=500) for col in DRIFT_FEATURE_COLUMNS})

    baseline = _feature_baseline(dataset)

    assert set(baseline) == set(DRIFT_FEATURE_COLUMNS)


def test_feature_baseline_mean_std_and_bin_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    dataset = _dataset_with(rsi=rng.normal(loc=50, scale=10, size=1000))

    entry = _feature_baseline(dataset)["rsi"]

    assert entry["mean"] == pytest.approx(dataset["rsi"].mean())
    assert entry["std"] == pytest.approx(dataset["rsi"].std())
    assert len(entry["bin_edges"]) == DRIFT_BASELINE_N_BINS + 1
    assert sum(entry["bin_fractions"]) == pytest.approx(1.0)


def test_feature_baseline_missing_column_is_none():
    # e.g. btc_ret when exporting BTCUSDT itself -- compute_features never
    # adds that column for the base asset.
    dataset = _dataset_with(**{col: np.zeros(10) for col in DRIFT_FEATURE_COLUMNS if col != "btc_ret"})
    dataset = dataset.drop(columns=["btc_ret"])

    baseline = _feature_baseline(dataset)

    assert baseline["btc_ret"] is None


def test_feature_baseline_all_nan_column_is_none():
    dataset = _dataset_with(atr=np.full(50, np.nan))

    baseline = _feature_baseline(dataset)

    assert baseline["atr"] is None


def test_feature_baseline_handles_duplicate_values_without_crashing():
    # A long flat/constant stretch collapses decile edges into fewer than
    # DRIFT_BASELINE_N_BINS bins via duplicates="drop" -- must not raise.
    dataset = _dataset_with(macd_diff=np.concatenate([np.zeros(900), np.array([1.0, 2.0, 3.0])]))

    entry = _feature_baseline(dataset)["macd_diff"]

    assert entry is not None
    assert sum(entry["bin_fractions"]) == pytest.approx(1.0)
    assert len(entry["bin_edges"]) <= DRIFT_BASELINE_N_BINS + 1
