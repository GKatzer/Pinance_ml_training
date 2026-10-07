import numpy as np
import pandas as pd

from pinance_ml.dataset import build_dataset, feature_columns


def _synthetic_candles(n=60, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")

    log_ret = rng.normal(0, 0.001, size=n)
    close = 100 * np.exp(np.cumsum(log_ret))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, size=n))
    volume = rng.uniform(10, 100, size=n)

    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_build_dataset_has_one_row_per_input_candle():
    candles = _synthetic_candles(n=60)
    dataset = build_dataset(candles)
    assert len(dataset) == len(candles)
    assert list(dataset["ts"]) == list(candles.sort_values("ts")["ts"])


def test_build_dataset_includes_all_12_target_columns():
    candles = _synthetic_candles(n=60)
    dataset = build_dataset(candles)
    for h in range(1, 13):
        assert f"r_{h}" in dataset.columns


def test_feature_columns_excludes_ohlcv_and_targets():
    candles = _synthetic_candles(n=60)
    dataset = build_dataset(candles)
    cols = feature_columns(dataset)

    for excluded in ["ts", "open", "high", "low", "close", "volume", "r_1", "r_12"]:
        assert excluded not in cols
    assert "rsi" in cols
    assert "ret_lag_1" in cols


def test_build_dataset_includes_long_horizon_target_column():
    candles = _synthetic_candles(n=60)
    dataset = build_dataset(candles)
    assert "r_288" in dataset.columns


def test_feature_columns_excludes_long_horizon_target():
    candles = _synthetic_candles(n=60)
    dataset = build_dataset(candles)
    assert "r_288" not in feature_columns(dataset)


def test_feature_columns_includes_btc_ret_only_when_provided():
    alt = _synthetic_candles(n=60, seed=1)
    btc = _synthetic_candles(n=60, seed=2)

    without_btc = feature_columns(build_dataset(alt))
    with_btc = feature_columns(build_dataset(alt, btc_candles=btc))

    assert "btc_ret" not in without_btc
    assert "btc_ret" in with_btc
