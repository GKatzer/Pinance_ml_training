import numpy as np
import pandas as pd

from pinance_ml.features.targets import compute_log_return_targets


def _candles(ts, close):
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "close": close})


def test_r1_matches_manual_log_return():
    ts = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    close = [100.0, 101.0, 99.0, 102.0, 103.0]
    df = compute_log_return_targets(_candles(ts, close), horizons=[1], candle_interval_minutes=5)

    expected = np.log(np.array(close[1:]) / np.array(close[:-1]))
    np.testing.assert_allclose(df["r_1"].to_numpy()[:-1], expected)
    assert np.isnan(df["r_1"].iloc[-1])


def test_gap_in_candles_produces_nan_target_instead_of_wrong_pairing():
    # Missing the candle at t+5min: row 0's r_1 must be NaN, not paired with t+10min's close.
    ts = ["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z", "2026-01-01T00:15:00Z"]
    close = [100.0, 110.0, 120.0]
    df = compute_log_return_targets(_candles(ts, close), horizons=[1], candle_interval_minutes=5)

    assert np.isnan(df["r_1"].iloc[0])
    # row at 00:10 -> 00:15 exists and is exactly one 5-min step, so it's valid.
    assert np.isclose(df["r_1"].iloc[1], np.log(120.0 / 110.0))


def test_multiple_horizons_produce_separate_columns():
    ts = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    close = [100.0, 101.0, 102.0, 103.0]
    df = compute_log_return_targets(_candles(ts, close), horizons=[1, 2], candle_interval_minutes=5)

    assert np.isclose(df["r_2"].iloc[0], np.log(102.0 / 100.0))
    assert np.isnan(df["r_2"].iloc[-1])
    assert np.isnan(df["r_2"].iloc[-2])
