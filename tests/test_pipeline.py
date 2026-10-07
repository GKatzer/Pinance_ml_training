import numpy as np
import pandas as pd
import pytest

from pinance_ml.features.pipeline import ATR_WINDOW, compute_features


def _synthetic_candles(n=300, seed=0, start="2026-01-01"):
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


def test_same_input_produces_identical_output():
    df = _synthetic_candles()
    a = compute_features(df)
    b = compute_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_no_lookahead_prefix_features_match_when_future_rows_removed():
    """The core train/serve-skew guard: a row's features must depend only on
    candles at or before it. If some feature secretly used future data, this
    truncation would change its value; here it must not."""
    df = _synthetic_candles(n=300)
    full = compute_features(df)
    truncated = compute_features(df.iloc[:200])

    pd.testing.assert_frame_equal(
        full.iloc[:200].reset_index(drop=True),
        truncated,
        check_exact=False,
        rtol=1e-9,
    )


def test_ret_lag_is_shifted_log_return():
    df = _synthetic_candles(n=50)
    out = compute_features(df)
    ret = np.log(df["close"] / df["close"].shift(1))

    np.testing.assert_allclose(out["ret_lag_1"].to_numpy()[1:], ret.to_numpy()[:-1])
    np.testing.assert_allclose(
        out["ret_lag_24"].to_numpy()[24:], ret.to_numpy()[:-24], rtol=1e-10
    )


def test_rolling_mean_matches_manual_window_calc():
    df = _synthetic_candles(n=50)
    out = compute_features(df)
    ret = np.log(df["close"] / df["close"].shift(1))

    expected = ret.rolling(6).mean()
    pd.testing.assert_series_equal(out["ret_mean_6"], expected, check_names=False)


def test_hour_and_dow_cyclical_encoding_at_known_timestamps():
    # n must clear the largest indicator warmup window (ta's AverageTrueRange
    # indexes raw numpy arrays assuming len >= window, and raises otherwise).
    df = _synthetic_candles(n=20, start="2026-01-05")  # 2026-01-05 is a Monday
    out = compute_features(df)

    assert out["ts"].iloc[0].hour == 0
    assert np.isclose(out["hour_sin"].iloc[0], 0.0, atol=1e-9)
    assert np.isclose(out["hour_cos"].iloc[0], 1.0, atol=1e-9)

    assert out["ts"].iloc[0].dayofweek == 0  # Monday
    assert np.isclose(out["dow_sin"].iloc[0], 0.0, atol=1e-9)
    assert np.isclose(out["dow_cos"].iloc[0], 1.0, atol=1e-9)


def test_btc_cross_asset_feature_aligns_on_exact_timestamp_and_nans_on_gap():
    alt = _synthetic_candles(n=20, seed=1)
    btc = _synthetic_candles(n=20, seed=2)
    btc_dropped = btc.drop(index=5).reset_index(drop=True)  # introduce a gap

    out = compute_features(alt, btc_candles=btc_dropped)
    expected_btc_ret = np.log(btc["close"] / btc["close"].shift(1))

    matching_ts = alt["ts"].iloc[10]
    expected_value = expected_btc_ret[btc["ts"] == matching_ts].iloc[0]
    assert np.isclose(out.loc[out["ts"] == matching_ts, "btc_ret"].iloc[0], expected_value)

    gap_ts = btc["ts"].iloc[5]
    assert np.isnan(out.loc[out["ts"] == gap_ts, "btc_ret"].iloc[0])


def test_btc_feature_absent_when_not_provided():
    df = _synthetic_candles(n=20)
    out = compute_features(df)
    assert "btc_ret" not in out.columns


def test_ret_atr_norm_has_no_inf_when_atr_warmup_is_zero():
    # ta's AverageTrueRange zero-fills (not NaN-fills) its warmup rows, which
    # previously produced +/-inf here via division by exactly 0.0.
    df = _synthetic_candles(n=60)
    out = compute_features(df)

    assert not np.isinf(out["ret_atr_norm"].to_numpy(dtype=float)).any()
    assert (out["atr"].iloc[: ATR_WINDOW - 1] == 0.0).all()
    assert out["ret_atr_norm"].iloc[: ATR_WINDOW - 1].isna().all()


SERVING_WINDOW = 150  # MIN_WARMUP_CANDLES in Pinance_ml_inference


def test_serving_window_features_match_full_history():
    """Train/serve skew guard: training computes features over years of history,
    serving over only the last MIN_WARMUP_CANDLES. The last row must agree.

    OBV was cumulative (differed by ~70% here) and is now windowed -> exact.
    The EMA-based indicators (rsi/macd/atr) forget their start only
    geometrically, leaving ~4e-6 relative residual at 150 candles, hence rtol 1e-4.
    """
    df = _synthetic_candles(n=5000)
    full = compute_features(df).iloc[-1].drop("ts").astype(float)
    window = compute_features(df.iloc[-SERVING_WINDOW:]).iloc[-1].drop("ts").astype(float)

    pd.testing.assert_series_equal(full, window, rtol=1e-4, atol=1e-12, check_names=False)
    for col in [c for c in full.index if c.startswith("obv")]:
        np.testing.assert_allclose(full[col], window[col], rtol=1e-9)
