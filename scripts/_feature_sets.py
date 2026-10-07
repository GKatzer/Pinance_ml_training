"""Named extra-feature builders for scripts/measure_directional_gain.py.

Each builder takes (dataset, symbol) -> (dataset_with_extra_cols, extra_col_names).
The dataset already carries the full technical feature matrix (dataset.py);
a builder only appends its own columns, all strictly causal. Kept in
scripts/ (not src/) because these are experiment candidates being screened,
not pipeline features -- one gets promoted into src/pinance_ml only if a
walk-forward A/B says it earns its place.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FUNDING_CACHE_DIR = Path("reports/funding_cache")
FUNDING_Z_WINDOW = "30D"
FUNDING_Z_MIN_PERIODS = 30


def attach_funding_features(dataset: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """f_rate, f_z, f_absz -- last realized funding rate, its trailing-30d
    z-score (computed on the funding series, current settlement included
    since its rate is known), |z|. As-of merged backward onto ts; rows
    before the symbol's first settlement stay NaN (LightGBM routes it)."""
    f = pd.read_parquet(FUNDING_CACHE_DIR / f"{symbol}.parquet").sort_values("funding_time").reset_index(drop=True)
    s = f.set_index("funding_time")["funding_rate"]
    roll = s.rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS)
    f["f_rate"] = f["funding_rate"].to_numpy()
    f["f_z"] = ((s - roll.mean()) / roll.std()).to_numpy()

    merged = pd.merge_asof(
        dataset.sort_values("ts"),
        f[["funding_time", "f_rate", "f_z"]].sort_values("funding_time"),
        left_on="ts", right_on="funding_time", direction="backward",
    )
    merged["f_absz"] = merged["f_z"].abs()
    return merged.drop(columns=["funding_time"]), ["f_rate", "f_z", "f_absz"]


def attach_microstructure_features(dataset: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """OHLCV-only microstructure proxies (accuracy-program check #4) -- zero
    new data, all from the O/H/L/C/V that build_dataset already passes
    through. All strictly causal (rolling on past bars, no forward shift).
    Chosen to target channels the 93 existing features don't cover:

      amihud_{12,36}  mean |ret| / dollar_volume  -- price impact per $ traded
                      (pipeline has ret_std / ATR but nothing volume-scaled)
      gk_vol_36       mean Garman-Klass vol from OHLC -- range-efficient vol
                      (pipeline vol is close-to-close ret_std + ATR only)
      body_frac_12    mean |C-O| / (H-L)          -- intrabar decisiveness
      clv_{12,36}     mean ((C-L)-(H-C)) / (H-L)  -- close location in bar,
                      a crude order-flow-direction proxy
      signed_vol_12   mean sign(ret) * volume / SMA(volume,12) -- stationary
                      signed volume (OBV is cumulative / non-stationary)
    """
    df = dataset.sort_values("ts").copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    ret = np.log(c / c.shift(1))
    rng = (h - l).replace(0.0, np.nan)
    dollar_vol = (v * c).replace(0.0, np.nan)

    amihud = ret.abs() / dollar_vol
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o.replace(0.0, np.nan)) ** 2
    body_frac = (c - o).abs() / rng
    clv = ((c - l) - (h - c)) / rng
    signed_vol = np.sign(ret) * v / v.rolling(12).mean()

    df["amihud_12"] = amihud.rolling(12).mean()
    df["amihud_36"] = amihud.rolling(36).mean()
    df["gk_vol_36"] = gk.rolling(36).mean()
    df["body_frac_12"] = body_frac.rolling(12).mean()
    df["clv_12"] = clv.rolling(12).mean()
    df["clv_36"] = clv.rolling(36).mean()
    df["signed_vol_12"] = signed_vol.rolling(12).mean()

    cols = ["amihud_12", "amihud_36", "gk_vol_36", "body_frac_12", "clv_12", "clv_36", "signed_vol_12"]
    return df, cols


OFI_CACHE_DIR = Path("reports/ofi_cache")


def _ofi_series(symbol: str) -> pd.DataFrame:
    """5-min order-flow series from the aggTrades cache, with imbalance
    ratios derived from the stored raw sums. NOT yet lagged."""
    o = pd.read_parquet(OFI_CACHE_DIR / f"{symbol}.parquet").sort_values("ts").reset_index(drop=True)
    bv, sv = o["buy_vol"], o["sell_vol"]
    bc, sc = o["buy_cnt"], o["sell_cnt"]
    lb, ls = o["large_buy_vol"], o["large_sell_vol"]
    o["ofi_vol"] = (bv - sv) / (bv + sv).replace(0.0, np.nan)
    o["ofi_cnt"] = (bc - sc) / (bc + sc).replace(0.0, np.nan)
    o["ofi_large"] = (lb - ls) / (lb + ls).replace(0.0, np.nan)
    o["avg_trade_size"] = o["total_vol"] / o["n_trades"].replace(0, np.nan)
    return o


def attach_ofi_features(dataset: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Trade-level order-flow-imbalance features (accuracy-program check #5) --
    first candidate NOT derived from OHLCV. Source: scripts/fetch_agg_trades_ofi.py.

    Every feature is lagged by >=1 5-min bin (.shift(1) after any rolling),
    matching how returns enter the pipeline as ret_lag_1..24 -- so nothing
    uses the same bar whose close anchors the target, no same-bar ambiguity.
    Exact ts merge (both series are openTime-stamped 5-min bins, verified
    corr=1.0 against candle volume). Rows outside OFI coverage stay NaN.
    """
    o = _ofi_series(symbol)
    feat = pd.DataFrame({"ts": o["ts"]})
    feat["ofi_vol_lag1"] = o["ofi_vol"].shift(1)
    feat["ofi_cnt_lag1"] = o["ofi_cnt"].shift(1)
    feat["ofi_large_lag1"] = o["ofi_large"].shift(1)
    feat["ofi_vol_roll12"] = o["ofi_vol"].rolling(12).mean().shift(1)
    feat["ofi_vol_roll36"] = o["ofi_vol"].rolling(36).mean().shift(1)
    feat["ofi_cnt_roll12"] = o["ofi_cnt"].rolling(12).mean().shift(1)
    feat["trade_intensity_lag1"] = (o["n_trades"] / o["n_trades"].rolling(36).mean()).shift(1)
    ats = o["avg_trade_size"]
    feat["avg_trade_size_z_lag1"] = ((ats - ats.rolling(288).mean()) / ats.rolling(288).std()).shift(1)

    cols = [c for c in feat.columns if c != "ts"]
    merged = dataset.merge(feat, on="ts", how="left")
    return merged, cols


def attach_news_features(dataset: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Level-1 FinBERT time-decay news features (news/decay.NEWS_FEATURE_COLUMNS:
    sentiment_weighted, news_intensity, max_magnitude_60m, sentiment_dispersion,
    event_hack_60m, event_regulation_60m). Full 2018-2026 coverage.

    Re-test of a feature set already closed negative for DIRECTION on a
    lower-power fold-level view (project_sentiment_features_deprioritized,
    screen_intermediate_horizons stop-rule) -- rerun here only because the
    day-block bootstrap is ~30x more sensitive and news is the one candidate
    orthogonal to price/volume. compute_news_features is a no-lookahead pure
    function (published_at <= t); exact ts merge.
    """
    from pinance_ml.data.news_db import load_news
    from pinance_ml.news.decay import NEWS_FEATURE_COLUMNS, base_asset, compute_news_features

    news = load_news(asset=base_asset(symbol))
    feats = compute_news_features(news, dataset["ts"]).reset_index(drop=True)
    merged = pd.concat([dataset.reset_index(drop=True), feats], axis=1)
    # keyword event flags are bool -> float so LightGBM treats them numerically
    for c in ("event_hack_60m", "event_regulation_60m"):
        if c in merged.columns:
            merged[c] = merged[c].astype("float64")
    return merged, list(NEWS_FEATURE_COLUMNS)


FEATURE_SETS = {
    "funding": attach_funding_features,
    "micro": attach_microstructure_features,
    "ofi": attach_ofi_features,
    "news": attach_news_features,
}
