"""Accuracy-program check #5, marginal screen: does lagged trade-level
order-flow imbalance predict short-horizon return? No LightGBM, no
walk-forward -- conditional means / HAC-OLS on the covered candle panel
plus a non-overlapping subsample test, same discipline as
screen_funding_signal.py. Escalate to the walk-forward directional A/B
(measure_directional_gain.py ofi) only if a channel clears.

Channels:
  * directional -- does ofi_*_lag1 predict SIGNED r_6 / r_12? Sign is a
    finding either way: positive = flow continuation, negative = the
    imbalance gets absorbed (liquidity provision / mean reversion).
  * magnitude  -- does |ofi_vol_lag1| predict |r|? corridor channel.

All OFI features are lagged >=1 bin (built in _feature_sets.attach_ofi_features),
so this is genuinely "last completed bar's flow -> next bar's return",
not a same-bar artifact. Panel HAC lag = 96 candles (8h). The
non-overlapping subsample (every h-th bar) removes the overlapping-target
autocorrelation for a clean ordinary-t / Spearman read.

Usage: python scripts/screen_ofi_signal.py [SYMBOL ...]   (default BTCUSDT)
Requires reports/ofi_cache/{symbol}.parquet (scripts/fetch_agg_trades_ofi.py).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from _feature_sets import attach_ofi_features
from pinance_ml.config import CANDLE_INTERVAL_MINUTES
from pinance_ml.data.db import load_candles
from pinance_ml.features.targets import compute_log_return_targets
from pinance_ml.tracking import log_research_run

HORIZONS = [6, 12]
HAC_LAG = 96
OFI_FEATURES = ["ofi_vol_lag1", "ofi_cnt_lag1", "ofi_large_lag1", "ofi_vol_roll12"]
OUT = Path("reports/ofi_screen_summary.csv")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ols_hac(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float, float, int]:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = x.size
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for lg in range(1, lag + 1):
        w = 1.0 - lg / (lag + 1.0)
        G = u[lg:].T @ u[:-lg]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(cov[1, 1]))
    r2 = 1.0 - float((resid**2).sum()) / float(((y - y.mean()) ** 2).sum())
    return float(beta[1]), (float(beta[1] / se) if se > 0 else np.nan), r2, n


def quintile_means(r: pd.Series, by: pd.Series) -> pd.Series:
    d = pd.DataFrame({"r": r.to_numpy(), "by": by.to_numpy()}).dropna()
    d["q"] = pd.qcut(d["by"], 5, labels=["lo", "2", "3", "4", "hi"], duplicates="drop")
    return d.groupby("q", observed=True)["r"].mean() * 1e4  # bps


def screen_symbol(symbol: str, rows: list[dict]) -> None:
    log(f"\n{'=' * 66}\n{symbol}\n{'=' * 66}")
    candles = load_candles(symbol)
    targets = compute_log_return_targets(candles, HORIZONS, CANDLE_INTERVAL_MINUTES)
    panel, _ = attach_ofi_features(targets[["ts", *[f"r_{h}" for h in HORIZONS]]], symbol)
    panel = panel[panel["ofi_vol_lag1"].notna()].reset_index(drop=True)
    log(f"{len(panel):,} candles with OFI coverage ({panel['ts'].min()} .. {panel['ts'].max()})")

    for feat in OFI_FEATURES:
        for h in HORIZONS:
            rcol = f"r_{h}"
            ps, pt, pr2, pn = ols_hac(panel[feat].to_numpy(), panel[rcol].to_numpy(), HAC_LAG)
            sub = panel.iloc[::h].dropna(subset=[feat, rcol])  # non-overlapping
            lr = stats.linregress(sub[feat], sub[rcol])
            rho, rho_p = stats.spearmanr(sub[feat], sub[rcol])
            qd = quintile_means(panel[rcol], panel[feat])
            spread = qd.iloc[-1] - qd.iloc[0]

            # magnitude (only for the main ofi_vol feature)
            mag = ""
            if feat == "ofi_vol_lag1":
                ms, mt, mr2, _ = ols_hac(panel[feat].abs().to_numpy(), panel[rcol].abs().to_numpy(), HAC_LAG)
                lra = stats.linregress(sub[feat].abs(), sub[rcol].abs())
                mag = f" || MAG |slope|={ms:.4f} t_HAC={mt:+.2f} sub_p={lra.pvalue:.1e}"

            log(f"[{feat} -> r_{h}] panel slope={ps:+.5f} t_HAC={pt:+.2f} R2={pr2:.1e} | "
                f"nonoverlap slope={lr.slope:+.5f} t={lr.slope / lr.stderr:+.2f} "
                f"rho={rho:+.4f} p={rho_p:.1e} | quintile spread={spread:+.2f}bps{mag}")
            log(f"    r_{h} by {feat} quintile (bps): " + ", ".join(f"{k}={v:+.2f}" for k, v in qd.items()))
            rows.append({
                "symbol": symbol, "feature": feat, "horizon": h,
                "panel_slope": ps, "panel_t_hac": pt, "panel_r2": pr2,
                "nonoverlap_slope": lr.slope, "nonoverlap_t": lr.slope / lr.stderr,
                "spearman_rho": rho, "spearman_p": rho_p, "quintile_spread_bps": spread,
            })


def main() -> None:
    symbols = sys.argv[1:] or ["BTCUSDT"]
    rows: list[dict] = []
    for s in symbols:
        screen_symbol(s, rows)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.5f}".format):
        log("\n=== SUMMARY ===\n" + df.to_string(index=False))
    log(f"\nSaved -> {OUT}")
    log("\nRead: a channel is worth the walk-forward A/B if the non-overlap Spearman p is << 0.01 "
        "with a CONSISTENT sign across r_6/r_12 AND a quintile spread of more than a few bps. "
        "Sign itself (continuation vs absorption) is the finding.")

    log_research_run(
        __file__,
        run_name=f"ofi-signal-{'+'.join(symbols)}" if len(symbols) <= 4 else f"ofi-signal-{len(symbols)}sym",
        params={
            "symbols": ",".join(symbols),
            "hac_lag": HAC_LAG,
        },
        report_paths=sorted(OUT.parent.glob("ofi_screen_*")),
    )


if __name__ == "__main__":
    main()
