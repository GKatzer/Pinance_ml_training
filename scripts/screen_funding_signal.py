"""One-off: cheap full-history marginal screen for funding-rate signal at
5-60 minute horizons (accuracy-improvement program check #2). No LightGBM,
no walk-forward -- conditional means / OLS on the full candle panel plus a
funding-period-level robustness test, the same screen-before-you-spend-
compute discipline as screen_quantile_magnitude_signal.py and
screen_intermediate_horizons.py. Only a channel that clears this is worth
escalating to a walk-forward LightGBM A/B.

Two channels:
  * directional -- does an extreme (z-scored) funding rate predict a
    SIGNED forward return? overheated-longs mean-reversion hypothesis:
    high funding -> negative forward return, i.e. a negative OLS slope.
  * magnitude -- does |funding z| predict |forward return|? feeds the
    confidence corridor, not the point forecast.

Horizons: r_6 (30m) and r_12 (60m) are what the program cares about;
r_96 (8h, one funding interval) is a sanity horizon -- the classic
funding effect is documented there, so a null at 8h too would point at a
data/regime problem rather than "real effect, just not at our horizon".

Funding features, all causal (funding_time is the settlement instant, the
rate is known from then on):
  f_rate  last realized funding rate
  f_z     (f_rate - trailing-30d mean) / trailing-30d std, computed on the
          funding series itself then as-of merged onto candles
  f_absz  |f_z|

Autocorrelation caveat: funding is piecewise-constant for 8h and returns
are autocorrelated, so full-panel OLS t-stats are overstated. Panel slopes
are reported with Newey-West HAC se (lag = 96 candles = 8h); the headline
significance is the funding-period-level test -- one forward return per
settlement, settlements ~8h apart vs a <=60m forward window, so
near-independent and an ordinary t / Spearman is valid.

Usage: python scripts/screen_funding_signal.py [SYMBOL ...]   (default: all 4)
Requires reports/funding_cache/{symbol}.parquet (scripts/fetch_funding_rates.py).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from pinance_ml.config import CANDLE_INTERVAL_MINUTES
from pinance_ml.data.db import load_candles
from pinance_ml.features.targets import compute_log_return_targets
from pinance_ml.tracking import log_research_run

HORIZONS = [6, 12, 96]  # 30m, 60m, 8h(sanity)
Z_WINDOW = "30D"
Z_MIN_PERIODS = 30
HAC_LAG = 96  # candles in 8h
CACHE_DIR = Path("reports/funding_cache")
OUT_DIR = Path("reports")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def funding_with_z(symbol: str) -> pd.DataFrame:
    f = pd.read_parquet(CACHE_DIR / f"{symbol}.parquet").sort_values("funding_time").reset_index(drop=True)
    s = f.set_index("funding_time")["funding_rate"]
    roll = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
    f["f_z"] = ((s - roll.mean()) / roll.std()).to_numpy()
    return f  # funding_time, funding_rate, f_z


def ols_hac(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float, float, int]:
    """OLS y = a + b*x; return (b, Newey-West t on b, R^2, n)."""
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
    se_b = float(np.sqrt(cov[1, 1]))
    r2 = 1.0 - float(np.sum(resid**2)) / float(np.sum((y - y.mean()) ** 2))
    return float(beta[1]), float(beta[1] / se_b) if se_b > 0 else np.nan, r2, n


def _binned(value: pd.Series, by: pd.Series, q: int) -> pd.DataFrame:
    d = pd.DataFrame({"value": value.to_numpy(), "by": by.to_numpy()}).dropna()
    d["b"] = pd.qcut(d["by"], q=q, labels=False, duplicates="drop")
    return d


def bucket_table(value: pd.Series, by: pd.Series, q: int, labels: list[str]) -> pd.DataFrame:
    d = _binned(value, by, q)
    t = d.groupby("b")["value"].agg(["count", "mean", "std"])
    t.index = [labels[i] for i in t.index]
    return t


def kruskal_by_bucket(value: pd.Series, by: pd.Series, q: int):
    d = _binned(value, by, q)
    return stats.kruskal(*[g["value"].to_numpy() for _, g in d.groupby("b")])


def screen_symbol(symbol: str, summary_rows: list[dict]) -> None:
    log(f"\n{'=' * 70}\n{symbol}\n{'=' * 70}")
    candles = load_candles(symbol)
    targets = compute_log_return_targets(candles, HORIZONS, CANDLE_INTERVAL_MINUTES)
    f = funding_with_z(symbol)

    panel = pd.merge_asof(
        targets[["ts", *[f"r_{h}" for h in HORIZONS]]].sort_values("ts"),
        f.rename(columns={"funding_time": "ts_f"}).sort_values("ts_f"),
        left_on="ts", right_on="ts_f", direction="backward",
    )
    cov = panel["f_z"].notna().mean()
    log(f"{len(panel):,} candles, funding-z coverage {cov:.1%} "
        f"(f_z range [{panel['f_z'].min():.2f}, {panel['f_z'].max():.2f}], "
        f"f_rate median {panel['funding_rate'].median() * 1e4:.2f}bps)")

    # one forward return per settlement: first candle at/after the settlement
    per = pd.merge_asof(
        f.sort_values("funding_time"),
        targets[["ts", *[f"r_{h}" for h in HORIZONS]]].sort_values("ts"),
        left_on="funding_time", right_on="ts", direction="forward",
        tolerance=pd.Timedelta("10min"),
    ).dropna(subset=["f_z"])
    log(f"{len(per):,} funding periods with a matched forward candle")

    for h in HORIZONS:
        rcol = f"r_{h}"

        # ---- directional channel ----
        pslope, pt, pr2, pn = ols_hac(panel["f_z"].to_numpy(), panel[rcol].to_numpy(), HAC_LAG)
        pe = per.dropna(subset=[rcol])
        lr = stats.linregress(pe["f_z"], pe[rcol])
        rho, rho_p = stats.spearmanr(pe["f_z"], pe[rcol])
        qd = bucket_table(panel[rcol], panel["f_z"], 5, ["z_lo", "z2", "z3", "z4", "z_hi"])
        spread_bps = (qd["mean"].iloc[-1] - qd["mean"].iloc[0]) * 1e4
        log(f"\n[{symbol} r_{h}] DIRECTIONAL  (mean-reversion => negative slope)")
        log(f"  panel  OLS slope={pslope:.5f}  t_HAC={pt:+.2f}  R2={pr2:.2e}  n={pn:,}")
        log(f"  period OLS slope={lr.slope:.5f}  t={lr.slope / lr.stderr:+.2f}  "
            f"Spearman rho={rho:+.4f} p={rho_p:.2e}  n={len(pe):,}")
        log("  mean r by funding-z quintile (bps):\n"
            + (qd["mean"] * 1e4).round(3).to_string().replace("\n", "\n  "))
        log(f"  Q(z_hi) - Q(z_lo) spread = {spread_bps:+.3f} bps")

        # ---- magnitude channel ----
        mslope, mt, mr2, mn = ols_hac(panel["f_z"].abs().to_numpy(), panel[rcol].abs().to_numpy(), HAC_LAG)
        lr_abs = stats.linregress(pe["f_z"].abs(), pe[rcol].abs())
        qm = bucket_table(panel[rcol].abs(), panel["f_z"].abs(), 4, ["a_lo", "a2", "a3", "a_hi"])
        kw = kruskal_by_bucket(panel[rcol].abs(), panel["f_z"].abs(), 4)
        log(f"\n[{symbol} r_{h}] MAGNITUDE  (|f_z| -> |r|, corridor channel; expect positive)")
        log(f"  panel  OLS |slope|={mslope:.5f}  t_HAC={mt:+.2f}  R2={mr2:.2e}")
        log(f"  period linregress slope={lr_abs.slope:.5f}  r={lr_abs.rvalue:+.4f}  p={lr_abs.pvalue:.2e}")
        log("  mean |r| by |funding-z| quartile (bps):\n"
            + (qm["mean"] * 1e4).round(3).to_string().replace("\n", "\n  "))
        log(f"  Kruskal-Wallis H={kw.statistic:.1f} p={kw.pvalue:.2e}")

        summary_rows.append({
            "symbol": symbol, "horizon": h,
            "dir_panel_slope": pslope, "dir_panel_t_hac": pt, "dir_panel_r2": pr2,
            "dir_period_slope": lr.slope, "dir_period_t": lr.slope / lr.stderr,
            "dir_period_spearman_rho": rho, "dir_period_spearman_p": rho_p,
            "dir_quintile_spread_bps": spread_bps,
            "mag_panel_slope": mslope, "mag_panel_t_hac": mt,
            "mag_period_slope": lr_abs.slope, "mag_period_r": lr_abs.rvalue, "mag_period_p": lr_abs.pvalue,
            "n_periods": len(pe),
        })


def main() -> None:
    symbols = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for s in symbols:
        screen_symbol(s, summary_rows)

    summary = pd.DataFrame(summary_rows)
    out = OUT_DIR / "funding_screen_summary.csv"
    summary.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        log("\n\n=== SUMMARY (all symbols x horizons) ===\n" + summary.to_string(index=False))
    log(f"\nSaved -> {out}")
    log("\nRead: directional needs a consistent NEGATIVE slope + Spearman p<<0.01 at r_6/r_12 "
        "AND a quintile spread of more than a few bps to be worth a walk-forward A/B. "
        "Magnitude needs a positive period slope with p<<0.01. r_96 is context only.")

    log_research_run(
        __file__,
        run_name=f"funding-signal-{'+'.join(symbols)}" if len(symbols) <= 4 else f"funding-signal-{len(symbols)}sym",
        params={
            "symbols": ",".join(symbols),
            "z_min_periods": Z_MIN_PERIODS,
            "hac_lag": HAC_LAG,
        },
        report_paths=sorted(OUT_DIR.glob("funding_screen_*")),
    )


if __name__ == "__main__":
    main()
