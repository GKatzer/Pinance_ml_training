"""Draw every figure in docs/media/ from the committed result files in reports/.

    python docs/figures/make_figures.py            # writes docs/media/*.png
    python docs/figures/make_figures.py > docs/figures/figure_numbers.txt

Needs only pandas, numpy, scipy and matplotlib; no database. The numbers
each figure is based on are printed to stdout so the README tables can be
checked against them. Fixed seeds and a fixed palette: re-running produces
the same pictures.

Conventions used throughout (same as reports/README.md):
  * a "fold" is one yearly walk-forward test window (8 for BTC/ETH/BNB, 6 for SOL);
  * per-fold values are averages over the 12 horizons (5 .. 60 minutes);
  * pooling across folds is weighted by each fold's n_test;
  * the naive baseline (r = 0 for MAE, majority sign of the training window
    for directional accuracy, DA) comes from the weekly folds in
    naive_baseline_folds.csv whose test start falls inside the yearly window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MEDIA = ROOT / "docs" / "media"
sys.path.insert(0, str(ROOT / "src"))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
C_MODEL, C_NAIVE, C_GREY, C_BAD, C_REF = "#1f6fb2", "#8a8f98", "#c9ccd1", "#c0392b", "#222222"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e8eb",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)


def read(name: str) -> pd.DataFrame:
    df = pd.read_csv(REPORTS / name)
    for c in ("test_start", "test_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def save(fig: plt.Figure, name: str) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, bbox_inches="tight")
    plt.close(fig)
    print(f"-> docs/media/{name}\n")


# ---------------------------------------------------------------- shared data

def lightgbm_vs_naive() -> pd.DataFrame:
    """One row per (symbol, fold, horizon): LightGBM and naive MAE / DA on the
    same yearly window."""
    lg = pd.concat([read("lightgbm_folds.csv"), read("lightgbm_folds_rest.csv")], ignore_index=True)
    nv = read("naive_baseline_folds.csv")
    rows = []
    for (sym, fold), g in lg.groupby(["symbol", "fold"]):
        start, end = g["test_start"].iloc[0], g["test_end"].iloc[0]
        weekly = nv[(nv["symbol"] == sym) & (nv["test_start"] >= start) & (nv["test_start"] < end)]
        for h, gh in g.groupby("horizon"):
            wh = weekly[weekly["horizon"] == h]
            w = wh["n_test"]
            rows.append(
                {
                    "symbol": sym,
                    "fold": fold,
                    "test_start": start,
                    "horizon": h,
                    "n_test": gh["n_test"].iloc[0],
                    "mae": gh["mae"].iloc[0],
                    "da": gh["directional_accuracy"].iloc[0],
                    "naive_mae": (wh["mae"] * w).sum() / w.sum(),
                    "naive_da": (wh["directional_accuracy"] * w).sum() / w.sum(),
                }
            )
    return pd.DataFrame(rows)


def wmean(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())


# ------------------------------------------------------------------- figures

def fig_da_by_horizon(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    print("DIRECTIONAL ACCURACY BY HORIZON (pooled over folds, n_test-weighted)")
    for ax, sym in zip(axes, SYMBOLS):
        g = d[d["symbol"] == sym]
        by_h = g.groupby("horizon").apply(
            lambda x: pd.Series(
                {"da": wmean(x["da"], x["n_test"]), "naive": wmean(x["naive_da"], x["n_test"]), "n": x["n_test"].sum()}
            ),
            include_groups=False,
        )
        ax.plot(by_h.index * 5, by_h["da"] * 100, color=C_MODEL, marker="o", ms=4, label="LightGBM")
        ax.plot(by_h.index * 5, by_h["naive"] * 100, color=C_NAIVE, marker="s", ms=3.5, label="majority-sign constant")
        ax.axhline(50, color=C_REF, lw=0.8, ls=":")
        ax.set_title(f"{sym}  ({g['fold'].nunique()} folds)", fontsize=10)
        ax.set_xlabel("horizon, minutes")
        print(
            f"{sym}: DA h=1 {by_h['da'].iloc[0]*100:.2f}%  h=2..12 range "
            f"{by_h['da'].iloc[1:].min()*100:.2f}-{by_h['da'].iloc[1:].max()*100:.2f}%  "
            f"naive range {by_h['naive'].min()*100:.2f}-{by_h['naive'].max()*100:.2f}%  rows/horizon {int(by_h['n'].iloc[0]):,}"
        )
    axes[0].set_ylabel("directional accuracy, %")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Directional accuracy by horizon against the majority-sign constant (walk-forward, yearly folds)", y=1.03, fontsize=11)
    save(fig, "da-by-horizon.png")


def fig_btc_by_year(d: pd.DataFrame) -> None:
    g = d[d["symbol"] == "BTCUSDT"].groupby("fold").apply(
        lambda x: pd.Series(
            {
                "start": x["test_start"].iloc[0],
                "da": x["da"].mean(),
                "naive": x["naive_da"].mean(),
                "n": x["n_test"].iloc[0],
            }
        ),
        include_groups=False,
    )
    labels = [f"{s.year}-{s.month:02d}" for s in g["start"]]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, g["da"] * 100, 0.4, color=C_MODEL, label="LightGBM")
    ax.bar(x + 0.2, g["naive"] * 100, 0.4, color=C_NAIVE, label="majority-sign constant")
    ax.axhline(50, color=C_REF, lw=0.8, ls=":")
    ax.set_ylim(48, 56)
    ax.set_xticks(x, [f"from\n{l}\n(n={int(n/1000)}k)" for l, n in zip(labels, g["n"])], fontsize=8)
    ax.set_ylabel("directional accuracy, %")
    ax.set_title("BTCUSDT: directional accuracy per yearly test window (mean over 12 horizons)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    save(fig, "da-btc-by-year.png")
    print("BTC DA per fold:", ", ".join(f"{l}: {v*100:.2f}% (naive {n*100:.2f}%)" for l, v, n in zip(labels, g["da"], g["naive"])))
    print()


def fig_mae_vs_naive(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    print("MAE AND DA VS NAIVE (the README results table)")
    print("symbol  folds  MAE_lgbm  MAE_naive  dMAE%   p(MAE)  folds_better  DA_lgbm  DA_naive  dDA_pp  folds_DA_better  p(DA)")
    for sym, color, mk in zip(SYMBOLS, [C_MODEL, "#e08a1e", "#2e9e6b", "#8e5bd0"], "osD^"):
        pf = (
            d[d["symbol"] == sym]
            .groupby("fold")
            .agg(start=("test_start", "first"), n=("n_test", "first"), mae=("mae", "mean"), nmae=("naive_mae", "mean"),
                 da=("da", "mean"), nda=("naive_da", "mean"))
            .reset_index()
        )
        rel = (pf["mae"] / pf["nmae"] - 1) * 100
        ax.plot(pf["start"].dt.year + pf["start"].dt.month / 12, rel, color=color, marker=mk, ms=4, lw=1.2, label=sym)
        k = len(pf)
        p_mae = stats.ttest_rel(pf["mae"], pf["nmae"]).pvalue
        p_da = stats.ttest_rel(pf["da"], pf["nda"]).pvalue
        print(
            f"{sym:8s}{k:4d}   {wmean(pf['mae'], pf['n']):.6f}  {wmean(pf['nmae'], pf['n']):.6f}  "
            f"{(wmean(pf['mae'], pf['n']) / wmean(pf['nmae'], pf['n']) - 1) * 100:+6.2f}  {p_mae:6.3f}   {(pf['mae'] < pf['nmae']).sum()}/{k}        "
            f"{wmean(pf['da'], pf['n'])*100:6.2f}  {wmean(pf['nda'], pf['n'])*100:6.2f}   {(wmean(pf['da'], pf['n'])-wmean(pf['nda'], pf['n']))*100:+5.2f}   "
            f"{(pf['da'] > pf['nda']).sum()}/{k}            {p_da:.4f}"
        )
    ax.axhline(0, color=C_REF, lw=0.8)
    ax.set_ylabel("MAE relative to naive r = 0, %\n(above 0 = worse than predicting zero)")
    ax.set_xlabel("start of the yearly test window")
    ax.set_title("MAE of the point model relative to 'predict zero', per yearly fold: no consistent gain", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    save(fig, "mae-vs-naive-by-fold.png")


def fig_corridor() -> None:
    q = read("quantile_gain_folds.csv")
    base = q[q["variant"] == "technical_only"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    print("CORRIDOR COVERAGE (BTCUSDT, technical_only, 9 folds, n_test-weighted)")
    for ax, level in zip(axes, (0.1, 0.9)):
        g = base[base["quantile"] == level]
        by_h = g.groupby("horizon").apply(lambda x: wmean(x["coverage"], x["n_test"]), include_groups=False)
        ax.bar(by_h.index * 5, by_h.values, width=4, color=C_MODEL)
        ax.axhline(level, color=C_BAD, lw=1.2, label=f"target {level}")
        ax.axhspan(level - 0.03, level + 0.03, color=C_BAD, alpha=0.1, label="tolerance +/- 0.03")
        ax.set_ylim(level - 0.06, level + 0.06)
        ax.set_xlabel("horizon, minutes")
        ax.set_title(f"share of outcomes at or below the q{level} bound", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        per_fold = g.groupby("fold").apply(lambda x: x["coverage"].mean(), include_groups=False)
        within = int(((per_fold - level).abs() <= 0.03).sum())
        pooled = wmean(g["coverage"], g["n_test"])
        print(f"q{level}: pooled coverage {pooled:.4f}; folds within +/-0.03: {within}/{len(per_fold)}; worst fold {per_fold.index[np.argmax((per_fold-level).abs())]} = {per_fold.max() if level==0.1 else per_fold.min():.3f}")
    fig.suptitle("Baseline confidence corridor: out-of-sample coverage by horizon (BTCUSDT, 9 walk-forward folds)", y=1.03, fontsize=11)
    save(fig, "corridor-coverage.png")


def delta_interval(variant_a: pd.DataFrame, variant_b: pd.DataFrame, col: str) -> tuple[float, float, float, int]:
    """Mean over folds of (b - a) of the per-fold horizon-average of `col`,
    and its 95% t-interval. Unweighted, because the interval is across folds."""
    a = variant_a.groupby("fold")[col].mean()
    b = variant_b.groupby("fold")[col].mean()
    delta = (b - a).dropna()
    k = len(delta)
    m = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(k))
    h = float(stats.t.ppf(0.975, k - 1) * se)
    return m, m - h, m + h, k


def fig_news_effects() -> None:
    items = []

    news = read("news_feature_gain_folds.csv")
    for sym in SYMBOLS:
        s = news[news["symbol"] == sym]
        items.append((f"FinBERT features, {sym[:3]}", s[s["variant"] == "technical_only"], s[s["variant"] == "with_news"]))

    lora = read("finbert_lora_gain_folds.csv")
    variants = sorted(lora["variant"].unique())
    items.append(("LoRA-tuned FinBERT, BTC", lora[lora["variant"] == "technical_only"], lora[lora["variant"] != "technical_only"]))

    llm = read("llm_event_gain_folds.csv")
    items.append(("LLM event type vs FinBERT, BTC", llm[llm["variant"] == "with_level1_news"], llm[llm["variant"] == "with_level1_and_llm_events"]))

    h24 = read("horizon_24h_baseline_folds.csv").dropna(subset=["directional_accuracy"])
    v24 = sorted(h24["variant"].unique())
    items.append(("24 h horizon: + LLM event, BTC", h24[h24["variant"] == v24[0]], h24[h24["variant"] == v24[-1]]))

    print("NEWS-FEATURE EXPERIMENTS: change in directional accuracy (pp), unweighted mean over folds, 95% t-interval")
    print(f"(LoRA variants present: {variants}; 24h variants present: {v24})")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (label, a, b) in enumerate(items):
        m, lo, hi, k = delta_interval(a, b, "directional_accuracy")
        ax.errorbar(m * 100, i, xerr=[[(m - lo) * 100], [(hi - m) * 100]], fmt="o", color=C_MODEL, capsize=3)
        ax.text(1.0 + 0.02, i, f"{m*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}]  k={k}", va="center", fontsize=8, transform=ax.get_yaxis_transform())
        print(f"{label:34s} {m*100:+.3f} pp  [{lo*100:+.3f}, {hi*100:+.3f}]  folds={k}")
    ax.axvline(0, color=C_REF, lw=0.8)
    ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("change in directional accuracy, percentage points (right = better)")
    ax.set_title("News features: every interval contains zero", fontsize=10)
    ax.set_xlim(-1, 1)
    save(fig, "news-effects.png")


def fig_obv() -> None:
    old = read("obv_fix_btc_old_folds.csv")
    new = read("obv_fix_btc_new_folds.csv")
    print("OBV FIX (BTCUSDT, walk-forward, current LightGBM defaults)")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.35})
    for ax, col, title in zip(axes[:2], ["directional_accuracy", "mae"], ["directional accuracy, %", "MAE (log-return)"]):
        a = old.groupby("fold")[col].mean()
        b = new.groupby("fold")[col].mean()
        n = old.groupby("fold")["n_test"].first()
        scale = 100 if col == "directional_accuracy" else 1
        x = np.arange(len(a))
        ax.plot(x, a * scale, color=C_NAIVE, marker="s", ms=4, label="cumulative obv (before)")
        ax.plot(x, b * scale, color=C_MODEL, marker="o", ms=4, label="obv_roc_36 (after)")
        ax.set_xlabel("fold")
        ax.set_title(title, fontsize=10)
        if col == "mae":
            ax.legend(frameon=False, fontsize=8)
        p = stats.ttest_rel(a, b).pvalue
        print(
            f"{col}: before {wmean(a, n)*scale:.5f}  after {wmean(b, n)*scale:.5f}  "
            f"delta {(wmean(b, n)-wmean(a, n))*scale:+.5f}  (relative {(wmean(b, n)/wmean(a, n)-1)*100:+.2f}%)  paired t-test p={p:.3f}  folds={len(a)}"
        )

    # Synthetic illustration of the skew (random-walk candles, NOT market data).
    import ta

    rng = np.random.default_rng(0)
    n = 30000
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    volume = pd.Series(rng.lognormal(3, 0.5, n))

    def old_obv(c: pd.Series, v: pd.Series) -> pd.Series:
        return ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()

    def new_obv(c: pd.Series, v: pd.Series, w: int = 36) -> pd.Series:
        return (np.sign(c.diff()) * v).rolling(w).sum() / v.rolling(w).sum()

    ts = list(range(20000, n, 500))
    full_old = np.array([old_obv(close.iloc[:t + 1], volume.iloc[:t + 1]).iloc[-1] for t in ts])
    win_old = np.array([old_obv(close.iloc[t - 149:t + 1].reset_index(drop=True), volume.iloc[t - 149:t + 1].reset_index(drop=True)).iloc[-1] for t in ts])
    full_new = new_obv(close, volume).iloc[ts].to_numpy()
    win_new = np.array([new_obv(close.iloc[t - 149:t + 1].reset_index(drop=True), volume.iloc[t - 149:t + 1].reset_index(drop=True)).iloc[-1] for t in ts])
    ax = axes[2]
    mismatch_old = np.abs(full_old - win_old).mean() / full_old.std()
    mismatch_new = max(np.abs(full_new - win_new).mean() / full_new.std(), 1e-17)
    ax.bar([0, 1], [mismatch_old, mismatch_new], color=[C_NAIVE, C_MODEL], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["cumulative obv", "obv_roc_36"])
    ax.set_ylabel("|full history - 150-candle window|\n/ std of the feature (log scale)")
    ax.set_title("Training vs serving mismatch, same candle\n(synthetic random-walk candles, not market data)", fontsize=9)
    print(f"synthetic mismatch ratio: cumulative obv {mismatch_old:.3f}, obv_roc_36 {mismatch_new:.2e}")
    print(
        f"synthetic parity check, {len(ts)} candles: max |full - window| cumulative obv = {np.abs(full_old - win_old).max():.1f} "
        f"(values span {full_old.min():.0f}..{full_old.max():.0f}); obv_roc_36 = {np.abs(full_new - win_new).max():.2e}"
    )
    fig.suptitle("The obv fix: no visible change in offline metrics, identical features between training and serving", y=1.04, fontsize=11)
    save(fig, "obv-fix.png")


def fig_tuning() -> None:
    from pinance_ml.evaluation import directional_block_bootstrap

    daily = pd.read_csv(REPORTS / "gridsearch" / "confirm_daily.csv", parse_dates=["day"])
    print("HYPERPARAMETER TUNING, BTCUSDT: pooled change in DA against the default, day-block bootstrap (7-day blocks, 4000 draws)")
    names = {
        "reg_strong": "reg_strong",
        "grid_nl7_mcs100_mid_k100": "grid winner:\n7 leaves, 100 trees",
        "grid_nl7_mcs300_strong_k100": "grid #2",
        "grid_nl7_mcs100_none_k100": "grid #3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True, gridspec_kw={"wspace": 0.55})
    for ax, (title, mask) in zip(axes, [("selection folds 0-4 (2018-2023)", daily["fold"] <= 4), ("held-out folds 5-8 (2023-2026)", daily["fold"] >= 5)]):
        d = daily[mask]
        for i, (v, label) in enumerate(names.items()):
            r = directional_block_bootstrap(d, "default", v, n_boot=4000)
            p = r[r["horizon"] == "pooled"].iloc[0]
            ok = p["ci_lo_pp"] > 0
            ax.errorbar(p["delta_da_pp"], i, xerr=[[p["delta_da_pp"] - p["ci_lo_pp"]], [p["ci_hi_pp"] - p["delta_da_pp"]]],
                        fmt="o", color=C_MODEL if ok else C_BAD, capsize=3)
            print(f"{title:34s} {v:30s} dDA {p['delta_da_pp']:+.3f} pp  CI [{p['ci_lo_pp']:+.3f}, {p['ci_hi_pp']:+.3f}]  dMAE {p['delta_mae_pct']:+.2f}%  days={int(p['n_days'])}")
        ax.axvline(0, color=C_REF, lw=0.8)
        ax.set_yticks(range(len(names)), list(names.values()), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("change in directional accuracy vs default, pp")
    fig.suptitle("Tuning chosen on folds 0-4 does not carry over to folds 5-8 (BTCUSDT, 12 horizons pooled)", y=1.03, fontsize=11)
    save(fig, "tuning-select-vs-judge.png")


def main() -> None:
    d = lightgbm_vs_naive()
    fig_da_by_horizon(d)
    fig_btc_by_year(d)
    fig_mae_vs_naive(d)
    fig_corridor()
    fig_news_effects()
    fig_obv()
    fig_tuning()


if __name__ == "__main__":
    main()
