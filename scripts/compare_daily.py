"""Re-score any *_daily.csv (from measure_directional_gain.py or another
harness that emits evaluation.daily_prediction_stats output) with the
calendar-day-block bootstrap. Standalone so an already-finished run can
get the higher-power DA read without retraining.

Usage:
    python scripts/compare_daily.py reports/directional_gain_funding_daily.csv \
        --a technical_only --b with_funding [--block-days 7] [--n-boot 10000]

Omit --a/--b to auto-pick the two variants present (errors if not exactly 2).
Pass the same value for --a and --b as a null check: the delta CI should
straddle 0 tightly.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.evaluation import directional_block_bootstrap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("daily_csv")
    ap.add_argument("--a", default=None, help="baseline variant")
    ap.add_argument("--b", default=None, help="candidate variant")
    ap.add_argument("--block-days", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    daily = pd.read_csv(args.daily_csv, parse_dates=["day"])
    variants = list(daily["variant"].unique())
    a = args.a or (variants[0] if len(variants) == 2 else None)
    b = args.b or (variants[1] if len(variants) == 2 else None)
    if a is None or b is None:
        sys.exit(f"pass --a/--b explicitly; variants present: {variants}")

    boot = directional_block_bootstrap(
        daily, a, b, block_days=args.block_days, n_boot=args.n_boot, seed=args.seed
    )
    out = Path(args.daily_csv).with_name(Path(args.daily_csv).stem.replace("_daily", "") + "_bootstrap.csv")
    boot.to_csv(out, index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(f"\n{b} vs {a}  ({args.n_boot} draws, {args.block_days}-day blocks)\n")
        print(boot.to_string(index=False))
    p = boot[boot["horizon"] == "pooled"].iloc[0]
    tag = (
        "DA IMPROVES (CI excludes 0)" if p["ci_lo_pp"] > 0
        else "DA REGRESSES (CI excludes 0)" if p["ci_hi_pp"] < 0
        else "no resolvable DA effect (CI spans 0)"
    )
    print(f"\n>>> pooled dDA = {p['delta_da_pp']:+.3f}pp [{p['ci_lo_pp']:+.3f}, {p['ci_hi_pp']:+.3f}] "
          f"P(>0)={p['p_delta_gt_0']:.3f} -> {tag}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
