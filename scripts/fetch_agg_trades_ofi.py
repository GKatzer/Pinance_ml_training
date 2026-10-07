"""Download Binance spot aggTrades daily dumps and reduce them to 5-minute
order-flow aggregates -> reports/ofi_cache/{symbol}.parquet.

Accuracy-program check #5: trade-level order-flow imbalance, the first
candidate built on data that is NOT an OHLCV transform (checks #1-#4, all
OHLCV-derived, were negative on one shared mechanism -- existing price/vol
features already carry the signal).

Source: https://data.binance.vision/data/spot/daily/aggTrades/{SYM}/
{SYM}-aggTrades-YYYY-MM-DD.zip -- one CSV, no header, columns:
  agg_id, price, qty, first_id, last_id, transact_time(us), is_buyer_maker, is_best_match
is_buyer_maker == True  => the aggressor was the SELLER (sell-side market order)
is_buyer_maker == False => the aggressor was the BUYER

Each day is streamed from memory (nothing but the final parquet touches
disk), reduced to 288 5-minute rows, appended. Idempotent: days already in
the cache are skipped. Per 5-min bin the parquet stores raw sums only --
buy/sell volume & count, plus the same split for large trades (qty above
that day's 90th percentile) -- so the imbalance ratios and any
normalisation are computed downstream, not baked in here.

Usage:
    python scripts/fetch_agg_trades_ofi.py [SYMBOL ...] [--start 2024-01-01] [--end 2026-08-27]
Default: BTCUSDT, --start 2024-01-01, --end = yesterday (UTC).
"""

import argparse
import io
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
CACHE_DIR = Path("reports/ofi_cache")
BIN = "5min"
LARGE_Q = 0.90
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "transact_time", "is_buyer_maker", "is_best_match"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def day_ofi(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(
        z.open(z.namelist()[0]), header=None, names=COLS,
        usecols=["qty", "transact_time", "is_buyer_maker"],
        dtype={"qty": "float64", "transact_time": "int64", "is_buyer_maker": "bool"},
    )
    # Binance changed aggTrades dump precision over time: pre-2025 files are
    # in ms, 2025+ in us. Detect from magnitude (epoch seconds ~1.7e9 in
    # this era, so ms ~1.7e12, us ~1.7e15, ns ~1.7e18).
    t0 = int(df["transact_time"].iloc[0])
    unit = "ns" if t0 > 1e17 else "us" if t0 > 1e14 else "ms" if t0 > 1e11 else "s"
    df["ts"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True).dt.floor(BIN)
    buy = ~df["is_buyer_maker"].to_numpy()
    q = df["qty"].to_numpy()
    large = q >= np.quantile(q, LARGE_Q)

    df["buy_vol"] = np.where(buy, q, 0.0)
    df["sell_vol"] = np.where(buy, 0.0, q)
    df["buy_cnt"] = buy.astype("int64")
    df["large_buy_vol"] = np.where(buy & large, q, 0.0)
    df["large_sell_vol"] = np.where(~buy & large, q, 0.0)

    out = df.groupby("ts").agg(
        n_trades=("qty", "size"),
        total_vol=("qty", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        buy_cnt=("buy_cnt", "sum"),
        large_buy_vol=("large_buy_vol", "sum"),
        large_sell_vol=("large_sell_vol", "sum"),
    )
    out["sell_cnt"] = out["n_trades"] - out["buy_cnt"]
    return out.reset_index()


def update_symbol(symbol: str, start: date, end: date) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}.parquet"
    have = set()
    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_parquet(path)
        have = set(pd.to_datetime(existing["ts"]).dt.date.unique())
        log(f"{symbol}: cache {len(existing)} rows, {len(have)} days through {max(have)}")

    frames = [existing] if len(existing) else []
    d = start
    n_new = 0
    while d <= end:
        if d in have:
            d += timedelta(days=1)
            continue
        url = f"{BASE}/{symbol}/{symbol}-aggTrades-{d:%Y-%m-%d}.zip"
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            log(f"{symbol} {d}: 404 (no dump yet)")
            d += timedelta(days=1)
            continue
        r.raise_for_status()
        frames.append(day_ofi(r.content))
        n_new += 1
        if n_new % 20 == 0:
            pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").to_parquet(path, index=False)
            log(f"{symbol}: {d:%Y-%m-%d} ({n_new} days fetched, {len(r.content)/1e6:.1f}MB last)")
        time.sleep(0.15)
        d += timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    log(f"{symbol}: +{n_new} days -> {len(combined)} rows, {combined['ts'].min()} .. {combined['ts'].max()} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*", default=["BTCUSDT"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat())
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    for s in args.symbols or ["BTCUSDT"]:
        update_symbol(s, start, end)


if __name__ == "__main__":
    main()
