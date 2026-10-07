"""Cache Binance USD-M perpetual funding-rate history to
reports/funding_cache/{symbol}.parquet.

Public endpoint GET /fapi/v1/fundingRate, paginated by startTime (limit
1000, ~8h between settlements, ~8 pages per symbol for the full history).
Idempotent: an existing cache is extended from its last stored settlement
rather than refetched.

This is a local experiment cache (reports/ is gitignored): the read-only
training DB credential can't create a new table, and funding history
isn't a production ingestion path -- it's input for the accuracy-program
check #2 screen (scripts/screen_funding_signal.py).

Usage: python scripts/fetch_funding_rates.py [SYMBOL ...]   (default: all 4 perps)
"""

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
CACHE_DIR = Path("reports/funding_cache")
# BTC-USDⓈ perp launched 2019-09-08; the other three are later. A start
# before the earliest settlement just returns the first real page.
EARLIEST_MS = int(dt.datetime(2019, 9, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_since(symbol: str, start_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while True:
        resp = requests.get(
            FAPI, params={"symbol": symbol, "startTime": cursor, "limit": 1000}, timeout=30
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if len(batch) < 1000 or last <= cursor:
            break
        cursor = last + 1
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame({"funding_time": pd.to_datetime([], utc=True), "funding_rate": []})
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return (
        df[["funding_time", "funding_rate"]]
        .drop_duplicates("funding_time")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def update_symbol(symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        start_ms = int(existing["funding_time"].max().timestamp() * 1000) + 1
        log(f"{symbol}: cache {len(existing)} rows through {existing['funding_time'].max()}, extending")
    else:
        existing = pd.DataFrame({"funding_time": pd.to_datetime([], utc=True), "funding_rate": []})
        start_ms = EARLIEST_MS
        log(f"{symbol}: no cache, full fetch")

    fresh = fetch_since(symbol, start_ms)
    combined = (
        pd.concat([existing, fresh], ignore_index=True)
        .drop_duplicates("funding_time")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )
    combined.to_parquet(path, index=False)
    span = f"{combined['funding_time'].min()} .. {combined['funding_time'].max()}" if len(combined) else "empty"
    log(f"{symbol}: +{len(fresh)} new -> {len(combined)} total ({span}) -> {path}")
    return path


def main() -> None:
    symbols = sys.argv[1:] or DEFAULT_SYMBOLS
    for s in symbols:
        update_symbol(s)


if __name__ == "__main__":
    main()
