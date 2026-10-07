"""Builds the self-supervised training set for Level 3 LoRA fine-tuning
(README "Уровень 3"): one row per news item, labeled by the actual sign
of the asset's return 60 minutes after publication (news/labels.py).

Runs per tracked asset (BTC/ETH/BNB/SOL) since each needs its own candles
join. Output is a single CSV meant to travel to wherever the fine-tune
itself runs (README: "Обучение на десктопе") -- this script only needs
DB access, no GPU, no torch.

Usage: python scripts/build_finbert_labels.py [--neutral-threshold 0.0015] [--out reports/finbert_labels.csv]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.data.db import list_symbols, load_candles
from pinance_ml.data.news_db import load_news
from pinance_ml.news.decay import base_asset
from pinance_ml.news.labels import label_news


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neutral-threshold", type=float, default=0.0015)
    parser.add_argument("--out", default="reports/finbert_labels.csv")
    parser.add_argument(
        "--include-body", action="store_true",
        help="join in scripts/backfill_article_bodies.py's crawled text (NULL for any asset not yet crawled)",
    )
    args = parser.parse_args()

    symbols = list_symbols()
    log(f"Symbols: {symbols}")

    all_labeled = []
    for symbol in symbols:
        asset = base_asset(symbol)
        log(f"{symbol} ({asset}): loading candles + news")
        candles = load_candles(symbol)
        news = load_news(asset=asset, include_body=args.include_body)
        log(f"{asset}: {len(news)} news items, {len(candles)} candles")

        labeled = label_news(news, candles, neutral_threshold=args.neutral_threshold)
        labeled["asset"] = asset
        labeled["symbol"] = symbol
        all_labeled.append(labeled)
        log(f"{asset}: {len(labeled)}/{len(news)} labeled (rest too close to end of candle history)")

    result = pd.concat(all_labeled, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    log(f"\nTotal labeled rows: {len(result)}")
    log("Label balance:\n" + result["label"].value_counts().to_string())
    log("Per-asset counts:\n" + result.groupby(["asset", "label"]).size().unstack(fill_value=0).to_string())
    log(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
