"""One-off historical news backfill via GDELT DOC 2.0 API (README status
item "RSS-парсер + FinBERT ... замер прироста" -- RSS alone only ever
covers the last ~20-30 items per feed, so it can't fill years of history
the way this can).

Walks [start, end) in `--chunk-days`-wide windows, oldest first. A chunk
that comes back at the API's (unpaginated) result cap is split in half and
re-queried recursively rather than silently dropping articles. Resumable
and safe to re-run: `news_items.url` is UNIQUE and `insert_news_items` does
`ON CONFLICT DO NOTHING`, so a chunk already ingested just no-ops.

Usage:

    python scripts/backfill_gdelt_news.py --start 2018-05-04 [--end 2026-07-22]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.data.news_db import insert_news_items
from pinance_ml.news.gdelt import fetch_articles, is_truncated, parse_articles
from pinance_ml.news.parser import dedup_by_url
from pinance_ml.news.sentiment import classify_event_types, extract_assets, score_sentiment

DEFAULT_QUERY = "(bitcoin OR ethereum OR cryptocurrency OR crypto)"

# GDELT's own published limit ("Please limit requests to one every 5
# seconds") -- confirmed by hitting it directly. A margin on top of the
# stated minimum, on top of fetch_articles' own 429 backoff, since we're
# not the only client sharing this budget.
REQUEST_PACING_SECONDS = 6.0

# Below this window width, stop splitting a truncated chunk and just take
# what the API returns -- a >=250-article window under an hour wide means
# genuinely saturated coverage, not a chunking problem worth chasing further.
MIN_CHUNK_WIDTH = pd.Timedelta(hours=1)


def _score_and_build_rows(items, parsed_at: pd.Timestamp) -> list[dict]:
    if not items:
        return []
    texts = [item.title for item in items]  # GDELT items have no summary
    sentiment = score_sentiment(texts)
    rows = []
    for item, text_, (_, scores) in zip(items, texts, sentiment.iterrows()):
        rows.append(
            {
                "url": item.url,
                "source": item.source,
                "title": item.title,
                "summary": item.summary,
                "published_at": item.published_at.to_pydatetime(),
                "parsed_at": parsed_at.to_pydatetime(),
                "assets": extract_assets(text_),
                "event_types": classify_event_types(text_),
                "sentiment_pos": float(scores["sentiment_pos"]),
                "sentiment_neu": float(scores["sentiment_neu"]),
                "sentiment_neg": float(scores["sentiment_neg"]),
                "sentiment_confidence": float(scores["sentiment_confidence"]),
            }
        )
    return rows


def backfill_window(query: str, start: pd.Timestamp, end: pd.Timestamp, parsed_at: pd.Timestamp) -> int:
    """Fetch+insert one window, recursively splitting on truncation. Returns rows inserted."""
    raw = fetch_articles(query, start, end)
    time.sleep(REQUEST_PACING_SECONDS)

    if is_truncated(raw) and (end - start) > MIN_CHUNK_WIDTH:
        mid = start + (end - start) / 2
        print(f"  {start} - {end}: truncated, splitting at {mid}")
        return backfill_window(query, start, mid, parsed_at) + backfill_window(query, mid, end, parsed_at)

    items = parse_articles(raw, parsed_at=parsed_at)
    items = dedup_by_url(items, known_urls=set())  # within-batch only; DB UNIQUE handles cross-run dedup
    rows = _score_and_build_rows(items, parsed_at)
    inserted = insert_news_items(rows)
    print(f"  {start} - {end}: {len(items)} parsed, {inserted} inserted")
    return inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2018-05-04")
    parser.add_argument("--end", default=None, help="ISO date; defaults to now")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--chunk-days", type=float, default=7.0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(days=args.chunk_days)
    parsed_at = pd.Timestamp.now(tz="UTC")

    print(f"Backfilling {start} .. {end} in {args.chunk_days}-day chunks, query={args.query!r}")

    total_inserted = 0
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + step, end)
        total_inserted += backfill_window(args.query, chunk_start, chunk_end, parsed_at)
        chunk_start = chunk_end

    print(f"\nDone. {total_inserted} total rows inserted.")


if __name__ == "__main__":
    main()
