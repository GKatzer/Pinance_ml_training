"""FinBERT sentiment pass over already-fetched, not-yet-scored news
(sentiment_pos IS NULL) -- the slow half of the historical backfill,
deliberately separate from scripts/backfill_outlet_history.py's fast
fetch-only pass.

Resumable by construction: every chunk is scored and UPDATEd
independently, and the next run's `pending_news()` query only ever sees
rows that are still NULL. Interrupt this at any point (Ctrl-C, a crash, a
reboot) and re-running it picks up exactly where it left off -- no
re-fetching, no re-scoring already-done rows.

Usage:

    python scripts/score_pending_news.py [--chunk-size 200]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinance_ml.data.news_db import pending_news, update_sentiment
from pinance_ml.news.sentiment import score_sentiment

DEFAULT_CHUNK_SIZE = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    pending = pending_news()
    print(f"{len(pending)} rows pending FinBERT scoring", flush=True)
    if pending.empty:
        return

    total_updated = 0
    for chunk_start in range(0, len(pending), args.chunk_size):
        chunk = pending.iloc[chunk_start : chunk_start + args.chunk_size]
        sentiment = score_sentiment(chunk["title"].tolist())
        rows = [
            {
                "url": url,
                "sentiment_pos": float(scores["sentiment_pos"]),
                "sentiment_neu": float(scores["sentiment_neu"]),
                "sentiment_neg": float(scores["sentiment_neg"]),
                "sentiment_confidence": float(scores["sentiment_confidence"]),
            }
            for url, (_, scores) in zip(chunk["url"], sentiment.iterrows())
        ]
        updated = update_sentiment(rows)
        total_updated += updated
        print(
            f"    scored {chunk_start + len(chunk)}/{len(pending)} ({updated} updated, {total_updated} total)",
            flush=True,
        )

    print(f"\nDone. {total_updated} rows scored.")


if __name__ == "__main__":
    main()
