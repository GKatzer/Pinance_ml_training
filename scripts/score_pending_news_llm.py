"""Level 2 event-category classification pass over already-fetched,
not-yet-classified news (`llm_event_type IS NULL`) -- same structure as
scripts/score_pending_news.py's FinBERT pass, for news/extraction.py's
Qwen2.5-3B classifier instead.

Resumable by construction: every chunk is classified and UPDATEd
independently, and the next run's `pending_llm_extraction()` query only
ever sees rows still NULL.

Usage:
    python scripts/score_pending_news_llm.py [--asset BTC] [--chunk-size 50] [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinance_ml.data.news_db import pending_llm_extraction, update_llm_extraction
from pinance_ml.news.extraction import classify_event_type_llm

DEFAULT_CHUNK_SIZE = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", default=None, help="scope the pass (e.g. BTC); omit for all pending rows")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pending = pending_llm_extraction(asset=args.asset)
    if args.limit:
        pending = pending.head(args.limit)
    total = len(pending)
    log(f"{total} rows pending Level 2 classification" + (f" (asset={args.asset})" if args.asset else ""))
    if pending.empty:
        return

    total_updated = 0
    none_count = 0
    t0 = time.time()
    for chunk_start in range(0, total, args.chunk_size):
        chunk = pending.iloc[chunk_start : chunk_start + args.chunk_size]
        texts = [
            (row.body if isinstance(row.body, str) and len(row.body) > 20 else row.title)
            for row in chunk.itertuples()
        ]
        event_types = classify_event_type_llm(texts)
        none_count += sum(1 for e in event_types if e is None)
        rows = [
            {"url": url, "llm_event_type": event_type}
            for url, event_type in zip(chunk["url"], event_types)
            if event_type is not None
        ]
        updated = update_llm_extraction(rows)
        total_updated += updated

        done = chunk_start + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (total - done) / rate / 60 if rate > 0 else float("nan")
        log(f"  {done}/{total} done ({total_updated} updated, {none_count} unparseable), {rate:.2f}/s, ETA {eta_min:.1f}min")

    log(f"\nDone. {total_updated} rows classified, {none_count} unparseable. Total time: {(time.time() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
