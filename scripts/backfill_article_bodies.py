"""Crawls full article text for news_items rows (Level 3 investigation):
title alone turned out ambiguous/thin for the self-supervised fine-tune
(checked -- no mojibake, but titles average ~64 chars / 10 words, a
headline, not the article). Uses trafilatura for generic boilerplate-
stripped extraction, which works across arbitrary site structures without
a per-site parser (unlike news/sitemap.py + news/wordpress_api.py's
fetch-only backfill, each written against one outlet's structure).

Resumable by construction: `WHERE body IS NULL`, and every row gets body
set to either the extracted text or '' (never left NULL) -- '' means
"attempted, found nothing usable" (404/paywall/unparseable page),
distinct from "not yet attempted", so a restart never re-crawls a URL
that's permanently empty.

Usage: python scripts/backfill_article_bodies.py [--asset BTC] [--limit N] [--delay 0.4]
"""

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trafilatura

from pinance_ml.data.news_db import pending_article_bodies, update_article_body

_USER_AGENT = "Mozilla/5.0 (compatible; PinanceNewsBot/1.0)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fetch_body(url: str, timeout: float = 15.0, retries: int = 1) -> str:
    """Fetch + extract one article's body text, '' if unextractable.
    Same retry/timeout resilience as news/parser.py's fetch_feed --
    a transient network hiccup shouldn't permanently mark a URL empty.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                html = response.read()
            text = trafilatura.extract(html)
            return text or ""
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
    log(f"  giving up on {url}: {type(last_exc).__name__}: {last_exc}")
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", default=None, help="scope the crawl (e.g. BTC); omit for all pending rows")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="concurrent fetches -- this is I/O-bound (network wait, not CPU), so threads help a lot; "
        "single-threaded measured ~3s/article (mostly connection latency, not our own delay)",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="rows per incremental DB write")
    args = parser.parse_args()

    pending = pending_article_bodies(asset=args.asset)
    if args.limit:
        pending = pending.head(args.limit)
    total = len(pending)
    log(f"{total} rows pending article body crawl" + (f" (asset={args.asset})" if args.asset else "") + f", {args.workers} workers")

    ok = 0
    empty = 0
    batch = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_body, row.url): row.url for row in pending.itertuples()}
        for i, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            body = future.result()
            if body:
                ok += 1
            else:
                empty += 1
            batch.append({"url": url, "body": body})

            if len(batch) >= args.batch_size:
                update_article_body(batch)
                batch = []

            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (total - i) / rate / 60 if rate > 0 else float("nan")
                log(f"  {i}/{total} done ({ok} extracted, {empty} empty), {rate:.2f}/s, ETA {eta_min:.1f}min")

    if batch:
        update_article_body(batch)

    log(f"Done. {ok} extracted, {empty} empty, {total} total. Total time: {(time.time() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()
