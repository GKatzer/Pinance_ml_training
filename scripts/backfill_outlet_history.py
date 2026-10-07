"""Historical news backfill directly from outlet sitemaps / WordPress APIs
(README status item "RSS-парсер + FinBERT ... замер прироста").

GDELT's DOC 2.0 API (news/gdelt.py) was the first approach tried, but its
rate limiting proved unusable in practice (429s persisting across multiple
independent networks and many-minute cooldowns). These 4 outlets -- a
subset of NEWS_FEEDS -- expose real publish timestamps directly:
cointelegraph.com/decrypt.co/bitcoinmagazine.com via sitemap (lastmod),
cryptopotato.com via its WordPress REST API (date_gmt). The other 3
NEWS_FEEDS domains (coindesk, theblock, beincrypto) returned 403/timeout
on every path tried and are skipped here -- they still contribute via live
RSS going forward, just without historical depth.

Deliberately restricted to exactly these outlets rather than a broader
keyword search (unlike GDELT): the with-news/without-news backtest is only
valid if the historical feed has the same source distribution as the live
RSS feed it's meant to stand in for.

This script only fetches + inserts (url, title, published_at, ... with
sentiment_* left NULL) -- it does not call FinBERT. Fetching is network-
bound and fast (minutes); FinBERT is CPU-bound and, on constrained
hardware, badly throttled under sustained load (a 200-title chunk went
from ~20s in a short isolated test to ~9 minutes in a long-running one).
Coupling the two meant a slow scoring pass blocked seeing the full scope
of available data, and every restart re-walked sitemaps it didn't need
to. Run scripts/score_pending_news.py separately afterward for the
FinBERT pass -- it's resumable by construction (`WHERE sentiment_pos IS
NULL`), so it can be interrupted and restarted freely.

Usage:

    python scripts/backfill_outlet_history.py --start 2018-05-04 [--end 2026-07-22]
"""

import argparse
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.data.news_db import insert_news_items, known_urls
from pinance_ml.news.parser import dedup_by_url
from pinance_ml.news.sentiment import classify_event_types, extract_assets
from pinance_ml.news.sitemap import to_news_items, walk_sitemap
from pinance_ml.news.wordpress_api import fetch_posts_page, parse_posts

# Politeness delay between requests to a given outlet. Started at 1s;
# bumped to 3s after cryptopotato.com started returning 429 partway
# through a real run (confirmed, not hypothetical) at the lower pacing.
REQUEST_PACING_SECONDS = 3.0

# Purely a DB-safety chunk size (a single ~11.5k-row INSERT previously
# dropped the connection mid-query) -- unrelated to FinBERT now that
# scoring isn't happening in this script at all.
INSERT_CHUNK_SIZE = 500

SITEMAP_OUTLETS = {
    "cointelegraph.com": {
        "entry": "https://cointelegraph.com/sitemap/articles/1.xml",
        "url_filter": None,  # already an article-only leaf urlset
    },
    "decrypt.co": {
        "entry": "https://decrypt.co/sitemap_index.xml",
        "url_filter": lambda u: "post-sitemap" in u,
    },
    "bitcoinmagazine.com": {
        "entry": "https://bitcoinmagazine.com/sitemap.xml",
        "url_filter": lambda u: "post-sitemap" in u,
    },
}

WORDPRESS_API_OUTLETS = ["cryptopotato.com"]


def _insert_pending(items, parsed_at: pd.Timestamp, already_known: set[str]) -> int:
    """Inserts url/title/published_at/... with sentiment_* left NULL --
    scripts/score_pending_news.py fills those in later, separately.

    `already_known` (a real `known_urls()` snapshot, not an empty set) is
    what makes a restart skip already-fetched URLs instead of re-inserting
    (harmless, `ON CONFLICT DO NOTHING` handles that) but still re-walking
    the same ground.
    """
    items = dedup_by_url(items, known_urls=already_known)
    if not items:
        return 0

    total_inserted = 0
    for chunk_start in range(0, len(items), INSERT_CHUNK_SIZE):
        chunk = items[chunk_start : chunk_start + INSERT_CHUNK_SIZE]
        rows = [
            {
                "url": item.url,
                "source": item.source,
                "title": item.title,
                "summary": item.summary,
                "published_at": item.published_at.to_pydatetime(),
                "parsed_at": parsed_at.to_pydatetime(),
                "assets": extract_assets(item.title),
                "event_types": classify_event_types(item.title),
                "sentiment_pos": None,
                "sentiment_neu": None,
                "sentiment_neg": None,
                "sentiment_confidence": None,
            }
            for item in chunk
        ]
        inserted = insert_news_items(rows)
        total_inserted += inserted
        print(
            f"    inserted {chunk_start + len(chunk)}/{len(items)} ({inserted} new, {total_inserted} total)",
            flush=True,
        )
    return total_inserted


def backfill_sitemap_outlet(
    domain: str, config: dict, start: pd.Timestamp, end: pd.Timestamp, parsed_at: pd.Timestamp, already_known: set[str]
) -> int:
    print(f"{domain}: walking sitemap from {config['entry']}")
    entries = walk_sitemap(config["entry"], url_filter=config["url_filter"])
    print(f"{domain}: {len(entries)} URLs found total")

    entries = [(url, lastmod) for url, lastmod in entries if lastmod is not None and start <= lastmod < end]
    print(f"{domain}: {len(entries)} URLs in [{start}, {end})")

    items = to_news_items(entries, source=domain, parsed_at=parsed_at)
    inserted = _insert_pending(items, parsed_at, already_known)
    print(f"{domain}: {inserted} rows inserted")
    return inserted


def backfill_wordpress_outlet(
    domain: str, start: pd.Timestamp, end: pd.Timestamp, parsed_at: pd.Timestamp, already_known: set[str]
) -> int:
    print(f"{domain}: querying WordPress API for [{start}, {end})", flush=True)

    total_inserted = 0
    page = 1
    total_pages = 1
    while page <= total_pages:
        try:
            raw, total_pages = fetch_posts_page(domain, start, end, page=page)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  [{page}/{total_pages}] skipping page ({type(exc).__name__}: {exc})", flush=True)
            page += 1
            continue
        time.sleep(REQUEST_PACING_SECONDS)

        items = parse_posts(raw, source=domain, parsed_at=parsed_at)
        inserted = _insert_pending(items, parsed_at, already_known)
        total_inserted += inserted
        print(f"  [{page}/{total_pages}] {len(items)} posts, {inserted} inserted (total {total_inserted})", flush=True)
        page += 1

    print(f"{domain}: {total_inserted} rows inserted")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2018-05-04")
    parser.add_argument("--end", default=None, help="ISO date; defaults to now")
    parser.add_argument("--outlets", nargs="*", default=None, help="Subset of domains; default: all configured")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    parsed_at = pd.Timestamp.now(tz="UTC")
    outlets = set(args.outlets) if args.outlets else None

    print(f"Backfilling {start} .. {end}")

    already_known = known_urls()
    print(f"{len(already_known)} URLs already in news_items -- resuming past those\n")

    total_inserted = 0
    for domain, config in SITEMAP_OUTLETS.items():
        if outlets is not None and domain not in outlets:
            continue
        total_inserted += backfill_sitemap_outlet(domain, config, start, end, parsed_at, already_known)

    for domain in WORDPRESS_API_OUTLETS:
        if outlets is not None and domain not in outlets:
            continue
        total_inserted += backfill_wordpress_outlet(domain, start, end, parsed_at, already_known)

    print(f"\nDone. {total_inserted} total rows inserted (sentiment pending -- run score_pending_news.py next).")


if __name__ == "__main__":
    main()
