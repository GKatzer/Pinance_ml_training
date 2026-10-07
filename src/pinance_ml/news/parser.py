"""RSS fetching, parsing and dedup (README "Текстовый слой" / "Источники").

Split into a network edge (`fetch_feed`) and pure functions (everything
else) so the parsing/dedup logic is testable against fixture feed content
without a live HTTP call, the same separation `data/db.py` draws around
`features/pipeline.py`.
"""

import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pinance_ml.config import NEWS_NEAR_DUP_SIMILARITY_THRESHOLD

# Some sources (CoinDesk, at least) return an empty body to feedparser's
# own default urllib User-Agent — a plain browser-like one is enough.
_USER_AGENT = "Mozilla/5.0 (compatible; PinanceNewsBot/1.0)"


@dataclass
class NewsItem:
    url: str
    source: str
    title: str
    summary: str
    published_at: pd.Timestamp  # tz-aware UTC; from the feed entry itself
    parsed_at: pd.Timestamp  # tz-aware UTC; wall-clock time of this poll


def fetch_feed(url: str, timeout: float = 15.0, retries: int = 1) -> bytes:
    """GET raw feed bytes. The only network-touching function in this module.

    One retry by default: on the training host this goes through redsocks -> an
    upstream SOCKS relay (infra detail, not in this repo), which was
    observed intermittently stalling a single connection ("splice(from
    relay): Connection timed out" in redsocks's own log) rather than being
    reliably down -- a plain retry succeeded seconds later against the
    same host. timeout*(retries+1) is kept small per source so a fully-down
    cycle across every NEWS_FEEDS entry still finishes comfortably inside
    the 5-minute timer interval (see deploy/news_parser/news-parser.timer).
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1)
    raise last_exc


def parse_feed(raw_feed: bytes | str, source: str, parsed_at: pd.Timestamp | None = None) -> list[NewsItem]:
    """Parse one already-fetched RSS feed into NewsItems.

    Entries without a parseable publish date are dropped: both the
    time-decay features (`news/decay.py`) and the look-ahead guard
    (README: train on publication time, not parse time) key off
    `published_at`, so an item without one can't be placed on the
    timeline at all.
    """
    import feedparser  # lazy: keeps `import pinance_ml.news.parser` cheap for callers that only need dedup/dataclasses

    feed = feedparser.parse(raw_feed)
    parsed_at = parsed_at if parsed_at is not None else pd.Timestamp.now(tz="UTC")

    items = []
    for entry in feed.entries:
        published = _entry_timestamp(entry)
        if published is None:
            continue
        items.append(
            NewsItem(
                url=entry.get("link", ""),
                source=source,
                title=entry.get("title", ""),
                summary=entry.get("summary", ""),
                published_at=published,
                parsed_at=parsed_at,
            )
        )
    return items


def _entry_timestamp(entry) -> pd.Timestamp | None:
    for field in ("published_parsed", "updated_parsed"):
        struct = entry.get(field)
        if struct is not None:
            return pd.Timestamp(datetime(*struct[:6], tzinfo=timezone.utc))
    return None


def dedup_by_url(items: list[NewsItem], known_urls: set[str]) -> list[NewsItem]:
    """Drop items whose URL is already known — already stored in DB, or
    seen earlier in this same batch (two feeds can carry the same wire
    story at the same URL)."""
    seen = set(known_urls)
    out = []
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        out.append(item)
    return out


def drop_near_duplicates(
    items: list[NewsItem],
    reference_texts: Sequence[str] = (),
    threshold: float = NEWS_NEAR_DUP_SIMILARITY_THRESHOLD,
) -> list[NewsItem]:
    """Drop items whose title+summary is a near-duplicate of an
    already-known recent item or of another item earlier in this batch
    (README: "near-duplicate по эмбеддингам") — e.g. the same story
    republished with a different URL by two outlets.

    Similarity is TF-IDF cosine over title+summary text: a bag-of-words
    embedding rather than a neural one, chosen so dedup runs on the training host's CPU
    budget alongside FinBERT without a second model.
    """
    if not items:
        return []

    new_texts = [f"{it.title} {it.summary}".strip() for it in items]
    corpus = list(reference_texts) + new_texts
    if not any(corpus):
        return list(items)

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    except ValueError:
        # Empty vocabulary after stopword removal (e.g. all-punctuation
        # titles in a test fixture) — nothing meaningful to compare on.
        return list(items)

    n_ref = len(reference_texts)
    kept_items = []
    kept_rows = list(range(n_ref))  # row indices into `matrix` already treated as "known"
    for i, item in enumerate(items):
        row = n_ref + i
        if kept_rows and cosine_similarity(matrix[row], matrix[kept_rows]).max() >= threshold:
            continue
        kept_items.append(item)
        kept_rows.append(row)
    return kept_items
