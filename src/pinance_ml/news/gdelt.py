"""GDELT DOC 2.0 API client for historical crypto news backfill.

RSS (news/parser.py) only ever exposes the last ~20-30 items per feed --
fine for polling forward every 5 minutes, useless for matching years of
candle history. GDELT's DOC 2.0 API is a searchable index of global news
back to 2017-01-01, queryable by keyword + date range
(https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/), so it's the
practical way to fill years of history instead of just accumulating
forward from whenever fetch_news.py first started running.

Produces the same NewsItem as news/parser.py so every downstream step
(sentiment scoring, ticker/event heuristics, DB insert) is shared code,
not a parallel path.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from pinance_ml.news.parser import NewsItem

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# The DOC API doesn't paginate -- a response with exactly this many
# articles almost certainly lost the rest of the window and needs a
# narrower date range, not a bigger page.
MAX_RECORDS_PER_QUERY = 250

_USER_AGENT = "Mozilla/5.0 (compatible; PinanceNewsBot/1.0)"
_SEENDATE_FORMAT = "%Y%m%dT%H%M%SZ"
_QUERY_DATETIME_FORMAT = "%Y%m%d%H%M%S"


def _build_url(query: str, start: pd.Timestamp, end: pd.Timestamp, max_records: int) -> str:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": start.strftime(_QUERY_DATETIME_FORMAT),
        "enddatetime": end.strftime(_QUERY_DATETIME_FORMAT),
    }
    return f"{GDELT_DOC_API_URL}?{urllib.parse.urlencode(params)}"


def fetch_articles(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_records: int = MAX_RECORDS_PER_QUERY,
    max_retries: int = 5,
    initial_backoff_seconds: float = 5.0,
    timeout: float = 30.0,
) -> bytes:
    """GET raw JSON bytes for one [start, end) window. The only
    network-touching function in this module (mirrors news/parser.py's
    fetch_feed split).

    Retries with exponential backoff on HTTP 429 -- GDELT throttles
    aggressively enough that this is a routine condition to handle, not a
    hypothetical edge case (confirmed hitting it immediately in practice
    while building this).
    """
    url = _build_url(query, start, end, max_records)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    delay = initial_backoff_seconds
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable: loop above always returns or raises")


def parse_articles(raw: bytes | str, parsed_at: pd.Timestamp | None = None) -> list[NewsItem]:
    """Parse one already-fetched DOC API response into NewsItems.

    `mode=artlist` gives title + url + seendate + domain -- no article
    body the way an RSS entry's <description> does, so `summary` is always
    empty for GDELT-sourced items; FinBERT scores off the title alone for
    these. Articles missing url/title/seendate, or with an unparseable
    seendate, are dropped -- same policy as parser.py's missing-publish-
    date rows, since `published_at` is load-bearing for every downstream
    step (time-decay features, the look-ahead guard).
    """
    parsed_at = parsed_at if parsed_at is not None else pd.Timestamp.now(tz="UTC")
    payload = json.loads(raw)

    items = []
    for article in payload.get("articles", []):
        url = article.get("url")
        title = article.get("title")
        seendate = article.get("seendate")
        if not url or not title or not seendate:
            continue
        try:
            published = pd.Timestamp(
                datetime.strptime(seendate, _SEENDATE_FORMAT).replace(tzinfo=timezone.utc)
            )
        except ValueError:
            continue
        items.append(
            NewsItem(
                url=url,
                source=f"gdelt:{article.get('domain', '')}",
                title=title,
                summary="",
                published_at=published,
                parsed_at=parsed_at,
            )
        )
    return items


def is_truncated(raw: bytes | str, max_records: int = MAX_RECORDS_PER_QUERY) -> bool:
    """True if a query likely hit the (unpaginated) result cap and needs
    re-querying with a narrower [start, end) window to see the rest.

    Checks the raw response's article count, not `parse_articles`' output
    -- some raw articles get dropped for missing fields, so counting the
    parsed list would undercount and could miss a truncated window.
    """
    payload = json.loads(raw)
    return len(payload.get("articles", [])) >= max_records
