"""WordPress REST API client for outlets that expose /wp-json/wp/v2/posts
directly to unauthenticated requests (confirmed working: cryptopotato.com;
several other NEWS_FEEDS domains return 403 -- presumably WAF/bot
protection -- see news/sitemap.py for those instead).

Produces the same NewsItem as the rest of news/*, with real titles --
unlike the sitemap path, this API returns the actual headline, not a
URL-slug approximation.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape

import pandas as pd

from pinance_ml.news.parser import NewsItem

_USER_AGENT = "Mozilla/5.0 (compatible; PinanceNewsBot/1.0)"
PER_PAGE = 100


def _posts_url(domain: str, after: pd.Timestamp, before: pd.Timestamp, page: int) -> str:
    params = {
        "after": after.strftime("%Y-%m-%dT%H:%M:%S"),
        "before": before.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_page": str(PER_PAGE),
        "page": str(page),
        "_fields": "title,date_gmt,link",
        "orderby": "date",
        "order": "asc",
    }
    return f"https://{domain}/wp-json/wp/v2/posts?{urllib.parse.urlencode(params)}"


def fetch_posts_page(
    domain: str,
    after: pd.Timestamp,
    before: pd.Timestamp,
    page: int = 1,
    timeout: float = 20.0,
    max_retries: int = 4,
    initial_backoff_seconds: float = 3.0,
) -> tuple[bytes, int]:
    """GET one page of raw JSON plus the API's reported total page count
    (`X-WP-TotalPages` header) -- reading that up front means the caller
    can loop `range(2, total_pages + 1)` instead of paging past the end
    and having to distinguish "empty page" from the 400 WP returns for
    that. The only network-touching function here.

    Retries with backoff on transient connection failures (URLError,
    TimeoutError) -- a run against dozens/hundreds of pages will hit an
    occasional SSL/connection reset from this environment's network path
    (confirmed repeatedly while building this, not a hypothetical), and
    one flaky request shouldn't lose everything fetched so far.
    """
    url = _posts_url(domain, after, before, page)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    delay = initial_backoff_seconds
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                total_pages = int(response.headers.get("X-WP-TotalPages", "1"))
                return raw, total_pages
        except (urllib.error.URLError, TimeoutError):
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable: loop above always returns or raises")


def parse_posts(raw: bytes | str, source: str, parsed_at: pd.Timestamp | None = None) -> list[NewsItem]:
    """Parse one already-fetched posts page into NewsItems. Entries
    missing title/date_gmt/link are dropped, same missing-data policy as
    the rest of news/*."""
    parsed_at = parsed_at if parsed_at is not None else pd.Timestamp.now(tz="UTC")
    posts = json.loads(raw)
    if not isinstance(posts, list):
        return []

    items = []
    for post in posts:
        link = post.get("link")
        date_gmt = post.get("date_gmt")
        title = (post.get("title") or {}).get("rendered")
        if not link or not date_gmt or not title:
            continue
        items.append(
            NewsItem(
                url=link,
                source=source,
                title=unescape(title),
                summary="",
                published_at=pd.Timestamp(date_gmt, tz="UTC"),
                parsed_at=parsed_at,
            )
        )
    return items
