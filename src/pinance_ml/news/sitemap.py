"""Sitemap-based historical news fetching, for outlets that block API/
scraper access but still serve their sitemap.xml (most sites want these
crawlable -- that's the point of a sitemap, and it's why this path works
where news/wordpress_api.py's direct API calls get a 403).

Produces the same NewsItem as news/parser.py and news/gdelt.py, so
downstream sentiment scoring / DB insert is shared code. Unlike RSS or a
CMS API, a sitemap gives no article title -- only URL + lastmod -- so
`title` here is derived from the URL slug, a lower-fidelity stand-in that
still carries real signal (slugs are usually a compressed paraphrase of
the real headline, not an arbitrary ID) for FinBERT-style scoring.
"""

import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from pinance_ml.news.parser import NewsItem

_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_USER_AGENT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"


def fetch_sitemap(url: str, timeout: float = 20.0) -> bytes:
    """GET raw sitemap XML bytes. The only network-touching function here."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_sitemap_index(raw: bytes | str) -> list[str]:
    """Child sitemap URLs from a <sitemapindex> document. Empty list if
    `raw` is a leaf <urlset> instead -- nothing to recurse into."""
    root = ET.fromstring(raw)
    if root.tag != f"{_NS}sitemapindex":
        return []
    return [el.text for el in root.iter(f"{_NS}loc") if el.text]


def parse_url_entries(raw: bytes | str) -> list[tuple[str, pd.Timestamp | None]]:
    """(url, lastmod) pairs from a leaf <urlset> document. lastmod is None
    for entries missing it -- caller (`to_news_items`) drops those, since
    published_at is load-bearing everywhere downstream."""
    root = ET.fromstring(raw)
    if root.tag != f"{_NS}urlset":
        return []

    entries = []
    for url_el in root.iter(f"{_NS}url"):
        loc_el = url_el.find(f"{_NS}loc")
        if loc_el is None or not loc_el.text:
            continue
        lastmod_el = url_el.find(f"{_NS}lastmod")
        lastmod = _parse_lastmod(lastmod_el.text) if lastmod_el is not None and lastmod_el.text else None
        entries.append((loc_el.text, lastmod))
    return entries


def _parse_lastmod(text: str) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError):
        return None
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def title_from_url(url: str) -> str:
    """Derive a pseudo-title from a URL's last path segment: hyphens/
    underscores to spaces, title-cased. Falls back to the segment before
    it if the final one is a bare numeric ID (carries no title info)."""
    path = url.split("?")[0].split("#")[0].rstrip("/")
    segment = path.rsplit("/", 1)[-1]
    if segment.isdigit():
        parts = path.rsplit("/", 2)
        segment = parts[-2] if len(parts) >= 2 else segment
    words = re.sub(r"[-_]+", " ", segment).strip()
    return words.title() or url


def walk_sitemap(
    entry_url: str,
    url_filter=None,
    max_depth: int = 3,
) -> list[tuple[str, pd.Timestamp | None]]:
    """Recursively resolve a sitemap URL down to (url, lastmod) entries
    from every leaf <urlset> reachable from it.

    `url_filter(child_url) -> bool`, if given, skips child sitemap URLs it
    rejects -- e.g. to avoid descending into a page/category/tag/video
    sitemap that was never going to contain news articles. Network-
    touching (unlike the rest of this module): composes `fetch_sitemap`
    with the pure parse functions above, so it's thin glue rather than
    logic that itself needs unit coverage.

    A child sitemap that fails to fetch (404, timeout, ...) is skipped
    with a warning rather than aborting the whole walk -- a years-deep
    sitemap tree with dozens of files realistically has the occasional
    stale/dead entry, and losing everything already fetched to one bad
    file would make this unusable for a long historical backfill.
    """
    raw = fetch_sitemap(entry_url)
    children = parse_sitemap_index(raw)
    if not children:
        return parse_url_entries(raw)

    entries = []
    for i, child_url in enumerate(children):
        if url_filter is not None and not url_filter(child_url):
            continue
        if max_depth <= 0:
            continue
        try:
            child_entries = walk_sitemap(child_url, url_filter=url_filter, max_depth=max_depth - 1)
            print(f"  walk_sitemap: [{i + 1}/{len(children)}] {child_url}: {len(child_entries)} entries", flush=True)
            entries.extend(child_entries)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  walk_sitemap: [{i + 1}/{len(children)}] skipping {child_url} ({type(exc).__name__}: {exc})", flush=True)
    return entries


def to_news_items(
    entries: list[tuple[str, pd.Timestamp | None]],
    source: str,
    parsed_at: pd.Timestamp | None = None,
) -> list[NewsItem]:
    """(url, lastmod) pairs -> NewsItems, dropping undated entries."""
    parsed_at = parsed_at if parsed_at is not None else pd.Timestamp.now(tz="UTC")
    return [
        NewsItem(
            url=url,
            source=source,
            title=title_from_url(url),
            summary="",
            published_at=lastmod,
            parsed_at=parsed_at,
        )
        for url, lastmod in entries
        if lastmod is not None
    ]
