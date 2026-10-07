import pandas as pd
import pytest

from pinance_ml.news.parser import NewsItem, dedup_by_url, drop_near_duplicates, parse_feed

_FEED_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Fixture Feed</title>
{items}
</channel>
</rss>
"""

_ITEM_TEMPLATE = """
<item>
<title>{title}</title>
<link>{link}</link>
<description>{summary}</description>
<pubDate>{pub_date}</pubDate>
</item>
"""


def _feed(items: list[dict]) -> str:
    return _FEED_TEMPLATE.format(items="".join(_ITEM_TEMPLATE.format(**item) for item in items))


def test_parse_feed_extracts_fields_and_published_at():
    feed = _feed(
        [
            {
                "title": "Bitcoin surges",
                "link": "https://example.com/a",
                "summary": "Analysts bullish",
                "pub_date": "Tue, 21 Jul 2026 14:27:03 +0000",
            }
        ]
    )
    items = parse_feed(feed, source="fixture")

    assert len(items) == 1
    item = items[0]
    assert item.title == "Bitcoin surges"
    assert item.url == "https://example.com/a"
    assert item.summary == "Analysts bullish"
    assert item.source == "fixture"
    assert item.published_at == pd.Timestamp("2026-07-21 14:27:03", tz="UTC")


def test_parse_feed_drops_entries_without_a_publish_date():
    feed = _feed([{"title": "No date", "link": "https://example.com/b", "summary": "x", "pub_date": ""}])
    items = parse_feed(feed, source="fixture")
    assert items == []


def test_parse_feed_uses_given_parsed_at_for_every_entry():
    feed = _feed(
        [
            {"title": "A", "link": "https://example.com/a", "summary": "x", "pub_date": "Tue, 21 Jul 2026 10:00:00 +0000"},
            {"title": "B", "link": "https://example.com/b", "summary": "y", "pub_date": "Tue, 21 Jul 2026 11:00:00 +0000"},
        ]
    )
    parsed_at = pd.Timestamp("2026-07-21 12:00:00", tz="UTC")
    items = parse_feed(feed, source="fixture", parsed_at=parsed_at)
    assert all(item.parsed_at == parsed_at for item in items)


def _item(url, title="t", summary="s", source="src", published_at=None, parsed_at=None):
    now = pd.Timestamp.now(tz="UTC")
    return NewsItem(
        url=url,
        source=source,
        title=title,
        summary=summary,
        published_at=published_at or now,
        parsed_at=parsed_at or now,
    )


def test_dedup_by_url_drops_known_and_within_batch_duplicates():
    items = [_item("https://a"), _item("https://b"), _item("https://a")]
    out = dedup_by_url(items, known_urls={"https://b"})
    assert [i.url for i in out] == ["https://a"]


def test_dedup_by_url_keeps_all_when_nothing_known():
    items = [_item("https://a"), _item("https://b")]
    out = dedup_by_url(items, known_urls=set())
    assert [i.url for i in out] == ["https://a", "https://b"]


def test_drop_near_duplicates_removes_similar_titles_within_batch():
    items = [
        _item("https://a", title="Bitcoin ETF approved by SEC today", summary=""),
        _item("https://b", title="Bitcoin ETF approved by the SEC today", summary=""),
        _item("https://c", title="Ethereum gas fees drop to yearly low", summary=""),
    ]
    out = drop_near_duplicates(items, reference_texts=[], threshold=0.85)
    assert [i.url for i in out] == ["https://a", "https://c"]


def test_drop_near_duplicates_checks_against_reference_texts():
    items = [_item("https://a", title="Bitcoin ETF approved by SEC today", summary="")]
    out = drop_near_duplicates(items, reference_texts=["Bitcoin ETF approved by SEC today"], threshold=0.85)
    assert out == []


def test_drop_near_duplicates_handles_empty_text_without_crashing():
    items = [_item("https://a", title="", summary="")]
    out = drop_near_duplicates(items, reference_texts=[])
    assert [i.url for i in out] == ["https://a"]


def test_drop_near_duplicates_empty_input_returns_empty():
    assert drop_near_duplicates([], reference_texts=["anything"]) == []
