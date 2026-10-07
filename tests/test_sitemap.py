import urllib.error
from unittest.mock import patch

import pandas as pd

from pinance_ml.news.sitemap import (
    parse_sitemap_index,
    parse_url_entries,
    title_from_url,
    to_news_items,
    walk_sitemap,
)

_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/post-sitemap1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/post-sitemap2.xml</loc></sitemap>
</sitemapindex>
"""

_URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/news/bitcoin-surges-past-66000</loc>
    <lastmod>2019-06-01T12:00:00.000Z</lastmod>
  </url>
  <url>
    <loc>https://example.com/8442/ethereum-gas-fees-drop</loc>
    <lastmod>2019-06-02T08:30:00+00:00</lastmod>
  </url>
  <url>
    <loc>https://example.com/no-date-article</loc>
  </url>
</urlset>
"""


def test_parse_sitemap_index_extracts_child_locs():
    urls = parse_sitemap_index(_INDEX)
    assert urls == ["https://example.com/post-sitemap1.xml", "https://example.com/post-sitemap2.xml"]


def test_parse_sitemap_index_returns_empty_for_leaf_urlset():
    assert parse_sitemap_index(_URLSET) == []


def test_parse_url_entries_extracts_url_and_lastmod():
    entries = parse_url_entries(_URLSET)
    assert len(entries) == 3

    url, lastmod = entries[0]
    assert url == "https://example.com/news/bitcoin-surges-past-66000"
    assert lastmod == pd.Timestamp("2019-06-01 12:00:00", tz="UTC")


def test_parse_url_entries_handles_offset_and_z_suffix_formats():
    entries = parse_url_entries(_URLSET)
    _, z_format = entries[0]
    _, offset_format = entries[1]
    assert z_format.tzinfo is not None
    assert offset_format == pd.Timestamp("2019-06-02 08:30:00", tz="UTC")


def test_parse_url_entries_leaves_lastmod_none_when_missing():
    entries = parse_url_entries(_URLSET)
    _, lastmod = entries[2]
    assert lastmod is None


def test_title_from_url_converts_slug_to_title_case():
    assert title_from_url("https://example.com/news/bitcoin-surges-past-66000") == "Bitcoin Surges Past 66000"


def test_title_from_url_skips_bare_numeric_final_segment():
    assert title_from_url("https://example.com/8442/ethereum-gas-fees-drop") == "Ethereum Gas Fees Drop"


def test_title_from_url_strips_query_and_fragment():
    assert title_from_url("https://example.com/news/some-article?utm_source=x#top") == "Some Article"


def test_to_news_items_drops_entries_without_lastmod():
    entries = parse_url_entries(_URLSET)
    items = to_news_items(entries, source="example")
    assert len(items) == 2
    assert all(item.published_at is not None for item in items)


def test_to_news_items_sets_empty_summary_and_given_source():
    entries = [("https://example.com/news/a-b-c", pd.Timestamp("2019-01-01", tz="UTC"))]
    items = to_news_items(entries, source="example", parsed_at=pd.Timestamp("2026-01-01", tz="UTC"))
    assert items[0].summary == ""
    assert items[0].source == "example"
    assert items[0].parsed_at == pd.Timestamp("2026-01-01", tz="UTC")


_LEAF_A = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a</loc><lastmod>2019-01-01T00:00:00Z</lastmod></url>
</urlset>
"""
_LEAF_B = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/b</loc><lastmod>2019-02-01T00:00:00Z</lastmod></url>
</urlset>
"""


@patch("pinance_ml.news.sitemap.fetch_sitemap")
def test_walk_sitemap_recurses_through_index_and_collects_leaf_entries(mock_fetch):
    mock_fetch.side_effect = [_INDEX, _LEAF_A, _LEAF_B]

    entries = walk_sitemap("https://example.com/sitemap_index.xml")

    assert [url for url, _ in entries] == ["https://example.com/a", "https://example.com/b"]
    assert mock_fetch.call_count == 3


@patch("pinance_ml.news.sitemap.fetch_sitemap")
def test_walk_sitemap_returns_leaf_entries_directly_without_recursing(mock_fetch):
    mock_fetch.return_value = _LEAF_A
    entries = walk_sitemap("https://example.com/post-sitemap1.xml")
    assert [url for url, _ in entries] == ["https://example.com/a"]
    assert mock_fetch.call_count == 1


@patch("pinance_ml.news.sitemap.fetch_sitemap")
def test_walk_sitemap_skips_children_rejected_by_url_filter(mock_fetch):
    mock_fetch.side_effect = [_INDEX, _LEAF_B]  # only sitemap2 fetched -- sitemap1 filtered out

    entries = walk_sitemap(
        "https://example.com/sitemap_index.xml",
        url_filter=lambda u: u.endswith("post-sitemap2.xml"),
    )

    assert [url for url, _ in entries] == ["https://example.com/b"]
    assert mock_fetch.call_count == 2


@patch("pinance_ml.news.sitemap.fetch_sitemap")
def test_walk_sitemap_skips_a_child_that_fails_to_fetch_instead_of_aborting(mock_fetch):
    mock_fetch.side_effect = [
        _INDEX,
        urllib.error.HTTPError(url="u", code=404, msg="Not Found", hdrs=None, fp=None),
        _LEAF_B,
    ]

    entries = walk_sitemap("https://example.com/sitemap_index.xml")

    assert [url for url, _ in entries] == ["https://example.com/b"]
