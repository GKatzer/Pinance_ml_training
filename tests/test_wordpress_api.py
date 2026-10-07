import json
import urllib.error
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pinance_ml.news.wordpress_api import fetch_posts_page, parse_posts


def _post(link, date_gmt, title):
    return {"link": link, "date_gmt": date_gmt, "title": {"rendered": title}}


def test_parse_posts_extracts_fields():
    raw = json.dumps([_post("https://example.com/a", "2019-06-01T12:00:00", "Bitcoin surges")]).encode()
    items = parse_posts(raw, source="example")

    assert len(items) == 1
    item = items[0]
    assert item.url == "https://example.com/a"
    assert item.title == "Bitcoin surges"
    assert item.summary == ""
    assert item.source == "example"
    assert item.published_at == pd.Timestamp("2019-06-01 12:00:00", tz="UTC")


def test_parse_posts_unescapes_html_entities_in_title():
    raw = json.dumps([_post("https://example.com/a", "2019-06-01T12:00:00", "Bitcoin &amp; Ethereum surge")]).encode()
    items = parse_posts(raw, source="example")
    assert items[0].title == "Bitcoin & Ethereum surge"


def test_parse_posts_drops_entries_missing_required_fields():
    raw = json.dumps(
        [
            {"link": "https://a", "date_gmt": "2019-06-01T12:00:00"},  # no title
            {"link": "https://b", "title": {"rendered": "t"}},  # no date_gmt
            {"date_gmt": "2019-06-01T12:00:00", "title": {"rendered": "t"}},  # no link
        ]
    ).encode()
    assert parse_posts(raw, source="example") == []


def test_parse_posts_returns_empty_for_non_list_payload():
    raw = json.dumps({"code": "rest_post_invalid_page_number"}).encode()
    assert parse_posts(raw, source="example") == []


def test_parse_posts_uses_given_parsed_at():
    raw = json.dumps([_post("https://a", "2019-06-01T12:00:00", "t")]).encode()
    parsed_at = pd.Timestamp("2026-01-01", tz="UTC")
    items = parse_posts(raw, source="example", parsed_at=parsed_at)
    assert items[0].parsed_at == parsed_at


@patch("urllib.request.urlopen")
def test_fetch_posts_page_returns_raw_bytes_and_total_pages(mock_urlopen):
    response = MagicMock()
    response.read.return_value = b"[]"
    response.headers = {"X-WP-TotalPages": "7"}
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    mock_urlopen.return_value = response

    raw, total_pages = fetch_posts_page(
        "example.com", pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-02-01", tz="UTC"), page=1
    )

    assert raw == b"[]"
    assert total_pages == 7


@patch("urllib.request.urlopen")
def test_fetch_posts_page_defaults_total_pages_to_one_when_header_missing(mock_urlopen):
    response = MagicMock()
    response.read.return_value = b"[]"
    response.headers = {}
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    mock_urlopen.return_value = response

    _, total_pages = fetch_posts_page(
        "example.com", pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-02-01", tz="UTC")
    )
    assert total_pages == 1


@patch("urllib.request.urlopen")
def test_fetch_posts_page_builds_expected_query_params(mock_urlopen):
    response = MagicMock()
    response.read.return_value = b"[]"
    response.headers = {}
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    mock_urlopen.return_value = response

    fetch_posts_page(
        "example.com",
        pd.Timestamp("2019-01-01", tz="UTC"),
        pd.Timestamp("2019-02-01", tz="UTC"),
        page=3,
    )

    requested_url = mock_urlopen.call_args[0][0].full_url
    assert "example.com/wp-json/wp/v2/posts" in requested_url
    assert "page=3" in requested_url
    assert "after=2019-01-01T00%3A00%3A00" in requested_url


def _url_error():
    return urllib.error.URLError("SSL: UNEXPECTED_EOF_WHILE_READING")


@patch("pinance_ml.news.wordpress_api.time.sleep")
@patch("urllib.request.urlopen")
def test_fetch_posts_page_retries_transient_network_errors_then_succeeds(mock_urlopen, mock_sleep):
    ok_response = MagicMock()
    ok_response.read.return_value = b"[]"
    ok_response.headers = {"X-WP-TotalPages": "1"}
    ok_response.__enter__.return_value = ok_response
    ok_response.__exit__.return_value = False
    mock_urlopen.side_effect = [_url_error(), _url_error(), ok_response]

    raw, total_pages = fetch_posts_page(
        "example.com", pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-02-01", tz="UTC")
    )

    assert raw == b"[]"
    assert total_pages == 1
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("pinance_ml.news.wordpress_api.time.sleep")
@patch("urllib.request.urlopen")
def test_fetch_posts_page_gives_up_after_max_retries(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [_url_error()] * 4

    with pytest.raises(urllib.error.URLError):
        fetch_posts_page(
            "example.com",
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2019-02-01", tz="UTC"),
            max_retries=4,
            initial_backoff_seconds=0.01,
        )
    assert mock_urlopen.call_count == 4
