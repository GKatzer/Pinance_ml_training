import json
import urllib.error
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pinance_ml.news.gdelt import fetch_articles, is_truncated, parse_articles


def _response(articles: list[dict]) -> bytes:
    return json.dumps({"articles": articles}).encode("utf-8")


def test_parse_articles_extracts_fields_with_empty_summary():
    raw = _response(
        [
            {
                "url": "https://example.com/a",
                "title": "Bitcoin surges past all-time high",
                "seendate": "20190601T120000Z",
                "domain": "coindesk.com",
            }
        ]
    )
    items = parse_articles(raw)

    assert len(items) == 1
    item = items[0]
    assert item.url == "https://example.com/a"
    assert item.title == "Bitcoin surges past all-time high"
    assert item.summary == ""  # DOC API artlist mode has no article body
    assert item.source == "gdelt:coindesk.com"
    assert item.published_at == pd.Timestamp("2019-06-01 12:00:00", tz="UTC")


def test_parse_articles_uses_given_parsed_at():
    raw = _response(
        [{"url": "https://a", "title": "t", "seendate": "20190601T120000Z", "domain": "x.com"}]
    )
    parsed_at = pd.Timestamp("2026-07-22 00:00:00", tz="UTC")
    items = parse_articles(raw, parsed_at=parsed_at)
    assert items[0].parsed_at == parsed_at


@pytest.mark.parametrize(
    "article",
    [
        {"title": "no url", "seendate": "20190601T120000Z", "domain": "x.com"},
        {"url": "https://a", "seendate": "20190601T120000Z", "domain": "x.com"},  # no title
        {"url": "https://a", "title": "no seendate", "domain": "x.com"},
        {"url": "https://a", "title": "bad seendate", "seendate": "not-a-date", "domain": "x.com"},
    ],
)
def test_parse_articles_drops_entries_missing_required_fields(article):
    raw = _response([article])
    assert parse_articles(raw) == []


def test_is_truncated_true_when_article_count_hits_cap():
    raw = _response([{"url": f"https://a/{i}"} for i in range(250)])
    assert is_truncated(raw, max_records=250) is True


def test_is_truncated_false_when_below_cap():
    raw = _response([{"url": f"https://a/{i}"} for i in range(10)])
    assert is_truncated(raw, max_records=250) is False


def test_is_truncated_checks_raw_count_not_parsed_count():
    # 250 raw articles but only 1 has the fields parse_articles needs --
    # truncation must still be flagged from the raw count, not len(parsed).
    articles = [{"url": "https://a", "title": "t", "seendate": "20190601T120000Z", "domain": "x"}]
    articles += [{"title": "missing url/seendate"} for _ in range(249)]
    raw = _response(articles)
    assert is_truncated(raw, max_records=250) is True


def _http_error(code):
    return urllib.error.HTTPError(url="u", code=code, msg="err", hdrs=None, fp=None)


@patch("pinance_ml.news.gdelt.time.sleep")
@patch("urllib.request.urlopen")
def test_fetch_articles_retries_on_429_then_succeeds(mock_urlopen, mock_sleep):
    ok_response = MagicMock()
    ok_response.read.return_value = _response([])
    ok_response.__enter__.return_value = ok_response
    ok_response.__exit__.return_value = False

    mock_urlopen.side_effect = [_http_error(429), _http_error(429), ok_response]

    result = fetch_articles("bitcoin", pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-01-02", tz="UTC"))

    assert result == _response([])
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("pinance_ml.news.gdelt.time.sleep")
@patch("urllib.request.urlopen")
def test_fetch_articles_backs_off_exponentially(mock_urlopen, mock_sleep):
    ok_response = MagicMock()
    ok_response.read.return_value = _response([])
    ok_response.__enter__.return_value = ok_response
    ok_response.__exit__.return_value = False
    mock_urlopen.side_effect = [_http_error(429), _http_error(429), ok_response]

    fetch_articles(
        "bitcoin",
        pd.Timestamp("2019-01-01", tz="UTC"),
        pd.Timestamp("2019-01-02", tz="UTC"),
        initial_backoff_seconds=2.0,
    )

    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)


@patch("pinance_ml.news.gdelt.time.sleep")
@patch("urllib.request.urlopen")
def test_fetch_articles_gives_up_after_max_retries(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [_http_error(429)] * 5

    with pytest.raises(urllib.error.HTTPError):
        fetch_articles(
            "bitcoin",
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2019-01-02", tz="UTC"),
            max_retries=5,
            initial_backoff_seconds=0.01,
        )
    assert mock_urlopen.call_count == 5


@patch("urllib.request.urlopen")
def test_fetch_articles_raises_immediately_on_non_429_error(mock_urlopen):
    mock_urlopen.side_effect = _http_error(500)
    with pytest.raises(urllib.error.HTTPError):
        fetch_articles("bitcoin", pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-01-02", tz="UTC"))
    assert mock_urlopen.call_count == 1
