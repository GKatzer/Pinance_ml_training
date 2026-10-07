import pinance_ml.news.sentiment as sentiment_module
from pinance_ml.news.sentiment import classify_event_types, extract_assets, score_sentiment


def test_extract_assets_matches_ticker_and_alias():
    assert extract_assets("Bitcoin (BTC) rallies as Ethereum lags") == ["BTC", "ETH"]


def test_extract_assets_case_insensitive_and_no_match():
    assert extract_assets("BITCOIN hits new high") == ["BTC"]
    assert extract_assets("Stock market update") == []


def test_extract_assets_does_not_false_positive_on_substrings_of_other_words():
    # "eth" is a substring of "together"/"weather" -- word-boundary
    # matching must not treat that as an ETH mention.
    assert extract_assets("Investors get together as weather clears") == []


def test_classify_event_types_matches_keywords():
    assert classify_event_types("Exchange hacked, funds stolen") == ["hack"]
    assert classify_event_types("SEC regulation tightens on stablecoins") == ["regulation"]


def test_classify_event_types_no_match_returns_empty():
    assert classify_event_types("Market remains flat ahead of Fed decision") == []


def test_classify_event_types_does_not_false_positive_on_substrings_of_other_words():
    # "hack" is a substring of "hackathon" -- word-boundary matching must
    # not treat a hackathon announcement as a hack event.
    assert classify_event_types("ETH hackathon winners announced this weekend") == []


def test_classify_event_types_can_match_multiple():
    result = classify_event_types("New regulation follows a hack where funds were stolen")
    assert set(result) == {"hack", "regulation"}


class _FakeClassifier:
    def __call__(self, texts, batch_size=16, truncation=True, top_k=None):
        return [
            [
                {"label": "positive", "score": 0.7},
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.1},
            ]
            for _ in texts
        ]


def test_score_sentiment_maps_labels_to_columns(monkeypatch):
    monkeypatch.setattr(sentiment_module, "_get_pipeline", lambda: _FakeClassifier())

    out = score_sentiment(["Bitcoin rallies"])

    assert list(out.columns) == ["sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_confidence"]
    row = out.iloc[0]
    assert row["sentiment_pos"] == 0.7
    assert row["sentiment_neg"] == 0.2
    assert row["sentiment_neu"] == 0.1
    assert row["sentiment_confidence"] == 0.7


def test_score_sentiment_empty_input_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(sentiment_module, "_get_pipeline", lambda: _FakeClassifier())
    out = score_sentiment([])
    assert out.empty
    assert list(out.columns) == ["sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_confidence"]
