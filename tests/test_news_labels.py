import numpy as np
import pandas as pd

from pinance_ml.news.labels import label_news


def _candles(n: int, start: str = "2026-01-01", step_minutes: int = 5, close=None) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=f"{step_minutes}min", tz="UTC")
    if close is None:
        close = np.full(n, 100.0)
    return pd.DataFrame({"ts": ts, "close": close})


def test_positive_label_when_price_rises_over_next_60min():
    close = np.full(40, 100.0)
    close[12:] = 110.0  # jumps up exactly at the 60-minute mark (12 candles @ 5min)
    candles = _candles(40, close=close)
    news = pd.DataFrame({"published_at": [candles["ts"].iloc[0]]})

    result = label_news(news, candles, neutral_threshold=0.0015)

    assert len(result) == 1
    assert result["label"].iloc[0] == "positive"
    assert result["log_return"].iloc[0] > 0


def test_negative_label_when_price_falls_over_next_60min():
    close = np.full(40, 100.0)
    close[12:] = 90.0
    candles = _candles(40, close=close)
    news = pd.DataFrame({"published_at": [candles["ts"].iloc[0]]})

    result = label_news(news, candles, neutral_threshold=0.0015)

    assert result["label"].iloc[0] == "negative"
    assert result["log_return"].iloc[0] < 0


def test_neutral_label_when_move_is_within_threshold():
    candles = _candles(40)  # flat price throughout
    news = pd.DataFrame({"published_at": [candles["ts"].iloc[0]]})

    result = label_news(news, candles, neutral_threshold=0.0015)

    assert result["label"].iloc[0] == "neutral"
    assert result["log_return"].iloc[0] == 0.0


def test_rows_without_enough_future_candles_are_dropped():
    candles = _candles(40)
    # last candle has no room for a 60-minute-ahead (12-candle) label
    news = pd.DataFrame({"published_at": [candles["ts"].iloc[-1]]})

    result = label_news(news, candles, neutral_threshold=0.0015)

    assert result.empty


def test_entry_uses_first_candle_at_or_after_published_at():
    candles = _candles(40)
    between_candles = candles["ts"].iloc[5] + pd.Timedelta(minutes=1)
    news = pd.DataFrame({"published_at": [between_candles]})

    result = label_news(news, candles, neutral_threshold=0.0015)

    # entry candle should be index 6 (first candle >= between_candles),
    # label candle index 6+12=18 -- both well within the 40-candle range
    assert len(result) == 1


def test_preserves_other_news_columns():
    candles = _candles(40)
    news = pd.DataFrame({
        "published_at": [candles["ts"].iloc[0]],
        "title": ["Some headline"],
        "url": ["https://example.com/1"],
    })

    result = label_news(news, candles, neutral_threshold=0.0015)

    assert result["title"].iloc[0] == "Some headline"
    assert result["url"].iloc[0] == "https://example.com/1"
