import numpy as np
import pandas as pd

from pinance_ml.news.decay import (
    LLM_EVENT_FEATURE_COLUMNS,
    LLM_EVENT_FEATURE_COLUMNS_24H,
    NEWS_FEATURE_COLUMNS,
    base_asset,
    compute_llm_event_features,
    compute_news_features,
)


def _llm_news(published_at, llm_event_type):
    return pd.DataFrame({"published_at": published_at, "llm_event_type": llm_event_type})


def _news(published_at, pos, neu, neg, event_types=None):
    n = len(published_at)
    return pd.DataFrame(
        {
            "published_at": published_at,
            "sentiment_pos": pos,
            "sentiment_neu": neu,
            "sentiment_neg": neg,
            "event_types": event_types or [[] for _ in range(n)],
        }
    )


def test_base_asset_strips_known_quote_suffix():
    assert base_asset("BTCUSDT") == "BTC"
    assert base_asset("ETHUSDT") == "ETH"
    assert base_asset("XRPBUSD") == "XRP"


def test_base_asset_passes_through_unrecognized_suffix():
    assert base_asset("SOMEWEIRDPAIR") == "SOMEWEIRDPAIR"


def test_no_news_returns_zero_and_false_features():
    ts = pd.Series(pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"))
    news = pd.DataFrame(columns=["published_at", "sentiment_pos", "sentiment_neu", "sentiment_neg", "event_types"])

    out = compute_news_features(news, ts)

    assert len(out) == 5
    assert list(out.columns) == NEWS_FEATURE_COLUMNS
    assert (out["sentiment_weighted"] == 0.0).all()
    assert (out["news_intensity"] == 0.0).all()
    assert (out["max_magnitude_60m"] == 0.0).all()
    assert (out["sentiment_dispersion"] == 0.0).all()
    assert not out["event_hack_60m"].any()
    assert not out["event_regulation_60m"].any()


def test_single_item_weight_at_exactly_one_half_life():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    published = t - pd.Timedelta(minutes=90)
    news = _news([published], [0.8], [0.1], [0.1])

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    row = out.iloc[0]

    # exp(-ln(2)/90 * 90) == exp(-ln 2) == 0.5 exactly
    assert np.isclose(row["news_intensity"], 0.5, atol=1e-9)
    assert np.isclose(row["sentiment_weighted"], 0.5 * (0.8 - 0.1), atol=1e-9)


def test_news_intensity_is_sum_of_individual_weights():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    news = _news(
        [t - pd.Timedelta(minutes=90), t - pd.Timedelta(minutes=90)],
        [0.5, 0.5],
        [0.2, 0.2],
        [0.3, 0.3],
    )

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    assert np.isclose(out.iloc[0]["news_intensity"], 1.0, atol=1e-9)


def test_sentiment_dispersion_weighted_std_of_two_opposite_items():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    # both published at t (weight 1 each): sentiment = pos - neg = +1 and -1
    news = _news([t, t], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0])

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    assert np.isclose(out.iloc[0]["sentiment_dispersion"], 1.0, atol=1e-9)


def test_max_magnitude_60m_ignores_items_older_than_60_minutes():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    news = _news(
        [t - pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=90)],
        [0.1, 0.9],
        [0.8, 0.05],  # magnitude = 1 - neu: 0.2 (recent), 0.95 (older than 60m)
        [0.1, 0.05],
    )

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    assert np.isclose(out.iloc[0]["max_magnitude_60m"], 0.2, atol=1e-9)


def test_event_flag_true_only_within_60_minute_window():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    news = _news(
        [t - pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=90)],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.3, 0.3],
        event_types=[["hack"], ["regulation"]],
    )

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    row = out.iloc[0]
    assert bool(row["event_hack_60m"]) is True
    assert bool(row["event_regulation_60m"]) is False


def test_items_beyond_decay_horizon_contribute_nothing():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    # 16 half-lives (the default horizon) before t: negligible but the
    # cutoff drops it to exactly zero rather than a tiny nonzero residual.
    news = _news([t - pd.Timedelta(minutes=90 * 20)], [0.9], [0.05], [0.05])

    out = compute_news_features(news, pd.Series([t]), half_life_minutes=90.0)
    assert out.iloc[0]["news_intensity"] == 0.0
    assert out.iloc[0]["sentiment_weighted"] == 0.0


def test_no_lookahead_future_news_excluded_from_earlier_timestamp():
    t1 = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    t2 = t1 + pd.Timedelta(minutes=10)
    future_publish = t1 + pd.Timedelta(minutes=5)  # published strictly between t1 and t2
    news = _news([future_publish], [0.9], [0.05], [0.05])

    out = compute_news_features(news, pd.Series([t1, t2]), half_life_minutes=90.0)

    assert out.iloc[0]["news_intensity"] == 0.0
    assert out.iloc[1]["news_intensity"] > 0.0


def _synthetic_news(n=40, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(start, tz="UTC")
    offsets_minutes = np.sort(rng.uniform(0, 60 * 24, size=n))
    published_at = [base + pd.Timedelta(minutes=float(m)) for m in offsets_minutes]

    pos = rng.uniform(0, 1, size=n)
    neg = rng.uniform(0, 1 - pos)
    neu = 1 - pos - neg
    event_types = [(["hack"] if rng.random() < 0.1 else []) for _ in range(n)]

    return _news(published_at, pos, neu, neg, event_types)


def test_same_input_produces_identical_output():
    news = _synthetic_news()
    ts = pd.Series(pd.date_range("2026-01-01", periods=200, freq="5min", tz="UTC"))

    a = compute_news_features(news, ts)
    b = compute_news_features(news, ts)
    pd.testing.assert_frame_equal(a, b)


def test_llm_event_features_no_news_returns_all_false():
    ts = pd.Series(pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"))
    news = pd.DataFrame(columns=["published_at", "llm_event_type"])

    out = compute_llm_event_features(news, ts)

    assert len(out) == 3
    assert list(out.columns) == LLM_EVENT_FEATURE_COLUMNS
    assert not out.any().any()


def test_llm_event_features_true_only_within_60_minute_window():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    news = _llm_news(
        [t - pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=90)],
        ["hack", "regulation"],
    )

    out = compute_llm_event_features(news, pd.Series([t]))
    row = out.iloc[0]
    assert bool(row["event_llm_hack_60m"]) is True
    assert bool(row["event_llm_regulation_60m"]) is False


def test_llm_event_features_24h_window_is_wider_than_60m():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    # 5 hours ago: outside the 60m window, inside the 24h one.
    news = _llm_news([t - pd.Timedelta(hours=5)], ["hack"])

    default = compute_llm_event_features(news, pd.Series([t]))
    wide = compute_llm_event_features(news, pd.Series([t]), window_minutes=1440.0, suffix="24h")

    assert list(default.columns) == LLM_EVENT_FEATURE_COLUMNS
    assert list(wide.columns) == LLM_EVENT_FEATURE_COLUMNS_24H
    assert bool(default["event_llm_hack_60m"].iloc[0]) is False
    assert bool(wide["event_llm_hack_24h"].iloc[0]) is True


def test_llm_event_features_no_lookahead():
    t1 = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    t2 = t1 + pd.Timedelta(minutes=10)
    future_publish = t1 + pd.Timedelta(minutes=5)
    news = _llm_news([future_publish], ["hack"])

    out = compute_llm_event_features(news, pd.Series([t1, t2]))

    assert bool(out.iloc[0]["event_llm_hack_60m"]) is False
    assert bool(out.iloc[1]["event_llm_hack_60m"]) is True


def test_no_lookahead_prefix_matches_when_future_timestamps_removed():
    """The same train/serve-skew guard as test_pipeline.py's: a row's
    features must depend only on news at or before its timestamp, so
    computing over a truncated timestamp grid must reproduce the prefix
    of the full-grid output exactly."""
    news = _synthetic_news()
    ts = pd.Series(pd.date_range("2026-01-01", periods=200, freq="5min", tz="UTC"))

    full = compute_news_features(news, ts)
    truncated = compute_news_features(news, ts.iloc[:100])

    pd.testing.assert_frame_equal(full.iloc[:100].reset_index(drop=True), truncated, check_exact=False, rtol=1e-9)
