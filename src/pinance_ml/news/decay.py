"""Time-decay aggregation of scored news into per-timestamp features
(README "Агрегация: time-decay").

Individual news items collapse into a handful of numbers via an
exponential-decay weight `w = exp(-λ·Δt)`, Δt in minutes since publication.
This is a pure function over already-scored news + a timestamp grid, with
the same no-look-ahead property `features/pipeline.py` guarantees for
technical indicators: a row at time t only ever sees items with
`published_at <= t`. README's look-ahead note applies here directly —
callers must pass `published_at`, never `parsed_at`.
"""

import numpy as np
import pandas as pd

from pinance_ml.config import (
    NEWS_DECAY_HALF_LIFE_MINUTES,
    NEWS_DECAY_HORIZON_HALF_LIVES,
    NEWS_EVENT_KEYWORDS,
    NEWS_LLM_EVENT_TYPES,
    NEWS_RECENT_WINDOW_MINUTES,
    QUOTE_ASSET_SUFFIXES,
)

NEWS_FEATURE_COLUMNS = [
    "sentiment_weighted",
    "news_intensity",
    "max_magnitude_60m",
    "sentiment_dispersion",
    *(f"event_{event}_60m" for event in NEWS_EVENT_KEYWORDS),
]

# README "Уровень 2": LLM-classified event category (news/extraction.py,
# NEWS_LLM_EVENT_TYPES), a richer-taxonomy alternative to the keyword-
# matched event_{event}_60m flags above. Kept as a separate feature set
# rather than merged into NEWS_FEATURE_COLUMNS/compute_news_features --
# Level 2's downstream gain hasn't been measured yet (unlike Level 1's
# sentiment features, already in production use), so no other caller of
# compute_news_features should be affected by adding this.
def llm_event_feature_columns(suffix: str = "60m") -> list[str]:
    return [f"event_llm_{event}_{suffix}" for event in NEWS_LLM_EVENT_TYPES]


LLM_EVENT_FEATURE_COLUMNS = llm_event_feature_columns()

# the project notes: same flags, computed over a 24h recency
# window (NEWS_RECENT_WINDOW_24H_MINUTES) instead of 60 minutes --
# "did this event type occur recently" means something different to a
# model looking 24h ahead than to the 60-min one.
LLM_EVENT_FEATURE_COLUMNS_24H = llm_event_feature_columns("24h")


def base_asset(symbol: str) -> str:
    """Strip a quote-currency suffix: 'BTCUSDT' -> 'BTC'."""
    for suffix in QUOTE_ASSET_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _empty_row() -> dict:
    return {col: (False if col.startswith("event_") else 0.0) for col in NEWS_FEATURE_COLUMNS}


def _naive_utc_ns(series: pd.Series) -> np.ndarray:
    """Timestamps as a plain `datetime64[ns]` numpy array (UTC instant,
    tz label dropped) — tz-aware pandas Series otherwise convert to
    object-dtype arrays of Timestamp on `.to_numpy()`, which is both
    slower and not directly comparable to the plain-numpy windows above."""
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.reset_index(drop=True).to_numpy(dtype="datetime64[ns]")


def compute_news_features(
    news: pd.DataFrame,
    timestamps: pd.Series,
    half_life_minutes: float = NEWS_DECAY_HALF_LIFE_MINUTES,
) -> pd.DataFrame:
    """One row of time-decay features per entry in `timestamps`.

    `news` must carry `published_at` (tz-aware, already filtered to the
    relevant asset) plus per-item `sentiment_pos`/`sentiment_neg`/
    `sentiment_neu` and `event_types` (list[str]) — i.e. `data.news_db.
    load_news`'s output after FinBERT scoring. `timestamps` is typically a
    candle grid's `ts` column and must be sorted ascending.

    - `sentiment_weighted` — decay-weighted sum of (pos - neg).
    - `news_intensity` — sum of decay weights: volume of "fresh attention".
    - `max_magnitude_60m` — peak `1 - neutral` (distance from "nothing
      happened", regardless of direction) in the last 60 minutes, undecayed.
    - `sentiment_dispersion` — decay-weighted std of (pos - neg): a
      disagreeing news stream as a volatility proxy.
    - `event_{name}_60m` — whether any item of that event type published
      in the last 60 minutes.

    Implementation is a two-pointer sweep: both `timestamps` and
    `published_at` are sorted and both window starts advance monotonically
    with t, so each pointer only ever walks forward once across the whole
    grid — O(n_news + n_timestamps) total rather than O(n_news) per row.
    """
    ts = _naive_utc_ns(pd.Series(timestamps))

    if news.empty:
        return pd.DataFrame([_empty_row() for _ in range(len(ts))], columns=NEWS_FEATURE_COLUMNS)

    lam = np.log(2) / half_life_minutes
    # Windows as integer-second np.timedelta64 (not pd.Timedelta) so every
    # comparison/subtraction below stays plain numpy datetime64 arithmetic
    # — mixing datetime64 with pandas' Timedelta type works but is slower
    # and, per pandas, not guaranteed stable across versions.
    horizon = np.timedelta64(round(half_life_minutes * NEWS_DECAY_HORIZON_HALF_LIVES * 60), "s")
    recent_window = np.timedelta64(round(NEWS_RECENT_WINDOW_MINUTES * 60), "s")

    news_sorted = news.sort_values("published_at").reset_index(drop=True)
    published = _naive_utc_ns(news_sorted["published_at"])
    sentiment = (news_sorted["sentiment_pos"] - news_sorted["sentiment_neg"]).to_numpy(dtype=float)
    magnitude = (1.0 - news_sorted["sentiment_neu"]).to_numpy(dtype=float)
    event_masks = {
        event: news_sorted["event_types"].apply(lambda types, e=event: e in types).to_numpy()
        for event in NEWS_EVENT_KEYWORDS
    }

    n_news = len(news_sorted)
    lo = 0  # left edge of the decay window: published_arr[lo:] >= t - horizon
    recent_lo = 0  # left edge of the 60-minute window

    rows = []
    for t in ts:
        window_start = t - horizon
        recent_start = t - recent_window

        while lo < n_news and published[lo] < window_start:
            lo += 1
        while recent_lo < n_news and published[recent_lo] < recent_start:
            recent_lo += 1

        hi = int(np.searchsorted(published, t, side="right"))

        if hi <= lo:
            rows.append(_empty_row())
            continue

        delta_minutes = (t - published[lo:hi]) / np.timedelta64(1, "m")
        weights = np.exp(-lam * delta_minutes)
        sent_slice = sentiment[lo:hi]

        news_intensity = float(weights.sum())
        sentiment_weighted = float(np.sum(weights * sent_slice))

        weighted_mean = sentiment_weighted / news_intensity
        variance = float(np.sum(weights * (sent_slice - weighted_mean) ** 2)) / news_intensity
        sentiment_dispersion = float(np.sqrt(max(variance, 0.0)))

        row = {
            "sentiment_weighted": sentiment_weighted,
            "news_intensity": news_intensity,
            "sentiment_dispersion": sentiment_dispersion,
        }
        if hi > recent_lo:
            row["max_magnitude_60m"] = float(magnitude[recent_lo:hi].max())
            for event, mask in event_masks.items():
                row[f"event_{event}_60m"] = bool(mask[recent_lo:hi].any())
        else:
            row["max_magnitude_60m"] = 0.0
            for event in event_masks:
                row[f"event_{event}_60m"] = False
        rows.append(row)

    return pd.DataFrame(rows, columns=NEWS_FEATURE_COLUMNS)


def compute_llm_event_features(
    news: pd.DataFrame,
    timestamps: pd.Series,
    window_minutes: float = NEWS_RECENT_WINDOW_MINUTES,
    suffix: str = "60m",
) -> pd.DataFrame:
    """One row of `event_llm_{type}_{suffix}` flags per entry in
    `timestamps` -- README Level 2's LLM-classified extension of
    compute_news_features' keyword-matched `event_{name}_60m`, using
    news/extraction.py's richer taxonomy (NEWS_LLM_EVENT_TYPES) instead of
    NEWS_EVENT_KEYWORDS.

    Separate from compute_news_features (rather than merged into it)
    since `news` here needs `llm_event_type` (`data.news_db.load_news(...,
    include_llm=True)`), NULL for any row scripts/score_pending_news_llm.py
    hasn't reached yet, and this feature set's downstream gain is still
    being measured (scripts/measure_llm_event_gain.py) -- every other
    caller of compute_news_features stays on the unmodified Level-1-only
    feature set until that's settled.

    window_minutes/suffix default to the original 60-minute flags
    (`LLM_EVENT_FEATURE_COLUMNS`) -- pass `NEWS_RECENT_WINDOW_24H_MINUTES`/
    `"24h"` (`LLM_EVENT_FEATURE_COLUMNS_24H`) for the project notes'
    24h-horizon variant instead. Same undecayed "did this happen recently"
    window either way -- no decay weighting here since this is a boolean
    flag, not a magnitude sum.
    """
    columns = llm_event_feature_columns(suffix)
    ts = _naive_utc_ns(pd.Series(timestamps))

    def _empty_llm_row() -> dict:
        return dict.fromkeys(columns, False)

    if news.empty:
        return pd.DataFrame([_empty_llm_row() for _ in range(len(ts))], columns=columns)

    recent_window = np.timedelta64(round(window_minutes * 60), "s")

    news_sorted = news.sort_values("published_at").reset_index(drop=True)
    published = _naive_utc_ns(news_sorted["published_at"])
    event_masks = {
        event: (news_sorted["llm_event_type"] == event).to_numpy() for event in NEWS_LLM_EVENT_TYPES
    }

    n_news = len(news_sorted)
    recent_lo = 0

    rows = []
    for t in ts:
        recent_start = t - recent_window
        while recent_lo < n_news and published[recent_lo] < recent_start:
            recent_lo += 1
        hi = int(np.searchsorted(published, t, side="right"))

        if hi > recent_lo:
            rows.append(
                {f"event_llm_{event}_{suffix}": bool(mask[recent_lo:hi].any()) for event, mask in event_masks.items()}
            )
        else:
            rows.append(_empty_llm_row())

    return pd.DataFrame(rows, columns=columns)
