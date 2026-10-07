"""FinBERT sentiment scoring + text heuristics (README "Текстовый слой",
level 1: "sentiment (pos/neu/neg + confidence) ... Плюс эвристики по
тексту: упомянутые тикеры, категория события, источник").

Principle from the README ("LM извлекает признаки, регрессор
предсказывает"): every function here answers a question about the text
itself. None of them see price data or predict market movement — that
stays downstream, in the LightGBM models trained on `dataset.build_dataset`.
"""

import re
from functools import lru_cache

import pandas as pd

from pinance_ml.config import NEWS_ASSET_ALIASES, NEWS_EVENT_KEYWORDS, NEWS_FINBERT_MODEL

SENTIMENT_COLUMNS = ["sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_confidence"]


@lru_cache(maxsize=1)
def _get_pipeline():
    # transformers/torch (~1.5GB resident, per README) are only imported
    # once sentiment scoring is actually invoked, so importing this module
    # — or anything that imports it, like news/decay.py's callers — doesn't
    # require torch to be installed at all.
    from transformers import pipeline

    return pipeline("text-classification", model=NEWS_FINBERT_MODEL, top_k=None)


def score_sentiment(texts: list[str], batch_size: int = 16) -> pd.DataFrame:
    """FinBERT pos/neu/neg probabilities + confidence, one row per text.

    `sentiment_confidence` is the top predicted class's own probability —
    how sure FinBERT is, independent of which class won; a headline
    scored 0.9 positive is more confident than one scored 0.4
    positive/0.35 negative/0.25 neutral even though both "win" positive.
    """
    if not texts:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)

    classifier = _get_pipeline()
    raw = classifier(list(texts), batch_size=batch_size, truncation=True, top_k=None)

    rows = []
    for scores in raw:
        by_label = {s["label"].lower(): s["score"] for s in scores}
        rows.append(
            {
                "sentiment_pos": by_label.get("positive", 0.0),
                "sentiment_neu": by_label.get("neutral", 0.0),
                "sentiment_neg": by_label.get("negative", 0.0),
                "sentiment_confidence": max(by_label.values()),
            }
        )
    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)


def _contains_any(text: str, phrases: list[str]) -> bool:
    """Whole-word (`\\b`-bounded) match of any `phrases` entry in `text`,
    case-insensitive. Plain substring search would false-positive short
    aliases/keywords inside unrelated words -- "eth" inside "together",
    "hack" inside "hackathon" -- which word boundaries rule out."""
    return any(re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) for phrase in phrases)


def extract_assets(text: str) -> list[str]:
    """Tickers mentioned in `text` (README: "упомянутые тикеры"), matched
    against NEWS_ASSET_ALIASES' alias lists."""
    return [ticker for ticker, aliases in NEWS_ASSET_ALIASES.items() if _contains_any(text, aliases)]


def classify_event_types(text: str) -> list[str]:
    """Event categories present in `text` (README: "категория события" /
    "event_type_flags"), matched by keyword against NEWS_EVENT_KEYWORDS."""
    return [event for event, keywords in NEWS_EVENT_KEYWORDS.items() if _contains_any(text, keywords)]
