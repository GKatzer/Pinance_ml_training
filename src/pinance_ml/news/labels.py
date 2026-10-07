"""Self-supervised labels for Level 3 LoRA fine-tuning (README "Уровень 3"):
label = sign of the realized return some number of minutes after a news
item's publication -- free, comes from the market, no human labeling
needed, and testable without a live model the way the reverse "ask an LM
to guess price impact" approach (README's own explicit non-goal) never
could be.
"""

import numpy as np
import pandas as pd

from pinance_ml.config import CANDLE_INTERVAL_MINUTES

# README: "метка новости -- фактический знак return актива через 60 минут
# после публикации". At the default 5-minute candle interval this is
# exactly h=12 in this repo's own horizon numbering (config.HORIZONS),
# same close-to-close log-return definition as the regressor's own
# targets (r_h = log(P[t+5h] / P[t])) -- not a coincidence, both trace back
# to the same README target definition.
LABEL_HORIZON_MINUTES = 60


def label_news(news: pd.DataFrame, candles: pd.DataFrame, neutral_threshold: float = 0.0015) -> pd.DataFrame:
    """One row per `news` item (must have a `published_at` column) with an
    added `log_return` and `label` in {"positive", "neutral", "negative"} --
    FinBERT's own native 3-class shape, so a LoRA-tuned model can drop into
    news/sentiment.py's score_sentiment() with zero downstream changes.

    `candles` must be sorted by ts ascending (load_candles' own contract).
    Rows where LABEL_HORIZON_MINUTES ahead falls outside candles' range
    (too-recent news, or a symbol/date range with no later history) are
    dropped rather than given a label from data that doesn't exist yet.

    neutral_threshold is a log-return magnitude, not a percent -- 0.0015
    is roughly the same order as this repo's own measured h=12 MAE
    (~0.004 for BTC), i.e. "smaller than what the regressor can reliably
    resolve counts as flat," not an arbitrary round number. Tune via the
    resulting label balance (build_finbert_labels.py prints it), not by
    guessing a "nicer" threshold.
    """
    steps = LABEL_HORIZON_MINUTES // CANDLE_INTERVAL_MINUTES
    ts = candles["ts"]
    close = candles["close"].to_numpy()
    n = len(candles)

    entry_pos = ts.searchsorted(news["published_at"].to_numpy(), side="left")
    label_pos = entry_pos + steps
    valid = (entry_pos < n) & (label_pos < n)

    log_return = np.full(len(news), np.nan)
    log_return[valid] = np.log(close[label_pos[valid]] / close[entry_pos[valid]])

    label = np.where(
        log_return > neutral_threshold, "positive", np.where(log_return < -neutral_threshold, "negative", "neutral")
    )

    result = news.copy()
    result["log_return"] = log_return
    result["label"] = label
    return result[valid].reset_index(drop=True)
