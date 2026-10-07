import pandas as pd

from pinance_ml.config import CANDLE_INTERVAL_MINUTES, HORIZONS, LONG_HORIZONS
from pinance_ml.features.pipeline import compute_features
from pinance_ml.features.targets import compute_log_return_targets

# Raw OHLCV passthrough columns: not model inputs. README's X[t] is built
# entirely from engineered features (lags, rolling stats, indicators, ...),
# never raw price/volume level — those aren't stationary across years of
# history the way a return or an indicator ratio is.
NON_FEATURE_COLUMNS = {"ts", "open", "high", "low", "close", "volume"}
# Both HORIZONS (5-60min) and LONG_HORIZONS (24h, the project notes)
# targets are excluded here -- r_288 must never leak into feature_columns()
# for the short-horizon models (or vice versa), even though both target sets
# come from the same build_dataset() call below.
TARGET_COLUMNS = {f"r_{h}" for h in HORIZONS} | {f"r_{h}" for h in LONG_HORIZONS}


def build_dataset(candles: pd.DataFrame, btc_candles: pd.DataFrame | None = None) -> pd.DataFrame:
    """X (engineered features) + y (r_1..r_12 and r_288 targets), joined on ts.

    `targets.py` and `pipeline.py` stay independent of each other (neither
    needs to know the other exists); this is the one place they're brought
    together, at the point where a model actually needs both.

    Short and long horizons share this one feature matrix (same
    ROLLING_WINDOWS, deliberately left untouched for the 24h model's first
    pass -- see the project notes) -- only the target column and
    the walk-forward purge width differ between the two.
    """
    features = compute_features(candles, btc_candles=btc_candles)
    targets = compute_log_return_targets(candles, HORIZONS + LONG_HORIZONS, CANDLE_INTERVAL_MINUTES)

    target_cols = ["ts"] + sorted(TARGET_COLUMNS)
    return features.merge(targets[target_cols], on="ts", how="inner", validate="one_to_one")


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Every column in a `build_dataset` output that's a model input."""
    return [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS and c not in TARGET_COLUMNS]
