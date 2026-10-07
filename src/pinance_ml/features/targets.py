import numpy as np
import pandas as pd


def compute_log_return_targets(
    df: pd.DataFrame,
    horizons: list[int],
    candle_interval_minutes: int,
) -> pd.DataFrame:
    """Add r_1..r_N log-return target columns to a candle DataFrame.

    r_h[t] = log(close[t + h * interval] / close[t]), looked up by exact
    timestamp rather than by row offset, so gaps in the candle series produce
    NaN targets instead of silently pairing rows that aren't `h` candles apart.

    `df` must contain `ts` (sorted, unique, tz-aware) and `close` columns.
    """
    out = df.set_index("ts").sort_index()
    close = out["close"]

    for h in horizons:
        target_ts = out.index + pd.Timedelta(minutes=candle_interval_minutes * h)
        future_close = close.reindex(target_ts).to_numpy()
        out[f"r_{h}"] = np.log(future_close / close.to_numpy())

    return out.reset_index()
