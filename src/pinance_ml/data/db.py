from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from pinance_ml.config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(DATABASE_URL)


def list_symbols() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM candles ORDER BY symbol"))
        return [r[0] for r in rows]


def load_candles(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load OHLCV candles for a symbol, sorted by timestamp.

    start/end are optional ISO date strings; omit to load the full history.
    """
    query = """
        SELECT ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = %(symbol)s
          AND (%(start)s IS NULL OR ts >= %(start)s)
          AND (%(end)s IS NULL OR ts <= %(end)s)
        ORDER BY ts
    """
    df = pd.read_sql(
        query,
        get_engine(),
        params={"symbol": symbol, "start": start, "end": end},
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df
