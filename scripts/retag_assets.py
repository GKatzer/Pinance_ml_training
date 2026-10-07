"""One-off re-tag of news_items.assets after NEWS_ASSET_ALIASES gained
BNB/SOL entries (they were missing initially -- only BTC/ETH were tracked,
so measure_news_feature_gain.py would have seen zero news for BNBUSDT/
SOLUSDT and silently measured a fake "no gain" result for both).

extract_assets runs once at insert time and its result is stored, so
widening the alias list doesn't retroactively affect already-inserted
rows without this. Cheap (a regex match + UPDATE per row), not FinBERT --
unrelated to scripts/score_pending_news.py's slow pass.

Usage: python scripts/retag_assets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sqlalchemy import text

from pinance_ml.data.news_db import get_news_engine
from pinance_ml.news.sentiment import extract_assets

CHUNK_SIZE = 2000


def main():
    df = pd.read_sql("SELECT url, title FROM news_items ORDER BY id", get_news_engine())
    print(f"{len(df)} rows to retag", flush=True)

    query = text("UPDATE news_items SET assets = :assets WHERE url = :url")
    total = 0
    for start in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[start : start + CHUNK_SIZE]
        rows = [{"url": row.url, "assets": extract_assets(row.title)} for row in chunk.itertuples()]
        with get_news_engine().begin() as conn:
            conn.execute(query, rows)
        total += len(rows)
        print(f"  retagged {total}/{len(df)}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
