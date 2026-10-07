from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from pinance_ml.config import NEWS_DATABASE_URL


@lru_cache(maxsize=1)
def get_news_engine():
    # Without this, psycopg2 sends one network round-trip per row for any
    # multi-row execute() (INSERT list-of-dicts, UPDATE-many, ...) --
    # confirmed directly: a 2000-row UPDATE took 131s (~66ms/row, i.e. bare
    # round-trip latency) before this. "values_plus_batch" makes psycopg2
    # batch INSERTs via execute_values and UPDATE/DELETE via execute_batch
    # instead of one statement per row.
    return create_engine(NEWS_DATABASE_URL, executemany_mode="values_plus_batch")


def known_urls(since: pd.Timestamp | None = None) -> set[str]:
    """URLs already stored, for `news.parser.dedup_by_url`. `since`
    restricts the scan to a recent window -- a duplicate arriving days
    after the original wouldn't collide with any RSS poll anyway, so
    scanning the whole (ever-growing) table for this on every poll would
    be pure waste."""
    # `text()` binds `:name`-style parameters, unlike `load_news`/`db.load_candles`
    # below which hand a raw `%(name)s` string straight to `pd.read_sql`.
    query = text("SELECT url FROM news_items WHERE (:since IS NULL OR published_at >= :since)")
    with get_news_engine().connect() as conn:
        rows = conn.execute(query, {"since": since})
        return {r[0] for r in rows}


def recent_texts(since: pd.Timestamp) -> list[str]:
    """title+summary of recently stored items, for
    `news.parser.drop_near_duplicates`'s `reference_texts`."""
    query = text("SELECT title, summary FROM news_items WHERE published_at >= :since")
    with get_news_engine().connect() as conn:
        rows = conn.execute(query, {"since": since})
        return [f"{title} {summary}" for title, summary in rows]


def insert_news_items(rows: list[dict]) -> int:
    """Insert already-scored news rows and return how many landed.

    Each dict needs exactly `news_items`'s non-generated columns: url,
    source, title, summary, published_at, parsed_at, assets, event_types,
    sentiment_pos, sentiment_neu, sentiment_neg, sentiment_confidence --
    i.e. `scripts/fetch_news.py`'s output after `parser.parse_feed` +
    `sentiment.score_sentiment` + the ticker/event-type heuristics have all
    run. Callers are expected to have already run `dedup_by_url` against
    `known_urls()`, so a unique-constraint hit here means a race with a
    concurrent poll rather than routine duplication -- `ON CONFLICT DO
    NOTHING` absorbs that instead of failing the whole batch.
    """
    if not rows:
        return 0

    query = text(
        """
        INSERT INTO news_items
            (url, source, title, summary, published_at, parsed_at, assets, event_types,
             sentiment_pos, sentiment_neu, sentiment_neg, sentiment_confidence)
        VALUES
            (:url, :source, :title, :summary, :published_at, :parsed_at, :assets, :event_types,
             :sentiment_pos, :sentiment_neu, :sentiment_neg, :sentiment_confidence)
        ON CONFLICT (url) DO NOTHING
        """
    )
    with get_news_engine().begin() as conn:
        conn.execute(query, rows)
        # NOT result.rowcount: same executemany_mode="values_plus_batch"
        # issue as update_sentiment below -- psycopg2 only reports the
        # *last* statement's rowcount for a batched executemany, not the
        # sum across the batch (a 156-row poll logged "Inserted 1 rows").
        # len(rows) isn't quite as airtight here as in update_sentiment
        # (ON CONFLICT DO NOTHING can genuinely skip a row on a real
        # concurrent-poll race), but that's the rare case this docstring
        # already calls out, not the routine one -- len(rows) is right far
        # more often than result.rowcount is.
        return len(rows)


def pending_news() -> pd.DataFrame:
    """Rows already fetched (url/title/published_at, ...) but not yet
    FinBERT-scored (`sentiment_pos IS NULL`) -- what
    scripts/score_pending_news.py works through. Fetching and scoring are
    separate passes for a historical backfill: fetching is network-bound
    and fast, FinBERT is CPU-bound and, on constrained hardware, badly
    throttled under sustained load -- coupling them meant a slow scoring
    run blocked seeing the full scope of fetched data, and made every
    restart re-walk sitemaps it didn't need to.
    """
    query = "SELECT url, title FROM news_items WHERE sentiment_pos IS NULL ORDER BY id"
    return pd.read_sql(query, get_news_engine())


def update_sentiment(rows: list[dict]) -> int:
    """Fill in sentiment_pos/neu/neg/confidence for already-inserted rows,
    matched by url (each dict needs exactly those 4 keys plus `url`) --
    the write side of `pending_news`."""
    if not rows:
        return 0

    query = text(
        """
        UPDATE news_items
        SET sentiment_pos = :sentiment_pos,
            sentiment_neu = :sentiment_neu,
            sentiment_neg = :sentiment_neg,
            sentiment_confidence = :sentiment_confidence
        WHERE url = :url
        """
    )
    with get_news_engine().begin() as conn:
        conn.execute(query, rows)
        # NOT result.rowcount: with executemany_mode="values_plus_batch"
        # (get_news_engine's psycopg2 batching, see its own docstring),
        # psycopg2 only reports the *last* statement's rowcount for a
        # batched executemany, not the sum across the batch -- confirmed
        # on the training host's first real run, where this logged "1 updated" for a
        # 31-row batch that the DB itself showed as fully applied
        # (sentiment_pos IS NULL count went to 0). url is UNIQUE and every
        # row here came straight from pending_news()'s own SELECT, so
        # len(rows) is what actually happened, not a hopeful guess.
        return len(rows)


def pending_article_bodies(asset: str | None = None) -> pd.DataFrame:
    """Rows with a URL but no crawled article body yet (`body IS NULL`) --
    what scripts/backfill_article_bodies.py works through. Same fetch-fast/
    enrich-slow separation as pending_news(): crawling full pages is much
    slower and less reliable (network-bound, some URLs 404/paywall/never
    resolve) than the RSS fetch that inserted the row in the first place.

    asset optionally scopes the crawl (e.g. "BTC") -- crawling all ~67k
    rows at a polite per-domain rate is hours; testing a hypothesis on
    one asset first is cheaper than committing to the full backfill.
    """
    query = text(
        "SELECT id, url FROM news_items WHERE body IS NULL AND (:asset IS NULL OR :asset = ANY(assets)) ORDER BY id"
    )
    return pd.read_sql(query, get_news_engine(), params={"asset": asset})


def update_article_body(rows: list[dict]) -> int:
    """Fill in body for already-inserted rows, matched by url (each dict
    needs `url` and `body`) -- the write side of pending_article_bodies().
    body='' is a valid, deliberate value here (see news_schema.sql's
    comment on the column: "attempted, found nothing usable"), so this
    only ever writes what the caller decided, never re-derives "empty"
    itself.
    """
    if not rows:
        return 0

    query = text("UPDATE news_items SET body = :body WHERE url = :url")
    with get_news_engine().begin() as conn:
        conn.execute(query, rows)
        return len(rows)


def pending_llm_extraction(asset: str | None = None) -> pd.DataFrame:
    """Rows not yet run through Level 2's event_type classification
    (`llm_event_type IS NULL`) -- what scripts/score_pending_news_llm.py
    works through. Same fetch-fast/enrich-slow separation as
    pending_news()/pending_article_bodies(): Qwen2.5-3B (README: "~3-5 сек
    на новость" on CPU) is far slower than the RSS fetch that inserted
    the row.

    asset scopes the pass the same way pending_article_bodies() does --
    backfilling one asset's history first is cheaper than committing to
    the full table before knowing whether the extracted field carries any
    downstream signal.
    """
    query = text(
        "SELECT id, url, title, body FROM news_items "
        "WHERE llm_event_type IS NULL AND (:asset IS NULL OR :asset = ANY(assets)) ORDER BY id"
    )
    return pd.read_sql(query, get_news_engine(), params={"asset": asset})


def update_llm_extraction(rows: list[dict]) -> int:
    """Fill in llm_event_type for already-inserted rows, matched by url
    (each dict needs `url` and `llm_event_type`) -- the write side of
    pending_llm_extraction()."""
    if not rows:
        return 0

    query = text("UPDATE news_items SET llm_event_type = :llm_event_type WHERE url = :url")
    with get_news_engine().begin() as conn:
        conn.execute(query, rows)
        return len(rows)


def load_news(
    asset: str | None = None, start=None, end=None, include_body: bool = False, include_llm: bool = False
) -> pd.DataFrame:
    """Scored news for feature computation (`news.decay.compute_news_features`),
    optionally filtered to items mentioning `asset` (e.g. "BTC" --
    `news.decay.base_asset("BTCUSDT")`, not the trading symbol itself) and
    a `published_at` range.

    include_body pulls in the crawled article text (scripts/
    backfill_article_bodies.py) for Level 3 -- off by default since
    decay.py's feature-computation callers never need it, only pay for
    the extra transfer when actually asked.

    include_llm pulls in llm_event_type (Level 2, scripts/score_pending_news_llm.py)
    for decay.py's event_{type}_60m flags -- off by default for the same
    reason, and NULL for any row not yet run through that pass.
    """
    body_col = ", body" if include_body else ""
    llm_col = ", llm_event_type" if include_llm else ""
    query = f"""
        SELECT url, source, title, published_at, assets, event_types,
               sentiment_pos, sentiment_neu, sentiment_neg, sentiment_confidence{body_col}{llm_col}
        FROM news_items
        WHERE (%(asset)s IS NULL OR %(asset)s = ANY(assets))
          AND (%(start)s IS NULL OR published_at >= %(start)s)
          AND (%(end)s IS NULL OR published_at <= %(end)s)
        ORDER BY published_at
    """
    df = pd.read_sql(query, get_news_engine(), params={"asset": asset, "start": start, "end": end})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
