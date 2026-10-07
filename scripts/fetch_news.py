"""One RSS poll cycle: fetch -> parse -> dedup -> FinBERT score -> store.

README: "RSS: CoinDesk, CoinTelegraph, The Block ... Опрос раз в 5 минут на
the training host, дедупликация по URL и near-duplicate по эмбеддингам." The 5-minute
cadence is a systemd timer / APScheduler job on the training host that runs this script;
this script itself does exactly one poll and exits. Usage:

    python scripts/fetch_news.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pinance_ml.config import NEWS_FEEDS
from pinance_ml.data.news_db import insert_news_items, known_urls, recent_texts
from pinance_ml.news.parser import dedup_by_url, drop_near_duplicates, fetch_feed, parse_feed
from pinance_ml.news.sentiment import classify_event_types, extract_assets, score_sentiment

# How far back "already known" reaches when deduping this poll's items --
# generous relative to the 5-minute poll cadence, cheap relative to the
# table's full history.
DEDUP_LOOKBACK_HOURS = 24


def main():
    parsed_at = pd.Timestamp.now(tz="UTC")
    since = parsed_at - pd.Timedelta(hours=DEDUP_LOOKBACK_HOURS)

    items = []
    for source, url in NEWS_FEEDS.items():
        try:
            raw = fetch_feed(url)
        except Exception as exc:
            print(f"  {source}: fetch failed ({exc})")
            continue
        feed_items = parse_feed(raw, source, parsed_at=parsed_at)
        print(f"  {source}: {len(feed_items)} entries")
        items.extend(feed_items)

    items = dedup_by_url(items, known_urls(since=since))
    items = drop_near_duplicates(items, recent_texts(since=since))
    print(f"{len(items)} new items after dedup")

    if not items:
        return

    # Title only, not title+summary -- matches score_pending_news.py (which
    # only ever had title, since load_news() doesn't select summary) and,
    # as of this commit, scripts/build_finbert_labels.py's Level 3 training
    # data. summary is populated on <1% of rows in practice (checked: 99.6%
    # empty across the whole table, and empty even on most *live* RSS
    # sources bar coindesk) -- title+summary was really just title with
    # extra steps, at the cost of a real train/serve mismatch for any
    # sentiment model (base or fine-tuned) that's ever scored consistently.
    texts = [item.title for item in items]
    try:
        sentiment = score_sentiment(texts)
    except Exception as exc:
        # Insert with sentiment_* NULL rather than losing this whole poll's
        # items -- the schema's nullable sentiment columns exist for
        # exactly this, and score_pending_news.py's `WHERE sentiment_pos
        # IS NULL` scan picks these back up on its own next run.
        print(f"score_sentiment failed ({exc}); inserting with sentiment NULL, score_pending_news.py will catch up")
        sentiment = None

    rows = []
    for i, (item, text_) in enumerate(zip(items, texts)):
        scores = sentiment.iloc[i] if sentiment is not None else None
        rows.append(
            {
                "url": item.url,
                "source": item.source,
                "title": item.title,
                "summary": item.summary,
                "published_at": item.published_at.to_pydatetime(),
                "parsed_at": item.parsed_at.to_pydatetime(),
                "assets": extract_assets(text_),
                "event_types": classify_event_types(text_),
                "sentiment_pos": float(scores["sentiment_pos"]) if scores is not None else None,
                "sentiment_neu": float(scores["sentiment_neu"]) if scores is not None else None,
                "sentiment_neg": float(scores["sentiment_neg"]) if scores is not None else None,
                "sentiment_confidence": float(scores["sentiment_confidence"]) if scores is not None else None,
            }
        )

    inserted = insert_news_items(rows)
    print(f"Inserted {inserted} rows")


if __name__ == "__main__":
    main()
