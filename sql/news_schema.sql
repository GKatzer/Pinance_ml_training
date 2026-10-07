-- News articles ingested from RSS/API sources (README "Текстовый слой").
-- Applied once against TimescaleDB on the training host with a write-capable role
-- (NEWS_DATABASE_URL) -- distinct from the read-only `pinance_ro`
-- connection used for training reads elsewhere in this repo.
CREATE TABLE IF NOT EXISTS news_items (
    id                   BIGSERIAL PRIMARY KEY,
    url                  TEXT NOT NULL UNIQUE,
    source               TEXT NOT NULL,
    title                TEXT NOT NULL,
    summary              TEXT NOT NULL DEFAULT '',
    published_at         TIMESTAMPTZ NOT NULL,
    parsed_at            TIMESTAMPTZ NOT NULL,
    assets               TEXT[] NOT NULL DEFAULT '{}',
    event_types          TEXT[] NOT NULL DEFAULT '{}',
    -- Nullable: fetching (URL/title/date, network-bound, fast) and FinBERT
    -- scoring (CPU-bound, slow -- and on constrained hardware, throttled
    -- badly under sustained load) are deliberately separate passes for a
    -- historical backfill. A row lands here the moment it's fetched, with
    -- sentiment_* NULL; scripts/score_pending_news.py fills them in later,
    -- `WHERE sentiment_pos IS NULL` making that pass naturally resumable.
    sentiment_pos        DOUBLE PRECISION,
    sentiment_neu        DOUBLE PRECISION,
    sentiment_neg        DOUBLE PRECISION,
    sentiment_confidence DOUBLE PRECISION
);

-- Added later (Level 3 investigation, README "Уровень 3"): full article
-- text, crawled from `url` separately from the RSS fetch -- title/summary
-- alone turned out ambiguous/thin for a decent chunk of self-supervised
-- fine-tuning examples. Same nullable-then-backfilled pattern as
-- sentiment_* above: NULL means "not yet attempted", '' means "attempted,
-- extraction found nothing usable" (a 404, a paywall, a page trafilatura
-- couldn't parse) -- distinguishing the two is what makes
-- scripts/backfill_article_bodies.py's `WHERE body IS NULL` scan
-- naturally resumable without retrying permanently-unextractable URLs
-- forever.
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS body TEXT;

-- Added later (Level 2, README "Уровень 2"): event-category classification
-- from a small instruct model (Qwen2.5-3B-Instruct via llama.cpp). Started
-- as a 6-field structured extraction (assets/novelty/specificity/
-- magnitude/sentiment_direction/is_speculative alongside event_type), but
-- a pilot on 300 real BTC items (2026-07-31, reports/pilot_level2_extraction.csv)
-- found those other fields either degenerate zero-shot (novelty/
-- is_speculative collapsed to a single constant value) or, even after a
-- few-shot fix removed the degeneracy, uncorrelated-to-negatively-correlated
-- with realized 60-min return -- consistent with Level 3's LoRA fine-tune
-- finding no price signal in article text either. event_type was the one
-- field with healthy, non-degenerate output and a plausible mechanism (an
-- LLM-classified category is a natural extension of NEWS_EVENT_KEYWORDS'
-- keyword-matched hack/regulation), so the other fields were dropped
-- rather than kept as always-NULL dead columns.
--
-- Same nullable-then-backfilled pattern as sentiment_*/body above: NULL
-- means "not yet attempted" -- scripts/score_pending_news_llm.py's `WHERE
-- llm_event_type IS NULL` scan is what makes that pass resumable.
-- NEWS_LLM_EVENT_TYPES (config.py) has an explicit "other" category so a
-- real extraction always produces *some* non-NULL value here.
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS llm_event_type TEXT;

-- compute_news_features (news/decay.py) scans forward from the oldest
-- relevant item for every candle timestamp -- this index is what makes
-- that scan (and load_news's time-range filter) start from a range scan
-- instead of a sequential one as the table grows.
CREATE INDEX IF NOT EXISTS news_items_published_at_idx ON news_items (published_at);

-- load_news's `%(symbol)s = ANY(assets)` filter.
CREATE INDEX IF NOT EXISTS news_items_assets_idx ON news_items USING GIN (assets);

-- score_pending_news.py's `WHERE sentiment_pos IS NULL` scan -- a partial
-- index keeps that cheap regardless of how large the (mostly-scored) table
-- grows, since it only ever indexes the still-pending rows.
CREATE INDEX IF NOT EXISTS news_items_pending_score_idx ON news_items (id) WHERE sentiment_pos IS NULL;

-- score_pending_news_llm.py's `WHERE llm_event_type IS NULL` scan -- same
-- reasoning as news_items_pending_score_idx above.
CREATE INDEX IF NOT EXISTS news_items_pending_llm_idx ON news_items (id) WHERE llm_event_type IS NULL;
