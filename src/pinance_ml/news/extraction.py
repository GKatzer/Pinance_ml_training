"""Level 2 event-category classification via a small instruct model
(README "Текстовый слой", "Уровень 2": Qwen2.5-3B-Instruct, GGUF Q4,
llama.cpp) -- an LLM-classified extension of NEWS_EVENT_KEYWORDS'
keyword-matched hack/regulation flags (news/sentiment.py's
classify_event_types), on top of level 1's FinBERT sentiment.

Started as a 6-field structured extraction (assets/novelty/specificity/
magnitude/sentiment_direction/is_speculative alongside event_type) per
the README's original spec. A pilot on 300 real BTC items (2026-07-31,
reports/pilot_level2_extraction.csv) found the other fields either
degenerate zero-shot (novelty and is_speculative collapsed to a single
constant value across all 300 items) or, even after a few-shot prompt
fix removed the degeneracy, uncorrelated-to-negatively-correlated with
realized 60-minute return -- consistent with Level 3's LoRA fine-tune
finding no price signal in article text either (project memory:
the project notes). event_type was the one field
with healthy, non-degenerate output, so the rest were dropped rather
than shipped as always-close-to-constant features.

Same principle as sentiment.py (README "LM извлекает признаки, регрессор
предсказывает"): this answers a question about the text itself, never
about future price.
"""

import json
from functools import lru_cache

import pandas as pd

from pinance_ml.config import NEWS_LLM_EVENT_TYPES, NEWS_LLM_GPU_LAYERS, NEWS_LLM_MODEL_PATH

_JSON_SCHEMA = {
    "type": "object",
    "properties": {"event_type": {"type": "string", "enum": NEWS_LLM_EVENT_TYPES}},
    "required": ["event_type"],
}

_SYSTEM_PROMPT = (
    "You are a financial-news classification tool for a crypto trading system. "
    "Classify the article's event into exactly one category -- a property of the text itself, "
    "never a prediction of future price movement.\n\n"
    f"Categories: {NEWS_LLM_EVENT_TYPES}. Use \"other\" only if none of the specific categories fit.\n\n"
    'Respond with a single JSON object, nothing else: {"event_type": "..."}'
)

# Keeps prompts short enough for the README's "~3-5 сек на новость" CPU
# budget -- a full crawled body (news/backfill_article_bodies.py) can run
# to ~4000 chars, most of which isn't needed to classify event type.
_MAX_TEXT_CHARS = 1500


@lru_cache(maxsize=1)
def _get_llm():
    # llama_cpp is only imported once extraction is actually invoked, same
    # deferred-import reasoning as sentiment.py's _get_pipeline for torch --
    # nothing that just imports this module needs llama-cpp-python installed.
    from llama_cpp import Llama

    return Llama(model_path=NEWS_LLM_MODEL_PATH, n_ctx=2048, n_gpu_layers=NEWS_LLM_GPU_LAYERS, verbose=False)


@lru_cache(maxsize=1)
def _get_grammar():
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(json.dumps(_JSON_SCHEMA))


def classify_event_type_llm(texts: list[str]) -> list[str | None]:
    """One event_type per text, in order, or None if the model's output
    didn't parse as the expected JSON shape (grammar-constrained decoding
    makes this rare, not impossible) -- callers write these through
    unchanged, so news_items.llm_event_type stays NULL and
    scripts/score_pending_news_llm.py's `WHERE llm_event_type IS NULL`
    picks the row back up on a later run rather than silently dropping it.

    Sequential, one completion per text -- llama.cpp itself is
    multi-threaded per call (n_threads), but a single Llama instance
    doesn't batch independent prompts the way a transformers pipeline
    does, so this matches the README's per-item timing budget rather than
    trying to parallelize across items.
    """
    if not texts:
        return []

    llm = _get_llm()
    grammar = _get_grammar()

    results = []
    for text in texts:
        completion = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": (text or "")[:_MAX_TEXT_CHARS]},
            ],
            grammar=grammar,
            temperature=0.0,
        )
        try:
            obj = json.loads(completion["choices"][0]["message"]["content"])
            event_type = obj.get("event_type")
        except json.JSONDecodeError:
            event_type = None
        results.append(event_type if event_type in NEWS_LLM_EVENT_TYPES else None)
    return results
