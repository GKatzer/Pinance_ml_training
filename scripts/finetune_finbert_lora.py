"""LoRA fine-tune of FinBERT on self-supervised market-reaction labels
(README "Уровень 3": "метка новости -- фактический знак return актива
через 60 минут после публикации"). Input is
scripts/build_finbert_labels.py's output.

Meant to run on GPU (README: "Обучение на десктопе... ~20 минут на
2060S") -- CPU works but is slow, mainly useful for --smoke-test's fast
correctness check before handing this off to run for real.

Acceptance is deliberately NOT this script's job. README is explicit:
"Приёмка по downstream-метрике регрессора, а не по accuracy самой LM: LM
здесь генератор признаков, а не самостоятельная цель" -- this script's
own eval accuracy/F1 is a sanity check that training worked at all, not
the ship/no-ship decision. That decision needs
scripts/measure_news_feature_gain.py rerun with NEWS_FINBERT_MODEL
pointed at this script's --out directory, compared against the current
baseline.

Usage:
    python scripts/finetune_finbert_lora.py
        [--labels reports/finbert_labels.csv] [--out models/finbert-lora]
        [--epochs 3] [--val-fraction 0.1] [--smoke-test]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pinance_ml.config import NEWS_FINBERT_MODEL


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class TitleLabelDataset(torch.utils.data.Dataset):
    """Tokenized "[ASSET] body-or-title" strings + integer label ids --
    scripts/backfill_article_bodies.py's crawled text when available
    (falls back to title, which is all live scoring still uses -- see
    fetch_news.py's own commit for why that's title-only), asset prefixed
    so a multi-asset headline's per-asset labels aren't contradictory
    inputs to the same text."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="reports/finbert_labels.csv")
    parser.add_argument("--out", default="models/finbert-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="2e-4 (default) left eval_loss stuck near ln(3)=1.0986 for 10 full epochs on this data -- "
        "conservative even for LoRA (typically 5e-4-1e-3). Try higher before assuming there's no signal.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="tiny subset (200 rows) + 1 epoch, to verify the pipeline runs correctly before a real run",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.labels, parse_dates=["published_at"])
    df = df.sort_values("published_at").reset_index(drop=True)
    if args.smoke_test:
        df = df.tail(200).reset_index(drop=True)
        args.epochs = 1
        log("--smoke-test: using last 200 rows, 1 epoch")

    # Time-based split, not random -- same reason this repo's walk-forward
    # eval never shuffles: a random split would leak near-future titles
    # into training right next to their validation-set neighbors.
    val_size = max(1, int(len(df) * args.val_fraction))
    train_df, val_df = df.iloc[:-val_size], df.iloc[-val_size:]
    log(f"{len(train_df)} train rows, {len(val_df)} val rows (time-based split, val = most recent)")
    log("Train label balance:\n" + train_df["label"].value_counts().to_string())

    log(f"Loading base model {NEWS_FINBERT_MODEL}")
    config = AutoConfig.from_pretrained(NEWS_FINBERT_MODEL)
    label2id = config.label2id  # e.g. {"positive": 0, "negative": 1, "neutral": 2} -- not alphabetical, don't assume
    tokenizer = AutoTokenizer.from_pretrained(NEWS_FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NEWS_FINBERT_MODEL)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        target_modules=["query", "value"],  # BERT-family self-attention projections
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def encode(frame: pd.DataFrame) -> TitleLabelDataset:
        # Asset prefix: build_finbert_labels.py emits one row per (article,
        # asset) it mentions, so a multi-asset headline ("BTC, ETH, XRP
        # Collapse") appears once per asset with potentially different
        # labels (checked: 13.8% of rows, 4491/32635, come from such
        # titles) -- without telling the model which asset's reaction
        # it's being asked about, that's the same input text mapped to
        # contradictory targets. "[ASSET] title" resolves the ambiguity
        # instead of leaving it for the model to somehow guess.
        #
        # Body (scripts/backfill_article_bodies.py), falling back to title
        # when a row wasn't crawled (--labels without --include-body) or
        # crawling found nothing usable (body=''): title alone is ~64
        # chars/10 words, a headline, not the article -- max_length bumped
        # to BERT's 512-token ceiling to match, up from 128 for titles.
        if "body" in frame.columns:
            body = frame["body"].where(frame["body"].fillna("").str.len() > 20, frame["title"])
        else:
            body = frame["title"]
        texts = ("[" + frame["asset"] + "] " + body).tolist()
        enc = tokenizer(texts, truncation=True, max_length=512)
        labels = [label2id[lbl] for lbl in frame["label"]]
        return TitleLabelDataset(enc, labels)

    train_ds, val_ds = encode(train_df), encode(val_df)

    training_args = TrainingArguments(
        output_dir=f"{args.out}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        learning_rate=args.learning_rate,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    log("Training starting")
    trainer.train()
    metrics = trainer.evaluate()
    log(f"Final eval metrics: {metrics}")

    log("Merging LoRA adapters into base weights for standalone serving")
    merged = model.merge_and_unload()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    provenance = {
        "base_model": NEWS_FINBERT_MODEL,
        "labels_file": args.labels,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "learning_rate": args.learning_rate,
        "eval_metrics": metrics,
        "smoke_test": args.smoke_test,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    log(f"Saved standalone model -> {out_dir}")
    log(
        "NOT auto-adopted: rerun measure_news_feature_gain.py with NEWS_FINBERT_MODEL "
        f"pointed at {out_dir} and compare against the current baseline before using this for real."
    )


if __name__ == "__main__":
    main()
