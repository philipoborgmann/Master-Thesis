#!/usr/bin/env python3
"""
Sentiment_Score_Finbert.py
==========================
Scores Reddit post titles and selftexts using ProsusAI/finbert.

Reads the cleaned, combined sentiment CSV and adds FinBERT scores
for both text fields separately. Saves a checkpoint every N batches
so interruptions don't cost the whole run.

Input:
    - Data/Processed/Sentiment/sentiment_combined.csv

Output:
    - Data/Transformed/Sentiment_Scored_Finbert.csv

Intermediate:
    - Same path as output, written once after title scoring completes.
      Means if selftext scoring crashes, titles are already on disk.

Checkpoint files (auto-created, auto-cleaned on success):
    - Data/Transformed/.finbert_checkpoints/title.pkl
    - Data/Transformed/.finbert_checkpoints/selftext.pkl

Columns added:
    Title scoring:
        title_sentiment         label (positive / negative / neutral)
        title_score             P(positive) - P(negative), range [-1, +1]
        title_prob_pos          P(positive)
        title_prob_neg          P(negative)
        title_prob_neu          P(neutral)

    Selftext scoring (only for posts with usable text):
        selftext_sentiment      label / NaN
        selftext_score          P(positive) - P(negative) / NaN
        selftext_prob_pos       P(positive) / NaN
        selftext_prob_neg       P(negative) / NaN
        selftext_prob_neu       P(neutral) / NaN

Requirements:
    pip install pandas transformers torch tqdm

Usage:
    python Sentiment_Score_Finbert.py

    # Quick test on first 7 days
    python Sentiment_Score_Finbert.py --test_days 7

    # Custom batch size (reduce if GPU runs out of memory)
    python Sentiment_Score_Finbert.py --batch_size 32

    # Save checkpoint every 5 batches (default is 10)
    python Sentiment_Score_Finbert.py --checkpoint_every 5

    # Ignore any existing checkpoint and start fresh
    python Sentiment_Score_Finbert.py --restart
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME         = "ProsusAI/finbert"
MAX_SEQ_LENGTH     = 512
EMPTY_SELFTEXT     = {"[removed]", "[deleted]", "", "nan", "None"}
CHECKPOINT_DIRNAME = ".finbert_checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_finbert():
    """
    Loads the ProsusAI/finbert tokenizer and model.
    Automatically selects GPU if available.

    Returns:
        tokenizer, model, device
    """
    print(f"[INFO] Loading FinBERT model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    return tokenizer, model, device


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _texts_fingerprint(texts: list) -> str:
    """
    Cheap fingerprint of the input text list. Lets us detect if the data
    has changed between runs (e.g. different --test_days, different input file).
    """
    if not texts:
        return hashlib.md5(b"empty").hexdigest()
    sample = "|".join(texts[:5]) + "|" + "|".join(texts[-5:])
    payload = f"{len(texts)}::{sample}".encode("utf-8", errors="replace")
    return hashlib.md5(payload).hexdigest()


def _save_checkpoint(path: str, data: dict) -> None:
    """Atomic save: write to .tmp then os.replace (crash-safe)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 4. BATCH SCORING WITH CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def score_texts(texts: list, tokenizer, model, device,
                batch_size: int, desc: str,
                checkpoint_path: str, checkpoint_every: int,
                restart: bool = False) -> dict:
    """
    Scores a list of text strings with FinBERT. Resumable: saves progress
    every `checkpoint_every` batches, reloads on restart if fingerprint matches.

    FinBERT label mapping:
        0 = positive
        1 = negative
        2 = neutral

    Returns:
        dict with keys: labels, scores, prob_pos, prob_neg, prob_neu
    """
    label_map   = {0: "positive", 1: "negative", 2: "neutral"}
    fingerprint = _texts_fingerprint(texts)
    n_total     = len(texts)

    # ── Initialize accumulators ─────────────────────────────────────────
    labels, scores        = [], []
    prob_pos, prob_neg    = [], []
    prob_neu              = []
    start_idx             = 0

    # ── Optional: wipe checkpoint before reading ────────────────────────
    if restart and os.path.isfile(checkpoint_path):
        print(f"[INFO] --restart: removing existing checkpoint {checkpoint_path}")
        os.remove(checkpoint_path)

    # ── Load existing checkpoint if compatible ──────────────────────────
    if os.path.isfile(checkpoint_path):
        try:
            ckpt = _load_checkpoint(checkpoint_path)
            same_fp  = ckpt.get("fingerprint") == fingerprint
            same_tot = ckpt.get("n_total") == n_total
            if same_fp and same_tot:
                labels   = ckpt["labels"]
                scores   = ckpt["scores"]
                prob_pos = ckpt["prob_pos"]
                prob_neg = ckpt["prob_neg"]
                prob_neu = ckpt["prob_neu"]
                start_idx = len(labels)
                print(f"[INFO] Resuming from checkpoint: "
                      f"{start_idx:,} / {n_total:,} already scored "
                      f"({start_idx / n_total * 100:.1f}%)")
            else:
                print(f"[WARN] Checkpoint fingerprint mismatch — input data "
                      f"changed since last run. Starting fresh.")
                os.remove(checkpoint_path)
        except Exception as e:
            print(f"[WARN] Could not load checkpoint ({e}). Starting fresh.")
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)

    # ── Short-circuit: already done ─────────────────────────────────────
    if start_idx >= n_total:
        print(f"[INFO] All {n_total:,} texts already scored in checkpoint.")
        return {"labels": labels, "scores": scores,
                "prob_pos": prob_pos, "prob_neg": prob_neg, "prob_neu": prob_neu}

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _flush():
        _save_checkpoint(checkpoint_path, {
            "fingerprint": fingerprint,
            "n_total":  n_total,
            "labels":   labels,
            "scores":   scores,
            "prob_pos": prob_pos,
            "prob_neg": prob_neg,
            "prob_neu": prob_neu,
        })

    # ── Score remaining batches ─────────────────────────────────────────
    batches_since_save = 0

    with tqdm(total=n_total, initial=start_idx, desc=desc) as pbar:
        try:
            for i in range(start_idx, n_total, batch_size):
                batch = texts[i : i + batch_size]

                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**encoded)

                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)

                for j in range(len(batch)):
                    labels.append(label_map[preds[j]])
                    scores.append(float(probs[j][0] - probs[j][1]))
                    prob_pos.append(float(probs[j][0]))
                    prob_neg.append(float(probs[j][1]))
                    prob_neu.append(float(probs[j][2]))

                pbar.update(len(batch))
                batches_since_save += 1

                if batches_since_save >= checkpoint_every:
                    _flush()
                    batches_since_save = 0

        except BaseException as e:
            # Catches KeyboardInterrupt AND exceptions — flush before propagating
            _flush()
            print(f"\n[WARN] Interrupted at {len(labels):,} / {n_total:,}. "
                  f"Progress saved to {checkpoint_path}")
            raise

    # Final flush
    _flush()

    return {"labels": labels, "scores": scores,
            "prob_pos": prob_pos, "prob_neg": prob_neg, "prob_neu": prob_neu}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORE TITLE COLUMN
# ══════════════════════════════════════════════════════════════════════════════

def score_titles(df: pd.DataFrame, tokenizer, model, device,
                 batch_size: int, checkpoint_path: str,
                 checkpoint_every: int, restart: bool) -> pd.DataFrame:
    """
    Scores every post title with FinBERT.
    Titles are expected to be non-null (cleaned in Sentiment_Data_Load).
    """
    print(f"\n[INFO] Scoring {len(df):,} titles...")
    titles = df["title"].fillna("").tolist()
    results = score_texts(
        titles, tokenizer, model, device,
        batch_size=batch_size, desc="Title scoring",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        restart=restart,
    )

    df["title_sentiment"] = results["labels"]
    df["title_score"]     = results["scores"]
    df["title_prob_pos"]  = results["prob_pos"]
    df["title_prob_neg"]  = results["prob_neg"]
    df["title_prob_neu"]  = results["prob_neu"]

    dist = pd.Series(results["labels"]).value_counts()
    print(f"[INFO] Title sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        count = dist.get(label, 0)
        print(f"         {label:10s}  {count:>8,}  ({count / len(df) * 100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORE SELFTEXT COLUMN
# ══════════════════════════════════════════════════════════════════════════════

def score_selftexts(df: pd.DataFrame, tokenizer, model, device,
                    batch_size: int, checkpoint_path: str,
                    checkpoint_every: int, restart: bool) -> pd.DataFrame:
    """
    Scores selftext for posts that have usable text content.
    Posts with [removed], [deleted], or empty selftext get NaN scores.
    """
    df["_selftext_clean"] = df["selftext"].fillna("").astype(str).str.strip()
    has_text = ~df["_selftext_clean"].isin(EMPTY_SELFTEXT) & (df["_selftext_clean"].str.len() > 0)

    n_with_text = has_text.sum()
    n_without   = (~has_text).sum()
    print(f"\n[INFO] Selftext: {n_with_text:,} usable, "
          f"{n_without:,} empty/removed/deleted → skipped")

    # Initialize with NaN
    df["selftext_sentiment"] = None          # object dtype — accepts strings + NaN
    df["selftext_score"]     = np.nan
    df["selftext_prob_pos"]  = np.nan
    df["selftext_prob_neg"]  = np.nan
    df["selftext_prob_neu"]  = np.nan

    if n_with_text == 0:
        print("[WARN] No usable selftext found, skipping selftext scoring")
        df = df.drop(columns=["_selftext_clean"])
        return df

    texts = df.loc[has_text, "_selftext_clean"].tolist()
    results = score_texts(
        texts, tokenizer, model, device,
        batch_size=batch_size, desc="Selftext scoring",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        restart=restart,
    )

    idx = df.index[has_text]
    df.loc[idx, "selftext_sentiment"] = results["labels"]
    df.loc[idx, "selftext_score"]     = results["scores"]
    df.loc[idx, "selftext_prob_pos"]  = results["prob_pos"]
    df.loc[idx, "selftext_prob_neg"]  = results["prob_neg"]
    df.loc[idx, "selftext_prob_neu"]  = results["prob_neu"]

    dist = pd.Series(results["labels"]).value_counts()
    print(f"[INFO] Selftext sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        count = dist.get(label, 0)
        print(f"         {label:10s}  {count:>8,}  ({count / n_with_text * 100:.1f}%)")

    df = df.drop(columns=["_selftext_clean"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAFE OUTPUT SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_output(df: pd.DataFrame, path: str, stage_label: str = "") -> None:
    """
    Atomic CSV save: write to .tmp, rename into place, verify non-zero size,
    log the absolute path so there's no ambiguity about where the file lives.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    tmp = abs_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, abs_path)

    size = os.path.getsize(abs_path)
    if size == 0:
        raise RuntimeError(f"Output file {abs_path} is empty after write!")

    prefix = f"[INFO] {stage_label}: " if stage_label else "[INFO] "
    print(f"{prefix}Saved → {abs_path} ({size / 1024 / 1024:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK VALIDATION (--validate)
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(batch_size: int = 64, n_samples: int = 1000):
    """
    Validates the FinBERT scoring pipeline against the Financial PhraseBank
    benchmark (Malo et al., 2014) — the dataset FinBERT was trained on.

    Loads a random subset, scores it with the same pipeline used for
    Reddit posts, and computes classification metrics (accuracy, precision,
    recall, F1) per sentiment class.

    Reference accuracy: ~86% on Sentences_50Agree (Araci, 2019).
    """
    from datasets import load_dataset
    from sklearn.metrics import classification_report, accuracy_score

    print("\n" + "=" * 60)
    print("BENCHMARK VALIDATION: FinBERT on Financial PhraseBank")
    print("=" * 60)

    # Load dataset (50% annotator agreement subset)
    ds = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        revision="refs/pr/10",
        split="train"
    )
    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    # Random subset
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    texts = ds["sentence"]
    y_true = [label_names[l] for l in ds["label"]]
    print(f"[INFO] Sampled {len(texts)} sentences from Financial PhraseBank")

    # Load model and score
    tokenizer, model, device = load_finbert()

    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    y_pred = []

    print("[INFO] Scoring benchmark texts...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        preds = torch.softmax(outputs.logits, dim=-1).cpu().numpy().argmax(axis=1)
        y_pred.extend([label_map[p] for p in preds])

    # Classification report
    print(f"\n{'─' * 60}")
    print(f"RESULTS: FinBERT on Financial PhraseBank (n={len(texts)})")
    print(f"{'─' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"\n{classification_report(y_true, y_pred, digits=4)}")


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment_Score_Finbert — Score titles & selftexts with FinBERT"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join("Data", "Processed", "Sentiment", "sentiment_combined.csv"),
        help="Path to the combined sentiment CSV"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("Data", "Transformed", "Sentiment_Scored_Finbert.csv"),
        help="Output path for scored CSV"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="FinBERT batch size (reduce to 16-32 if GPU OOM)"
    )
    parser.add_argument(
        "--test_days", type=int, default=None,
        help="Only score the first N days for a quick test run"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=10,
        help="Save checkpoint every N batches (default: 10)"
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore existing checkpoints and start fresh"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run benchmark validation on Financial PhraseBank instead of scoring"
    )
    parser.add_argument(
        "--validate_n", type=int, default=1000,
        help="Number of benchmark samples (default: 1000)"
    )
    args = parser.parse_args()

    # ── Benchmark validation mode ────────────────────────────
    if args.validate:
        run_validation(batch_size=args.batch_size, n_samples=args.validate_n)
        return

    # ── Validate input ──────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print(f"        Run Sentiment_Data_Load.py first to create it.")
        sys.exit(1)

    # ── Resolve output + checkpoint paths up front ─────────────
    output_path = args.output
    if args.test_days is not None:
        base, ext   = os.path.splitext(args.output)
        output_path = f"{base}_test{args.test_days}d{ext}"

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir     = os.path.join(output_dir, CHECKPOINT_DIRNAME)
    test_suffix  = f"_test{args.test_days}d" if args.test_days is not None else ""
    ckpt_title   = os.path.join(ckpt_dir, f"title{test_suffix}.pkl")
    ckpt_seltext = os.path.join(ckpt_dir, f"selftext{test_suffix}.pkl")

    print(f"[INFO] Input:           {os.path.abspath(args.input)}")
    print(f"[INFO] Output:          {os.path.abspath(output_path)}")
    print(f"[INFO] Checkpoint dir:  {ckpt_dir}")

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading processed sentiment data")
    print("=" * 60)

    df = pd.read_csv(args.input, parse_dates=["date"])
    print(f"[INFO] Loaded {len(df):,} posts")

    if args.test_days is not None:
        cutoff = df["date"].min() + pd.Timedelta(days=args.test_days)
        before = len(df)
        df = df[df["date"] < cutoff].reset_index(drop=True)
        print(f"[INFO] TEST MODE: first {args.test_days} days → "
              f"{before:,} → {len(df):,} posts")

    # ── Step 2: Load FinBERT ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading FinBERT model")
    print("=" * 60)

    tokenizer, model, device = load_finbert()

    # ── Step 3: Score titles ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Scoring titles")
    print("=" * 60)

    t0 = time.time()
    df = score_titles(df, tokenizer, model, device, args.batch_size,
                      ckpt_title, args.checkpoint_every, args.restart)
    title_time = time.time() - t0
    print(f"[INFO] Title scoring completed in {title_time:.0f}s "
          f"({len(df) / max(title_time, 1e-9):.0f} posts/sec)")

    # Intermediate save — titles are now durable even if selftext step dies
    save_output(df, output_path, stage_label="After titles")

    # ── Step 4: Score selftexts ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Scoring selftexts")
    print("=" * 60)

    t1 = time.time()
    df = score_selftexts(df, tokenizer, model, device, args.batch_size,
                         ckpt_seltext, args.checkpoint_every, args.restart)
    selftext_time = time.time() - t1
    print(f"[INFO] Selftext scoring completed in {selftext_time:.0f}s")

    # ── Step 5: Final export ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Exporting scored data")
    print("=" * 60)

    save_output(df, output_path, stage_label="Final")

    # ── Clean up checkpoints on success ────────────────────────
    for p in [ckpt_title, ckpt_seltext]:
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        if os.path.isdir(ckpt_dir) and not os.listdir(ckpt_dir):
            os.rmdir(ckpt_dir)
    except OSError:
        pass

    # ── Summary ────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Posts scored:       {len(df):,}")
    print(f"  Title scores:       {df['title_sentiment'].notna().sum():,}")
    print(f"  Selftext scores:    {df['selftext_sentiment'].notna().sum():,}")
    print(f"  Total time:         {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Output:             {os.path.abspath(output_path)}")
    print()


if __name__ == "__main__":
    main()