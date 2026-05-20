#!/usr/bin/env python3
# This wrapper is kept for backward compatibility. Preferred usage: python -m thesis_pipeline.cli score-sentiment --model vader
"""
Sentiment_Score_Vader.py
==========================
Scores Reddit post titles and selftexts using VADER (Valence Aware Dictionary
and sEntiment Reasoner). Reads the cleaned, combined sentiment CSV and adds
VADER scores for both text fields separately.

Saves a checkpoint every N texts so interruptions don't cost the whole run.

Input:
    - Data/Processed/Sentiment/sentiment_combined.csv

Output:
    - Data/Transformed/Sentiment_Scored_Vader.csv

Intermediate:
    - Same path as output, written once after title scoring completes.
      Means if selftext scoring crashes, titles are already on disk.

Checkpoint files (auto-created, auto-cleaned on success):
    - Data/Transformed/.vader_checkpoints/title.pkl
    - Data/Transformed/.vader_checkpoints/selftext.pkl

Columns added:
    Title scoring:
        title_sentiment         label (positive / negative / neutral)
        title_score             compound score, range [-1, +1]
        title_prob_pos          VADER pos proportion
        title_prob_neg          VADER neg proportion
        title_prob_neu          VADER neu proportion

    Selftext scoring (only for posts with usable text):
        selftext_sentiment      label (positive / negative / neutral / NaN)
        selftext_score          compound score, range [-1, +1] / NaN
        selftext_prob_pos       VADER pos proportion / NaN
        selftext_prob_neg       VADER neg proportion / NaN
        selftext_prob_neu       VADER neu proportion / NaN

    Label thresholds (standard VADER):
        compound >=  0.05  →  positive
        compound <= -0.05  →  negative
        else               →  neutral

Requirements:
    pip install pandas vaderSentiment tqdm

Usage:
    python Sentiment_Score_Vader.py

    # Quick test on first 7 days
    python Sentiment_Score_Vader.py --test_days 7

    # Save checkpoint every 5000 texts (default is 10000)
    python Sentiment_Score_Vader.py --checkpoint_every 5000

    # Ignore any existing checkpoint and start fresh
    python Sentiment_Score_Vader.py --restart
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Standard VADER compound thresholds
POS_THRESHOLD      =  0.05
NEG_THRESHOLD      = -0.05

# Selftext values that indicate no usable text
EMPTY_SELFTEXT     = {"[removed]", "[deleted]", "", "nan", "None"}

CHECKPOINT_DIRNAME = ".vader_checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_vader():
    """
    Initialises the VADER SentimentIntensityAnalyzer.

    Returns:
        analyzer   (SentimentIntensityAnalyzer)
    """
    print("[INFO] Loading VADER lexicon...")
    analyzer = SentimentIntensityAnalyzer()
    print("[INFO] VADER ready (CPU-only, no GPU needed)")
    return analyzer


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
# 4. SCORING WITH CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def score_texts(texts: list, analyzer: SentimentIntensityAnalyzer,
                desc: str, checkpoint_path: str,
                checkpoint_every: int,
                restart: bool = False) -> dict:
    """
    Scores a list of text strings with VADER. Resumable: saves progress
    every `checkpoint_every` texts, reloads on restart if fingerprint matches.

    VADER polarity_scores returns:
        pos, neg, neu  — proportions summing to 1.0
        compound       — normalised weighted composite, range [-1, +1]

    Returns:
        dict with keys: labels, scores, prob_pos, prob_neg, prob_neu
    """
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

    # ── Score remaining texts ───────────────────────────────────────────
    texts_since_save = 0

    try:
        for i in tqdm(range(start_idx, n_total), initial=start_idx,
                      total=n_total, desc=desc):
            vs = analyzer.polarity_scores(texts[i])

            compound = vs["compound"]
            if compound >= POS_THRESHOLD:
                label = "positive"
            elif compound <= NEG_THRESHOLD:
                label = "negative"
            else:
                label = "neutral"

            labels.append(label)
            scores.append(compound)
            prob_pos.append(vs["pos"])
            prob_neg.append(vs["neg"])
            prob_neu.append(vs["neu"])

            texts_since_save += 1
            if texts_since_save >= checkpoint_every:
                _flush()
                texts_since_save = 0

    except BaseException:
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

def score_titles(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer,
                 checkpoint_path: str, checkpoint_every: int,
                 restart: bool) -> pd.DataFrame:
    """
    Scores every post title with VADER.
    Titles are expected to be non-null (cleaned in Sentiment_Data_Load).
    """
    print(f"\n[INFO] Scoring {len(df):,} titles...")
    titles = df["title"].fillna("").tolist()
    results = score_texts(
        titles, analyzer,
        desc="Title scoring",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        restart=restart,
    )

    df["title_sentiment"] = results["labels"]
    df["title_score"]     = results["scores"]
    df["title_prob_pos"]  = results["prob_pos"]
    df["title_prob_neg"]  = results["prob_neg"]
    df["title_prob_neu"]  = results["prob_neu"]

    # Print distribution
    dist = pd.Series(results["labels"]).value_counts()
    print(f"[INFO] Title sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        count = dist.get(label, 0)
        print(f"         {label:10s}  {count:>8,}  ({count / len(df) * 100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORE SELFTEXT COLUMN
# ══════════════════════════════════════════════════════════════════════════════

def score_selftexts(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer,
                    checkpoint_path: str, checkpoint_every: int,
                    restart: bool) -> pd.DataFrame:
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

    # Initialize score columns
    df["selftext_sentiment"] = None          # object dtype — accepts strings + NaN
    df["selftext_score"]     = np.nan
    df["selftext_prob_pos"]  = np.nan
    df["selftext_prob_neg"]  = np.nan
    df["selftext_prob_neu"]  = np.nan

    if n_with_text == 0:
        print("[WARN] No usable selftext found, skipping selftext scoring")
        df = df.drop(columns=["_selftext_clean"])
        return df

    # Score only usable selftexts
    texts = df.loc[has_text, "_selftext_clean"].tolist()
    results = score_texts(
        texts, analyzer,
        desc="Selftext scoring",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        restart=restart,
    )

    # Write results back to the correct row positions
    idx = df.index[has_text]
    df.loc[idx, "selftext_sentiment"] = results["labels"]
    df.loc[idx, "selftext_score"]     = results["scores"]
    df.loc[idx, "selftext_prob_pos"]  = results["prob_pos"]
    df.loc[idx, "selftext_prob_neg"]  = results["prob_neg"]
    df.loc[idx, "selftext_prob_neu"]  = results["prob_neu"]

    # Print distribution
    dist = pd.Series(results["labels"]).value_counts()
    print(f"[INFO] Selftext sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        count = dist.get(label, 0)
        print(f"         {label:10s}  {count:>8,}  ({count / n_with_text * 100:.1f}%)")

    # Drop helper column
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

def run_validation(n_samples: int = 1000):
    """
    Validates the VADER scoring pipeline against the TweetEval sentiment
    benchmark (Barbieri et al., 2020), derived from SemEval-2013 Task 2
    (Nakov et al., 2013) — the standard benchmark for social media
    sentiment analysis.

    Loads a random subset, scores it with the same VADER pipeline used
    for Reddit posts, and computes classification metrics per class.
    """
    from datasets import load_dataset
    from sklearn.metrics import classification_report, accuracy_score

    print("\n" + "=" * 60)
    print("BENCHMARK VALIDATION: VADER on TweetEval Sentiment")
    print("=" * 60)

    # Load dataset (test split for proper evaluation)
    ds = load_dataset("tweet_eval", "sentiment", split="test")

    # TweetEval labels: 0=negative, 1=neutral, 2=positive
    ds_label_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Random subset
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=42).select(range(n_samples))
    texts = ds["text"]
    y_true = [ds_label_map[l] for l in ds["label"]]
    print(f"[INFO] Sampled {len(texts)} tweets from TweetEval Sentiment")

    # Load VADER
    analyzer = load_vader()

    # Score each text using the same thresholds as the main pipeline
    y_pred = []
    for text in texts:
        vs = analyzer.polarity_scores(text)
        compound = vs["compound"]
        if compound >= POS_THRESHOLD:
            y_pred.append("positive")
        elif compound <= NEG_THRESHOLD:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")

    # Classification report
    print(f"\n{'─' * 60}")
    print(f"RESULTS: VADER on TweetEval Sentiment (n={len(texts)})")
    print(f"{'─' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"\n{classification_report(y_true, y_pred, digits=4)}")


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment_Score_Vader — Score titles & selftexts with VADER"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join("Data", "Processed", "Sentiment", "sentiment_combined.csv"),
        help="Path to the combined sentiment CSV"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("Data", "Transformed", "Sentiment_Scored_Vader.csv"),
        help="Output path for scored CSV"
    )
    parser.add_argument(
        "--test_days", type=int, default=None,
        help="Only score the first N days for a quick test run"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=10000,
        help="Save checkpoint every N texts (default: 10000)"
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore existing checkpoints and start fresh"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run benchmark validation on TweetEval Sentiment instead of scoring"
    )
    parser.add_argument(
        "--validate_n", type=int, default=1000,
        help="Number of benchmark samples (default: 1000)"
    )
    args = parser.parse_args()

    # ── Benchmark validation mode ────────────────────────────
    if args.validate:
        run_validation(n_samples=args.validate_n)
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

    # ── Step 2: Load VADER ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading VADER lexicon")
    print("=" * 60)

    analyzer = load_vader()

    # ── Step 3: Score titles ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Scoring titles")
    print("=" * 60)

    t0 = time.time()
    df = score_titles(df, analyzer,
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
    df = score_selftexts(df, analyzer,
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
