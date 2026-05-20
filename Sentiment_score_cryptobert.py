#!/usr/bin/env python3
# This wrapper is kept for backward compatibility. Preferred usage: python -m thesis_pipeline.cli score-sentiment --model cryptobert
"""
Sentiment_Score_Cryptobert.py
==============================
Scores Reddit post titles and selftexts using ElKulako/cryptobert.

CryptoBERT is a RoBERTa-based model post-trained on 3.2M crypto social-media
posts and fine-tuned for 3-class sentiment (Bearish / Neutral / Bullish) on
2M StockTwits posts. Recommended max sequence length is 128 tokens.

Reads the cleaned, combined sentiment CSV and adds CryptoBERT scores
for both text fields separately. Saves a checkpoint every N batches
so interruptions don't cost the whole run.

Input:
    - Data/Processed/Sentiment/sentiment_combined.csv

Output:
    - Data/Transformed/Sentiment_Scored_Cryptobert.csv

Intermediate:
    - Same path as output, written once after title scoring completes.
      Means if selftext scoring crashes, titles are already on disk.

Checkpoint files (auto-created, auto-cleaned on success):
    - Data/Transformed/.cryptobert_checkpoints/title.pkl
    - Data/Transformed/.cryptobert_checkpoints/selftext.pkl

Columns added:
    Title scoring:
        title_sentiment         label (positive / negative / neutral)
        title_score             P(bullish) - P(bearish), range [-1, +1]
        title_prob_pos          P(bullish)
        title_prob_neg          P(bearish)
        title_prob_neu          P(neutral)

    Selftext scoring (only for posts with usable text):
        selftext_sentiment      label / NaN
        selftext_score          P(bullish) - P(bearish) / NaN
        selftext_prob_pos       P(bullish) / NaN
        selftext_prob_neg       P(bearish) / NaN
        selftext_prob_neu       P(neutral) / NaN

    CryptoBERT native labels → unified labels:
        Bearish  →  negative
        Neutral  →  neutral
        Bullish  →  positive

Requirements:
    pip install pandas transformers torch tqdm openpyxl pyarrow

Usage:
    python Sentiment_Score_Cryptobert.py

    # Quick test on first 7 days
    python Sentiment_Score_Cryptobert.py --test_days 7

    # Custom batch size (reduce if GPU runs out of memory)
    python Sentiment_Score_Cryptobert.py --batch_size 32

    # Save checkpoint every 5 batches (default is 10)
    python Sentiment_Score_Cryptobert.py --checkpoint_every 5

    # Ignore any existing checkpoint and start fresh
    python Sentiment_Score_Cryptobert.py --restart
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

MODEL_NAME         = "ElKulako/cryptobert"
MAX_SEQ_LENGTH     = 128          # CryptoBERT trained with 128; up to 514 possible but not recommended
EMPTY_SELFTEXT     = {"[removed]", "[deleted]", "", "nan", "None"}
CHECKPOINT_DIRNAME = ".cryptobert_checkpoints"

# Local benchmark file for --validate
BENCHMARK_XLSX     = os.path.join("Data", "Raw", "Sentiment", "st-data-full.xlsx")
BENCHMARK_PARQUET  = os.path.join("Data", "Raw", "Sentiment", "st-data-full.parquet")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_cryptobert():
    """
    Loads the ElKulako/cryptobert tokenizer and model.
    Automatically selects GPU if available.

    Returns:
        tokenizer, model, device, label_map
    """
    print(f"[INFO] Loading CryptoBERT model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Read label mapping from model config, fall back to known default
    id2label = model.config.id2label
    print(f"[INFO] Model id2label: {id2label}")

    # Map CryptoBERT's native labels to the unified scheme used across scripts
    NATIVE_TO_UNIFIED = {
        "Bearish":  "negative",
        "Neutral":  "neutral",
        "Bullish":  "positive",
        # lower-case fallbacks just in case
        "bearish":  "negative",
        "neutral":  "neutral",
        "bullish":  "positive",
    }

    label_map = {}
    bullish_idx = None
    bearish_idx = None
    for idx, native_label in id2label.items():
        unified = NATIVE_TO_UNIFIED.get(native_label)
        if unified is None:
            print(f"[WARN] Unknown native label '{native_label}' at index {idx}, "
                  f"mapping to 'neutral'")
            unified = "neutral"
        label_map[int(idx)] = unified
        if unified == "positive":
            bullish_idx = int(idx)
        elif unified == "negative":
            bearish_idx = int(idx)

    if bullish_idx is None or bearish_idx is None:
        print(f"[WARN] Could not identify bullish/bearish indices from config. "
              f"Falling back to default: 0=Bearish, 1=Neutral, 2=Bullish")
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        bullish_idx = 2
        bearish_idx = 0

    print(f"[INFO] Unified label map: {label_map}")
    print(f"[INFO] Score = P(idx={bullish_idx}) - P(idx={bearish_idx})")

    return tokenizer, model, device, label_map, bullish_idx, bearish_idx


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
                label_map: dict, bullish_idx: int, bearish_idx: int,
                batch_size: int, desc: str,
                checkpoint_path: str, checkpoint_every: int,
                restart: bool = False) -> dict:
    """
    Scores a list of text strings with CryptoBERT. Resumable: saves progress
    every `checkpoint_every` batches, reloads on restart if fingerprint matches.

    Returns:
        dict with keys: labels, scores, prob_pos, prob_neg, prob_neu
    """
    fingerprint = _texts_fingerprint(texts)
    n_total     = len(texts)

    # Find neutral index (the one that is neither bullish nor bearish)
    neutral_idx = [i for i in label_map if i not in (bullish_idx, bearish_idx)][0]

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
                    scores.append(float(probs[j][bullish_idx] - probs[j][bearish_idx]))
                    prob_pos.append(float(probs[j][bullish_idx]))
                    prob_neg.append(float(probs[j][bearish_idx]))
                    prob_neu.append(float(probs[j][neutral_idx]))

                pbar.update(len(batch))
                batches_since_save += 1

                if batches_since_save >= checkpoint_every:
                    _flush()
                    batches_since_save = 0

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

def score_titles(df: pd.DataFrame, tokenizer, model, device,
                 label_map: dict, bullish_idx: int, bearish_idx: int,
                 batch_size: int, checkpoint_path: str,
                 checkpoint_every: int, restart: bool) -> pd.DataFrame:
    """
    Scores every post title with CryptoBERT.
    Titles are expected to be non-null (cleaned in Sentiment_Data_Load).
    """
    print(f"\n[INFO] Scoring {len(df):,} titles...")
    titles = df["title"].fillna("").tolist()
    results = score_texts(
        titles, tokenizer, model, device,
        label_map=label_map, bullish_idx=bullish_idx, bearish_idx=bearish_idx,
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

def score_selftexts(df: pd.DataFrame, tokenizer, model, device,
                    label_map: dict, bullish_idx: int, bearish_idx: int,
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
        texts, tokenizer, model, device,
        label_map=label_map, bullish_idx=bullish_idx, bearish_idx=bearish_idx,
        batch_size=batch_size, desc="Selftext scoring",
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
# 8. BENCHMARK DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_benchmark_parquet(xlsx_path: str = BENCHMARK_XLSX,
                             parquet_path: str = BENCHMARK_PARQUET) -> str:
    """
    Converts the local ElKulako StockTwits benchmark from XLSX to Parquet once.
    If the Parquet file already exists and is newer than the XLSX file, it is reused.

    The uploaded workbook contains two sheets:
        - stocktwits_1
        - stocktwits_2

    Both sheets contain:
        - column A: text
        - column B: label

    The text column contains a few non-string cells, e.g. numeric 0 and Excel
    errors like #NAME?. Therefore this function parses the XLSX directly,
    cleans invalid rows, and writes a stable Parquet schema:
        text: string
        label: string
        source_sheet: string

    Expected source:
        Data/Raw/Sentiment/st-data-full.xlsx

    Created target:
        Data/Raw/Sentiment/st-data-full.parquet
    """
    import posixpath
    import re
    import zipfile
    import xml.etree.ElementTree as ET

    import pyarrow as pa
    import pyarrow.parquet as pq

    xlsx_abs = os.path.abspath(xlsx_path)
    parquet_abs = os.path.abspath(parquet_path)

    if not os.path.isfile(xlsx_abs):
        raise FileNotFoundError(
            f"Benchmark XLSX not found: {xlsx_abs}\n"
            f"Place st-data-full.xlsx in Data/Raw/Sentiment/ first."
        )

    needs_conversion = (
        not os.path.isfile(parquet_abs)
        or os.path.getmtime(xlsx_abs) > os.path.getmtime(parquet_abs)
    )

    if not needs_conversion:
        print(f"[INFO] Using existing benchmark Parquet: {parquet_abs}")
        return parquet_abs

    print(f"[INFO] Converting benchmark XLSX → Parquet")
    print(f"[INFO] XLSX:    {xlsx_abs}")
    print(f"[INFO] Parquet: {parquet_abs}")

    os.makedirs(os.path.dirname(parquet_abs), exist_ok=True)

    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    bad_text_values = {
        "",
        "0",
        "nan",
        "None",
        "#NAME?",
        "#VALUE!",
        "#REF!",
        "#N/A",
        "#DIV/0!",
        "#NULL!",
        "#NUM!",
    }

    def _load_shared_strings(zf):
        """Loads shared strings once. Needed because XLSX stores most text there."""
        shared = []
        with zf.open("xl/sharedStrings.xml") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if elem.tag == main_ns + "si":
                    value = "".join(t.text or "" for t in elem.iter(main_ns + "t"))
                    shared.append(value)
                    elem.clear()
        return shared

    def _sheet_paths(zf):
        """Returns [(sheet_name, sheet_xml_path), ...]."""
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        id_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        sheets = []
        for sheet in workbook.find(main_ns + "sheets"):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[rel_ns + "id"]
            target = id_to_target[rel_id]

            if target.startswith("/xl/"):
                sheet_path = target.lstrip("/")
            else:
                sheet_path = posixpath.normpath(posixpath.join("xl", target))

            sheets.append((sheet_name, sheet_path))

        return sheets

    def _cell_col(ref: str) -> str:
        match = re.match(r"([A-Z]+)", ref or "")
        return match.group(1) if match else ""

    def _cell_value(cell, shared):
        cell_type = cell.attrib.get("t")

        if cell_type == "inlineStr":
            return "".join(t.text or "" for t in cell.iter(main_ns + "t"))

        value_node = cell.find(main_ns + "v")
        if value_node is None:
            return None

        raw_value = value_node.text

        if cell_type == "s":
            try:
                return shared[int(raw_value)]
            except (ValueError, IndexError, TypeError):
                return None

        # Numeric cells and Excel error cells arrive here as strings.
        return raw_value

    schema = pa.schema([
        ("text", pa.string()),
        ("label", pa.string()),
        ("source_sheet", pa.string()),
    ])

    tmp = parquet_abs + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    writer = pq.ParquetWriter(tmp, schema=schema, compression="snappy")

    n_raw = 0
    n_clean = 0
    chunk_size = 100_000
    buffer_text = []
    buffer_label = []
    buffer_sheet = []

    def _flush():
        nonlocal buffer_text, buffer_label, buffer_sheet

        if not buffer_text:
            return

        table = pa.Table.from_pydict(
            {
                "text": buffer_text,
                "label": buffer_label,
                "source_sheet": buffer_sheet,
            },
            schema=schema,
        )
        writer.write_table(table)

        buffer_text = []
        buffer_label = []
        buffer_sheet = []

    try:
        with zipfile.ZipFile(xlsx_abs) as zf:
            shared = _load_shared_strings(zf)
            sheets = _sheet_paths(zf)

            print(f"[INFO] Workbook sheets: {[name for name, _ in sheets]}")
            print(f"[INFO] Shared strings:  {len(shared):,}")

            for sheet_name, sheet_path in sheets:
                print(f"[INFO] Reading sheet:   {sheet_name}")

                with zf.open(sheet_path) as f:
                    for _, elem in ET.iterparse(f, events=("end",)):
                        if elem.tag != main_ns + "row":
                            continue

                        row_number = int(elem.attrib.get("r", "0"))

                        # Header row: A=text, B=label
                        if row_number == 1:
                            elem.clear()
                            continue

                        n_raw += 1
                        text_value = None
                        label_value = None

                        for cell in elem.findall(main_ns + "c"):
                            col = _cell_col(cell.attrib.get("r"))

                            if col == "A":
                                text_value = _cell_value(cell, shared)
                            elif col == "B":
                                label_value = _cell_value(cell, shared)

                        text_value = "" if text_value is None else str(text_value).strip()
                        label_value = "" if label_value is None else str(label_value).strip()

                        if (
                            text_value not in bad_text_values
                            and len(text_value) > 1
                            and label_value in {"0", "1", "2"}
                        ):
                            buffer_text.append(text_value)
                            buffer_label.append(label_value)
                            buffer_sheet.append(sheet_name)
                            n_clean += 1

                            if len(buffer_text) >= chunk_size:
                                _flush()

                        elem.clear()

            _flush()
    finally:
        writer.close()

    os.replace(tmp, parquet_abs)

    size_mb = os.path.getsize(parquet_abs) / 1024 / 1024
    print(f"[INFO] Benchmark rows raw:   {n_raw:,}")
    print(f"[INFO] Benchmark rows clean: {n_clean:,}")
    print(f"[INFO] Dropped rows:         {n_raw - n_clean:,}")
    print(f"[INFO] Saved benchmark Parquet → {parquet_abs} ({size_mb:.1f} MB)")

    return parquet_abs


def _normalize_colname(col) -> str:
    """Normalizes column names for robust matching."""
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(df: pd.DataFrame, candidates: list, role: str) -> str:
    """Finds a column by trying several likely names."""
    normalized = {_normalize_colname(c): c for c in df.columns}
    for candidate in candidates:
        key = _normalize_colname(candidate)
        if key in normalized:
            return normalized[key]

    raise RuntimeError(
        f"Could not identify the benchmark {role} column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Tried candidates: {candidates}"
    )


def _map_benchmark_label(label) -> str:
    """
    Maps benchmark labels to the unified scheme used in this project:
        negative / neutral / positive
    """
    if pd.isna(label):
        raise ValueError("Missing benchmark label")

    # Existing script assumption: 0=negative, 1=neutral, 2=positive
    if isinstance(label, (int, np.integer)):
        numeric_map = {0: "negative", 1: "neutral", 2: "positive"}
        if int(label) in numeric_map:
            return numeric_map[int(label)]

    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        numeric_map = {0: "negative", 1: "neutral", 2: "positive"}
        if int(label) in numeric_map:
            return numeric_map[int(label)]

    s = str(label).strip()
    s_lower = s.lower()

    string_map = {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "bearish": "negative",
        "bullish": "positive",
        "bear": "negative",
        "bull": "positive",
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
    }

    if s_lower in string_map:
        return string_map[s_lower]

    raise ValueError(
        f"Unknown benchmark label: {label!r}. "
        f"Extend _map_benchmark_label() if your XLSX uses a different coding."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK VALIDATION (--validate)
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(batch_size: int = 64, n_samples: int = 1000):
    """
    Validates the CryptoBERT scoring pipeline against the local ElKulako
    StockTwits benchmark.

    The local XLSX file is converted to Parquet once and reused afterwards:
        Data/Raw/Sentiment/st-data-full.xlsx
        Data/Raw/Sentiment/st-data-full.parquet

    Loads a random subset, scores it with the same pipeline used for
    Reddit posts, and computes classification metrics per class.
    """
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    print("\n" + "=" * 60)
    print("BENCHMARK VALIDATION: CryptoBERT on ElKulako Dataset")
    print("=" * 60)

    # Convert XLSX to Parquet if needed, then load benchmark data
    parquet_path = ensure_benchmark_parquet()
    df_bench = pd.read_parquet(parquet_path)

    print(f"[INFO] Benchmark rows:    {len(df_bench):,}")
    print(f"[INFO] Benchmark columns: {df_bench.columns.tolist()}")

    # Try likely column names. If this fails, inspect the printed columns
    # and add the actual names to these candidate lists.
    text_col = _find_column(
        df_bench,
        candidates=["text", "message", "body", "post", "tweet", "content", "clean_text"],
        role="text"
    )
    label_col = _find_column(
        df_bench,
        candidates=["sentiment", "label", "labels", "class", "classification", "sentiment_label"],
        role="label"
    )

    print(f"[INFO] Using text column:  {text_col!r}")
    print(f"[INFO] Using label column: {label_col!r}")

    # Clean and map labels to the unified scheme
    df_bench = df_bench[[text_col, label_col]].dropna().copy()
    df_bench[text_col] = df_bench[text_col].astype(str).str.strip()
    df_bench = df_bench[df_bench[text_col].str.len() > 0].copy()
    df_bench["_label_unified"] = df_bench[label_col].apply(_map_benchmark_label)

    # Random subset
    if len(df_bench) > n_samples:
        df_bench = df_bench.sample(n=n_samples, random_state=42)

    texts = df_bench[text_col].astype(str).tolist()
    y_true = df_bench["_label_unified"].tolist()

    print(f"[INFO] Sampled {len(texts)} posts from local ElKulako StockTwits benchmark")
    print("[INFO] Label distribution:")
    for label, count in pd.Series(y_true).value_counts().items():
        print(f"         {label:10s}  {count:>8,}  ({count / len(y_true) * 100:.1f}%)")

    # Load model
    tokenizer, model, device, label_map, bullish_idx, bearish_idx = load_cryptobert()

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
    print(f"RESULTS: CryptoBERT on ElKulako StockTwits (n={len(texts)})")
    print(f"{'─' * 60}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"\n{classification_report(y_true, y_pred, digits=4)}")

    # Confusion matrix: rows = true labels, columns = predicted labels
    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels_order],
        columns=[f"pred_{label}" for label in labels_order],
    )

    print(f"\n{'─' * 60}")
    print("CONFUSION MATRIX: counts")
    print(f"{'─' * 60}")
    print(cm_df.to_string())

    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0.0)
    print(f"\n{'─' * 60}")
    print("CONFUSION MATRIX: row-normalized shares")
    print(f"{'─' * 60}")
    print(cm_norm.round(4).to_string())

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment_Score_Cryptobert — Score titles & selftexts with CryptoBERT"
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join("Data", "Processed", "Sentiment", "sentiment_combined.csv"),
        help="Path to the combined sentiment CSV"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("Data", "Transformed", "Sentiment_Scored_Cryptobert.csv"),
        help="Output path for scored CSV"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="CryptoBERT batch size (reduce to 16-32 if GPU OOM)"
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
        help="Run benchmark validation on ElKulako StockTwits dataset"
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

    # ── Step 2: Load CryptoBERT ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading CryptoBERT model")
    print("=" * 60)

    tokenizer, model, device, label_map, bullish_idx, bearish_idx = load_cryptobert()

    # ── Step 3: Score titles ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Scoring titles")
    print("=" * 60)

    t0 = time.time()
    df = score_titles(df, tokenizer, model, device,
                      label_map, bullish_idx, bearish_idx,
                      args.batch_size, ckpt_title,
                      args.checkpoint_every, args.restart)
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
    df = score_selftexts(df, tokenizer, model, device,
                         label_map, bullish_idx, bearish_idx,
                         args.batch_size, ckpt_seltext,
                         args.checkpoint_every, args.restart)
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
