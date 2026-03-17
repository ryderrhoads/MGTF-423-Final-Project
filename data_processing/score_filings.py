"""
Score filing text sections with FinBERT, following sentence-level process used in sentiment_scoring.py.

Method:
- Split section text into sentences
- Chunk long sentences (~400 words)
- Score each unit with FinBERT
- Section score = positive_count / (positive_count + negative_count)
  (if no pos/neg units, fallback 0.5)

Input:
- data/text_sections.csv

Output:
- data/text_sections_scored.csv (full rows + numeric score)
- data/text_section_sentence_details.csv (per-unit diagnostics)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

MODEL_NAME = "ProsusAI/finbert"
DATA_DIR = Path("data")
TEXT_PATH = DATA_DIR / "text_sections.csv"
OUT_PATH = DATA_DIR / "text_sections_scored.csv"
DETAILS_PATH = DATA_DIR / "text_section_sentence_details.csv"
BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "64"))
PROGRESS_EVERY = int(os.getenv("FINBERT_PROGRESS_EVERY", "500"))
MAX_WORDS_PER_SENTENCE = int(os.getenv("FINBERT_MAX_WORDS_PER_SENTENCE", "400"))
INCREMENTAL = os.getenv("FINBERT_INCREMENTAL", "1") == "1"
HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(DATA_DIR / "hf_cache")))


def choose_torch_device() -> str:
    """Prefer Apple Metal (MPS) on Mac, then CUDA, else CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def split_into_sentences(text: str) -> list[str]:
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [x.strip() for x in s if x and x.strip()]


def chunk_long_sentence(sentence: str, max_words: int = 400) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    out = []
    for i in range(0, len(words), max_words):
        ch = " ".join(words[i : i + max_words])
        if len(ch.split()) >= 5:
            out.append(ch)
    return out


def score_units(units: list[str], clf) -> list[dict]:
    if not units:
        return []
    res = clf(units, truncation=True, max_length=512, batch_size=BATCH_SIZE, top_k=None)
    out = []
    for txt, rr in zip(units, res):
        probs = {x["label"]: x["score"] for x in rr}
        label = max(probs, key=probs.get)
        score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        out.append(
            {
                "text": txt,
                "label": label,
                "score_raw": score,
                "prob_positive": probs.get("positive", 0.0),
                "prob_negative": probs.get("negative", 0.0),
                "prob_neutral": probs.get("neutral", 0.0),
            }
        )
    return out


def ratio_sentiment(text: str, clf) -> tuple[float, int, int, int, int, float, list[dict]]:
    if not text or not text.strip():
        return 0.5, 0, 0, 0, 0, 0.0, []

    sentences = split_into_sentences(text)
    units = []
    for s in sentences:
        units.extend(chunk_long_sentence(s, max_words=MAX_WORDS_PER_SENTENCE))

    scored = score_units(units, clf)
    if not scored:
        return 0.5, 0, 0, 0, 0, 0.0, []

    pos = sum(1 for x in scored if x["label"] == "positive")
    neg = sum(1 for x in scored if x["label"] == "negative")
    neu = sum(1 for x in scored if x["label"] == "neutral")
    n = len(scored)

    ratio = 0.5 if (pos + neg) == 0 else pos / (pos + neg)
    std = float(np.std([x["score_raw"] for x in scored])) if n > 1 else 0.0
    return float(ratio), n, pos, neg, neu, std, scored


def main() -> None:
    if not TEXT_PATH.exists():
        raise FileNotFoundError(f"Missing {TEXT_PATH}. Run extract_filings.py first.")

    df = pd.read_csv(TEXT_PATH)
    if df.empty:
        print("No text rows found.")
        return

    key_cols = ["ticker", "accession", "section"]
    existing_scored = pd.DataFrame()
    existing_keys: set[tuple] = set()
    if INCREMENTAL and OUT_PATH.exists():
        try:
            existing_scored = pd.read_csv(OUT_PATH)
            if all(c in existing_scored.columns for c in key_cols):
                existing_scored["score"] = pd.to_numeric(existing_scored.get("score"), errors="coerce")
                ok = existing_scored[existing_scored["score"].notna()].copy()
                existing_keys = set(map(tuple, ok[key_cols].astype(str).values.tolist()))
        except Exception:
            existing_scored = pd.DataFrame()
            existing_keys = set()

    if existing_keys:
        mask = df[key_cols].astype(str).apply(tuple, axis=1).isin(existing_keys)
        todo = df.loc[~mask].copy().reset_index(drop=True)
        print(f"Incremental mode: skipping {mask.sum()} already-scored rows, scoring {len(todo)} new rows")
    else:
        todo = df.copy().reset_index(drop=True)

    if todo.empty:
        print("No new rows to score. Incremental run complete.")
        return

    # Pin Hugging Face cache path (avoids redownloading + keeps cache in project data dir by default).
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

    # Metal optimization for Apple Silicon (falls back automatically if unsupported op).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = choose_torch_device()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    pipe_device = -1 if device == "cpu" else device
    print(f"Loading FinBERT: {MODEL_NAME} on device={device} (batch_size={BATCH_SIZE}, progress_every={PROGRESS_EVERY})")
    print(f"HF cache dir: {HF_CACHE_DIR}")
    clf = pipeline("sentiment-analysis", model=MODEL_NAME, device=pipe_device)

    scores = []
    details = []

    for i, r in todo.iterrows():
        txt = str(r.get("text") or "")
        ratio, n, pos, neg, neu, std, scored = ratio_sentiment(txt, clf)

        scores.append(
            {
                "score": ratio,
                "num_units": n,
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "score_std": std,
                "score_method": "pos_neg_ratio_sentence_level",
            }
        )

        for j, u in enumerate(scored):
            details.append(
                {
                    "row_index": i,
                    "ticker": r.get("ticker"),
                    "accession": r.get("accession"),
                    "section": r.get("section"),
                    "unit_index": j,
                    "label": u["label"],
                    "score_raw": u["score_raw"],
                    "prob_positive": u["prob_positive"],
                    "prob_negative": u["prob_negative"],
                    "prob_neutral": u["prob_neutral"],
                    "text_preview": " ".join(u["text"].split()[:40]),
                }
            )

        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"Scored {i+1}/{len(todo)} rows...")

    sc = pd.DataFrame(scores)
    new_out = pd.concat([todo.reset_index(drop=True).drop(columns=["score"], errors="ignore"), sc], axis=1)

    if INCREMENTAL and not existing_scored.empty:
        merged = pd.concat([existing_scored, new_out], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        out = merged
    else:
        out = new_out

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # Details: append in incremental mode when possible.
    new_details = pd.DataFrame(details)
    if INCREMENTAL and DETAILS_PATH.exists():
        try:
            old_details = pd.read_csv(DETAILS_PATH)
            all_details = pd.concat([old_details, new_details], ignore_index=True)
        except Exception:
            all_details = new_details
    else:
        all_details = new_details
    all_details.to_csv(DETAILS_PATH, index=False)

    print(f"Saved: {OUT_PATH} ({len(out)} rows)")
    print(f"Saved: {DETAILS_PATH} ({len(all_details)} rows, +{len(new_details)} new)")


if __name__ == "__main__":
    main()
