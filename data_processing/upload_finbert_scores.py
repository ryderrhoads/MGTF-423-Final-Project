#!/usr/bin/env python3
from __future__ import annotations

import gc
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from data_processing.build_postgres_dataset import apply_basic_rules, create_indexes

if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise SystemExit("Set DATABASE_URL or POSTGRES_URL.")

CHUNK_ROWS = int(os.getenv("FAST_FINBERT_CHUNK_ROWS", os.getenv("FAST_UPLOAD_CHUNK_ROWS", "10000")))
DATA_DIR = ROOT / "data"
TEXT_SCORED_PATH = DATA_DIR / "text_sections_scored.csv"


def _copy_append(table_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False)
            buf.seek(0)
            cur.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT csv)", buf)
            conn.commit()


def dedupe_table(engine, table_name: str, key_cols: list[str]) -> None:
    sql = f"""
    DELETE FROM {table_name} a
    USING {table_name} b
    WHERE a.ctid < b.ctid
      AND {" AND ".join([f"a.{c} = b.{c}" for c in key_cols])};
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def main() -> None:
    engine = create_engine(DATABASE_URL)
    if not TEXT_SCORED_PATH.exists():
        raise FileNotFoundError("Missing text_sections_scored.csv")

    print(f"streaming finbert_section_scores from {TEXT_SCORED_PATH} with chunk_rows={CHUNK_ROWS}")
    chunks = pd.read_csv(TEXT_SCORED_PATH, chunksize=CHUNK_ROWS)

    keep_cols = [
        "filing_id", "ticker", "accession", "form", "filed_date", "section", "score", "num_units",
        "positive_count", "negative_count", "neutral_count", "score_std", "score_method",
    ]

    table_created = False
    total_rows = 0
    start = time.perf_counter()

    for i, chunk in enumerate(chunks, start=1):
        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
        chunk["filed_date"] = pd.to_datetime(chunk.get("filed_date"), errors="coerce", utc=True).dt.date
        chunk["filing_id"] = chunk["ticker"].astype(str) + "|" + chunk["accession"].astype(str)
        cols = [c for c in keep_cols if c in chunk.columns]
        chunk = apply_basic_rules(chunk[cols], "finbert_section_scores")

        if not table_created:
            chunk.head(0).to_sql("finbert_section_scores", engine, if_exists="replace", index=False)
            table_created = True

        _copy_append("finbert_section_scores", chunk)
        total_rows += len(chunk)
        elapsed = time.perf_counter() - start
        rate = total_rows / elapsed if elapsed > 0 else 0
        print(f"finbert_section_scores chunk {i} loaded_rows={total_rows:,} rate={rate:,.0f}/s")

        del chunk
        gc.collect()

    if table_created:
        dedupe_table(engine, "finbert_section_scores", ["filing_id", "section"])
    create_indexes(engine)
    print("done: finbert_section_scores")


if __name__ == "__main__":
    main()
