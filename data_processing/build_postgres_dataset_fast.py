#!/usr/bin/env python3
"""
Fast Postgres uploader for core project dataset.

Core tables uploaded here:
- market_data_daily
- financials
- filings
- features

Split uploads (run separately):
- universe  -> data_processing/upload_universe.py
- text_sections -> data_processing/upload_text_sections.py
- finbert_section_scores -> data_processing/upload_finbert_scores.py

Usage:
  python data_processing/build_postgres_dataset_fast.py

Env:
  DATABASE_URL / POSTGRES_URL   required
  FAST_UPLOAD_CHUNK_ROWS        default 20000
  FAST_SKIP_MARKET_DATA         1 to skip market_data_daily
"""

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

from data_processing.build_postgres_dataset import (
    apply_basic_rules,
    create_indexes,
    load_features,
    load_financials,
    load_filings,
    load_market_data_daily,
)

if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise SystemExit("Set DATABASE_URL or POSTGRES_URL (or put one in .env).")

CHUNK_ROWS = int(os.getenv("FAST_UPLOAD_CHUNK_ROWS", "20000"))
SKIP_MARKET = os.getenv("FAST_SKIP_MARKET_DATA", "0") == "1"


def fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def upload_df_copy(engine, table_name: str, df: pd.DataFrame, chunk_rows: int = 20000) -> None:
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)

    total = len(df)
    if total == 0:
        print(f"  {table_name}: 0 rows (skipped)")
        return

    start = time.perf_counter()
    chunks = (total + chunk_rows - 1) // chunk_rows

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for i in range(chunks):
                a = i * chunk_rows
                b = min((i + 1) * chunk_rows, total)
                chunk = df.iloc[a:b]

                buf = io.StringIO()
                chunk.to_csv(buf, index=False, header=False)
                buf.seek(0)
                cur.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT csv)", buf)
                conn.commit()

                loaded = b
                elapsed = time.perf_counter() - start
                rate = loaded / elapsed if elapsed > 0 else 0
                eta = (total - loaded) / rate if rate > 0 else 0
                print(
                    f"  {table_name} chunk {i+1}/{chunks} rows={loaded:,}/{total:,} "
                    f"({100*loaded/total:.1f}%) eta={fmt_eta(eta)}"
                )


def _prepare_table(name: str, loader) -> pd.DataFrame:
    t = time.perf_counter()
    df = apply_basic_rules(loader(), name)
    print(f"  prepared {name}: {len(df):,} rows in {fmt_eta(time.perf_counter() - t)}")
    return df


def main() -> None:
    t0 = time.perf_counter()
    engine = create_engine(DATABASE_URL)

    print("Preparing + uploading core tables (low-memory mode)...")

    steps: list[tuple[str, callable, int, bool]] = [
        ("market_data_daily", load_market_data_daily, CHUNK_ROWS, not SKIP_MARKET),
        ("financials", load_financials, max(5000, CHUNK_ROWS // 4), True),
        ("features", load_features, max(5000, CHUNK_ROWS // 4), True),
    ]

    total_steps = len([s for s in steps if s[3]]) + 1  # + filings
    idx = 0
    financials_df: pd.DataFrame | None = None

    for table_name, loader_fn, chunk_rows, enabled in steps:
        if not enabled:
            continue

        idx += 1
        print(f"[{idx}/{total_steps}] {table_name}")
        df = _prepare_table(table_name, loader_fn)
        upload_df_copy(engine, table_name, df, chunk_rows=chunk_rows)

        if table_name == "financials":
            financials_df = df[
                [
                    c
                    for c in df.columns
                    if c
                    in {
                        "filing_id",
                        "ticker",
                        "accession",
                        "form",
                        "filed_date",
                        "end_period",
                        "start_period",
                        "currency",
                        "units",
                    }
                ]
            ].copy()

        del df
        gc.collect()

        if table_name == "financials" and financials_df is not None:
            idx += 1
            print(f"[{idx}/{total_steps}] filings")
            filings = apply_basic_rules(load_filings(financials_df), "filings")
            print(f"  prepared filings: {len(filings):,} rows")
            upload_df_copy(engine, "filings", filings, chunk_rows=max(5000, CHUNK_ROWS // 4))
            del filings
            del financials_df
            financials_df = None
            gc.collect()

    print("Creating indexes...")
    create_indexes(engine)

    print(f"Done in {fmt_eta(time.perf_counter() - t0)}")
    print("Run separate scripts for: universe, text_sections, finbert_section_scores")


if __name__ == "__main__":
    main()
