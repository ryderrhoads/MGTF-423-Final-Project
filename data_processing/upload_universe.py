#!/usr/bin/env python3
from __future__ import annotations

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

from data_processing.build_postgres_dataset import apply_basic_rules, create_indexes, load_universe

if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise SystemExit("Set DATABASE_URL or POSTGRES_URL.")


def fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"


def upload_df_copy(engine, table_name: str, df: pd.DataFrame, chunk_rows: int = 5000) -> None:
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)
    total = len(df)
    if total == 0:
        print(f"{table_name}: 0 rows")
        return
    chunks = (total + chunk_rows - 1) // chunk_rows
    start = time.perf_counter()
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for i in range(chunks):
                a, b = i * chunk_rows, min((i + 1) * chunk_rows, total)
                buf = io.StringIO()
                df.iloc[a:b].to_csv(buf, index=False, header=False)
                buf.seek(0)
                cur.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT csv)", buf)
                conn.commit()
                loaded = b
                elapsed = time.perf_counter() - start
                rate = loaded / elapsed if elapsed > 0 else 0
                eta = (total - loaded) / rate if rate > 0 else 0
                print(f"{table_name} chunk {i+1}/{chunks} rows={loaded:,}/{total:,} eta={fmt_eta(eta)}")


def main() -> None:
    engine = create_engine(DATABASE_URL)
    universe = apply_basic_rules(load_universe(), "universe")
    print(f"prepared universe: {len(universe):,} rows")
    upload_df_copy(engine, "universe", universe, chunk_rows=5000)
    create_indexes(engine)
    print("done: universe")


if __name__ == "__main__":
    main()
