#!/usr/bin/env python3
"""
Build and upload core project tables to Postgres.

Tables created:
- market_data_daily
- filings
- text_sections
- finbert_section_scores
- financials
- features
- universe

Usage:
  export DATABASE_URL='postgresql://...'
  python build_postgres_dataset.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(".env")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RETURNS_DIR = DATA_DIR / "daily_returns"
PRICE_VOL_DIR = DATA_DIR / "price_volume"
STOCK_UNIVERSE_DIR = ROOT / "stock_universe"
STOCKS_FILE = STOCK_UNIVERSE_DIR / "stocks.txt"
CONSUMER_DISCRETIONARY_PATH = STOCK_UNIVERSE_DIR / "consumer_discretionary.csv"
CONSUMER_STAPLES_PATH = STOCK_UNIVERSE_DIR / "consumer_staples.csv"

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise SystemExit("Set DATABASE_URL or POSTGRES_URL first.")


def _norm_ticker(s: str) -> str:
    return str(s).strip().upper()


def apply_basic_rules(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Apply basic dataset hygiene rules before upload."""
    out = df.copy()

    # Common key/date cleanup
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].map(_norm_ticker)
        out = out[out["ticker"].str.len() > 0]

    for dc in ["date", "filed_date", "start_period", "end_period"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce", utc=True).dt.date

    # Table-specific rules
    if table == "market_data_daily":
        out = out.dropna(subset=["ticker", "date"])
        if "volume" in out.columns:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0)
        if "close" in out.columns:
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            out.loc[out["close"] <= 0, "close"] = pd.NA
        if "ret" in out.columns:
            out["ret"] = pd.to_numeric(out["ret"], errors="coerce").clip(-1.0, 5.0)
        out = out.drop_duplicates(["ticker", "date"], keep="last")

    elif table in {"filings", "financials", "features"}:
        key_cols = [c for c in ["filing_id", "ticker", "accession"] if c in out.columns]
        if key_cols:
            out = out.dropna(subset=key_cols)
        if "shares_outstanding" in out.columns:
            out["shares_outstanding"] = pd.to_numeric(out["shares_outstanding"], errors="coerce").clip(lower=0)
        if "market_cap" in out.columns:
            out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
            out.loc[(out["market_cap"] < 1e6) | (out["market_cap"] > 5e12), "market_cap"] = pd.NA
        if "filing_id" in out.columns:
            out = out.drop_duplicates(["filing_id"], keep="last")

    elif table in {"text_sections", "finbert_section_scores"}:
        need = [c for c in ["filing_id", "section"] if c in out.columns]
        if need:
            out = out.dropna(subset=need)
        if "section" in out.columns:
            out["section"] = out["section"].astype(str).str.strip()
            out = out[out["section"].str.len() > 0]
        if "score" in out.columns:
            out["score"] = pd.to_numeric(out["score"], errors="coerce").clip(0.0, 1.0)
        out = out.drop_duplicates([c for c in ["filing_id", "section"] if c in out.columns], keep="last")

    elif table == "universe":
        out = out.dropna(subset=["ticker"])
        out["ticker"] = out["ticker"].map(_norm_ticker)
        out = out[out["ticker"].str.len() > 0]
        if "in_universe" in out.columns:
            out["in_universe"] = out["in_universe"].fillna(False).astype(bool)
        out = out.drop_duplicates(["ticker"], keep="last")

    return out.reset_index(drop=True)


def load_market_data_daily() -> pd.DataFrame:
    rows = []

    # returns
    ret_map: dict[str, pd.DataFrame] = {}
    for p in RETURNS_DIR.glob("*.csv"):
        t = _norm_ticker(p.stem)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        val_col = t if t in df.columns else df.columns[-1]
        out = pd.DataFrame(
            {
                "ticker": t,
                "date": pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.date,
                "ret": pd.to_numeric(df[val_col], errors="coerce"),
            }
        ).dropna(subset=["date"])
        ret_map[t] = out[["ticker", "date", "ret"]]

    # price/volume
    pxv_map: dict[str, pd.DataFrame] = {}
    for p in PRICE_VOL_DIR.glob("*.csv"):
        t = _norm_ticker(p.stem)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "date" not in df.columns:
            continue
        out = pd.DataFrame(
            {
                "ticker": t,
                "date": pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date,
                "close": pd.to_numeric(df.get("close"), errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            }
        ).dropna(subset=["date"])
        pxv_map[t] = out[["ticker", "date", "close", "volume"]]

    tickers = sorted(set(ret_map) | set(pxv_map))
    for t in tickers:
        r = ret_map.get(t)
        p = pxv_map.get(t)
        if r is not None and p is not None:
            m = p.merge(r, on=["ticker", "date"], how="outer")
        elif p is not None:
            m = p.copy()
            m["ret"] = pd.NA
        else:
            m = r.copy()
            m["close"] = pd.NA
            m["volume"] = pd.NA
        rows.append(m)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume", "ret"])

    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_financials() -> pd.DataFrame:
    p = DATA_DIR / "financials.csv"
    df = pd.read_csv(p)
    df["ticker"] = df["ticker"].map(_norm_ticker)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True).dt.date
    if "end_period" in df.columns:
        df["end_period"] = pd.to_datetime(df["end_period"], errors="coerce", utc=True).dt.date
    if "start_period" in df.columns:
        df["start_period"] = pd.to_datetime(df["start_period"], errors="coerce", utc=True).dt.date
    df["filing_id"] = df["ticker"].astype(str) + "|" + df["accession"].astype(str)
    return df


def load_filings(financials: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["filing_id", "ticker", "accession", "form", "filed_date", "end_period", "start_period", "currency", "units"] if c in financials.columns]
    return financials[cols].drop_duplicates("filing_id").reset_index(drop=True)


def load_text_sections() -> pd.DataFrame:
    p = DATA_DIR / "text_sections.csv"
    if p.exists():
        df = pd.read_csv(p)
    else:
        # fallback to scored file if raw not present
        df = pd.read_csv(DATA_DIR / "text_sections_scored.csv")

    df["ticker"] = df["ticker"].map(_norm_ticker)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True).dt.date
    df["filing_id"] = df["ticker"].astype(str) + "|" + df["accession"].astype(str)

    keep = [c for c in ["filing_id", "ticker", "accession", "form", "filed_date", "section", "text", "num_units", "filing"] if c in df.columns]
    return df[keep].drop_duplicates(["filing_id", "section"]).reset_index(drop=True)


def load_finbert_section_scores() -> pd.DataFrame:
    p = DATA_DIR / "text_sections_scored.csv"
    df = pd.read_csv(p)
    df["ticker"] = df["ticker"].map(_norm_ticker)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True).dt.date
    df["filing_id"] = df["ticker"].astype(str) + "|" + df["accession"].astype(str)

    keep = [
        c
        for c in [
            "filing_id",
            "ticker",
            "accession",
            "form",
            "filed_date",
            "section",
            "score",
            "num_units",
            "positive_count",
            "negative_count",
            "neutral_count",
            "score_std",
            "score_method",
        ]
        if c in df.columns
    ]
    return df[keep].drop_duplicates(["filing_id", "section"]).reset_index(drop=True)


def load_features() -> pd.DataFrame:
    p = DATA_DIR / "features.csv"
    df = pd.read_csv(p)
    df["ticker"] = df["ticker"].map(_norm_ticker)
    if "filed_date" in df.columns:
        df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True).dt.date
    if "accession" in df.columns:
        df["filing_id"] = df["ticker"].astype(str) + "|" + df["accession"].astype(str)
    return df


def load_universe() -> pd.DataFrame:
    rows: list[dict] = []

    stocks: set[str] = set()
    if STOCKS_FILE.exists():
        stocks = {
            _norm_ticker(x)
            for x in STOCKS_FILE.read_text(encoding="utf-8").splitlines()
            if str(x).strip()
        }

    consumer_map: dict[str, tuple[str, str]] = {}
    if CONSUMER_DISCRETIONARY_PATH.exists():
        disc = pd.read_csv(CONSUMER_DISCRETIONARY_PATH)
        if "Ticker" in disc.columns:
            for t in disc["Ticker"].dropna().map(_norm_ticker):
                consumer_map[t] = ("consumer_discretionary", "VCR")

    if CONSUMER_STAPLES_PATH.exists():
        stap = pd.read_csv(CONSUMER_STAPLES_PATH)
        if "Ticker" in stap.columns:
            for t in stap["Ticker"].dropna().map(_norm_ticker):
                consumer_map[t] = ("consumer_staples", "VDC")

    all_tickers = sorted(set(stocks) | set(consumer_map.keys()))
    for t in all_tickers:
        label = consumer_map.get(t)
        rows.append(
            {
                "ticker": t,
                "in_universe": t in stocks if stocks else True,
                "consumer_label": label[0] if label else pd.NA,
                "etf_label": label[1] if label else pd.NA,
                "source": "stock_universe",
            }
        )

    return pd.DataFrame(rows, columns=["ticker", "in_universe", "consumer_label", "etf_label", "source"])


def create_indexes(engine) -> None:
    stmts = [
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_market_data_daily_ticker_date ON market_data_daily (ticker, date)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_filings_filing_id ON filings (filing_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_text_sections_filing_section ON text_sections (filing_id, section)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_finbert_scores_filing_section ON finbert_section_scores (filing_id, section)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_financials_filing_id ON financials (filing_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_features_filing_id ON features (filing_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_universe_ticker ON universe (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_market_data_daily_date ON market_data_daily (date)",
        "CREATE INDEX IF NOT EXISTS idx_features_filed_date ON features (filed_date)",
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))


def _fmt_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def main() -> None:
    t0 = time.perf_counter()
    engine = create_engine(DATABASE_URL)

    steps_total = 8
    step = 0

    step += 1
    market_data_daily = apply_basic_rules(load_market_data_daily(), "market_data_daily")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] market_data_daily prepared ({len(market_data_daily):,} rows) | elapsed={_fmt_eta(elapsed)}")

    step += 1
    financials = apply_basic_rules(load_financials(), "financials")
    filings = apply_basic_rules(load_filings(financials), "filings")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] financials+filings prepared ({len(financials):,}/{len(filings):,}) | elapsed={_fmt_eta(elapsed)}")

    step += 1
    text_sections = apply_basic_rules(load_text_sections(), "text_sections")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] text_sections prepared ({len(text_sections):,} rows) | elapsed={_fmt_eta(elapsed)}")

    step += 1
    finbert_scores = apply_basic_rules(load_finbert_section_scores(), "finbert_section_scores")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] finbert_section_scores prepared ({len(finbert_scores):,} rows) | elapsed={_fmt_eta(elapsed)}")

    step += 1
    features = apply_basic_rules(load_features(), "features")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] features prepared ({len(features):,} rows) | elapsed={_fmt_eta(elapsed)}")

    step += 1
    universe = apply_basic_rules(load_universe(), "universe")
    elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] universe prepared ({len(universe):,} rows) | elapsed={_fmt_eta(elapsed)}")

    uploads = [
        ("market_data_daily", market_data_daily, 5000),
        ("filings", filings, 5000),
        ("text_sections", text_sections, 2000),
        ("finbert_section_scores", finbert_scores, 2000),
        ("financials", financials, 5000),
        ("features", features, 2000),
        ("universe", universe, 5000),
    ]

    step += 1
    print(f"[{step}/{steps_total}] uploading tables...")
    t_up = time.perf_counter()
    up_total = len(uploads)
    for i, (name, frame, chunk) in enumerate(uploads, start=1):
        frame.to_sql(name, engine, if_exists="replace", index=False, method="multi", chunksize=chunk)
        up_elapsed = time.perf_counter() - t_up
        rate = i / up_elapsed if up_elapsed > 0 else 0
        eta = (up_total - i) / rate if rate > 0 else 0
        print(f"  upload {i}/{up_total}: {name} ({len(frame):,} rows) | eta={_fmt_eta(eta)}")

    step += 1
    create_indexes(engine)
    total_elapsed = time.perf_counter() - t0
    print(f"[{step}/{steps_total}] indexes created | total_elapsed={_fmt_eta(total_elapsed)}")
    print("Upload complete.")


if __name__ == "__main__":
    main()
