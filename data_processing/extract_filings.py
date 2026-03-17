"""
Extract the filings to store the following

Financials
{currency:str,
units:str,
filed_date:date,
end_period:date,
start_period:date,
revenue:int,
cogs:int,
sgna:int,
inventory:int,
debt:int,
equity:int,
net_income:int,
shares_outstanding:int,
}

Text_Sections (MD&A, Risk Factors, Business Overview, Liquidity / Capital Resources)
{section:str,
text:str,
score:str,
filing:str,
}
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine = None
    text = None

# ── Constants ──────────────────────────────────────────────────────────────────

STOCKS_FILE = "stock_universe/stocks.txt"
DATA_DIR = Path("data")
FILINGS_DIR = DATA_DIR / "sec-edgar-filings"
TEN_YEARS_AGO = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

HEADERS = {"User-Agent": "Ryder Rhoads ryder.rhoads10@gmail.com"}

FORM_TYPES = {"10-K", "10-Q"}

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
FINANCIALS_TABLE = "financials"
TEXT_SECTIONS_TABLE = "text_sections"

# GAAP concept fallback lists — companies use different tags across years
CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cogs": [
        "CostOfGoodsSold",
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],
    "sgna": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "inventory": ["InventoryNet"],
    "debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtCurrent",
        "DebtCurrent",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "net_income": ["NetIncomeLoss"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_tickers(filepath: str = STOCKS_FILE) -> list[str]:
    with open(filepath) as f:
        tickers = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


# ── FINANCIALS: SEC XBRL API ───────────────────────────────────────────────────

def load_cik_map(tickers: list[str]) -> dict[str, str]:
    """Download SEC ticker→CIK map and filter to our universe."""
    print("Fetching CIK map from SEC...")
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    lookup = {v["ticker"].upper(): str(v["cik_str"]) for v in resp.json().values()}

    cik_map: dict[str, str] = {}
    for t in tickers:
        cik = lookup.get(t.upper())
        if cik:
            cik_map[t] = cik
        else:
            print(f"  [WARN] No CIK found for {t}")
    print(f"  {len(cik_map)}/{len(tickers)} tickers mapped\n")
    return cik_map


def fetch_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _collect_facts(us_gaap: dict, concepts: list[str], unit: str = "USD") -> list[dict]:
    """Collect facts across all candidate concept names (not just first non-empty).

    Important: companies often switch tags over time (e.g., SalesRevenueNet ->
    RevenueFromContractWithCustomerExcludingAssessedTax). We need all tags to
    avoid sparse/null series by accession.
    """
    out: list[dict] = []
    for concept in concepts:
        out.extend(us_gaap.get(concept, {}).get("units", {}).get(unit, []) or [])
    return out


def _forward_fill_shares_from_10k(df: pd.DataFrame, max_days: int = 370) -> pd.DataFrame:
    """Forward-fill shares_outstanding from latest 10-K into subsequent filings.

    This handles the common case where shares are only tagged in annual filings.
    We only carry values for up to ~1 year to avoid stale leakage.
    """
    if df.empty or "shares_outstanding" not in df.columns:
        return df

    out = df.sort_values(["ticker", "filed_date"]).copy()
    out["filed_date"] = pd.to_datetime(out["filed_date"], errors="coerce")

    filled_vals = []
    for _, g in out.groupby("ticker", sort=False):
        last_10k_shares = None
        last_10k_date = None
        for _, r in g.iterrows():
            shares = r.get("shares_outstanding")
            form = str(r.get("form") or "")
            fdate = r.get("filed_date")

            if pd.notna(shares):
                if form == "10-K" and pd.notna(fdate):
                    last_10k_shares = shares
                    last_10k_date = fdate
                filled_vals.append(shares)
                continue

            use_val = None
            if last_10k_shares is not None and pd.notna(fdate) and last_10k_date is not None:
                if (fdate - last_10k_date).days <= max_days:
                    use_val = last_10k_shares
            filled_vals.append(use_val)

    out["shares_outstanding"] = filled_vals
    return out


def extract_financials(tickers: list[str]) -> pd.DataFrame:
    cik_map = load_cik_map(tickers)
    rows: list[dict] = []

    for ticker, cik in cik_map.items():
        print(f"  {ticker} (CIK {cik})...")
        try:
            data = fetch_company_facts(cik)
        except Exception as e:
            print(f"    [ERROR] {e}")
            time.sleep(0.5)
            continue
        time.sleep(0.12)  # SEC fair-use: ≤ 10 req/s

        us_gaap = data.get("facts", {}).get("us-gaap", {})

        def _duration_days(f: dict) -> int | None:
            try:
                s = pd.to_datetime(f.get("start"), errors="coerce")
                e = pd.to_datetime(f.get("end"), errors="coerce")
                if pd.isna(s) or pd.isna(e):
                    return None
                return int((e - s).days)
            except Exception:
                return None

        def _duration_score(form: str, dd: int | None) -> float:
            if dd is None:
                return 1e9
            if form == "10-Q":
                # Prefer true quarter (~90d), avoid YTD 6m/9m values.
                target = 91
                penalty = 0.0 if 75 <= dd <= 120 else 10_000.0
                return abs(dd - target) + penalty
            if form == "10-K":
                target = 365
                penalty = 0.0 if 300 <= dd <= 430 else 10_000.0
                return abs(dd - target) + penalty
            return 1e9

        def index_by_accn(concepts: list[str], unit: str = "USD", require_duration: bool = False) -> dict[str, dict]:
            """Index facts by accession number, keeping only filings in scope.

            When require_duration=True, select facts with period length aligned to
            filing form (10-Q ~ quarter, 10-K ~ year).
            """
            facts = _collect_facts(us_gaap, concepts, unit)
            result: dict[str, dict] = {}
            for f in facts:
                form = f.get("form")
                if form not in FORM_TYPES:
                    continue
                if f.get("filed", "") < TEN_YEARS_AGO:
                    continue
                accn = f.get("accn", "")
                if not accn:
                    continue

                dd = _duration_days(f)
                if require_duration and dd is None:
                    continue

                existing = result.get(accn)
                if existing is None:
                    result[accn] = f
                    continue

                if require_duration:
                    cur_score = _duration_score(form, _duration_days(existing))
                    new_score = _duration_score(form, dd)
                    if new_score < cur_score:
                        result[accn] = f
                    elif new_score == cur_score and (f.get("filed") or "") >= (existing.get("filed") or ""):
                        result[accn] = f
                else:
                    # Point-in-time concepts: latest filed wins.
                    if (f.get("filed") or "") >= (existing.get("filed") or ""):
                        result[accn] = f
            return result

        rev_map  = index_by_accn(CONCEPTS["revenue"], require_duration=True)
        cogs_map = index_by_accn(CONCEPTS["cogs"], require_duration=True)
        sgna_map = index_by_accn(CONCEPTS["sgna"], require_duration=True)
        inv_map  = index_by_accn(CONCEPTS["inventory"], require_duration=False)
        debt_map = index_by_accn(CONCEPTS["debt"], require_duration=False)
        eq_map   = index_by_accn(CONCEPTS["equity"], require_duration=False)
        ni_map   = index_by_accn(CONCEPTS["net_income"], require_duration=True)
        sh_map   = index_by_accn(CONCEPTS["shares_outstanding"], unit="shares", require_duration=False)

        all_accns: set[str] = (
            rev_map.keys() | cogs_map.keys() | sgna_map.keys()
            | inv_map.keys() | debt_map.keys() | eq_map.keys()
            | ni_map.keys() | sh_map.keys()
        )

        ticker_rows = 0
        for accn in sorted(all_accns):
            # Use the richest available fact as the anchor for period metadata
            anchor = (
                rev_map.get(accn) or ni_map.get(accn) or cogs_map.get(accn)
                or next(
                    (m[accn] for m in (sgna_map, inv_map, debt_map, eq_map, sh_map) if accn in m),
                    None,
                )
            )
            if anchor is None:
                continue

            def val_m(fmap: dict) -> int | None:
                f = fmap.get(accn)
                return round(f["val"] / 1_000_000) if f and f.get("val") is not None else None

            rows.append({
                "ticker": ticker,
                "currency": "USD",
                "units": "M",
                "form": anchor.get("form"),
                "filed_date": anchor.get("filed"),
                "end_period": anchor.get("end"),
                "start_period": anchor.get("start"),
                "revenue": val_m(rev_map),
                "cogs": val_m(cogs_map),
                "sgna": val_m(sgna_map),
                "inventory": val_m(inv_map),
                "debt": val_m(debt_map),
                "equity": val_m(eq_map),
                "net_income": val_m(ni_map),
                "shares_outstanding": sh_map[accn]["val"] if accn in sh_map else None,
                "accession": accn,
            })
            ticker_rows += 1

        print(f"    -> {ticker_rows} filing periods")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
        df = _forward_fill_shares_from_10k(df)
    return df


# ── TEXT SECTIONS: local full-submission.txt ───────────────────────────────────

def get_filing_date(filepath: str | Path) -> datetime | None:
    """Parse FILED AS OF DATE from SGML header."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            if "FILED AS OF DATE:" in line:
                date_str = line.split(":")[-1].strip()
                return datetime.strptime(date_str, "%Y%m%d")
    return None


def extract_main_document(filepath: str | Path, form_type: str) -> str:
    """Pull the primary filing document body out of the SGML wrapper."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    doc_pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>([^\n<]+).*?<TEXT>(.*?)</TEXT>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in doc_pattern.finditer(content):
        if match.group(1).strip().upper().startswith(form_type.upper()):
            return match.group(2)
    return content


def clean_filing_text(raw_text: str) -> str:
    """Strip HTML tags, XBRL inline markup, and HTML entities."""
    text = raw_text
    text = re.sub(r"</?XBRL>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?ix:[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\?.*?\?>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text, flags=re.DOTALL)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_section(
    full_text: str,
    text_lower: str,
    patterns: list[str],
    max_chars: int = 120_000,
) -> str | None:
    """Return the longest candidate match for any of the given heading patterns."""
    item_header = re.compile(r"\bitem\s*\d+[a-z]?\b[.\s]", re.IGNORECASE)
    candidates: list[tuple[int, int]] = []

    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            start = m.start()
            end_m = item_header.search(text_lower, start + 300)
            end = end_m.start() if end_m else min(start + max_chars, len(full_text))
            candidates.append((start, end))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[1] - c[0])
    section_text = full_text[best[0]:best[1]].strip()
    return section_text if len(section_text.split()) > 50 else None


def extract_all_sections(full_text: str) -> dict[str, str | None]:
    """Extract the four required narrative sections from cleaned filing text."""
    tl = full_text.lower()
    return {
        "Business Overview": _find_section(full_text, tl, [
            r"item\s*1[.\s]+business\b",
            r"business\s+overview",
        ]),
        "Risk Factors": _find_section(full_text, tl, [
            r"item\s*1a[.\s]+risk\s*factors",
            r"risk\s+factors",
        ]),
        "MD&A": _find_section(full_text, tl, [
            r"item\s*7[.\s]+management.s\s+discussion\s+and\s+analysis",
            r"item\s*2[.\s]+management.s\s+discussion\s+and\s+analysis",
            r"management.s\s+discussion\s+and\s+analysis",
        ]),
        "Liquidity / Capital Resources": _find_section(full_text, tl, [
            r"liquidity\s+and\s+capital\s+resources",
            r"item\s*7a[.\s]+quantitative",
        ], max_chars=60_000),
    }


def extract_text_sections(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    for ticker in tickers:
        for form_type in ("10-K", "10-Q"):
            form_dir = FILINGS_DIR / ticker / form_type
            if not form_dir.exists():
                continue

            for accession_dir in sorted(p for p in form_dir.iterdir() if p.is_dir()):
                fpath = accession_dir / "full-submission.txt"
                if not fpath.exists():
                    continue

                filed_date = get_filing_date(fpath)
                if filed_date is None or filed_date.strftime("%Y-%m-%d") < TEN_YEARS_AGO:
                    continue

                try:
                    raw_doc = extract_main_document(fpath, form_type)
                    full_clean = clean_filing_text(raw_doc)
                    sections = extract_all_sections(full_clean)
                except Exception as e:
                    print(f"  [ERROR] {ticker} {form_type} {accession_dir.name}: {e}")
                    continue

                found = {k: v for k, v in sections.items() if v}
                summary = ", ".join(f"{k} {len(v.split())}w" for k, v in found.items())
                print(f"  {ticker} {form_type} {filed_date.date()} ({accession_dir.name}): "
                      f"{summary or 'no sections found'}")

                for section_name, text in sections.items():
                    if text:
                        rows.append({
                            "ticker": ticker,
                            "form": form_type,
                            "filed_date": filed_date.strftime("%Y-%m-%d"),
                            "accession": accession_dir.name,
                            "section": section_name,
                            "text": text,
                            "score": None,       # populated by score_filings.py
                            "filing": accession_dir.name,
                        })

    return pd.DataFrame(rows)


def write_to_postgres(df: pd.DataFrame, table_name: str, database_url: str, *, replace: bool = True) -> None:
    if create_engine is None:
        raise RuntimeError(
            "SQLAlchemy is required for Postgres writes. Install with: pip install sqlalchemy psycopg2-binary"
        )
    engine = create_engine(database_url)
    if_exists = "replace" if replace else "append"
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    use_postgres = bool(DATABASE_URL)
    if use_postgres:
        print("Postgres mode enabled (DATABASE_URL found).")
    else:
        print("DATABASE_URL not set -> using CSV fallback mode.")

    tickers = load_tickers(STOCKS_FILE)
    print(f"Loaded {len(tickers)} tickers\n")

    # --- Financials ---
    print("=" * 60)
    print("EXTRACTING FINANCIALS (SEC XBRL API)")
    print("=" * 60)
    df_fin = extract_financials(tickers)

    if use_postgres:
        write_to_postgres(df_fin, FINANCIALS_TABLE, DATABASE_URL, replace=True)
        print(f"\nWrote table {FINANCIALS_TABLE} ({len(df_fin)} rows)")
    fin_path = DATA_DIR / "financials.csv"
    df_fin.to_csv(fin_path, index=False)
    print(f"Saved CSV {fin_path} ({len(df_fin)} rows)\n")

    # --- Text Sections ---
    print("=" * 60)
    print("EXTRACTING TEXT SECTIONS (local filings)")
    print("=" * 60)
    df_text = extract_text_sections(tickers)

    if use_postgres:
        write_to_postgres(df_text, TEXT_SECTIONS_TABLE, DATABASE_URL, replace=True)
        print(f"\nWrote table {TEXT_SECTIONS_TABLE} ({len(df_text)} rows)")
    text_path = DATA_DIR / "text_sections.csv"
    df_text.to_csv(text_path, index=False)
    print(f"Saved CSV {text_path} ({len(df_text)} rows)")

    if not df_text.empty:
        print(df_text.groupby("section").size().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
