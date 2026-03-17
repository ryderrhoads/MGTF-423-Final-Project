"""
First-pass feature builder for filing-level modeling.

Outputs one row per filing (ticker + accession + filed_date), with:
- Market/returns features computed from daily returns *before filing date*
- Financial ratio/growth features from SEC financials (QoQ / YoY)
- Text-section aggregates (sentiment/length/tone composition)

Writes:
- data/features.csv (always)
- Postgres table `features` if DATABASE_URL / POSTGRES_URL is set
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import yfinance as yf

try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None

DATA_DIR = Path("data")
RETURNS_DIR = DATA_DIR / "daily_returns"
PRICE_VOL_DIR = DATA_DIR / "price_volume"
FINANCIALS_PATH = DATA_DIR / "financials.csv"
TEXT_PATH = DATA_DIR / "text_sections.csv"
TEXT_SCORED_PATH = DATA_DIR / "text_sections_scored.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
SECTOR_LABELS_PATH = DATA_DIR / "ticker_sector_labels.csv"
CONSUMER_DISCRETIONARY_PATH = Path("stock_universe/consumer_discretionary.csv")
CONSUMER_STAPLES_PATH = Path("stock_universe/consumer_staples.csv")
STOCKS_FILE = Path("stock_universe/stocks.txt")

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
FEATURES_TABLE = "features"
FEATURE_SOURCE = "postgres"
USE_POSTGRES_SOURCE = True

_ENGINE = None

def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        if create_engine is None:
            raise SystemExit("sqlalchemy is required for Postgres source mode.")
        if not DATABASE_URL:
            raise SystemExit("DATABASE_URL/POSTGRES_URL must be set for Postgres source mode.")
        _ENGINE = create_engine(DATABASE_URL)
    return _ENGINE

def _read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    eng = _get_engine()
    return pd.read_sql(query, eng, params=params)

pd.options.mode.copy_on_write = True

# Headline-model pruning: drop weak/redundant columns from final feature output.
DROP_FEATURES = {
    "growth_vs_momentum",
    "inventory_sales_gap", "operating_leverage_proxy", "debt_equity", "debt_equity_slog",
    "sales_inventory_change_qoq", "sales_inventory_change_yoy_slog", "inventory_buildup",
    "interaction_sentiment_momentum", "risk_sentiment_form_z", "gross_margin_change_qoq_slog",
    "gross_profit_growth_yoy_slog", "growth_to_valuation", "gross_margin_change_yoy_slog",
    "inventory_turnover_proxy", "sales_inventory_change_qoq_slog", "sales_inventory_change_yoy",
    "etf_momentum_20d", "gross_margin_change_qoq", "acceleration_revenue", "revenue_change_qoq_slog",
    "market_cap", "dollar_volume_20d", "shares_outstanding", "cogs", "revenue",
    "filing_quarter", "filing_length", "filing_length_form_z", "filing_length_change",
    "mdna_sentiment_form_z", "risk_sentiment", "mdna_sentiment", "avg_text_score_form_z",
}


def normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def load_or_build_consumer_labels(tickers: list[str]) -> pd.DataFrame:
    """Build ticker -> {consumer_label, etf_label} from local source files.

    Source of truth:
      - consumer_discretionary.csv
      - consumer_staples.csv
      - stocks.txt (universe filter)

    Labels:
      - consumer_discretionary -> VCR
      - consumer_staples -> VDC
    """
    if not CONSUMER_DISCRETIONARY_PATH.exists() or not CONSUMER_STAPLES_PATH.exists():
        raise FileNotFoundError(
            "Missing consumer sector source CSVs. Expected consumer_discretionary.csv and consumer_staples.csv"
        )

    disc = pd.read_csv(CONSUMER_DISCRETIONARY_PATH)
    stap = pd.read_csv(CONSUMER_STAPLES_PATH)

    if "Ticker" not in disc.columns or "Ticker" not in stap.columns:
        raise ValueError("Both consumer CSVs must contain a 'Ticker' column")

    disc_set = set(disc["Ticker"].dropna().map(normalize_ticker))
    stap_set = set(stap["Ticker"].dropna().map(normalize_ticker))

    if STOCKS_FILE.exists():
        stocks_universe = {
            normalize_ticker(x)
            for x in STOCKS_FILE.read_text(encoding="utf-8").splitlines()
            if str(x).strip()
        }
    else:
        stocks_universe = set()

    requested = {normalize_ticker(t) for t in tickers}
    universe = requested & stocks_universe if stocks_universe else requested

    rows = []
    for t in sorted(universe):
        if t in stap_set:
            rows.append({"ticker": t, "consumer_label": "consumer_staples", "etf_label": "VDC"})
        elif t in disc_set:
            rows.append({"ticker": t, "consumer_label": "consumer_discretionary", "etf_label": "VCR"})

    out = pd.DataFrame(rows, columns=["ticker", "consumer_label", "etf_label"]).drop_duplicates("ticker")
    SECTOR_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SECTOR_LABELS_PATH, index=False)
    return out


def safe_read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read CSV safely; return empty DataFrame for empty/corrupt files."""
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def load_returns(ticker: str) -> pd.DataFrame:
    if USE_POSTGRES_SOURCE:
        q = """
        SELECT date, ret
        FROM market_data_daily
        WHERE ticker = %(ticker)s
        ORDER BY date
        """
        try:
            out = _read_sql(q, {"ticker": normalize_ticker(ticker)})
        except Exception:
            return pd.DataFrame(columns=["date", "ret"])
        if out.empty:
            return pd.DataFrame(columns=["date", "ret"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
        out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        if hasattr(out["date"].dt, "tz") and out["date"].dt.tz is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        return out.reset_index(drop=True)

    p = RETURNS_DIR / f"{ticker}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "ret"])

    df = safe_read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["date", "ret"])

    val_col = ticker if ticker in df.columns else df.columns[-1]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], errors="coerce", utc=True),
        "ret": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna().sort_values("date")
    if hasattr(out["date"].dt, "tz") and out["date"].dt.tz is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    return out.reset_index(drop=True)


def load_price_volume(ticker: str) -> pd.DataFrame:
    """Load daily close/volume from Postgres source or local cache/yfinance."""
    if USE_POSTGRES_SOURCE:
        q = """
        SELECT date, close, volume
        FROM market_data_daily
        WHERE ticker = %(ticker)s
        ORDER BY date
        """
        try:
            out = _read_sql(q, {"ticker": normalize_ticker(ticker)})
        except Exception:
            return pd.DataFrame(columns=["date", "close", "volume"])
        if out.empty:
            return pd.DataFrame(columns=["date", "close", "volume"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out = out.dropna(subset=["date"])
        if hasattr(out["date"].dt, "tz") and out["date"].dt.tz is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        return out.sort_values("date").reset_index(drop=True)

    PRICE_VOL_DIR.mkdir(parents=True, exist_ok=True)
    p = PRICE_VOL_DIR / f"{ticker}.csv"

    if p.exists():
        df = safe_read_csv(p)
        if df.empty:
            return pd.DataFrame(columns=["date", "close", "volume"])
    else:
        if os.getenv("FEATURE_BUILDER_OFFLINE", "0") == "1":
            return pd.DataFrame(columns=["date", "close", "volume"])
        try:
            hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
            if hist.empty:
                return pd.DataFrame(columns=["date", "close", "volume"])
            df = hist.reset_index()[["Date", "Close", "Volume"]].rename(
                columns={"Date": "date", "Close": "close", "Volume": "volume"}
            )
            df.to_csv(p, index=False)
        except Exception:
            return pd.DataFrame(columns=["date", "close", "volume"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce", utc=True),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    ).dropna(subset=["date"])
    if hasattr(out["date"].dt, "tz") and out["date"].dt.tz is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    return out.sort_values("date").reset_index(drop=True)


def _missing_price_volume_tickers(tickers: list[str]) -> list[str]:
    PRICE_VOL_DIR.mkdir(parents=True, exist_ok=True)
    return [t for t in tickers if not (PRICE_VOL_DIR / f"{t}.csv").exists()]


def batch_download_price_volume(tickers: list[str]) -> None:
    """Download missing price/volume in one yfinance batch call, then cache per ticker."""
    if not tickers:
        return

    try:
        hist = yf.download(
            tickers=tickers,
            period="max",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return

    if hist is None or len(hist) == 0:
        return

    PRICE_VOL_DIR.mkdir(parents=True, exist_ok=True)

    # Single ticker returns flat columns; multi ticker returns MultiIndex columns.
    if not isinstance(hist.columns, pd.MultiIndex):
        if {"Close", "Volume"}.issubset(hist.columns):
            t = tickers[0]
            out = pd.DataFrame({
                "date": hist.index,
                "close": hist["Close"],
                "volume": hist["Volume"],
            }).reset_index(drop=True)
            out.to_csv(PRICE_VOL_DIR / f"{t}.csv", index=False)
        return

    for t in tickers:
        try:
            sub = hist[t]
        except Exception:
            continue
        if sub is None or sub.empty or not {"Close", "Volume"}.issubset(sub.columns):
            continue
        out = pd.DataFrame({
            "date": sub.index,
            "close": sub["Close"],
            "volume": sub["Volume"],
        }).reset_index(drop=True)
        out.to_csv(PRICE_VOL_DIR / f"{t}.csv", index=False)


def load_caches_parallel(tickers: list[str], max_workers: int = 8) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    def load_one(t: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
        return t, load_returns(t), load_price_volume(t)

    workers = max(1, min(max_workers, len(tickers) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(load_one, tickers))

    ret_cache = {t: ret for t, ret, _ in results}
    pxv_cache = {t: pxv for t, _, pxv in results}
    return ret_cache, pxv_cache


def _format_eta(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def to_date_value_arrays(df: pd.DataFrame, value_col: str = "ret") -> tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)
    d = pd.to_datetime(df["date"], errors="coerce")
    v = pd.to_numeric(df[value_col], errors="coerce")
    mask = d.notna() & v.notna()
    if not mask.any():
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)
    return d[mask].to_numpy(dtype="datetime64[ns]"), v[mask].to_numpy(dtype=float)


def tail_before(dates: np.ndarray, vals: np.ndarray, asof: pd.Timestamp, n: int) -> np.ndarray:
    if dates.size == 0 or vals.size == 0:
        return np.array([], dtype=float)
    i = np.searchsorted(dates, np.datetime64(asof), side="left")
    if i <= 0:
        return np.array([], dtype=float)
    start = max(0, i - n)
    return vals[start:i]


def comp_return(s: np.ndarray) -> float | None:
    if s.size == 0:
        return None
    return float(np.prod(1.0 + s) - 1.0)


def realized_vol_annualized(s: np.ndarray) -> float | None:
    if s.size < 2:
        return None
    return float(np.std(s, ddof=1) * np.sqrt(252))


def beta_252(stock_252: np.ndarray, spy_252: np.ndarray) -> float | None:
    n = min(stock_252.size, spy_252.size)
    if n < 30:
        return None
    x = stock_252[-n:]
    y = spy_252[-n:]
    var = np.var(y, ddof=1)
    if var == 0 or np.isnan(var):
        return None
    cov = np.cov(x, y, ddof=1)[0, 1]
    return float(cov / var)


def drawdown_60(s: np.ndarray) -> float | None:
    if s.size == 0:
        return None
    curve = np.cumprod(1.0 + s)
    roll_max = np.maximum.accumulate(curve)
    dd = (curve / roll_max) - 1.0
    return float(np.min(dd))


def market_stress_proxy_cached(
    spy_dates: np.ndarray,
    spy_vol20: np.ndarray,
    asof: pd.Timestamp,
    short: int = 20,
    long: int = 120,
) -> float | None:
    i = np.searchsorted(spy_dates, np.datetime64(asof), side="left")
    if i < long + short:
        return None
    current = spy_vol20[i - 1]
    window = spy_vol20[max(0, i - long):i]
    if np.isnan(current):
        return None
    longer = np.nanmean(window)
    if np.isnan(longer) or longer == 0:
        return None
    return float(current / longer)


def spy_vol_regime_cached(spy_dates: np.ndarray, spy_vals: np.ndarray, asof: pd.Timestamp) -> float | None:
    i = np.searchsorted(spy_dates, np.datetime64(asof), side="left")
    if i < 60:
        return None
    s5 = spy_vals[max(0, i - 5):i]
    s60 = spy_vals[max(0, i - 60):i]
    if s5.size < 2 or s60.size < 2:
        return None
    v5 = float(np.std(s5, ddof=1) * np.sqrt(252))
    v60 = float(np.std(s60, ddof=1) * np.sqrt(252))
    if not np.isfinite(v5) or not np.isfinite(v60) or v60 == 0:
        return None
    return float(v5 / v60)


def pct_change(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev in (None, 0) or pd.isna(curr) or pd.isna(prev):
        return None
    return float((curr - prev) / abs(prev))


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0) or pd.isna(a) or pd.isna(b):
        return None
    return float(a / b)


def ratio_with_floor(num: pd.Series, den: pd.Series, floor: float = 1.0) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    d_safe = d.where(d.abs() >= floor, np.nan)
    return n / d_safe


def winsorize_columns(
    df: pd.DataFrame,
    cols: list[str],
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() < 20:
            continue
        lo, hi = s.quantile(lower_q), s.quantile(upper_q)
        if pd.isna(lo) or pd.isna(hi) or lo >= hi:
            continue
        out[c] = s.clip(lo, hi)
    return out


def signed_log_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))


def prep_text_agg() -> pd.DataFrame:
    if USE_POSTGRES_SOURCE:
        q = """
        SELECT ticker, accession, filed_date, section, score, num_units,
               positive_count, negative_count, neutral_count, score_std
        FROM finbert_section_scores
        """
        tx = _read_sql(q)
        if tx.empty:
            return pd.DataFrame(columns=["ticker", "accession", "filed_date"])
    else:
        # Strict mode: require FinBERT-scored text sections.
        src = TEXT_SCORED_PATH
        if not src.exists():
            raise FileNotFoundError(
                f"Missing {TEXT_SCORED_PATH}. Run score_filings.py first to enforce FinBERT sentiment usage."
            )

        header = safe_read_csv(src, nrows=0)
        if len(header.columns) == 0:
            raise ValueError(f"{src} is empty. Re-run score_filings.py.")

        # Fast path default: avoid loading raw text unless explicitly requested.
        use_full_text = os.getenv("FEATURE_BUILDER_FULL_TEXT", "0") == "1"
        base_cols = [
            "ticker", "accession", "filed_date", "section", "score", "num_units",
            "positive_count", "negative_count", "neutral_count", "score_std",
        ]
        use_cols = [c for c in base_cols if c in header.columns]
        if use_full_text and "text" in header.columns:
            use_cols.append("text")

        tx = safe_read_csv(src, usecols=use_cols)
        if tx.empty:
            return pd.DataFrame(columns=["ticker", "accession", "filed_date"])

    if "score" not in tx.columns:
        tx["score"] = np.nan

    tx["filed_date"] = pd.to_datetime(tx["filed_date"], errors="coerce")
    tx["score_num"] = pd.to_numeric(tx["score"], errors="coerce")

    if "filing_length" not in tx.columns:
        if "num_units" in tx.columns:
            # sentence-count proxy when raw text is not loaded
            tx["filing_length"] = pd.to_numeric(tx["num_units"], errors="coerce").fillna(0)
        else:
            tx["filing_length"] = 0

    def sec_mean(section: str) -> pd.Series:
        g = tx.loc[tx["section"].eq(section)]
        if g.empty:
            return pd.Series(dtype=float)
        return g.groupby(["ticker", "accession"])["score_num"].mean()

    mdna = sec_mean("MD&A").rename("mdna_sentiment")
    risk = sec_mean("Risk Factors").rename("risk_sentiment")

    # numeric hygiene for section-level NLP fields
    for c in ["positive_count", "negative_count", "neutral_count", "score_std", "num_units"]:
        if c not in tx.columns:
            tx[c] = 0
        tx[c] = pd.to_numeric(tx[c], errors="coerce").fillna(0)

    grp = tx.groupby(["ticker", "accession", "filed_date"], as_index=False).agg(
        filing_length=("filing_length", "sum"),
        avg_text_score=("score_num", "mean"),
        score_std_mean=("score_std", "mean"),
        score_std_max=("score_std", "max"),
        positive_count_total=("positive_count", "sum"),
        negative_count_total=("negative_count", "sum"),
        neutral_count_total=("neutral_count", "sum"),
        section_count=("section", "nunique"),
        num_units_total=("num_units", "sum"),
    )

    grp = grp.merge(mdna.reset_index(), on=["ticker", "accession"], how="left")
    grp = grp.merge(risk.reset_index(), on=["ticker", "accession"], how="left")

    total_sent = (
        pd.to_numeric(grp["positive_count_total"], errors="coerce").fillna(0)
        + pd.to_numeric(grp["negative_count_total"], errors="coerce").fillna(0)
        + pd.to_numeric(grp["neutral_count_total"], errors="coerce").fillna(0)
    )
    den = total_sent.replace(0, np.nan)
    grp["positive_ratio"] = pd.to_numeric(grp["positive_count_total"], errors="coerce") / den
    grp["negative_ratio"] = pd.to_numeric(grp["negative_count_total"], errors="coerce") / den
    grp["neutral_ratio"] = pd.to_numeric(grp["neutral_count_total"], errors="coerce") / den
    grp["sentiment_polarity"] = grp["positive_ratio"] - grp["negative_ratio"]
    grp["units_per_section"] = pd.to_numeric(grp["num_units_total"], errors="coerce") / pd.to_numeric(grp["section_count"], errors="coerce").replace(0, np.nan)

    grp = grp.sort_values(["ticker", "filed_date"])
    grp["filing_length_change"] = grp.groupby("ticker")["filing_length"].pct_change()
    grp["sentiment_change_qoq"] = grp.groupby("ticker")["avg_text_score"].diff()
    return grp


def build_feature_rows() -> pd.DataFrame:
    print("[build_features] source=postgres (financials/market_data_daily/finbert_section_scores)")
    fin = _read_sql("SELECT * FROM financials")
    fin["filed_date"] = pd.to_datetime(fin["filed_date"], errors="coerce")
    fin["end_period"] = pd.to_datetime(fin.get("end_period"), errors="coerce")
    fin = fin.sort_values(["ticker", "filed_date", "accession"]).reset_index(drop=True)

    # financial derived columns
    fin["gross_margin"] = (fin["revenue"] - fin["cogs"]) / fin["revenue"].replace(0, np.nan)
    fin["sales_inventory"] = fin["revenue"] / fin["inventory"].replace(0, np.nan)
    fin["debt_equity"] = fin["debt"] / fin["equity"].replace(0, np.nan)

    for col in ["revenue", "gross_margin", "net_income", "sgna", "inventory", "sales_inventory"]:
        fin[f"{col}_chg_qoq"] = fin.groupby("ticker")[col].pct_change()
        fin[f"{col}_chg_yoy"] = fin.groupby("ticker")[col].pct_change(4)

    # seasonality
    fin["filing_month"] = fin["filed_date"].dt.month
    fin["filing_quarter"] = fin["filed_date"].dt.quarter

    # text aggregates
    text_agg = prep_text_agg()
    feat = fin.merge(text_agg, on=["ticker", "accession", "filed_date"], how="left")

    # If a section score is missing outside 10-K, carry the last available 10-K score
    # (casual fallback for sections that are effectively 10-K-only in practice).
    feat = feat.sort_values(["ticker", "filed_date", "accession"]).reset_index(drop=True)
    form_str = feat["form"].astype(str) if "form" in feat.columns else pd.Series("", index=feat.index)
    is_10k = form_str.str.upper().str.startswith("10-K")

    for col in ["mdna_sentiment", "risk_sentiment"]:
        if col not in feat.columns:
            continue
        last_10k_col = (
            feat[col]
            .where(is_10k)
            .groupby(feat["ticker"])
            .ffill()
        )
        feat[col] = pd.to_numeric(feat[col], errors="coerce").where(
            pd.to_numeric(feat[col], errors="coerce").notna(),
            last_10k_col,
        )

    # Casual per-form normalization (10-K vs 10-Q apples-to-apples)
    # z = (x - mean_form) / std_form, fallback to 0 if std ~ 0
    for col in [
        "mdna_sentiment", "risk_sentiment", "avg_text_score", "filing_length",
        "score_std_mean", "score_std_max", "positive_ratio", "negative_ratio", "neutral_ratio",
        "sentiment_polarity", "section_count", "num_units_total", "units_per_section",
    ]:
        if col in feat.columns:
            grp = feat.groupby("form")[col]
            mu = grp.transform("mean")
            sd = grp.transform("std")
            z = (feat[col] - mu) / sd.replace(0, np.nan)
            feat[f"{col}_form_z"] = z.fillna(0.0)

    # consumer labels + ETF mapping (VCR/VDC)
    tickers = sorted(set(feat["ticker"].dropna().astype(str)))
    labels = load_or_build_consumer_labels(tickers)
    feat = feat.merge(labels, on="ticker", how="left")

    # cache returns + price/volume (array form for fast as-of slicing)
    # 1) optionally pre-warm missing price/volume caches with one batched yfinance request
    if (not USE_POSTGRES_SOURCE) and os.getenv("FEATURE_BUILDER_BATCH_YF", "1") == "1":
        missing_pxv = _missing_price_volume_tickers(tickers)
        if missing_pxv:
            batch_download_price_volume(missing_pxv)

    # 2) parallel local loading of per-ticker CSVs
    workers = int(os.getenv("FEATURE_BUILDER_WORKERS", "8"))
    ret_cache, pxv_cache = load_caches_parallel(tickers, max_workers=workers)

    spy = load_returns("SPY")
    vcr = load_returns("VCR")
    vdc = load_returns("VDC")
    spy_pxv = load_price_volume("SPY")

    ret_arrays: Dict[str, tuple[np.ndarray, np.ndarray]] = {
        t: to_date_value_arrays(df, "ret") for t, df in ret_cache.items()
    }
    pxv_arrays: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for t, df in pxv_cache.items():
        if df is None or df.empty:
            pxv_arrays[t] = (
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
            continue
        d = pd.to_datetime(df["date"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        v = pd.to_numeric(df["volume"], errors="coerce")
        m = d.notna()
        pxv_arrays[t] = (
            d[m].to_numpy(dtype="datetime64[ns]"),
            c[m].to_numpy(dtype=float),
            v[m].to_numpy(dtype=float),
        )

    spy_dates, spy_vals = to_date_value_arrays(spy, "ret")
    vcr_dates, vcr_vals = to_date_value_arrays(vcr, "ret")
    vdc_dates, vdc_vals = to_date_value_arrays(vdc, "ret")

    spy_px_dates, spy_px_close = to_date_value_arrays(spy_pxv, "close")

    spy_vol20 = pd.Series(spy_vals).rolling(20).std(ddof=1).to_numpy() * np.sqrt(252)

    rows = []
    n_rows = len(feat)
    progress_every = int(os.getenv("FEATURE_BUILDER_PROGRESS_EVERY", "250"))
    show_eta = os.getenv("FEATURE_BUILDER_SHOW_ETA", "1") == "1"
    t0 = time.perf_counter()

    for i, r in enumerate(feat.itertuples(index=False), start=1):
        t = str(r.ticker)
        filed = r.filed_date
        if pd.isna(filed):
            rows.append({})
            if show_eta and (i % progress_every == 0 or i == n_rows):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_rows - i) / rate if rate > 0 else 0
                print(f"[build_features] {i}/{n_rows} ({(100*i/n_rows):.1f}%) elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}")
            continue

        stock_dates, stock_vals = ret_arrays.get(
            t, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
        )
        if stock_vals.size == 0:
            rows.append({})
            if show_eta and (i % progress_every == 0 or i == n_rows):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_rows - i) / rate if rate > 0 else 0
                print(f"[build_features] {i}/{n_rows} ({(100*i/n_rows):.1f}%) elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}")
            continue

        s5 = tail_before(stock_dates, stock_vals, filed, 5)
        s10 = tail_before(stock_dates, stock_vals, filed, 10)
        s20 = tail_before(stock_dates, stock_vals, filed, 20)
        s60 = tail_before(stock_dates, stock_vals, filed, 60)
        s252 = tail_before(stock_dates, stock_vals, filed, 252)
        m20 = tail_before(spy_dates, spy_vals, filed, 20)
        m60 = tail_before(spy_dates, spy_vals, filed, 60)
        m252 = tail_before(spy_dates, spy_vals, filed, 252)

        etf_label = str(getattr(r, "etf_label", "VCR") or "VCR")
        etf_dates, etf_vals = (vdc_dates, vdc_vals) if etf_label == "VDC" else (vcr_dates, vcr_vals)
        e20 = tail_before(etf_dates, etf_vals, filed, 20)
        e60 = tail_before(etf_dates, etf_vals, filed, 60)

        stock_m20 = comp_return(s20)
        stock_m60 = comp_return(s60)
        spy_m20 = comp_return(m20)
        spy_m60 = comp_return(m60)
        etf_m20 = comp_return(e20)
        etf_m60 = comp_return(e60)
        stock_vol20 = realized_vol_annualized(s20)
        etf_vol20 = realized_vol_annualized(e20)

        # close/volume-derived features
        last_close = None
        high_252d = None
        distance_to_52w_high = None
        dollar_vol_20d = None
        turnover_20d = None
        px_dates, px_close, px_vol = pxv_arrays.get(
            t,
            (
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype=float),
                np.array([], dtype=float),
            ),
        )
        if px_dates.size > 0:
            idx = np.searchsorted(px_dates, np.datetime64(filed), side="left")
            if idx > 0:
                last_close = float(px_close[idx - 1]) if np.isfinite(px_close[idx - 1]) else None

                # Distance to 52w high: (price - high_252d) / high_252d
                k = max(0, idx - 252)
                c252 = px_close[k:idx]
                if c252.size > 0:
                    h252 = np.nanmax(c252)
                    if np.isfinite(h252) and h252 > 0:
                        high_252d = float(h252)
                        if last_close is not None:
                            distance_to_52w_high = float((last_close - h252) / h252)

                j = max(0, idx - 20)
                c20 = px_close[j:idx]
                v20 = px_vol[j:idx]
                if c20.size > 0 and v20.size > 0:
                    dv = c20 * v20
                    dv_mean = np.nanmean(dv)
                    dollar_vol_20d = float(dv_mean) if np.isfinite(dv_mean) else None

                    shares = pd.to_numeric(getattr(r, "shares_outstanding", np.nan), errors="coerce")
                    # Guard against tiny/invalid share counts that explode turnover.
                    min_shares = float(os.getenv("FEATURE_MIN_SHARES_FOR_TURNOVER", "100000"))
                    daily_turnover_cap = float(os.getenv("FEATURE_DAILY_TURNOVER_CAP", "2.0"))
                    if pd.notna(shares) and shares >= min_shares:
                        daily_turnover = np.clip(v20 / shares, 0.0, daily_turnover_cap)
                        t_mean = np.nanmean(daily_turnover)
                        turnover_20d = float(t_mean) if np.isfinite(t_mean) else None

        shares = pd.to_numeric(getattr(r, "shares_outstanding", np.nan), errors="coerce")
        market_cap = None
        if pd.notna(last_close) and pd.notna(shares) and shares > 0:
            mcap = float(last_close * shares)
            mcap_min = float(os.getenv("FEATURE_MARKET_CAP_MIN", "1000000"))
            mcap_max = float(os.getenv("FEATURE_MARKET_CAP_MAX", "5000000000000"))
            if np.isfinite(mcap) and (mcap_min <= mcap <= mcap_max):
                market_cap = mcap

        # SPY drawdown regime: (spy_price - spy_high_252d) / spy_high_252d
        spy_drawdown_252d = None
        if spy_px_dates.size > 0:
            sidx = np.searchsorted(spy_px_dates, np.datetime64(filed), side="left")
            if sidx > 0:
                spy_last = spy_px_close[sidx - 1]
                sh = np.nanmax(spy_px_close[max(0, sidx - 252):sidx])
                if np.isfinite(spy_last) and np.isfinite(sh) and sh > 0:
                    spy_drawdown_252d = float((spy_last - sh) / sh)

        rows.append({
            "spy_momentum_20d": spy_m20,
            "spy_momentum_60d": spy_m60,
            "spy_vol_20d": realized_vol_annualized(m20),
            "etf_momentum_20d": etf_m20,
            "etf_momentum_60d": etf_m60,
            "etf_vol_20d": etf_vol20,
            "sector_regime_20d": None if etf_m20 is None or spy_m20 is None else float(etf_m20 - spy_m20),
            "spy_drawdown_252d": spy_drawdown_252d,
            "stock_momentum_20d": stock_m20,
            "stock_momentum_60d": stock_m60,
            "stock_vol_20d": stock_vol20,
            "stock_20d_excess_return": None if stock_m20 is None or spy_m20 is None else float(stock_m20 - spy_m20),
            "stock_beta_252d": beta_252(s252, m252),
            "stock_drawdown_60d": drawdown_60(s60),
            "pre_event_runup_5d": comp_return(s5),
            "pre_event_runup_10d": comp_return(s10),
            "pre_event_runup_60d": stock_m60,
            "relative_strength_20d": None if stock_m20 is None or spy_m20 is None else float(stock_m20 - spy_m20),
            "relative_strength_60d": None if stock_m60 is None or spy_m60 is None else float(stock_m60 - spy_m60),
            "etf_relative_strength_20d": None if stock_m20 is None or etf_m20 is None else float(stock_m20 - etf_m20),
            "etf_relative_strength_60d": None if stock_m60 is None or etf_m60 is None else float(stock_m60 - etf_m60),
            "vol_ratio": None if stock_vol20 in (None, 0) or etf_vol20 in (None, 0) else float(stock_vol20 / etf_vol20),
            "market_stress_proxy": market_stress_proxy_cached(spy_dates, spy_vol20, filed),
            "spy_vol_regime_5d_60d": spy_vol_regime_cached(spy_dates, spy_vals, filed),
            "high_252d": high_252d,
            "distance_to_52w_high": distance_to_52w_high,
            "market_cap": market_cap,
            "log_market_cap": None if market_cap in (None, 0) else float(np.log(max(market_cap, 1.0))),
            "earnings_yield": None if market_cap in (None, 0) else safe_div(pd.to_numeric(getattr(r, "net_income", np.nan), errors="coerce"), market_cap),
            "growth_vs_momentum": None if stock_m60 is None else float(pd.to_numeric(getattr(r, "revenue_change_yoy", np.nan), errors="coerce") - stock_m60),
            "dollar_volume_20d": dollar_vol_20d,
            "log_dollar_volume_20d": None if dollar_vol_20d in (None, 0) else float(np.log(max(dollar_vol_20d, 1.0))),
            "turnover_20d": turnover_20d,
        })
        if show_eta and (i % progress_every == 0 or i == n_rows):
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_rows - i) / rate if rate > 0 else 0
            print(f"[build_features] {i}/{n_rows} ({(100*i/n_rows):.1f}%) elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}")

    ret_feats = pd.DataFrame(rows)
    out = pd.concat([feat.reset_index(drop=True), ret_feats], axis=1)

    # requested naming normalization
    out = out.rename(columns={
        "revenue_chg_qoq": "revenue_change_qoq",
        "revenue_chg_yoy": "revenue_change_yoy",
        "gross_margin_chg_qoq": "gross_margin_change_qoq",
        "gross_margin_chg_yoy": "gross_margin_change_yoy",
        "net_income_chg_qoq": "net_income_growth_qoq",
        "net_income_chg_yoy": "net_income_growth_yoy",
        "sgna_chg_qoq": "sgna_growth_qoq",
        "sgna_chg_yoy": "sgna_growth_yoy",
        "inventory_chg_qoq": "inventory_growth_qoq",
        "inventory_chg_yoy": "inventory_growth_yoy",
        "sales_inventory_chg_qoq": "sales_inventory_change_qoq",
        "sales_inventory_chg_yoy": "sales_inventory_change_yoy",
    })

    # Additional financial/margin/surprise features requested
    rev = pd.to_numeric(out.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(out.get("cogs"), errors="coerce")
    sgna = pd.to_numeric(out.get("sgna"), errors="coerce")
    inv = pd.to_numeric(out.get("inventory"), errors="coerce")
    debt = pd.to_numeric(out.get("debt"), errors="coerce")
    eq = pd.to_numeric(out.get("equity"), errors="coerce")
    ni = pd.to_numeric(out.get("net_income"), errors="coerce")

    ratio_floor = float(os.getenv("FEATURE_RATIO_DENOM_FLOOR", "1.0"))

    out["net_margin"] = ratio_with_floor(ni, rev, floor=ratio_floor)
    out["cogs_margin"] = ratio_with_floor(cogs, rev, floor=ratio_floor)
    out["sgna_margin"] = ratio_with_floor(sgna, rev, floor=ratio_floor)
    out["inventory_to_equity"] = ratio_with_floor(inv, eq, floor=ratio_floor)
    out["debt_to_assets_proxy"] = ratio_with_floor(debt, debt + eq, floor=ratio_floor)
    out["equity_to_assets_proxy"] = ratio_with_floor(eq, debt + eq, floor=ratio_floor)

    out["gross_profit"] = rev - cogs
    out["gross_profit_margin"] = ratio_with_floor(out["gross_profit"], rev, floor=ratio_floor)
    out["debt_to_gross_profit"] = ratio_with_floor(debt, out["gross_profit"], floor=ratio_floor)

    out["acceleration_revenue"] = out["revenue_change_qoq"] - out["revenue_change_yoy"]
    out["acceleration_net_income"] = out["net_income_growth_qoq"] - out["net_income_growth_yoy"]
    out["margin_surprise_proxy"] = out["gross_margin_change_qoq"] - out["gross_margin_change_yoy"]

    out["inventory_intensity"] = ratio_with_floor(inv, rev, floor=ratio_floor)
    out["inventory_turnover_proxy"] = ratio_with_floor(cogs, inv, floor=ratio_floor)
    out["inventory_buildup"] = out["inventory_growth_yoy"] - out["revenue_change_yoy"]
    out["inventory_sales_gap"] = out["inventory_growth_yoy"] - out["revenue_change_yoy"]
    out["operating_leverage_proxy"] = out["revenue_change_yoy"] - out["sgna_growth_yoy"]

    # Valuation-style event features
    out["price_to_sales"] = ratio_with_floor(pd.to_numeric(out.get("market_cap"), errors="coerce"), rev, floor=ratio_floor)
    out["growth_to_valuation"] = ratio_with_floor(
        pd.to_numeric(out.get("revenue_change_yoy"), errors="coerce"),
        pd.to_numeric(out.get("price_to_sales"), errors="coerce"),
        floor=1e-6,
    )

    # Explicit interaction features
    out["interaction_profitability_momentum"] = (
        pd.to_numeric(out.get("net_margin"), errors="coerce")
        * pd.to_numeric(out.get("stock_momentum_60d"), errors="coerce")
    )
    out["interaction_growth_volatility"] = (
        pd.to_numeric(out.get("revenue_change_yoy"), errors="coerce")
        * pd.to_numeric(out.get("stock_vol_20d"), errors="coerce")
    )
    out["interaction_sentiment_momentum"] = (
        pd.to_numeric(out.get("avg_text_score_form_z"), errors="coerce")
        * pd.to_numeric(out.get("stock_momentum_20d"), errors="coerce")
    )

    out = out.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
    out["gross_profit_growth_yoy"] = out.groupby("ticker")["gross_profit"].pct_change(4)

    # Filing length surprise = filing_length / rolling mean of prior 4 filings.
    if "filing_length" in out.columns:
        prior_rolling_4 = (
            out.groupby("ticker")["filing_length"]
            .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        )
        out["filing_length_surprise"] = ratio_with_floor(
            pd.to_numeric(out["filing_length"], errors="coerce"),
            pd.to_numeric(prior_rolling_4, errors="coerce"),
            floor=1.0,
        )

    # Stabilize explosive ratio/growth features.
    unstable_cols = [
        "debt_equity", "gross_margin_change_yoy", "gross_margin_change_qoq",
        "net_income_growth_qoq", "net_income_growth_yoy", "gross_profit_growth_yoy",
        "revenue_change_qoq", "revenue_change_yoy", "sales_inventory_change_qoq",
        "sales_inventory_change_yoy", "acceleration_net_income", "acceleration_revenue",
    ]
    if os.getenv("FEATURE_SIGNED_LOG", "1") == "1":
        for c in unstable_cols:
            if c in out.columns:
                out[f"{c}_slog"] = signed_log_series(out[c])

    if os.getenv("FEATURE_HARD_CLIP", "1") == "1":
        clip_abs = float(os.getenv("FEATURE_HARD_CLIP_ABS", "10"))
        for c in unstable_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(-clip_abs, clip_abs)

    # Winsorize unstable ratios/accelerations to tame outliers before modeling.
    if os.getenv("FEATURE_WINSORIZE", "1") == "1":
        lo = float(os.getenv("FEATURE_WINSOR_LOWER", "0.005"))
        hi = float(os.getenv("FEATURE_WINSOR_UPPER", "0.995"))
        winsor_cols = [
            "net_margin", "cogs_margin", "sgna_margin", "inventory_to_equity",
            "debt_to_assets_proxy", "equity_to_assets_proxy", "acceleration_revenue",
            "acceleration_net_income", "margin_surprise_proxy", "inventory_sales_gap",
            "inventory_intensity", "inventory_turnover_proxy", "inventory_buildup",
            "operating_leverage_proxy", "gross_profit_growth_yoy", "debt_to_gross_profit",
            "vol_ratio", "turnover_20d", "market_cap", "dollar_volume_20d",
            "relative_strength_20d", "relative_strength_60d", "etf_relative_strength_20d",
            "etf_relative_strength_60d", "pre_event_runup_5d", "pre_event_runup_10d",
            "pre_event_runup_60d", "distance_to_52w_high", "earnings_yield",
            "growth_vs_momentum", "filing_length_surprise", "spy_vol_regime_5d_60d",
            "spy_drawdown_252d", "sector_regime_20d", "price_to_sales", "growth_to_valuation",
            "interaction_profitability_momentum", "interaction_growth_volatility",
            "interaction_sentiment_momentum",
        ]
        out = winsorize_columns(out, winsor_cols, lower_q=lo, upper_q=hi)

    # Final pruning for lean headline feature set.
    drop_now = [c for c in DROP_FEATURES if c in out.columns]
    if drop_now:
        out = out.drop(columns=drop_now)

    return out


def enforce_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preserve identifiers / labels as-is (fill string empties).
    id_like = {
        "ticker", "currency", "units", "form", "filed_date", "end_period", "start_period",
        "accession", "filing", "section"
    }

    # 1) Numeric columns: clean inf/-inf, then per-ticker forward/back-fill
    num_cols = [c for c in out.columns if c not in id_like and pd.api.types.is_numeric_dtype(out[c])]
    if num_cols:
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
        out = out.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
        # Forward-fill only (no backward fill) to avoid future-data leakage.
        out[num_cols] = out.groupby("ticker", group_keys=False)[num_cols].ffill()

        # 2) Cross-sectional median by filing_date
        if "filed_date" in out.columns:
            for c in num_cols:
                med_by_date = out.groupby("filed_date")[c].transform("median")
                out[c] = out[c].fillna(med_by_date)

        # 3) Hard zero fallback (no global median)
        for c in num_cols:
            out[c] = out[c].fillna(0.0)

    # 4) Date-ish fields: fill from filed_date when missing
    for c in ("start_period", "end_period"):
        if c in out.columns and "filed_date" in out.columns:
            out[c] = out[c].fillna(out["filed_date"])

    # 5) Non-numeric columns: safe string fill for missing values
    for c in out.columns:
        if c in num_cols:
            continue
        if out[c].dtype == object:
            out[c] = out[c].fillna("")
        else:
            out[c] = out[c].ffill()

    return out


def write_to_postgres(df: pd.DataFrame, table_name: str, database_url: str) -> None:
    if create_engine is None:
        raise RuntimeError("Install SQLAlchemy + psycopg2-binary to write Postgres")
    engine = create_engine(database_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi", chunksize=1000)


def main() -> None:
    df = build_feature_rows()
    df = enforce_no_nulls(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    nulls = int(df.isna().sum().sum())
    print(f"Saved {FEATURES_PATH} ({len(df)} rows, null_cells={nulls})")

    added_feature_list = [
        "net_margin", "cogs_margin", "sgna_margin", "inventory_to_equity",
        "debt_to_assets_proxy", "equity_to_assets_proxy", "acceleration_revenue",
        "acceleration_net_income", "margin_surprise_proxy", "inventory_sales_gap",
        "inventory_intensity", "inventory_turnover_proxy", "inventory_buildup",
        "operating_leverage_proxy", "gross_profit", "gross_profit_growth_yoy",
        "gross_profit_margin", "debt_to_gross_profit", "pre_event_runup_5d",
        "pre_event_runup_10d", "pre_event_runup_60d", "relative_strength_20d",
        "relative_strength_60d", "etf_relative_strength_20d", "etf_relative_strength_60d",
        "vol_ratio", "market_stress_proxy", "spy_vol_regime_5d_60d", "spy_drawdown_252d",
        "sector_regime_20d", "market_cap", "log_market_cap", "high_252d", "distance_to_52w_high",
        "earnings_yield", "growth_vs_momentum", "filing_length_surprise",
        "price_to_sales", "growth_to_valuation", "interaction_profitability_momentum",
        "interaction_growth_volatility", "interaction_sentiment_momentum",
        "dollar_volume_20d", "log_dollar_volume_20d", "turnover_20d",
    ]
    present = [c for c in added_feature_list if c in df.columns]
    if present:
        print("\nAdded feature sanity stats:")
        print(df[present].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(6).to_string())

    if DATABASE_URL:
        write_to_postgres(df, FEATURES_TABLE, DATABASE_URL)
        print(f"Wrote Postgres table {FEATURES_TABLE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
