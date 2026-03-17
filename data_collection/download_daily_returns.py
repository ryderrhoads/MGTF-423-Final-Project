"""
Download daily returns 10 years back for the stocks in stocks.txt and indexes in indexes.txt. Saves them to data/daily_returns
"""

import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

OUTPUT_DIR = "data/daily_returns"
STOCKS_FILE = "stocks.txt"
INDEXES_FILE = "indexes.txt"

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365 * 10)


def load_tickers(filepath: str) -> list[str]:
    with open(filepath) as f:
        tickers = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(tickers))  # deduplicate while preserving order


def _has_fresh_returns(out_path: str, expected_end: date) -> bool:
    """Return True if an existing file looks complete/fresh enough to skip."""
    if not os.path.exists(out_path):
        return False
    try:
        df = pd.read_csv(out_path)
        if df.empty or "Date" not in df.columns:
            return False
        last = pd.to_datetime(df["Date"], errors="coerce").max()
        if pd.isna(last):
            return False
        # market calendar noise-safe: accept if within 7 days of end date
        return (expected_end - last.date()).days <= 7 and len(df) > 200
    except Exception:
        return False


def _yf_symbol_candidates(ticker: str) -> list[str]:
    """Return Yahoo-compatible candidates for symbols like BRK.B / AKO.A."""
    cands = [ticker]
    if "." in ticker:
        cands.append(ticker.replace(".", "-"))
    return list(dict.fromkeys(cands))


def download_returns(tickers: list[str], start: date, end: date, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        out_path = os.path.join(output_dir, f"{ticker}.csv")
        if _has_fresh_returns(out_path, end):
            print(f"Skipping {ticker} (existing file is fresh)")
            continue

        print(f"Downloading {ticker}...")
        ok = False
        last_err = None

        for sym in _yf_symbol_candidates(ticker):
            try:
                df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
                if df.empty:
                    continue

                returns = df["Close"].pct_change().dropna()
                if returns.empty:
                    continue

                returns.name = ticker  # keep original ticker as output column
                returns.to_csv(out_path, header=True)
                print(f"  Saved to {out_path} ({len(returns)} rows) [yf:{sym}]")
                ok = True
                break
            except Exception as e:
                last_err = e
                continue

        if not ok:
            if last_err:
                print(f"  Error downloading {ticker}: {last_err}")
            else:
                print(f"  No data found for {ticker}, skipping.")


def main() -> None:
    stocks = load_tickers(STOCKS_FILE)
    indexes = load_tickers(INDEXES_FILE)
    all_tickers = list(dict.fromkeys(stocks + indexes))

    print(f"Downloading daily returns for {len(all_tickers)} tickers from {START_DATE} to {END_DATE}")
    download_returns(all_tickers, START_DATE, END_DATE, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
