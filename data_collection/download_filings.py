"""
Download 10 years of 10-K and 10-Q filings for all stocks in stocks.txt.
Saves them to data/sec-edgar-filings.

Speedups:
- Skip symbols/forms already "complete" by count thresholds
- Cache known no-data ticker/form combos in data/download_meta.json
- Parallelized workers with centralized rate limit (10 starts/sec by default)
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from sec_edgar_downloader import Downloader

STOCKS_FILE = "stocks.txt"
DATA_DIR = Path("data")
FORM_TYPES = ["10-K", "10-Q"]
TEN_YEARS_AGO = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
META_PATH = DATA_DIR / "download_meta.json"

# Practical completeness thresholds for ~10 years.
MIN_COMPLETE = {
    "10-K": 8,
    "10-Q": 24,
}

MAX_WORKERS = int(os.getenv("EDGAR_MAX_WORKERS", "4"))
RATE_LIMIT_PER_SEC = float(os.getenv("EDGAR_REQS_PER_SEC", "10"))


class RateLimiter:
    """Thread-safe global rate limiter using a sliding window."""

    def __init__(self, max_calls_per_sec: float) -> None:
        self.max_calls = max(1, int(max_calls_per_sec))
        self.window = 1.0
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.calls = [t for t in self.calls if now - t < self.window]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_for = self.window - (now - self.calls[0])
            time.sleep(max(0.005, sleep_for))


def load_tickers(filepath: str = STOCKS_FILE) -> list[str]:
    with open(filepath) as f:
        tickers = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(tickers))


def _load_meta(path: Path = META_PATH) -> dict:
    if not path.exists():
        return {"no_data": {}, "errors": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"no_data": {}, "errors": {}}


def _save_meta(meta: dict, path: Path = META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _existing_count(download_dir: Path, ticker: str, form_type: str) -> int:
    out_path = download_dir / "sec-edgar-filings" / ticker / form_type
    if not out_path.exists():
        return 0
    try:
        return sum(1 for p in out_path.iterdir() if p.is_dir())
    except Exception:
        return 0


def _is_complete(download_dir: Path, ticker: str, form_type: str) -> tuple[bool, int]:
    cnt = _existing_count(download_dir, ticker, form_type)
    need = MIN_COMPLETE.get(form_type, 1)
    return cnt >= need, cnt


def _download_one(
    ticker: str,
    form_type: str,
    after: str,
    download_dir: Path,
    limiter: RateLimiter,
) -> tuple[str, str, int, int, str | None]:
    """Return: ticker, form_type, before_count, after_count, error"""
    before = _existing_count(download_dir, ticker, form_type)

    # Centralized global throttle across all workers.
    limiter.acquire()

    try:
        dl = Downloader("Ryder Rhoads", "ryder.rhoads10@gmail.com", str(download_dir))
        dl.get(form_type, ticker, after=after)
        after_cnt = _existing_count(download_dir, ticker, form_type)
        return ticker, form_type, before, after_cnt, None
    except Exception as e:
        return ticker, form_type, before, before, str(e)


def download_filings(
    tickers: list[str],
    form_types: list[str] = FORM_TYPES,
    after: str = TEN_YEARS_AGO,
    download_dir: Path = DATA_DIR,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta()
    no_data = meta.setdefault("no_data", {})
    errors = meta.setdefault("errors", {})

    # Build work queue after cheap skips.
    tasks: list[tuple[str, str]] = []
    total = len(tickers) * len(form_types)
    inspected = 0
    for ticker in tickers:
        for form_type in form_types:
            inspected += 1
            key = f"{ticker}:{form_type}"
            complete, count = _is_complete(download_dir, ticker, form_type)
            if no_data.get(key) == "true":
                print(f"[{inspected}/{total}] Skipping {form_type} for {ticker} (cached no-data)")
                continue
            if complete:
                print(f"[{inspected}/{total}] Skipping {form_type} for {ticker} ({count} existing filing(s), complete)")
                continue
            tasks.append((ticker, form_type))

    print(f"\nQueued {len(tasks)} download tasks (workers={MAX_WORKERS}, global rate={RATE_LIMIT_PER_SEC}/sec)\n")
    if not tasks:
        _save_meta(meta)
        return

    limiter = RateLimiter(RATE_LIMIT_PER_SEC)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(_download_one, t, f, after, download_dir, limiter): (t, f)
            for t, f in tasks
        }

        done = 0
        for fut in as_completed(futs):
            done += 1
            ticker, form_type, before, after_cnt, err = fut.result()
            key = f"{ticker}:{form_type}"

            if err is not None:
                errors[key] = err
                print(f"[{done}/{len(tasks)}] Error downloading {form_type} for {ticker}: {err}")
            else:
                added = max(0, after_cnt - before)
                if after_cnt == 0:
                    no_data[key] = "true"
                elif added > 0:
                    no_data.pop(key, None)
                errors.pop(key, None)
                print(
                    f"[{done}/{len(tasks)}] {ticker} {form_type}: "
                    f"now {after_cnt} filing(s) (added {added})"
                )

            if done % 25 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                rem = len(tasks) - done
                eta_sec = int(rem / rate) if rate > 0 else -1
                if eta_sec >= 0:
                    print(f"  Progress: {done}/{len(tasks)} | ETA ~{eta_sec//60}m{eta_sec%60:02d}s")

    _save_meta(meta)


if __name__ == "__main__":
    tickers = load_tickers(STOCKS_FILE)
    print(f"Downloading 10-K and 10-Q filings for {len(tickers)} tickers since {TEN_YEARS_AGO}\n")
    download_filings(tickers)
    print("\nDone.")
