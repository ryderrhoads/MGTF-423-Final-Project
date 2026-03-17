# MGTF 423 Final Project

## Overview
This project studies whether SEC filing information can predict post-filing **excess returns** versus SPY.

The final submission flow is intentionally simple:

```bash
python run_professor.py
```

That command runs the concise evaluation pipeline and writes a clean markdown summary to:
- `docs/MAIN_TRAIN_EVAL_SUMMARY.md`

---

## Data Collection / Sources
Universe and labels:
- `stock_universe/stocks.txt`
- `stock_universe/indexes.txt`
- `stock_universe/consumer_discretionary.csv`
- `stock_universe/consumer_staples.csv`

Raw sources:
- SEC EDGAR filings (10-K / 10-Q)
- SEC Company Facts (XBRL financial statement data)
- Yahoo Finance daily returns / price-volume history

Collection scripts:
- `data_collection/download_filings.py`
- `data_collection/download_daily_returns.py`

---

## Data Processing / Feature Design
Processing scripts:
- `data_processing/extract_filings.py`
- `data_processing/score_filings.py`
- `data_processing/build_features.py`

Feature groups used in the final modeling pipeline:
- Market regime and momentum (SPY/ETF momentum, volatility regime, sector regime)
- Technical state (drawdown, 52-week high distance, beta, volatility)
- Fundamentals and valuation (net margin, net income, earnings yield, equity, etc.)
- Event/text features (filing-derived sentiment and filing structure signals)
- Selected interaction terms

The feature set was pruned for the headline model to remove weak or redundant columns.

---

## Modeling & Targets
Modeling scripts:
- Full experiment script: `modeling/train_eval.py`
- Headline script (used for final writeup): `modeling/train_eval_main.py`

Headline targets and metrics (20D):
1. Direction classification (AUC)
2. Big-move classification (AUC)
3. Magnitude regression (R²)

Headline reporting also includes:
- SHAP top features for best model (20D Big Move, XGB)
- Decile spread check for 20D direction scores

---

## Summary / Findings
Final reported results are in:
- `docs/MAIN_TRAIN_EVAL_SUMMARY.md`
- `docs/MODEL_FINDINGS_2026-03-14_FINAL.md`

Core conclusion:
- Directional signal is modest but present.
- Big-move prediction is stronger than pure direction.
- Decile spread at 20D shows meaningful separation between low-score and high-score buckets.

---

## Methodological Notes
Safeguards and limitations are documented in:
- `docs/LIMITATIONS.md`

Key points:
- Chronological split (train/val/test) is used.
- Targets are forward-looking from filing date.
- Feature null handling avoids backward-fill leakage.
- Some preprocessing (e.g., global normalization/winsorization) remains a known caveat and is explicitly disclosed.
