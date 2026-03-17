# DATA

## Scope
This project models post-filing stock behavior for a consumer-sector universe using SEC filing data, market data, and text sentiment.

## Core tables/files
- `financials` — filing-level accounting fields (revenue, COGS, SG&A, inventory, debt, equity, net income, shares)
- `market_data_daily` — daily `close`, `volume`, `ret` by ticker/date
- `finbert_section_scores` — section-level sentiment outputs from FinBERT
- `features` — final filing-level modeling matrix
- `universe` — ticker membership + consumer labels/ETF mapping

## Row granularity
One feature row per filing event:
- key: `ticker + accession + filed_date`

## Target construction
Targets are derived from forward post-filing returns vs SPY:
- `target_excess_{h}d`
- `target_cls_up_{h}d`
- `target_mag_abs_excess_{h}d`
- `target_big_move_{h}d`

## Data quality controls
- ticker normalization to uppercase
- date parsing + key-based deduplication
- clipping/sanity checks for implausible values
- missing-value handling before modeling (imputation pipeline)

## What is intentionally excluded
Raw SEC text archive (~100GB `sec-edgar-filings/*.txt`) is not required for model training/inference once structured tables are created.