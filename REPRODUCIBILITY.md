# REPRODUCIBILITY

## Environment
- Python virtual environment in `.venv`
- Dependencies from `requirements.txt`
- Database connection via `DATABASE_URL` (or `POSTGRES_URL`)

## Rebuild steps
1. Activate environment:
   - `source .venv/bin/activate`
2. Ensure Postgres tables are loaded (`financials`, `market_data_daily`, `finbert_section_scores`, etc.).
3. Build features:
   - `FEATURE_SOURCE=postgres python data_processing/build_features.py`
4. Train/evaluate:
   - `python modeling/train_eval_main.py`

## Determinism notes
- Time-ordered splits are fixed by data ordering (`filed_date`).
- Model randomness controlled by explicit `random_state` in model definitions.
- Output artifacts are regenerated each run.

## Verification checks
- Confirm source mode line appears: `[build_features] source=postgres ...`
- Confirm feature count in report: `Features used after pruning: 69`
- Confirm report artifacts exist after run.