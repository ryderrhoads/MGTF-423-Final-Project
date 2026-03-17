# PIPELINE

## End-to-end flow
1. **Ingest filings + financials** into structured tables.
2. **Score text sections** with FinBERT (`finbert_section_scores`).
3. **Assemble features** at filing granularity (`build_features.py`).
4. **Train/evaluate models** with strict time-ordered split (`train_eval_main.py`).
5. **Produce report artifacts** (metrics, SHAP ranking, decile checks).

## Source-of-truth mode
`build_features.py` supports Postgres-native sourcing when `DATABASE_URL` is set:
- financials from `financials`
- returns/price-volume from `market_data_daily`
- sentiment aggregates from `finbert_section_scores`

## Upload design
- Core fast upload: `build_postgres_dataset_fast.py` (market, financials, filings, features)
- Split uploads:
  - `upload_universe.py`
  - `upload_finbert_scores.py`
  - `upload_text_sections.py` (optional, heavy)

## Why split uploads
Text sections are the largest/slowest payload. Splitting prevents long bottlenecks and makes reruns practical.

## Runtime and timeline
Total end-to-end runtime was approximately **~6 hours**, with most time spent in transformer inference.

### Phase breakdown
1. **FinBERT sentiment scoring (~4 hours)**
   - Executed on a remote GPU node (RunPod A40)
   - Produced `text_sections_scored.csv` used for sentiment aggregates

2. **Feature build (~minutes)**
   - Filing-level feature engineering from financial, market, and sentiment sources

3. **Training/evaluation (~minutes)**
   - XGBoost classification/regression runs
   - Artifact generation (metrics, SHAP ranking, decile checks)

## Infrastructure note
On-demand GPU for FinBERT kept cost low while avoiding local hardware bottlenecks. Remaining steps were CPU-light and ran locally.