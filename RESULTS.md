# RESULTS

## Main 20D test metrics
- **Direction (XGB):** AUC **0.5353**
- **Big Move (XGB):** AUC **0.6656**
- **Magnitude (XGB):** R² **0.1122**

## Decile portfolio check (20D Direction model)
- Decile 1 avg excess return: **-4.0608%**
- Decile 10 avg excess return: **3.9519%**
- Decile spread (D10 - D1): **8.0127%**
- Top-3 deciles avg excess return: **1.9777%**
- Bottom-3 deciles avg excess return: **-2.1964%**

## Interpretation
- The pipeline shows stronger signal for **Big Move** detection than pure direction.
- Decile sorting indicates meaningful rank-ordering signal in the direction model.

## Artifacts
- `MAIN_TRAIN_EVAL_SUMMARY.md` (human-readable report)
- `MAIN_TRAIN_EVAL_SHAP_FULL.csv` (full SHAP ranking)
- `FEATURE_PRUNING_BUCKETS.csv` (kept vs cut feature map)