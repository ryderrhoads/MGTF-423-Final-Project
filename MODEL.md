# MODEL

## Tasks (20D horizon)
- **Direction (classification):** will excess return vs SPY be positive?
- **Big Move (classification):** will absolute excess move exceed threshold?
- **Magnitude (regression):** size of absolute excess return.

## Main model family
XGBoost models (classifier/regressor) with time-ordered train/val/test split.

## Feature set
Final headline model uses the **69 post-pruned features** documented in `FEATURES.md`.

## Interpretation approach
- SHAP values used for feature attribution on the Big Move model.
- Report includes full ranked SHAP table, correlation-based directional interpretation, and feature group tags.

## Key caveats
- Directional SHAP interpretation is correlational, not causal.
- Performance is sensitive to target definition and market regime.