# FEATURES.md

This reference intentionally covers **only the 69 post-pruned features** used in the final main model/report.

Source of truth: `docs/FEATURE_PRUNING_BUCKETS.csv` (`bucket=keep`).

## Row granularity

One row per filing event (`ticker` + `accession` + `filed_date`).

## Post-pruned feature set (69)

- `acceleration_net_income`
- `acceleration_net_income_slog`
- `acceleration_revenue_slog`
- `avg_text_score`
- `cogs_margin`
- `debt`
- `debt_to_assets_proxy`
- `debt_to_gross_profit`
- `distance_to_52w_high`
- `earnings_yield`
- `equity`
- `equity_to_assets_proxy`
- `etf_momentum_60d`
- `etf_relative_strength_20d`
- `etf_relative_strength_60d`
- `etf_vol_20d`
- `filing_length_surprise`
- `filing_month`
- `gross_margin`
- `gross_margin_change_yoy`
- `gross_profit`
- `gross_profit_growth_yoy`
- `gross_profit_margin`
- `interaction_growth_volatility`
- `interaction_profitability_momentum`
- `inventory`
- `inventory_growth_qoq`
- `inventory_growth_yoy`
- `inventory_intensity`
- `inventory_to_equity`
- `log_dollar_volume_20d`
- `log_market_cap`
- `margin_surprise_proxy`
- `market_stress_proxy`
- `net_income`
- `net_income_growth_qoq`
- `net_income_growth_qoq_slog`
- `net_income_growth_yoy`
- `net_income_growth_yoy_slog`
- `net_margin`
- `pre_event_runup_10d`
- `pre_event_runup_5d`
- `pre_event_runup_60d`
- `price_to_sales`
- `relative_strength_20d`
- `relative_strength_60d`
- `revenue_change_qoq`
- `revenue_change_yoy`
- `revenue_change_yoy_slog`
- `sales_inventory`
- `sector_regime_20d`
- `sentiment_change_qoq`
- `sgna`
- `sgna_growth_qoq`
- `sgna_growth_yoy`
- `sgna_margin`
- `spy_drawdown_252d`
- `spy_momentum_20d`
- `spy_momentum_60d`
- `spy_vol_20d`
- `spy_vol_regime_5d_60d`
- `stock_20d_excess_return`
- `stock_beta_252d`
- `stock_drawdown_60d`
- `stock_momentum_20d`
- `stock_momentum_60d`
- `stock_vol_20d`
- `turnover_20d`
- `vol_ratio`

## Plain-language notes for key engineered features

- `acceleration_revenue_slog`: smoothed revenue acceleration signal (short-term vs long-term growth behavior).
- `acceleration_net_income`: near-term net-income trend minus longer-run trend; positive implies earnings momentum improving.
- `margin_surprise_proxy`: current gross-margin change vs its longer-run pattern; positive implies better-than-trend margin shift.
- `inventory_buildup` is **not** in the post-pruned 69 and is intentionally excluded from final-model docs.
- `operating_leverage_proxy` is **not** in the post-pruned 69 and is intentionally excluded from final-model docs.

## Targets used in modeling

Targets are built in `modeling/train_eval.py:add_targets` from forward post-filing returns:

- `target_excess_{h}d = forward_stock_return(h) - forward_spy_return(h)`
- `target_cls_up_{h}d = 1[target_excess_{h}d > 0]`
- `target_mag_abs_excess_{h}d = abs(target_excess_{h}d)`
- `target_big_move_{h}d` = big absolute excess move classifier
- `target_pos_surprise_{h}d`, `target_neg_surprise_{h}d`
- `target_rank_pct_{h}d`, `target_quintile_{h}d`
- `target_vol_expansion_10d`

Main report (`modeling/train_eval_main.py`) emphasizes 20D tasks:
- Direction AUC: `target_cls_up_20d`
- Big Move AUC: `target_big_move_20d`
- Magnitude R²: `target_mag_abs_excess_20d`
- Decile check: `target_cls_up_20d` scores vs realized `target_excess_20d`

## Feature family list (used in SHAP-family charts)

### Text / NLP
- `avg_text_score`
- `filing_length_surprise`
- `sentiment_change_qoq`
- `score_std_mean`
- `score_std_max`
- `positive_count_total`
- `negative_count_total`
- `neutral_count_total`
- `positive_ratio`
- `negative_ratio`
- `neutral_ratio`
- `sentiment_polarity`
- `section_count`
- `num_units_total`
- `units_per_section`

### Fundamental
- `acceleration_net_income`
- `acceleration_net_income_slog`
- `acceleration_revenue_slog`
- `cogs_margin`
- `debt`
- `debt_to_assets_proxy`
- `debt_to_gross_profit`
- `earnings_yield`
- `equity`
- `equity_to_assets_proxy`
- `gross_margin`
- `gross_margin_change_yoy`
- `gross_profit`
- `gross_profit_growth_yoy`
- `gross_profit_margin`
- `inventory`
- `inventory_growth_qoq`
- `inventory_growth_yoy`
- `inventory_intensity`
- `inventory_to_equity`
- `margin_surprise_proxy`
- `net_income`
- `net_income_growth_qoq`
- `net_income_growth_qoq_slog`
- `net_income_growth_yoy`
- `net_income_growth_yoy_slog`
- `net_margin`
- `price_to_sales`
- `revenue_change_qoq`
- `revenue_change_yoy`
- `revenue_change_yoy_slog`
- `sales_inventory`
- `sgna`
- `sgna_growth_qoq`
- `sgna_growth_yoy`
- `sgna_margin`

### Market / Technical
- `distance_to_52w_high`
- `log_dollar_volume_20d`
- `log_market_cap`
- `pre_event_runup_10d`
- `pre_event_runup_5d`
- `pre_event_runup_60d`
- `relative_strength_20d`
- `relative_strength_60d`
- `spy_drawdown_252d`
- `spy_momentum_20d`
- `spy_momentum_60d`
- `spy_vol_20d`
- `stock_20d_excess_return`
- `stock_beta_252d`
- `stock_drawdown_60d`
- `stock_momentum_20d`
- `stock_momentum_60d`
- `stock_vol_20d`
- `turnover_20d`
- `vol_ratio`

### Sector / Regime
- `etf_momentum_60d`
- `etf_relative_strength_20d`
- `etf_relative_strength_60d`
- `etf_vol_20d`
- `market_stress_proxy`
- `sector_regime_20d`

### Interaction / Seasonality
- `filing_month`
- `interaction_growth_volatility`
- `interaction_profitability_momentum`

**Note:** Family assignment follows the same keyword-based grouping logic used by the chart script (`modeling/generate_charts.py`).
