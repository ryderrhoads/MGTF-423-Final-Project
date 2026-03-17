# Features

69 features after pruning. One row per filing event (`ticker + accession + filed_date`).

SHAP rankings below are from the 20D Big Move XGBoost classifier (the strongest model). Full ranking in `docs/MAIN_TRAIN_EVAL_SHAP_FULL.csv`.

---

## Market / Technical (17 features)

The two most important features in the entire model are in this group. `stock_vol_20d` and `distance_to_52w_high` together account for more SHAP contribution than any other family — stocks that are volatile and far from their highs are more likely to make big post-filing moves.

| Feature | What it is |
|---|---|
| `stock_vol_20d` | Annualized 20-day trailing volatility of the stock. **SHAP #1.** |
| `distance_to_52w_high` | `close / 52-week high - 1`. Always ≤ 0. **SHAP #2.** |
| `spy_momentum_20d` | SPY compound return over trailing 20 days. **SHAP #7.** |
| `etf_relative_strength_20d` | Sector ETF 20d return minus SPY 20d return. **SHAP #8.** |
| `relative_strength_60d` | Stock 60d return minus SPY 60d return. **SHAP #9.** |
| `etf_momentum_60d` | Sector ETF compound return over trailing 60 days. |
| `etf_relative_strength_60d` | Sector ETF 60d return minus SPY 60d return. |
| `relative_strength_20d` | Stock 20d return minus SPY 20d return. |
| `spy_momentum_60d` | SPY compound return over trailing 60 days. |
| `spy_vol_20d` | Annualized 20-day trailing volatility of SPY. |
| `stock_20d_excess_return` | Stock 20d return minus SPY 20d return (same as `relative_strength_20d`). |
| `stock_beta_252d` | Rolling 252-day beta vs SPY. |
| `stock_drawdown_60d` | Max drawdown of the stock over trailing 60 days. |
| `stock_momentum_20d` | Stock compound return over trailing 20 days. |
| `stock_momentum_60d` | Stock compound return over trailing 60 days. |
| `turnover_20d` | Average daily share volume over trailing 20 days. |
| `vol_ratio` | `stock_vol_20d / spy_vol_20d`. Relative volatility. |

## Fundamental (22 features)

Profitability and size features dominate here. Larger, more profitable companies are less likely to make big post-filing moves — they're better covered and their filings contain fewer surprises.

| Feature | What it is |
|---|---|
| `net_income` | Net income from the filing. **SHAP #3.** |
| `equity` | Total shareholders' equity. **SHAP #4.** |
| `net_margin` | `net_income / revenue`. **SHAP #6.** |
| `gross_margin_change_yoy` | YoY change in gross margin. **SHAP #17.** |
| `revenue_change_yoy` | YoY revenue growth rate. |
| `inventory_to_equity` | `inventory / equity`. |
| `net_income_growth_qoq` | QoQ net income growth. |
| `revenue_change_qoq` | QoQ revenue growth rate. |
| `debt_to_gross_profit` | `debt / gross_profit`. |
| `gross_margin` | `(revenue - cogs) / revenue`. |
| `gross_profit` | `revenue - cogs`. |
| `gross_profit_growth_yoy` | YoY gross profit growth. |
| `gross_profit_margin` | Same as `gross_margin` (kept for legacy reasons). |
| `cogs_margin` | `cogs / revenue`. |
| `debt` | Total debt from the filing. |
| `debt_to_assets_proxy` | `debt / (debt + equity)`. |
| `earnings_yield` | Inverse of P/E, roughly. |
| `equity_to_assets_proxy` | `equity / (debt + equity)`. |
| `inventory` | Inventory from the filing. |
| `inventory_growth_qoq` | QoQ inventory growth. |
| `inventory_growth_yoy` | YoY inventory growth. |
| `inventory_intensity` | Inventory relative to revenue. |

Plus: `acceleration_net_income`, `acceleration_net_income_slog`, `acceleration_revenue_slog` (momentum-of-momentum signals), `margin_surprise_proxy` (gross margin change vs its own trend), `net_income_growth_qoq_slog`, `net_income_growth_yoy`, `net_income_growth_yoy_slog`, `price_to_sales`, `revenue_change_yoy_slog`, `sales_inventory`, `sgna`, `sgna_growth_qoq`, `sgna_growth_yoy`, `sgna_margin`.

The `_slog` suffix means signed-log transform: `sign(x) * log(1 + |x|)`. Compresses outliers in growth rates.

## Text / NLP (23 features)

These come from running FinBERT over MD&A and Risk Factors sections of each filing. The text features matter more for the Big Move and Magnitude tasks than for Direction (see the SHAP family chart in the README).

| Feature | What it is |
|---|---|
| `sentiment_polarity` | `positive_ratio - negative_ratio` across all scored sections. **SHAP #16.** |
| `negative_count_total` | Total negative-sentiment sentence count. **SHAP #18.** |
| `avg_text_score` | Mean FinBERT sentiment score across all sections. **SHAP #24.** |
| `score_std_mean` | Mean within-section sentiment standard deviation. Captures mixed-signal filings. |
| `score_std_max` | Max within-section sentiment standard deviation. |
| `sentiment_change_qoq` | Change in `avg_text_score` from prior filing. |
| `filing_length_surprise` | Filing length vs form-type average (z-scored). |
| `positive_ratio` | Share of sentences classified as positive. |
| `negative_ratio` | Share of sentences classified as negative. |
| `neutral_ratio` | Share of sentences classified as neutral. |
| `positive_count_total` | Total positive-sentiment sentence count. |
| `neutral_count_total` | Total neutral-sentiment sentence count. |
| `num_units_total` | Total scored text units (sentences) in the filing. |
| `units_per_section` | `num_units_total / section_count`. |
| `section_count` | Number of distinct sections scored. |

Plus form-type z-scored variants (`_form_z` suffix) for: `score_std_mean`, `score_std_max`, `positive_ratio`, `negative_ratio`, `neutral_ratio`, `sentiment_polarity`, `num_units_total`, `units_per_section`, `section_count`.

The form-type z-score normalizes each feature relative to its 10-K or 10-Q distribution, since filing structure differs systematically between the two form types.

## Sector / Regime (6 features)

| Feature | What it is |
|---|---|
| `etf_vol_20d` | Sector ETF annualized 20-day vol. |
| `market_stress_proxy` | Composite market stress indicator. |
| `sector_regime_20d` | Sector momentum regime classification. |
| `log_dollar_volume_20d` | `log(1 + avg daily dollar volume)`. Liquidity proxy. |
| `log_market_cap` | `log(1 + market cap)`. Size proxy. |
| `spy_drawdown_252d` | SPY max drawdown over trailing 252 days. |

## Interaction / Seasonality (3 features)

| Feature | What it is |
|---|---|
| `filing_month` | Calendar month of the filing date (1–12). |
| `interaction_growth_volatility` | Revenue growth × stock volatility cross-term. |
| `interaction_profitability_momentum` | Net margin × stock momentum cross-term. |

---

## Targets

All targets are computed from *forward* returns starting the day after `filed_date`. Excess = stock minus SPY over the same window.

| Target | Definition | Task |
|---|---|---|
| `target_cls_up_20d` | 1 if 20d excess return > 0 | Direction (classification) |
| `target_big_move_20d` | 1 if \|20d excess return\| > 5% | Big Move (classification) |
| `target_mag_abs_excess_20d` | \|20d excess return\| | Magnitude (regression) |