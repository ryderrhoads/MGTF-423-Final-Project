# Final results (20D)

- Features used after pruning: 70

## Test metrics

| Task | Model | Test |
|---|---|---:|
| Direction | XGB | AUC 0.5448 |
| Big Move | XGB | AUC 0.6657 |
| Magnitude | XGB | R² 0.1362 |

## Full SHAP ranking (20D Big Move, XGB)

| Rank | Feature | Mean abs SHAP | Corr(feature, SHAP) | Directional interpretation | Group |
|---:|---|---:|---:|---|---|
| 1 | distance_to_52w_high | 0.215031 | -0.973 | higher value → lower Big-Move probability | Market / Technical |
| 2 | stock_vol_20d | 0.201673 | 0.513 | higher value → higher Big-Move probability | Market / Technical |
| 3 | net_income | 0.117083 | -0.314 | higher value → lower Big-Move probability | Fundamental |
| 4 | equity | 0.084575 | -0.269 | higher value → lower Big-Move probability | Fundamental |
| 5 | spy_momentum_20d | 0.064355 | -0.765 | higher value → lower Big-Move probability | Market / Technical |
| 6 | gross_profit | 0.064129 | -0.204 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 7 | net_margin | 0.058834 | -0.152 | higher value → lower Big-Move probability | Fundamental |
| 8 | log_dollar_volume_20d | 0.052408 | -0.195 | higher value → lower Big-Move probability | Market / Technical |
| 9 | revenue_change_yoy | 0.049121 | 0.140 | higher value → higher Big-Move probability | Fundamental |
| 10 | filing_month | 0.036055 | 0.289 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 11 | turnover_20d | 0.036025 | 0.024 | weak/mixed directional effect | Market / Technical |
| 12 | sgna | 0.034269 | -0.218 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 13 | etf_relative_strength_60d | 0.029944 | 0.890 | higher value → higher Big-Move probability | Market / Technical |
| 14 | etf_relative_strength_20d | 0.026227 | 0.832 | higher value → higher Big-Move probability | Market / Technical |
| 15 | gross_margin_change_yoy | 0.025160 | 0.210 | higher value → higher Big-Move probability | Fundamental |
| 16 | relative_strength_60d | 0.025035 | 0.638 | higher value → higher Big-Move probability | Market / Technical |
| 17 | num_units_total | 0.023896 | 0.714 | higher value → higher Big-Move probability | Text / NLP |
| 18 | net_income_change_qoq | 0.022623 | 0.314 | higher value → higher Big-Move probability | Fundamental |
| 19 | score_std_max_form_z | 0.021417 | -0.830 | higher value → lower Big-Move probability | Text / NLP |
| 20 | negative_count_total | 0.021404 | 0.889 | higher value → higher Big-Move probability | Text / NLP |
| 21 | pre_event_runup_60d | 0.020717 | 0.552 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 22 | sgna_change_qoq | 0.020243 | 0.713 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 23 | score_std_mean | 0.018230 | 0.407 | higher value → higher Big-Move probability | Text / NLP |
| 24 | etf_vol_20d | 0.017736 | -0.700 | higher value → lower Big-Move probability | Market / Technical |
| 25 | inventory_change_yoy | 0.017540 | -0.303 | higher value → lower Big-Move probability | Fundamental |
| 26 | sentiment_change_qoq | 0.016921 | 0.725 | higher value → higher Big-Move probability | Text / NLP |
| 27 | inventory_to_equity | 0.016697 | 0.277 | higher value → higher Big-Move probability | Fundamental |
| 28 | sentiment_polarity | 0.016638 | -0.903 | higher value → lower Big-Move probability | Text / NLP |
| 29 | etf_momentum_60d | 0.016437 | -0.140 | higher value → lower Big-Move probability | Market / Technical |
| 30 | pre_event_runup_10d | 0.016400 | -0.302 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 31 | spy_momentum_60d | 0.015152 | -0.772 | higher value → lower Big-Move probability | Market / Technical |
| 32 | negative_ratio | 0.015015 | 0.835 | higher value → higher Big-Move probability | Text / NLP |
| 33 | neutral_count_total | 0.014219 | 0.739 | higher value → higher Big-Move probability | Text / NLP |
| 34 | sgna_change_yoy | 0.013304 | 0.111 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 35 | sales_inventory | 0.013114 | -0.174 | higher value → lower Big-Move probability | Fundamental |
| 36 | relative_strength_20d | 0.012808 | 0.164 | higher value → higher Big-Move probability | Market / Technical |
| 37 | cogs_margin | 0.011389 | 0.299 | higher value → higher Big-Move probability | Fundamental |
| 38 | score_std_max | 0.011178 | -0.480 | higher value → lower Big-Move probability | Text / NLP |
| 39 | stock_momentum_60d | 0.010854 | 0.509 | higher value → higher Big-Move probability | Market / Technical |
| 40 | score_std_mean_form_z | 0.010767 | -0.159 | higher value → lower Big-Move probability | Text / NLP |
| 41 | revenue_change_qoq | 0.010491 | -0.003 | weak/mixed directional effect | Fundamental |
| 42 | units_per_section | 0.010181 | 0.660 | higher value → higher Big-Move probability | Text / NLP |
| 43 | gross_margin | 0.010157 | -0.050 | weak/mixed directional effect | Fundamental |
| 44 | debt_to_gross_profit | 0.009910 | -0.113 | higher value → lower Big-Move probability | Fundamental |
| 45 | inventory_change_qoq | 0.009815 | 0.538 | higher value → higher Big-Move probability | Fundamental |
| 46 | positive_count_total | 0.009512 | 0.130 | higher value → higher Big-Move probability | Text / NLP |
| 47 | margin_surprise_proxy | 0.009155 | 0.219 | higher value → higher Big-Move probability | Fundamental |
| 48 | debt | 0.009028 | -0.153 | higher value → lower Big-Move probability | Fundamental |
| 49 | positive_ratio_form_z | 0.008864 | -0.781 | higher value → lower Big-Move probability | Text / NLP |
| 50 | avg_text_score | 0.008786 | 0.851 | higher value → higher Big-Move probability | Text / NLP |
| 51 | pre_event_runup_5d | 0.008511 | 0.181 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 52 | inventory | 0.007732 | -0.472 | higher value → lower Big-Move probability | Fundamental |
| 53 | debt_to_assets_proxy | 0.006826 | 0.251 | higher value → higher Big-Move probability | Fundamental |
| 54 | num_units_total_form_z | 0.006400 | 0.435 | higher value → higher Big-Move probability | Text / NLP |
| 55 | sentiment_polarity_form_z | 0.006241 | -0.757 | higher value → lower Big-Move probability | Text / NLP |
| 56 | positive_ratio | 0.006206 | -0.595 | higher value → lower Big-Move probability | Text / NLP |
| 57 | net_income_change_yoy | 0.006162 | -0.051 | weak/mixed directional effect | Fundamental |
| 58 | section_count_form_z | 0.005804 | 0.712 | higher value → higher Big-Move probability | Text / NLP |
| 59 | units_per_section_form_z | 0.005301 | 0.249 | higher value → higher Big-Move probability | Text / NLP |
| 60 | sgna_margin | 0.004464 | 0.059 | weak/mixed directional effect | Fundamental |
| 61 | spy_vol_20d | 0.004192 | 0.388 | higher value → higher Big-Move probability | Market / Technical |
| 62 | equity_to_assets_proxy | 0.004043 | -0.250 | higher value → lower Big-Move probability | Fundamental |
| 63 | neutral_ratio_form_z | 0.003875 | 0.064 | weak/mixed directional effect | Text / NLP |
| 64 | stock_20d_excess_return | 0.003770 | 0.144 | higher value → higher Big-Move probability | Market / Technical |
| 65 | negative_ratio_form_z | 0.003768 | 0.407 | higher value → higher Big-Move probability | Text / NLP |
| 66 | stock_momentum_20d | 0.003608 | 0.049 | weak/mixed directional effect | Market / Technical |
| 67 | gross_profit_margin | 0.003328 | 0.065 | weak/mixed directional effect | Fundamental |
| 68 | neutral_ratio | 0.001525 | 0.062 | weak/mixed directional effect | Text / NLP |
| 69 | section_count | 0.000073 | -0.313 | higher value → lower Big-Move probability | Text / NLP |
| 70 | log_market_cap | 0.000000 | nan | mixed/unclear | Market / Technical |

## Professor interpretation notes

- `Big Move` has the strongest discrimination signal (AUC materially above direction AUC), so this model appears better at identifying volatility events than pure sign.
- SHAP ranking is dominated by market-state features (distance to 52w high, short-term vol, drawdown) plus balance-sheet/profitability context, which is economically plausible for post-filing move magnitude.
- Directional interpretation column is correlational (feature value vs SHAP contribution), useful for spot-check narratives but not a causal claim.
- Use the full table above for row-level professor checks: rank, effect size, and direction are all explicit.

## Decile check (20D Direction, XGB)

- Decile 1 avg excess return: -5.1697%
- Decile 10 avg excess return: 1.1487%
- Decile spread (D10 - D1): 6.3184%
- Top-3 deciles avg excess return: 2.0058%
- Bottom-3 deciles avg excess return: -1.8591%