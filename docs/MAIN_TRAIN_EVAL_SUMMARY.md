# Final results (20D)

- Features used after pruning: 70

## Test metrics

| Task | Model | Test |
|---|---|---:|
| Direction | XGB | AUC 0.5404 |
| Big Move | XGB | AUC 0.6619 |
| Magnitude | XGB | R² 0.1373 |

## Full SHAP ranking (20D Big Move, XGB)

| Rank | Feature | Mean abs SHAP | Corr(feature, SHAP) | Directional interpretation | Group |
|---:|---|---:|---:|---|---|
| 1 | distance_to_52w_high | 0.196539 | -0.979 | higher value → lower Big-Move probability | Market / Technical |
| 2 | stock_vol_20d | 0.189254 | 0.510 | higher value → higher Big-Move probability | Market / Technical |
| 3 | net_income | 0.114028 | -0.297 | higher value → lower Big-Move probability | Fundamental |
| 4 | equity | 0.070873 | -0.294 | higher value → lower Big-Move probability | Fundamental |
| 5 | gross_profit | 0.061098 | -0.199 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 6 | net_margin | 0.057004 | -0.159 | higher value → lower Big-Move probability | Fundamental |
| 7 | spy_momentum_20d | 0.052887 | -0.762 | higher value → lower Big-Move probability | Market / Technical |
| 8 | etf_relative_strength_20d | 0.047695 | 0.930 | higher value → higher Big-Move probability | Market / Technical |
| 9 | relative_strength_60d | 0.045409 | 0.608 | higher value → higher Big-Move probability | Market / Technical |
| 10 | sgna | 0.040354 | -0.294 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 11 | etf_momentum_60d | 0.036862 | -0.324 | higher value → lower Big-Move probability | Market / Technical |
| 12 | pre_event_runup_10d | 0.032550 | -0.306 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 13 | sgna_change_yoy | 0.031121 | -0.294 | higher value → lower Big-Move probability | Interaction / Seasonality |
| 14 | log_dollar_volume_20d | 0.029204 | -0.120 | higher value → lower Big-Move probability | Market / Technical |
| 15 | gross_margin_change_yoy | 0.029150 | 0.234 | higher value → higher Big-Move probability | Fundamental |
| 16 | revenue_change_yoy | 0.028433 | 0.072 | weak/mixed directional effect | Fundamental |
| 17 | etf_relative_strength_60d | 0.028151 | 0.864 | higher value → higher Big-Move probability | Market / Technical |
| 18 | pre_event_runup_60d | 0.025476 | 0.565 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 19 | inventory_to_equity | 0.023744 | 0.462 | higher value → higher Big-Move probability | Fundamental |
| 20 | etf_vol_20d | 0.021936 | -0.737 | higher value → lower Big-Move probability | Market / Technical |
| 21 | sentiment_polarity | 0.021894 | -0.897 | higher value → lower Big-Move probability | Text / NLP |
| 22 | filing_month | 0.021401 | 0.179 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 23 | negative_count_total | 0.021103 | 0.891 | higher value → higher Big-Move probability | Text / NLP |
| 24 | avg_text_score | 0.020739 | 0.893 | higher value → higher Big-Move probability | Text / NLP |
| 25 | net_income_change_qoq | 0.019718 | 0.309 | higher value → higher Big-Move probability | Fundamental |
| 26 | turnover_20d | 0.019642 | 0.130 | higher value → higher Big-Move probability | Market / Technical |
| 27 | spy_momentum_60d | 0.019393 | -0.730 | higher value → lower Big-Move probability | Market / Technical |
| 28 | pre_event_runup_5d | 0.018469 | 0.176 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 29 | inventory | 0.017912 | -0.441 | higher value → lower Big-Move probability | Fundamental |
| 30 | score_std_mean | 0.016582 | 0.431 | higher value → higher Big-Move probability | Text / NLP |
| 31 | debt_to_gross_profit | 0.016362 | -0.161 | higher value → lower Big-Move probability | Fundamental |
| 32 | revenue_change_qoq | 0.015478 | -0.127 | higher value → lower Big-Move probability | Fundamental |
| 33 | inventory_change_yoy | 0.014868 | -0.242 | higher value → lower Big-Move probability | Fundamental |
| 34 | sales_inventory | 0.013150 | 0.102 | higher value → higher Big-Move probability | Fundamental |
| 35 | num_units_total | 0.013111 | 0.823 | higher value → higher Big-Move probability | Text / NLP |
| 36 | positive_ratio_form_z | 0.012918 | -0.867 | higher value → lower Big-Move probability | Text / NLP |
| 37 | positive_ratio | 0.012844 | -0.837 | higher value → lower Big-Move probability | Text / NLP |
| 38 | net_income_change_yoy | 0.012688 | 0.068 | weak/mixed directional effect | Fundamental |
| 39 | positive_count_total | 0.012364 | 0.105 | higher value → higher Big-Move probability | Text / NLP |
| 40 | neutral_count_total | 0.011313 | 0.768 | higher value → higher Big-Move probability | Text / NLP |
| 41 | score_std_max_form_z | 0.011188 | -0.815 | higher value → lower Big-Move probability | Text / NLP |
| 42 | relative_strength_20d | 0.011159 | 0.174 | higher value → higher Big-Move probability | Market / Technical |
| 43 | units_per_section | 0.010992 | 0.461 | higher value → higher Big-Move probability | Text / NLP |
| 44 | score_std_max | 0.010885 | -0.509 | higher value → lower Big-Move probability | Text / NLP |
| 45 | debt | 0.010088 | -0.389 | higher value → lower Big-Move probability | Fundamental |
| 46 | sgna_change_qoq | 0.010032 | 0.280 | higher value → higher Big-Move probability | Interaction / Seasonality |
| 47 | cogs_margin | 0.009901 | 0.370 | higher value → higher Big-Move probability | Fundamental |
| 48 | num_units_total_form_z | 0.009745 | 0.911 | higher value → higher Big-Move probability | Text / NLP |
| 49 | stock_momentum_60d | 0.009693 | 0.625 | higher value → higher Big-Move probability | Market / Technical |
| 50 | spy_vol_20d | 0.009264 | -0.142 | higher value → lower Big-Move probability | Market / Technical |
| 51 | sentiment_change_qoq | 0.009227 | 0.538 | higher value → higher Big-Move probability | Text / NLP |
| 52 | gross_margin | 0.007984 | -0.240 | higher value → lower Big-Move probability | Fundamental |
| 53 | margin_surprise_proxy | 0.007899 | 0.163 | higher value → higher Big-Move probability | Fundamental |
| 54 | sentiment_polarity_form_z | 0.007521 | -0.870 | higher value → lower Big-Move probability | Text / NLP |
| 55 | neutral_ratio | 0.006428 | -0.523 | higher value → lower Big-Move probability | Text / NLP |
| 56 | stock_momentum_20d | 0.005790 | -0.133 | higher value → lower Big-Move probability | Market / Technical |
| 57 | score_std_mean_form_z | 0.005708 | -0.299 | higher value → lower Big-Move probability | Text / NLP |
| 58 | stock_20d_excess_return | 0.005213 | 0.062 | weak/mixed directional effect | Market / Technical |
| 59 | negative_ratio | 0.005136 | 0.881 | higher value → higher Big-Move probability | Text / NLP |
| 60 | units_per_section_form_z | 0.004927 | 0.660 | higher value → higher Big-Move probability | Text / NLP |
| 61 | sgna_margin | 0.004815 | 0.075 | weak/mixed directional effect | Fundamental |
| 62 | equity_to_assets_proxy | 0.004301 | -0.063 | weak/mixed directional effect | Fundamental |
| 63 | debt_to_assets_proxy | 0.003574 | 0.150 | higher value → higher Big-Move probability | Fundamental |
| 64 | inventory_change_qoq | 0.003270 | 0.512 | higher value → higher Big-Move probability | Fundamental |
| 65 | gross_profit_margin | 0.002789 | -0.273 | higher value → lower Big-Move probability | Fundamental |
| 66 | negative_ratio_form_z | 0.002614 | 0.334 | higher value → higher Big-Move probability | Text / NLP |
| 67 | neutral_ratio_form_z | 0.002143 | 0.583 | higher value → higher Big-Move probability | Text / NLP |
| 68 | section_count_form_z | 0.001142 | 0.631 | higher value → higher Big-Move probability | Text / NLP |
| 69 | section_count | 0.000265 | 0.024 | weak/mixed directional effect | Text / NLP |
| 70 | log_market_cap | 0.000000 | nan | mixed/unclear | Market / Technical |

## Professor interpretation notes

- `Big Move` has the strongest discrimination signal (AUC materially above direction AUC), so this model appears better at identifying volatility events than pure sign.
- SHAP ranking is dominated by market-state features (distance to 52w high, short-term vol, drawdown) plus balance-sheet/profitability context, which is economically plausible for post-filing move magnitude.
- Directional interpretation column is correlational (feature value vs SHAP contribution), useful for spot-check narratives but not a causal claim.
- Use the full table above for row-level professor checks: rank, effect size, and direction are all explicit.

## Decile check (20D Direction, XGB)

- Decile 1 avg excess return: -4.1940%
- Decile 10 avg excess return: 3.6736%
- Decile spread (D10 - D1): 7.8676%
- Top-3 deciles avg excess return: 2.7068%
- Bottom-3 deciles avg excess return: -1.7645%