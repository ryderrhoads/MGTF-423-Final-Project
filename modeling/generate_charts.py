#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from xgboost import DMatrix, XGBClassifier, XGBRegressor

from train_eval import add_targets, get_feature_columns, load_features_source, prep_xy, time_split

ROOT = Path(__file__).resolve().parent.parent
VIS = ROOT / "visuals"
VIS.mkdir(parents=True, exist_ok=True)

CUT_FEATURES = {
    "uncertainty_score_form_z", "growth_vs_momentum", "uncertainty_delta", "uncertainty_score",
    "inventory_sales_gap", "operating_leverage_proxy", "debt_equity", "debt_equity_slog",
    "sales_inventory_change_qoq", "sales_inventory_change_yoy_slog", "inventory_buildup",
    "interaction_sentiment_momentum", "risk_sentiment_form_z", "gross_margin_change_qoq_slog",
    "gross_profit_growth_yoy_slog", "growth_to_valuation", "gross_margin_change_yoy_slog",
    "inventory_turnover_proxy", "sales_inventory_change_qoq_slog", "sales_inventory_change_yoy",
    "etf_momentum_20d", "gross_margin_change_qoq", "acceleration_revenue", "revenue_change_qoq_slog",
    "market_cap", "dollar_volume_20d", "shares_outstanding", "cogs", "revenue",
    "filing_quarter", "filing_length", "filing_length_form_z", "filing_length_change",
    "mdna_sentiment_form_z", "risk_sentiment", "mdna_sentiment", "avg_text_score_form_z",
}


def _feature_group(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["sentiment", "uncertainty", "text", "mdna", "risk_", "filing_length", "score_std", "positive_", "negative_", "neutral_", "section_count", "num_units", "units_per_section"]):
        return "Text / NLP"
    if any(k in n for k in ["margin", "revenue", "income", "equity", "cash", "debt", "assets", "inventory", "cogs"]):
        return "Fundamental"
    if any(k in n for k in ["stock_", "drawdown", "vol_", "momentum", "relative_strength", "52w", "return", "beta", "close", "volume", "market_cap", "turnover"]):
        return "Market / Technical"
    if any(k in n for k in ["sector", "etf", "regime"]):
        return "Sector / Regime"
    return "Interaction / Seasonality"


def _prepare_model_data():
    df = load_features_source()
    df = add_targets(df, horizons=(20,))
    df = df.dropna(subset=["filed_date"]).reset_index(drop=True)
    feat_cols = [c for c in get_feature_columns(df) if c not in CUT_FEATURES]
    d20 = df.dropna(subset=["target_excess_20d"]).reset_index(drop=True)
    sp = time_split(d20, train_frac=0.7, val_frac=0.1)
    return sp, feat_cols


def _xgb_clf() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1200,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=8,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=8.0,
        reg_alpha=2.0,
        gamma=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )


def _xgb_reg() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1200,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=10.0,
        reg_alpha=3.0,
        gamma=0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )


def make_roc_charts() -> None:
    sp, feat_cols = _prepare_model_data()

    # Direction ROC
    Xtr, _, Xt, ytr, _, yt = prep_xy(sp, feat_cols, "target_cls_up_20d")
    m = _xgb_clf()
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xt)[:, 1]
    fpr, tpr, _ = roc_curve(yt, p)
    auc_dir = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    ax.set_title("ROC Curve — Direction (20D, XGB)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.text(0.02, -0.12, f"AUC = {auc_dir:.4f}\nThe model showed limited ability to predict direction (modestly above chance).", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(VIS / "ROC_DIRECTION_20D.png", dpi=220)
    plt.close(fig)

    # Big move ROC
    Xtr, _, Xt, ytr, _, yt = prep_xy(sp, feat_cols, "target_big_move_20d")
    m = _xgb_clf()
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xt)[:, 1]
    fpr, tpr, _ = roc_curve(yt, p)
    auc_big = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    ax.set_title("ROC Curve — Big Move (20D, XGB)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.text(0.02, -0.12, f"AUC = {auc_big:.4f}\nThe model is stronger at flagging large post-filing moves than direction.", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(VIS / "ROC_BIG_MOVE_20D.png", dpi=220)
    plt.close(fig)


def make_magnitude_scatter() -> None:
    sp, feat_cols = _prepare_model_data()
    Xtr, _, Xt, ytr, _, yt = prep_xy(sp, feat_cols, "target_mag_abs_excess_20d")
    m = _xgb_reg()
    m.fit(Xtr, ytr)
    yp = m.predict(Xt)

    x = np.asarray(yp, dtype=float)
    y = np.asarray(yt, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return

    beta = np.polyfit(x, y, 1)
    xmin, xmax = 0.0, 0.6
    ylo = beta[0] * xmin + beta[1]
    yhi = beta[0] * xmax + beta[1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=12, alpha=0.35, color="#1d4ed8", edgecolors="none")
    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="#9ca3af", linewidth=1.5)
    ax.plot([xmin, xmax], [ylo, yhi], color="#dc2626", linewidth=2.3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_title("Predicted vs Actual — Magnitude (20D Excess, XGB)")
    ax.set_xlabel("Predicted absolute excess return (20D)")
    ax.set_ylabel("Actual absolute excess return (20D)")
    fig.tight_layout()
    fig.savefig(VIS / "MAGNITUDE_PREDICTED_VS_ACTUAL_20D.png", dpi=220)
    plt.close(fig)


def _mean_abs_contribs(model, X: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
    booster = model.get_booster()
    dm = DMatrix(X, feature_names=feat_cols)
    contrib = booster.predict(dm, pred_contribs=True)
    arr = np.asarray(contrib)
    if arr.ndim != 2 or arr.shape[1] < len(feat_cols):
        return pd.Series(np.zeros(len(feat_cols)), index=feat_cols)
    # drop bias term (last col)
    arr = arr[:, : len(feat_cols)]
    return pd.Series(np.abs(arr).mean(axis=0), index=feat_cols)


def make_shap_family_by_task() -> None:
    sp, feat_cols = _prepare_model_data()

    rows = []

    # Direction
    Xtr, _, Xt, ytr, _, _ = prep_xy(sp, feat_cols, "target_cls_up_20d")
    mc = _xgb_clf()
    mc.fit(Xtr, ytr)
    s = _mean_abs_contribs(mc, Xt, feat_cols)
    g = s.groupby([_feature_group(c) for c in s.index]).sum()
    for k, v in g.items():
        rows.append({"task": "Direction", "family": k, "mean_abs_shap": float(v)})

    # Big Move
    Xtr, _, Xt, ytr, _, _ = prep_xy(sp, feat_cols, "target_big_move_20d")
    mc = _xgb_clf()
    mc.fit(Xtr, ytr)
    s = _mean_abs_contribs(mc, Xt, feat_cols)
    g = s.groupby([_feature_group(c) for c in s.index]).sum()
    for k, v in g.items():
        rows.append({"task": "Big Move", "family": k, "mean_abs_shap": float(v)})

    # Magnitude
    Xtr, _, Xt, ytr, _, _ = prep_xy(sp, feat_cols, "target_mag_abs_excess_20d")
    mr = _xgb_reg()
    mr.fit(Xtr, ytr)
    s = _mean_abs_contribs(mr, Xt, feat_cols)
    g = s.groupby([_feature_group(c) for c in s.index]).sum()
    for k, v in g.items():
        rows.append({"task": "Magnitude", "family": k, "mean_abs_shap": float(v)})

    df = pd.DataFrame(rows)
    if df.empty:
        return

    pivot = df.pivot_table(index="family", columns="task", values="mean_abs_shap", aggfunc="sum").fillna(0.0)

    # Normalize each model column to sum to 1.0 so bars are relative shares (% importance)
    col_sums = pivot.sum(axis=0).replace(0.0, np.nan)
    pivot = pivot.div(col_sums, axis=1).fillna(0.0)

    order = pivot.mean(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(pivot.index))
    width = 0.24
    tasks = [c for c in ["Direction", "Big Move", "Magnitude"] if c in pivot.columns]
    offsets = np.linspace(-width, width, len(tasks))
    colors = {"Direction": "#4C78A8", "Big Move": "#F58518", "Magnitude": "#54A24B"}

    for off, t in zip(offsets, tasks):
        ax.barh(y + off, pivot[t].values, height=width, label=t, color=colors.get(t, "#777"))

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Relative feature-family importance (% of total mean |SHAP| within model)")
    ax.set_title("SHAP by Feature Family — Per Model (20D, normalized)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(VIS / "SHAP_FAMILY_BY_TASK_20D.png", dpi=220)
    plt.close(fig)


def write_chart_notes() -> None:
    txt = """# Chart Notes (Professor-facing)

## Direction ROC
The model showed limited ability to predict direction, with an AUC of 0.5353, only modestly above chance.

## Big Move ROC
The model was much better at identifying whether a filing would be followed by a large price move than predicting the exact direction of the move.

## Magnitude scatter
Predicted vs actual magnitude shows a loose positive relationship: imperfect fit, but non-zero signal.

## SHAP by family per model
Feature-family importance differs by task (Direction vs Big Move vs Magnitude), supporting task-specific information usage.
"""
    (VIS / "CHART_NOTES.md").write_text(txt, encoding="utf-8")


def remove_legacy_files() -> None:
    # Explicit removals requested
    for p in [
        ROOT / "ACTUAL_20D_EXCESS_RETURN_DISTRIBUTION.png",
        ROOT / "BIG_MOVE_PREDICTED_PROB_DISTRIBUTION.png",
    ]:
        if p.exists():
            p.unlink()

    # No more SVG charts
    for p in ROOT.glob("*.svg"):
        p.unlink()


def main() -> None:
    make_shap_family_by_task()
    make_roc_charts()
    make_magnitude_scatter()
    write_chart_notes()
    remove_legacy_files()
    print("Saved PNG charts to visuals/: SHAP family by task, direction ROC, big-move ROC, magnitude scatter")


if __name__ == "__main__":
    main()
