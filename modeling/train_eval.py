"""
Train/evaluate models on filing features with strict time-series split (70/10/20).

Models
------
For each horizon in {1d, 5d, 20d}:
- Classification (direction of excess return > 0, plus big-move targets):
  - RandomForestClassifier
  - XGBClassifier
- Regression (magnitude = abs(excess return)):
  - RandomForestRegressor
  - XGBRegressor

Outputs are printed to stdout with validation + test metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
# ElasticNet removed from headline runs to keep evaluation lean/defensible.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
# (intentionally removed pipeline-based linear classifier imports)

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as e:  # pragma: no cover
    raise SystemExit("xgboost is required. pip install xgboost") from e

try:
    import shap
except Exception:
    shap = None

try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None


DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features.csv"
RETURNS_DIR = DATA_DIR / "daily_returns"
IMPORTANCE_DIR = DATA_DIR / "feature_importance"
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
FEATURES_TABLE = os.getenv("FEATURES_TABLE", "features")


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_returns(ticker: str) -> pd.DataFrame:
    p = RETURNS_DIR / f"{ticker}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "ret"])
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["date", "ret"])
    val_col = ticker if ticker in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "ret": pd.to_numeric(df[val_col], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("date").reset_index(drop=True)


def forward_compound_return(df_ret: pd.DataFrame, asof: pd.Timestamp, horizon: int) -> float | None:
    if df_ret.empty or pd.isna(asof):
        return None
    s = df_ret.loc[df_ret["date"] > asof, "ret"].head(horizon)
    if len(s) < horizon:
        return None
    return float((1.0 + s).prod() - 1.0)


def trailing_vol(df_ret: pd.DataFrame, asof: pd.Timestamp, n: int) -> float | None:
    if df_ret.empty or pd.isna(asof):
        return None
    s = pd.to_numeric(df_ret.loc[df_ret["date"] <= asof, "ret"].tail(n), errors="coerce").dropna()
    if len(s) < max(2, n // 2):
        return None
    return float(s.std(ddof=1))


def forward_vol(df_ret: pd.DataFrame, asof: pd.Timestamp, n: int) -> float | None:
    if df_ret.empty or pd.isna(asof):
        return None
    s = pd.to_numeric(df_ret.loc[df_ret["date"] > asof, "ret"].head(n), errors="coerce").dropna()
    if len(s) < max(2, n // 2):
        return None
    return float(s.std(ddof=1))


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    out = df.copy()
    out["filed_date"] = pd.to_datetime(out["filed_date"], errors="coerce")

    tickers = sorted(set(out["ticker"].dropna().astype(str)))
    ret_cache = {t: load_returns(t) for t in tickers}
    spy = load_returns("SPY")

    targets = {h: [] for h in horizons}
    vol_exp_10d = []

    for _, r in out.iterrows():
        t = str(r.get("ticker", ""))
        d = r.get("filed_date")
        stock = ret_cache.get(t, pd.DataFrame(columns=["date", "ret"]))

        # Volatility expansion target: vol_10d_after / vol_20d_before
        v_after_10 = forward_vol(stock, d, 10)
        v_before_20 = trailing_vol(stock, d, 20)
        if v_after_10 is None or v_before_20 in (None, 0) or not np.isfinite(v_before_20):
            vol_exp_10d.append(np.nan)
        else:
            vol_exp_10d.append(float(v_after_10 / v_before_20))

        for h in horizons:
            r_stock = forward_compound_return(stock, d, h)
            r_spy = forward_compound_return(spy, d, h)
            if r_stock is None or r_spy is None:
                targets[h].append(np.nan)
            else:
                targets[h].append(r_stock - r_spy)

    out["target_vol_expansion_10d"] = vol_exp_10d

    for h in horizons:
        ex_col = f"target_excess_{h}d"
        out[ex_col] = targets[h]
        out[f"target_cls_up_{h}d"] = (out[ex_col] > 0).astype(float)
        out[f"target_mag_abs_excess_{h}d"] = out[ex_col].abs()

        # Large-move classification target (absolute excess return)
        if h == 5 and "stock_vol_20d" in out.columns:
            vol_thresh = 1.5 * pd.to_numeric(out["stock_vol_20d"], errors="coerce")
            out[f"target_big_move_{h}d"] = (
                out[ex_col].abs() > pd.concat([pd.Series(0.05, index=out.index), vol_thresh], axis=1).max(axis=1)
            ).astype(float)
        else:
            out[f"target_big_move_{h}d"] = (out[ex_col].abs() > 0.05).astype(float)

        # Positive/negative surprise classifiers
        out[f"target_pos_surprise_{h}d"] = (out[ex_col] > 0.03).astype(float)
        out[f"target_neg_surprise_{h}d"] = (out[ex_col] < -0.03).astype(float)

        # Rank-style targets
        out[f"target_rank_pct_{h}d"] = out[ex_col].rank(pct=True, method="average")
        out[f"target_quintile_{h}d"] = pd.qcut(out[ex_col], 5, labels=False, duplicates="drop")

    return out


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1) -> Split:
    s = df.sort_values("filed_date").reset_index(drop=True)
    n = len(s)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return Split(train=s.iloc[:i1], val=s.iloc[i1:i2], test=s.iloc[i2:])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    base_drop = {
        "ticker",
        "currency",
        "units",
        "form",
        "filed_date",
        "end_period",
        "start_period",
        "accession",
        "filing",
        "high_252d",  # use distance_to_52w_high instead
    }
    cols = []
    for c in df.columns:
        if c in base_drop:
            continue
        if c.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def prep_xy(split: Split, feature_cols: list[str], target_col: str) -> Tuple[np.ndarray, ...]:
    def _sanitize(x: pd.DataFrame) -> pd.DataFrame:
        z = x.copy()
        # convert +/-inf to NaN so imputer can safely handle them
        z = z.replace([np.inf, -np.inf], np.nan)
        # hard-cap extreme values to protect models from numeric blowups
        z = z.clip(lower=-1e12, upper=1e12)
        return z

    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    Xtr = imp.fit_transform(_sanitize(split.train[feature_cols]))
    Xv = imp.transform(_sanitize(split.val[feature_cols]))
    Xt = imp.transform(_sanitize(split.test[feature_cols]))

    ytr = split.train[target_col].to_numpy()
    yv = split.val[target_col].to_numpy()
    yt = split.test[target_col].to_numpy()
    return Xtr, Xv, Xt, ytr, yv, yt


def print_cls_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> None:
    y_hat = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    print(
        f"{name:18s} | acc={accuracy_score(y_true, y_hat):.4f} "
        f"prec={precision_score(y_true, y_hat, zero_division=0):.4f} "
        f"rec={recall_score(y_true, y_hat, zero_division=0):.4f} "
        f"f1={f1_score(y_true, y_hat, zero_division=0):.4f} auc={auc:.4f}"
    )


def print_reg_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(
        f"{name:18s} | mae={mean_absolute_error(y_true, y_pred):.6f} "
        f"rmse={rmse:.6f} r2={r2_score(y_true, y_pred):.4f}"
    )


def print_decile_table(y_score: np.ndarray, y_realized: np.ndarray, label: str) -> None:
    d = pd.DataFrame({"score": y_score, "realized": y_realized}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50:
        print(f"{label}: decile table skipped (insufficient rows)")
        return
    d["decile"] = pd.qcut(d["score"], 10, labels=False, duplicates="drop") + 1
    t = d.groupby("decile").agg(n=("realized", "size"), avg_realized=("realized", "mean")).reset_index()
    print(f"{label}: decile avg realized return")
    print(t.to_string(index=False))


def print_feature_distributions(df: pd.DataFrame, feature_cols: list[str], top_n: int = 30) -> None:
    """Print quick distro snapshot before training."""
    if not feature_cols:
        print("No feature columns available for distribution print.")
        return

    work = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    stats = work.describe(percentiles=[0.05, 0.5, 0.95]).T
    stats = stats.rename(columns={"5%": "p05", "50%": "p50", "95%": "p95"})

    # prioritize volatile/heavy-tail columns by std magnitude
    order = stats["std"].fillna(0).sort_values(ascending=False).head(top_n).index
    show = stats.loc[order, ["count", "mean", "std", "min", "p05", "p50", "p95", "max"]]

    print("\n" + "=" * 80)
    print(f"FEATURE DISTRIBUTIONS (top {min(top_n, len(show))} by std)")
    print("=" * 80)
    print(show.round(6).to_string())


def _safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _compute_shap_mean_abs(model, X: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    return np.abs(np.asarray(sv)).mean(axis=0)


def write_shap_importance_csv(
    model,
    X: np.ndarray,
    feature_cols: list[str],
    horizon: int,
    target_name: str,
    model_name: str,
    split_name: str = "test",
) -> None:
    if shap is None:
        return
    try:
        n = min(len(X), 1000)
        if n == 0:
            return
        Xs = X[:n]
        mean_abs = _compute_shap_mean_abs(model, Xs)
        out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
        fname = (
            f"h{horizon}d_{_safe_name(target_name)}_{_safe_name(model_name)}_{_safe_name(split_name)}_shap.csv"
        )
        out.to_csv(IMPORTANCE_DIR / fname, index=False)
    except Exception:
        return


def print_shap_top_features(model, X: np.ndarray, feature_cols: list[str], label: str, top_n: int = 12) -> None:
    if shap is None:
        print(f"{label}: SHAP skipped (install shap)")
        return
    try:
        n = min(len(X), 1000)
        if n == 0:
            print(f"{label}: SHAP skipped (no rows)")
            return
        mean_abs = _compute_shap_mean_abs(model, X[:n])
        idx = np.argsort(mean_abs)[::-1][:top_n]
        print(f"{label}: SHAP top {top_n} (mean |SHAP|)")
        for i in idx:
            print(f"  {feature_cols[i]:30s} {mean_abs[i]:.6f}")
    except Exception as e:
        print(f"{label}: SHAP failed ({e})")


def load_features_source() -> pd.DataFrame:
    if DATABASE_URL:
        if create_engine is None:
            raise RuntimeError("DATABASE_URL is set but SQLAlchemy is not installed.")
        engine = create_engine(DATABASE_URL)
        q = f"SELECT * FROM {FEATURES_TABLE}"
        print(f"Loading features from Postgres table '{FEATURES_TABLE}'...")
        return pd.read_sql(q, engine)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run build_features.py first.")
    print(f"Loading features from CSV: {FEATURES_PATH}")
    return pd.read_csv(FEATURES_PATH)


def main() -> None:
    horizons = (1, 5, 20)
    df = load_features_source()
    df = add_targets(df, horizons=horizons)
    df = df.dropna(subset=["filed_date"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Rows total (pre-horizon filter): {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print_feature_distributions(df, feature_cols, top_n=35)

    reg_models = {
        "RF Regressor": RandomForestRegressor(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "XGB Regressor": XGBRegressor(
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
        ),
    }

    for h in horizons:
        excess_col = f"target_excess_{h}d"
        cls_col = f"target_cls_up_{h}d"
        mag_col = f"target_mag_abs_excess_{h}d"

        dfx = df.dropna(subset=[excess_col]).reset_index(drop=True)
        split = time_split(dfx, train_frac=0.7, val_frac=0.1)

        print("\n" + "=" * 80)
        print(f"HORIZON: {h}D")
        print("=" * 80)
        print(f"Rows used: {len(dfx)}")
        print(f"Rows train/val/test: {len(split.train)}/{len(split.val)}/{len(split.test)}")
        print(f"Date range: {split.train['filed_date'].min()} -> {split.test['filed_date'].max()}")

        train_pos = float(split.train[cls_col].sum())
        train_neg = float(len(split.train) - train_pos)
        scale_pos_weight = (train_neg / train_pos) if train_pos > 0 else 1.0

        cls_models = {
            "RF Classifier": RandomForestClassifier(
                n_estimators=800,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "XGB Classifier": XGBClassifier(
                n_estimators=1200,
                max_depth=3,
                learning_rate=0.01,
                min_child_weight=8,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_lambda=8.0,
                reg_alpha=2.0,
                gamma=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
            ),
        }

        print("\nCLASSIFICATION: Direction of excess return")
        Xtr, Xv, Xt, ytr, yv, yt = prep_xy(split, feature_cols, cls_col)
        for name, model in cls_models.items():
            model.fit(Xtr, ytr)
            pv = model.predict_proba(Xv)[:, 1]
            pt = model.predict_proba(Xt)[:, 1]
            print(f"\n{name}")
            print_cls_metrics("  Val", yv, pv)
            print_cls_metrics("  Test", yt, pt)
            write_shap_importance_csv(model, Xt, feature_cols, h, cls_col, name, split_name="test")
            if name == "XGB Classifier":
                print_shap_top_features(model, Xt, feature_cols, label=f"  Test {h}d XGB Classifier")
                print_decile_table(pt, split.test[excess_col].to_numpy(), label=f"  Test {h}d XGB Classifier")

        # Additional binary targets (trimmed for defensibility)
        extra_targets: list[tuple[str, str]] = []
        if h != 5:  # 5D big-move target is degenerate in this dataset
            extra_targets.append((f"target_big_move_{h}d", "CLASSIFICATION: Big Move"))
        if h != 1:  # drop noisy 1D negative-surprise run
            extra_targets.append((f"target_neg_surprise_{h}d", "CLASSIFICATION: Negative Surprise"))

        for extra_target, title in extra_targets:
            dfx_extra = dfx.dropna(subset=[extra_target]).reset_index(drop=True)
            split_extra = time_split(dfx_extra, train_frac=0.7, val_frac=0.1)
            print(f"\n{title}")
            Xtr, Xv, Xt, ytr, yv, yt = prep_xy(split_extra, feature_cols, extra_target)
            for name, model in cls_models.items():
                model.fit(Xtr, ytr)
                pv = model.predict_proba(Xv)[:, 1]
                pt = model.predict_proba(Xt)[:, 1]
                print(f"\n{name}")
                print_cls_metrics("  Val", yv, pv)
                print_cls_metrics("  Test", yt, pt)
                write_shap_importance_csv(model, Xt, feature_cols, h, extra_target, name, split_name="test")

        print("\nREGRESSION: Magnitude abs(excess return)")
        Xtr, Xv, Xt, ytr, yv, yt = prep_xy(split, feature_cols, mag_col)
        for name, model in reg_models.items():
            model.fit(Xtr, ytr)
            pv = model.predict(Xv)
            pt = model.predict(Xt)
            print(f"\n{name}")
            print_reg_metrics("  Val", yv, pv)
            print_reg_metrics("  Test", yt, pt)
            write_shap_importance_csv(model, Xt, feature_cols, h, mag_col, name, split_name="test")
            if name == "XGB Regressor":
                print_shap_top_features(model, Xt, feature_cols, label=f"  Test {h}d XGB Regressor")

        # Vol-expansion regression intentionally omitted from headline evaluation

    print("\nDone.")


if __name__ == "__main__":
    main()
