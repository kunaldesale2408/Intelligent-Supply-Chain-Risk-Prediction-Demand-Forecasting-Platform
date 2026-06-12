"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
ML Model Architecture  |  Section 3
=============================================================
Three production ML models, each with:
  - Feature selection & train/test split strategy
  - Hyperparameter grid search
  - Full evaluation suite (metrics + plots saved to disk)
  - SHAP explainability (TreeExplainer or LinearExplainer)
  - Serialised artefacts (.joblib) for the FastAPI layer

Models
------
  MODEL 1  Demand Forecasting
           Algorithm : HistGradientBoostingRegressor (sklearn's
                       native gradient boosting — same theory as
                       XGBoost, no extra install needed)
           Target    : daily_demand  (continuous, regression)
           Split     : TimeSeriesSplit (walk-forward, no leakage)
           Metrics   : MAE, RMSE, MAPE

  MODEL 2  Stockout Prediction
           Algorithm : RandomForestClassifier (binary)
           Target    : stockout_in_horizon  (0/1)
           Split     : TimeSeriesSplit
           Metrics   : ROC-AUC, F1, Precision, Recall, Conf. Matrix

  MODEL 3  Supplier Risk Scoring
           Algorithm : GradientBoostingRegressor (delay_probability)
                       + ordinal label for risk_tier_encoded
           Target    : delay_probability  (0–1 regression)
                       risk_tier_encoded  (ordinal 0–3 classification)
           Split     : standard 80/20 (small dataset — 120 rows)
           Metrics   : MAE, RMSE (regression); ROC-AUC, F1 (classif.)

SHAP Explainability
-------------------
  - TreeExplainer for all tree-based models
  - Summary plots, waterfall plot for a single prediction,
    and beeswarm plot saved as PNG to data/model_outputs/
  - explain_prediction() function for the FastAPI /explain endpoint

MLflow stubs
------------
  Each train_* function wraps its training loop in
  mlflow_run_context() — a lightweight context manager that
  calls mlflow if installed, otherwise logs to console.
  Replace with `import mlflow` calls in production.

Requirements: scikit-learn >= 1.3, numpy, pandas, joblib
Optional:     shap (pip install shap), matplotlib, mlflow
=============================================================
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("models")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
FEATURES_DIR    = Path("./data/features")
MODELS_DIR      = Path("./models")
OUTPUTS_DIR     = Path("./data/model_outputs")

for d in (MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB  (optional — graceful fallback if headless)
# ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    log.warning("matplotlib not available — plot saving disabled")

# ─────────────────────────────────────────────────────────────
# SHAP  (optional — graceful fallback)
# ─────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
    log.info("SHAP available — explainability enabled")
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("SHAP not installed. Using permutation importance as fallback.")
    log.warning("Install with: pip install shap")

# ─────────────────────────────────────────────────────────────
# MLFLOW STUB
# Wraps training with MLflow tracking if mlflow is installed;
# falls back to structured console logging otherwise.
# ─────────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class _MLflowRun:
    """Lightweight context manager — real MLflow or stub."""

    def __init__(self, run_name: str, experiment: str = "supply_chain"):
        self.run_name   = run_name
        self.experiment = experiment
        self._run       = None

    def __enter__(self):
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.experiment)
            self._run = mlflow.start_run(run_name=self.run_name)
        log.info(f"[MLflow] Starting run: {self.run_name}")
        return self

    def __exit__(self, *_):
        if MLFLOW_AVAILABLE and self._run:
            mlflow.end_run()
        log.info(f"[MLflow] Run complete: {self.run_name}")

    def log_param(self, key: str, value: Any) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.log_param(key, value)
        log.info(f"  param  {key} = {value}")

    def log_params(self, params: dict) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value)
        log.info(f"  metric {key} = {value:.6f}")

    def log_metrics(self, metrics: dict) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_model(self, model: Any, artifact_name: str) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(model, artifact_name)
        log.info(f"  artifact → {artifact_name}")

    def log_artifact(self, path: str) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(path)


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def _load_features(name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"features_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run feature_engineering.py first."
        )
    return pd.read_csv(path, low_memory=False)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, capped at 100 % to avoid division-by-zero."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(
        np.mean(np.minimum(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]), 1.0)) * 100
    )


def _save_fig(fig, name: str) -> Path:
    if not MPL_AVAILABLE:
        return OUTPUTS_DIR / f"{name}.png"
    path = OUTPUTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot saved → {path.name}")
    return path


def _save_model(model: Any, name: str) -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    log.info(f"  Model saved → {path}")
    return path


def _load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE  (SHAP fallback)
# ─────────────────────────────────────────────────────────────

def _permutation_feature_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    scoring: str = "neg_mean_absolute_error",
    n_repeats: int = 10,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Permutation-based feature importance as a SHAP fallback.
    Works for ANY sklearn-compatible estimator.
    """
    result = permutation_importance(
        model, X_test, y_test,
        scoring=scoring, n_repeats=n_repeats, random_state=42
    )
    fi_df = (
        pd.DataFrame({
            "feature":    X_test.columns,
            "importance": result.importances_mean,
            "std":        result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return fi_df


def _plot_feature_importance(fi_df: pd.DataFrame, title: str, fname: str) -> None:
    if not MPL_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(fi_df) + 2))
    colors = ["#2563EB" if v >= 0 else "#EF4444" for v in fi_df["importance"]]
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1],
            xerr=fi_df["std"][::-1] if "std" in fi_df.columns else None,
            align="center", height=0.7)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean importance" if "std" in fi_df.columns else "Importance")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _save_fig(fig, fname)


# ─────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY UTILITIES
# ─────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X_explain: pd.DataFrame,
    model_type: str = "tree",           # "tree" | "linear" | "kernel"
) -> tuple[Any, np.ndarray]:
    """
    Create the right SHAP explainer for the model type and
    compute SHAP values.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_explain  : DataFrame of feature rows to explain
    model_type : explainer type ('tree' for RF/GBM, 'linear' for LR)

    Returns
    -------
    (explainer, shap_values_array)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Run: pip install shap")

    if model_type == "tree":
        # TreeExplainer is exact and fast for tree models
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    elif model_type == "linear":
        # LinearExplainer for logistic/linear regression
        explainer   = shap.LinearExplainer(model, X_explain,
                                             feature_perturbation="correlation_dependent")
        shap_values = explainer.shap_values(X_explain)
    else:
        # KernelExplainer works for any black-box model (slower)
        background  = shap.sample(X_explain, min(50, len(X_explain)))
        explainer   = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    return explainer, shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    title: str,
    fname: str,
    plot_type: str = "dot",             # "dot" (beeswarm) | "bar"
) -> None:
    """Save a SHAP summary plot (beeswarm or bar)."""
    if not SHAP_AVAILABLE or not MPL_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_explain,
        plot_type=plot_type,
        show=False,
        max_display=20,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(plt.gcf(), fname)
    plt.close("all")


def plot_shap_waterfall(
    explainer,
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    idx: int,
    title: str,
    fname: str,
) -> None:
    """
    Waterfall plot explaining a single prediction at row `idx`.
    Shows which features pushed the prediction above or below the baseline.
    """
    if not SHAP_AVAILABLE or not MPL_AVAILABLE:
        return
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = exp_val[0]

    sv = shap_values[idx] if shap_values.ndim == 2 else shap_values[:, idx]

    shap_exp = shap.Explanation(
        values         = sv,
        base_values    = exp_val,
        data           = X_explain.iloc[idx].values,
        feature_names  = X_explain.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(plt.gcf(), fname)
    plt.close("all")


def explain_prediction(
    model,
    X_row: pd.DataFrame,
    model_type: str = "tree",
    top_n: int = 10,
) -> dict:
    """
    Explain a single prediction row.
    Used by the FastAPI /explain endpoint.

    Returns
    -------
    {
        "prediction": float,
        "base_value": float,
        "top_features": [
            {"feature": str, "shap_value": float, "feature_value": float}, ...
        ]
    }
    """
    if SHAP_AVAILABLE:
        explainer, shap_values = compute_shap_values(model, X_row, model_type)
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            exp_val = float(exp_val[0])
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        top_features = [
            {
                "feature":       X_row.columns[i],
                "shap_value":    round(float(sv[i]), 6),
                "feature_value": round(float(X_row.iloc[0, i]), 6),
                "direction":     "↑ increases risk" if sv[i] > 0 else "↓ decreases risk",
            }
            for i in indices
        ]
    else:
        # Fallback: use built-in feature_importances_ if available
        prediction = float(
            model.predict(X_row)[0]
            if hasattr(model, "predict") else 0.0
        )
        if hasattr(model, "feature_importances_"):
            fi  = model.feature_importances_
            idx = np.argsort(fi)[::-1][:top_n]
            top_features = [
                {"feature": X_row.columns[i],
                 "importance": round(float(fi[i]), 6)}
                for i in idx
            ]
        else:
            top_features = []
        return {"prediction": prediction, "top_features": top_features,
                "note": "SHAP not installed; using feature_importances_ fallback"}

    pred = model.predict(X_row)[0]
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X_row)[0][1]   # P(positive class)

    return {
        "prediction":   round(float(pred), 6),
        "base_value":   round(exp_val, 6),
        "top_features": top_features,
    }


# ═════════════════════════════════════════════════════════════
# MODEL 1 — DEMAND FORECASTING
# Algorithm: HistGradientBoostingRegressor
# ═════════════════════════════════════════════════════════════

#  Features used by the demand forecasting model.
#  Excludes: identifiers, target leakage, and revenue (derived from target).
DEMAND_FEATURES = [
    "lag_7", "lag_14", "lag_21", "lag_30",
    "rolling_mean_7d",  "rolling_std_7d",
    "rolling_mean_14d", "rolling_std_14d",
    "rolling_mean_30d", "rolling_std_30d",
    "roll_price_7d", "roll_discount_7d",
    "demand_delta_7d", "demand_accel",
    "dow", "month", "quarter", "week_of_year", "day_of_year",
    "is_weekend", "is_month_start", "is_month_end", "is_quarter_end",
    "month_sin", "month_cos",
    "dow_sin",   "dow_cos",
    "woy_sin",   "woy_cos",
    "is_holiday_period", "is_back_to_school",
    "unit_price", "unit_cost", "lead_time_days", "reorder_point",
    "price_tier",
    "cat_Apparel", "cat_Electronics",
    "cat_Food & Beverage", "cat_Home & Garden",
]
DEMAND_TARGET = "daily_demand"


def train_demand_model(
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    run_gridsearch: bool = True,
) -> dict:
    """
    Train the demand forecasting model with walk-forward
    time-series cross-validation to prevent data leakage.

    Walk-forward split rationale
    ----------------------------
    Standard k-fold shuffle would leak future demand into past
    training windows. TimeSeriesSplit ensures each fold only
    trains on data strictly before the validation window.

    Hyperparameters
    ---------------
    HistGradientBoostingRegressor supports native NaN handling
    (no imputation needed for lag features at the start of series).
    Key parameters:
      max_iter        : number of boosting rounds (equiv. n_estimators)
      learning_rate   : step size shrinkage
      max_depth       : tree depth; controls variance/bias trade-off
      l2_regularization: L2 penalty on leaf weights (like XGBoost lambda)
      min_samples_leaf: minimum samples per leaf; prevents overfitting
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  MODEL 1: Demand Forecasting (GBM Regressor)")
    log.info("═══════════════════════════════════════════════════════")

    if df is None:
        df = _load_features("demand")

    # ── Feature & target preparation ─────────────────────────
    available_features = [f for f in DEMAND_FEATURES if f in df.columns]
    missing            = set(DEMAND_FEATURES) - set(available_features)
    if missing:
        log.warning(f"  Missing features (will skip): {missing}")

    df = df.copy().sort_values(["product_id", "sale_date"])
    X  = df[available_features].copy()
    y  = df[DEMAND_TARGET].copy()

    # HistGBM handles NaN natively — no imputation needed
    log.info(f"  Features: {len(available_features)}  |  Rows: {len(X):,}")
    log.info(f"  Target range: min={y.min():.1f}  mean={y.mean():.2f}  max={y.max():.1f}")

    # ── Walk-forward time-series split ───────────────────────
    tss = TimeSeriesSplit(n_splits=n_splits)

    # ── Hyperparameter grid ───────────────────────────────────
    base_model = HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    param_grid = {
        "max_iter":         [200, 400],
        "learning_rate":    [0.05, 0.10],
        "max_depth":        [4, 6],
        "min_samples_leaf": [20, 50],
        "l2_regularization":[0.0, 0.1],
    }

    with _MLflowRun("demand_forecasting_v1") as run:
        if run_gridsearch:
            log.info("  Running GridSearchCV (this takes ~1–2 min)…")
            gs = GridSearchCV(
                base_model, param_grid,
                cv=tss, scoring="neg_mean_absolute_error",
                n_jobs=-1, refit=True, verbose=0,
            )
            gs.fit(X, y)
            best_model  = gs.best_estimator_
            best_params = gs.best_params_
            log.info(f"  Best params: {best_params}")
        else:
            # Sensible defaults — use when iteration speed matters
            best_params = {
                "max_iter": 300, "learning_rate": 0.08,
                "max_depth": 5, "min_samples_leaf": 30,
                "l2_regularization": 0.05,
            }
            best_model = HistGradientBoostingRegressor(
                **best_params, random_state=42,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20,
            )
            best_model.fit(X, y)

        run.log_params(best_params)

        # ── Walk-forward evaluation ───────────────────────────
        cv_maes, cv_rmses, cv_mapes = [], [], []
        splits = list(tss.split(X))

        for fold, (tr_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            fold_model = HistGradientBoostingRegressor(
                **best_params, random_state=42,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20,
            )
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)

            mae_  = mean_absolute_error(y_val, preds)
            rmse_ = np.sqrt(mean_squared_error(y_val, preds))
            mape_ = _mape(y_val.values, preds)

            cv_maes.append(mae_)
            cv_rmses.append(rmse_)
            cv_mapes.append(mape_)
            log.info(f"  Fold {fold+1}/{n_splits}  MAE={mae_:.3f}  "
                     f"RMSE={rmse_:.3f}  MAPE={mape_:.2f}%")

        # ── Hold-out evaluation (last 20 % of time) ──────────
        split_idx    = int(len(X) * 0.8)
        X_train_fin  = X.iloc[:split_idx]
        X_test_fin   = X.iloc[split_idx:]
        y_train_fin  = y.iloc[:split_idx]
        y_test_fin   = y.iloc[split_idx:]

        best_model.fit(X_train_fin, y_train_fin)
        y_pred_fin   = best_model.predict(X_test_fin)

        mae_final  = mean_absolute_error(y_test_fin, y_pred_fin)
        rmse_final = np.sqrt(mean_squared_error(y_test_fin, y_pred_fin))
        mape_final = _mape(y_test_fin.values, y_pred_fin)

        metrics = {
            "cv_mae_mean":   float(np.mean(cv_maes)),
            "cv_mae_std":    float(np.std(cv_maes)),
            "cv_rmse_mean":  float(np.mean(cv_rmses)),
            "cv_mape_mean":  float(np.mean(cv_mapes)),
            "test_mae":      mae_final,
            "test_rmse":     rmse_final,
            "test_mape":     mape_final,
        }
        run.log_metrics(metrics)

        log.info(f"\n  ── HOLD-OUT RESULTS ──────────────────────────")
        log.info(f"  MAE  = {mae_final:.4f}  (avg units off per day)")
        log.info(f"  RMSE = {rmse_final:.4f}")
        log.info(f"  MAPE = {mape_final:.2f}%")

        # ── Actual vs Predicted plot ──────────────────────────
        if MPL_AVAILABLE:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))

            # Top: time-series overlay (sample 500 points for clarity)
            sample = min(500, len(y_test_fin))
            axes[0].plot(y_test_fin.values[:sample], label="Actual",
                         color="#2563EB", linewidth=1.0, alpha=0.85)
            axes[0].plot(y_pred_fin[:sample], label="Predicted",
                         color="#EF4444", linewidth=1.0, alpha=0.85, linestyle="--")
            axes[0].set_title("Demand Forecasting — Actual vs Predicted",
                               fontsize=13, fontweight="bold")
            axes[0].legend()
            axes[0].set_xlabel("Time step (test set)")
            axes[0].set_ylabel("Daily demand (units)")

            # Bottom: residual histogram
            residuals = y_test_fin.values - y_pred_fin
            axes[1].hist(residuals, bins=50, color="#7C3AED", alpha=0.75, edgecolor="white")
            axes[1].axvline(0, color="black", linewidth=1.2)
            axes[1].set_title(f"Residuals  (RMSE={rmse_final:.3f})")
            axes[1].set_xlabel("Residual (actual − predicted)")
            axes[1].set_ylabel("Count")

            plt.tight_layout()
            _save_fig(fig, "demand_actual_vs_predicted")

        # ── Feature importance ────────────────────────────────
        if SHAP_AVAILABLE:
            log.info("  Computing SHAP values for demand model…")
            # Explain on a sample to keep it fast
            X_sample = X_test_fin.sample(min(300, len(X_test_fin)), random_state=42)
            try:
                explainer, shap_vals = compute_shap_values(
                    best_model, X_sample, model_type="tree"
                )
                plot_shap_summary(
                    shap_vals, X_sample,
                    "SHAP Feature Importance — Demand Forecasting",
                    "demand_shap_summary",
                    plot_type="bar",
                )
                plot_shap_waterfall(
                    explainer, shap_vals, X_sample, idx=0,
                    title="SHAP Waterfall — Single Demand Prediction",
                    fname="demand_shap_waterfall",
                )
                joblib.dump(explainer, MODELS_DIR / "demand_shap_explainer.joblib")
            except Exception as e:
                log.warning(f"  SHAP computation failed: {e}")
        else:
            log.info("  Computing permutation importance (SHAP fallback)…")
            fi_df = _permutation_feature_importance(
                best_model, X_test_fin, y_test_fin,
                scoring="neg_mean_absolute_error"
            )
            _plot_feature_importance(fi_df, "Feature Importance — Demand (Permutation)",
                                     "demand_feature_importance")
            fi_df.to_csv(OUTPUTS_DIR / "demand_feature_importance.csv", index=False)

        # ── Save model artefact ───────────────────────────────
        model_path = _save_model(best_model, "demand_forecasting_v1")
        run.log_model(best_model, "demand_forecasting_v1")

        # Save evaluation summary
        eval_summary = {
            "model":   "demand_forecasting_v1",
            "metrics": metrics,
            "params":  best_params,
            "features": available_features,
        }
        with open(OUTPUTS_DIR / "demand_eval_summary.json", "w") as f:
            json.dump(eval_summary, f, indent=2)

    log.info("  MODEL 1 complete.\n")
    return {
        "model":    best_model,
        "metrics":  metrics,
        "params":   best_params,
        "features": available_features,
        "X_test":   X_test_fin,
        "y_test":   y_test_fin,
    }


# ═════════════════════════════════════════════════════════════
# MODEL 2 — STOCKOUT PREDICTION
# Algorithm: RandomForestClassifier (binary)
# ═════════════════════════════════════════════════════════════

STOCKOUT_FEATURES = [
    # Inventory position
    "quantity_on_hand", "quantity_available", "quantity_reserved",
    "reorder_point", "reorder_quantity",
    "is_below_reorder", "reorder_triggered",
    # Inventory dynamics
    "inventory_turnover_30d", "days_of_inventory",
    "lead_time_coverage", "avg_stock_30d",
    # Demand features
    "daily_demand",
    "demand_mean_7d",  "demand_std_7d",
    "demand_mean_14d", "demand_std_14d",
    "demand_mean_30d", "demand_std_30d",
    "demand_to_stock_ratio", "demand_accel_7d",
    # Risk indicators
    "is_zero_stock", "stockout_rate_30d",
    "is_low_stock", "consec_low_stock_days",
    # Product metadata
    "lead_time_days", "unit_cost_snapshot",
    # Calendar
    "dow", "month", "quarter",
    "is_weekend", "is_holiday_period", "is_back_to_school",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]
STOCKOUT_TARGET = "stockout_in_horizon"


def train_stockout_model(
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    run_gridsearch: bool = True,
) -> dict:
    """
    Train the binary stockout prediction classifier.

    Class imbalance strategy
    ------------------------
    With ~33 % positive rate the dataset is moderately imbalanced.
    RandomForestClassifier handles this via class_weight='balanced',
    which inversely weights each class by frequency.
    Evaluation prioritises ROC-AUC and F1 over raw accuracy.

    Feature leakage guard
    ---------------------
    'future_min_stock' is the raw look-ahead used to construct the
    target — it is explicitly excluded from model features.
    'is_zero_stock' from the SAME day is allowed (current state),
    but future stock info is not.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  MODEL 2: Stockout Prediction (Random Forest)")
    log.info("═══════════════════════════════════════════════════════")

    if df is None:
        df = _load_features("stockout")

    available_features = [
        f for f in STOCKOUT_FEATURES
        if f in df.columns and f != "future_min_stock"  # leakage guard
    ]
    missing = set(STOCKOUT_FEATURES) - set(available_features)
    if missing:
        log.warning(f"  Missing features (will skip): {missing}")

    df = df.copy().sort_values(["product_id", "snapshot_date"])
    X  = df[available_features].fillna(0).copy()
    y  = df[STOCKOUT_TARGET].astype(int).copy()

    pos_rate = y.mean()
    log.info(f"  Features: {len(available_features)}  |  Rows: {len(X):,}")
    log.info(f"  Target positive rate: {pos_rate*100:.1f}%")

    # ── Walk-forward split ────────────────────────────────────
    tss = TimeSeriesSplit(n_splits=n_splits)

    # ── Hyperparameter grid ───────────────────────────────────
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [8, 15, None],
        "min_samples_leaf": [5, 15],
        "max_features":     ["sqrt", 0.5],
    }

    with _MLflowRun("stockout_prediction_v1") as run:
        run.log_param("class_weight", "balanced")
        run.log_param("positive_rate", round(pos_rate, 4))

        if run_gridsearch:
            log.info("  Running GridSearchCV (this takes ~2–3 min)…")
            gs = GridSearchCV(
                base_model, param_grid,
                cv=tss, scoring="roc_auc",
                n_jobs=-1, refit=True, verbose=0,
            )
            gs.fit(X, y)
            best_model  = gs.best_estimator_
            best_params = gs.best_params_
            log.info(f"  Best params: {best_params}")
        else:
            best_params = {
                "n_estimators": 150, "max_depth": 12,
                "min_samples_leaf": 10, "max_features": "sqrt",
            }
            best_model = RandomForestClassifier(
                **best_params,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            best_model.fit(X, y)

        run.log_params(best_params)

        # ── Walk-forward CV ───────────────────────────────────
        cv_aucs, cv_f1s = [], []
        for fold, (tr_idx, val_idx) in enumerate(tss.split(X)):
            fm = RandomForestClassifier(
                **best_params,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            fm.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = fm.predict_proba(X.iloc[val_idx])[:, 1]
            preds = fm.predict(X.iloc[val_idx])
            auc_  = roc_auc_score(y.iloc[val_idx], proba)
            f1_   = f1_score(y.iloc[val_idx], preds, zero_division=0)
            cv_aucs.append(auc_)
            cv_f1s.append(f1_)
            log.info(f"  Fold {fold+1}/{n_splits}  ROC-AUC={auc_:.4f}  F1={f1_:.4f}")

        # ── Hold-out evaluation ───────────────────────────────
        split_idx   = int(len(X) * 0.8)
        X_tr_fin    = X.iloc[:split_idx]
        X_te_fin    = X.iloc[split_idx:]
        y_tr_fin    = y.iloc[:split_idx]
        y_te_fin    = y.iloc[split_idx:]

        best_model.fit(X_tr_fin, y_tr_fin)
        y_proba     = best_model.predict_proba(X_te_fin)[:, 1]
        y_pred_fin  = best_model.predict(X_te_fin)

        auc_final  = roc_auc_score(y_te_fin, y_proba)
        f1_final   = f1_score(y_te_fin, y_pred_fin, zero_division=0)
        prec_final = precision_score(y_te_fin, y_pred_fin, zero_division=0)
        rec_final  = recall_score(y_te_fin, y_pred_fin, zero_division=0)

        metrics = {
            "cv_auc_mean":   float(np.mean(cv_aucs)),
            "cv_auc_std":    float(np.std(cv_aucs)),
            "cv_f1_mean":    float(np.mean(cv_f1s)),
            "test_roc_auc":  auc_final,
            "test_f1":       f1_final,
            "test_precision":prec_final,
            "test_recall":   rec_final,
        }
        run.log_metrics(metrics)

        log.info(f"\n  ── HOLD-OUT RESULTS ──────────────────────────")
        log.info(f"  ROC-AUC  = {auc_final:.4f}")
        log.info(f"  F1       = {f1_final:.4f}")
        log.info(f"  Precision= {prec_final:.4f}")
        log.info(f"  Recall   = {rec_final:.4f}")
        log.info(f"\n{classification_report(y_te_fin, y_pred_fin, digits=4)}")

        # ── Evaluation plots ──────────────────────────────────
        if MPL_AVAILABLE:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # ROC curve
            fpr, tpr, _ = roc_curve(y_te_fin, y_proba)
            axes[0].plot(fpr, tpr, color="#2563EB", linewidth=2,
                         label=f"AUC = {auc_final:.4f}")
            axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
            axes[0].fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
            axes[0].set_title("ROC Curve — Stockout Prediction",
                               fontsize=11, fontweight="bold")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend(loc="lower right")

            # Confusion matrix
            cm = confusion_matrix(y_te_fin, y_pred_fin)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No Stockout", "Stockout"]
            )
            disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
            axes[1].set_title("Confusion Matrix", fontsize=11, fontweight="bold")

            # Prediction probability distribution
            axes[2].hist(y_proba[y_te_fin == 0], bins=40,
                         alpha=0.65, color="#22C55E", label="No Stockout")
            axes[2].hist(y_proba[y_te_fin == 1], bins=40,
                         alpha=0.65, color="#EF4444", label="Stockout")
            axes[2].axvline(0.5, color="black", linewidth=1.2, linestyle="--",
                            label="Default threshold")
            axes[2].set_title("Predicted Probability Distribution",
                               fontsize=11, fontweight="bold")
            axes[2].set_xlabel("P(stockout)")
            axes[2].legend()

            plt.tight_layout()
            _save_fig(fig, "stockout_evaluation")

        # ── SHAP / feature importance ─────────────────────────
        if SHAP_AVAILABLE:
            log.info("  Computing SHAP values for stockout model…")
            X_sample = X_te_fin.sample(min(200, len(X_te_fin)), random_state=42)
            try:
                explainer, shap_vals = compute_shap_values(
                    best_model, X_sample, model_type="tree"
                )
                # For binary RF, shap_values is a list [neg_class, pos_class]
                sv_pos = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                plot_shap_summary(
                    sv_pos, X_sample,
                    "SHAP Beeswarm — Stockout Prediction (Positive Class)",
                    "stockout_shap_beeswarm",
                    plot_type="dot",
                )
                # Waterfall for the highest-risk prediction
                max_risk_idx = int(np.argmax(
                    best_model.predict_proba(X_sample)[:, 1]
                ))
                plot_shap_waterfall(
                    explainer,
                    sv_pos,
                    X_sample,
                    idx=max_risk_idx,
                    title="SHAP Waterfall — Highest Risk Stockout Prediction",
                    fname="stockout_shap_waterfall",
                )
                joblib.dump(explainer, MODELS_DIR / "stockout_shap_explainer.joblib")
            except Exception as e:
                log.warning(f"  SHAP computation failed: {e}")
        else:
            log.info("  Computing permutation importance…")
            fi_df = _permutation_feature_importance(
                best_model, X_te_fin, y_te_fin,
                scoring="roc_auc",
            )
            _plot_feature_importance(fi_df, "Feature Importance — Stockout (Permutation)",
                                     "stockout_feature_importance")
            fi_df.to_csv(OUTPUTS_DIR / "stockout_feature_importance.csv", index=False)

        # ── Native RF importance (always available) ───────────
        fi_native = pd.DataFrame({
            "feature":    X.columns,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        _plot_feature_importance(
            fi_native.rename(columns={"importance": "importance"}),
            "RF Native Feature Importance — Stockout",
            "stockout_rf_native_importance",
        )
        fi_native.to_csv(OUTPUTS_DIR / "stockout_rf_importance.csv", index=False)

        # ── Save artefact ─────────────────────────────────────
        _save_model(best_model, "stockout_prediction_v1")
        run.log_model(best_model, "stockout_prediction_v1")

        eval_summary = {
            "model":    "stockout_prediction_v1",
            "metrics":  metrics,
            "params":   best_params,
            "features": available_features,
        }
        with open(OUTPUTS_DIR / "stockout_eval_summary.json", "w") as f:
            json.dump(eval_summary, f, indent=2)

    log.info("  MODEL 2 complete.\n")
    return {
        "model":    best_model,
        "metrics":  metrics,
        "params":   best_params,
        "features": available_features,
        "X_test":   X_te_fin,
        "y_test":   y_te_fin,
        "y_proba":  y_proba,
    }


# ═════════════════════════════════════════════════════════════
# MODEL 3 — SUPPLIER RISK SCORING
# Algorithm: GradientBoostingRegressor (delay probability)
#            + GradientBoostingClassifier (risk tier)
# ═════════════════════════════════════════════════════════════

SUPPLIER_RISK_FEATURES = [
    # Core delay statistics
    "hist_mean_delay_days",
    "hist_delay_variance",
    "hist_on_time_rate",
    "orders_evaluated",
    # Rolling / trend features
    "rolling_delay_mean_3m",
    "rolling_delay_std_3m",
    "rolling_on_time_rate_3m",
    "rolling_delay_prob_3m",
    "delay_trend_3m",
    "delay_spike_flag",
    # Lag features
    "lag_1m_delay_mean",
    "lag_1m_delay_prob",
    "lag_2m_delay_mean",
    # Supplier metadata
    "reliability_score",
    "payment_terms_days",
    "country_risk_tier",
    "years_active",
    # Interaction features
    "reliability_x_ontime",
    "delay_var_x_mean",
    # Region dummies
    "region_Americas",
    "region_Asia",
    "region_Europe",
]
SUPPLIER_REG_TARGET   = "delay_probability"       # 0–1 continuous
SUPPLIER_CLASS_TARGET = "risk_tier_encoded"        # 0=low … 3=critical


def train_supplier_risk_model(
    df: pd.DataFrame | None = None,
    run_gridsearch: bool = True,
) -> dict:
    """
    Train two complementary supplier risk models:

    3-A  Regression  → predict delay_probability (0–1)
         Ideal for a continuous risk score in the dashboard.

    3-B  Classification → predict risk_tier_encoded (0–3 ordinal)
         Ideal for the supplier scorecard tier label.

    Dataset size note
    -----------------
    With only 120 rows (10 suppliers × 12 months), standard
    k-fold cross-validation is used instead of TimeSeriesSplit.
    GridSearchCV uses 5-fold CV with stratification for classifier.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  MODEL 3: Supplier Risk (GBM Regressor + Classifier)")
    log.info("═══════════════════════════════════════════════════════")

    if df is None:
        df = _load_features("supplier_risk")

    available_features = [
        f for f in SUPPLIER_RISK_FEATURES if f in df.columns
    ]
    missing = set(SUPPLIER_RISK_FEATURES) - set(available_features)
    if missing:
        log.warning(f"  Missing features (will skip): {missing}")

    df = df.copy().sort_values(["supplier_id", "year_month"])
    X  = df[available_features].fillna(df[available_features].median()).copy()
    y_reg   = df[SUPPLIER_REG_TARGET].copy()
    y_class = df[SUPPLIER_CLASS_TARGET].astype(int).copy()

    log.info(f"  Features: {len(available_features)}  |  Rows: {len(X):,}")
    log.info(f"  Delay prob range: {y_reg.min():.3f} – {y_reg.max():.3f}")
    log.info(f"  Risk tier dist: {y_class.value_counts().sort_index().to_dict()}")

    # ── Train / test split (chronological, per supplier) ─────
    X_train, X_test, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te = train_test_split(
        X, y_reg, y_class,
        test_size=0.20, shuffle=False, random_state=42,
    )

    # ──────────────────────────────────────────────────────────
    # 3-A  REGRESSION
    # ──────────────────────────────────────────────────────────
    reg_param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.10],
        "max_depth":     [3, 5],
        "subsample":     [0.8, 1.0],
    }
    base_reg = GradientBoostingRegressor(random_state=42)

    with _MLflowRun("supplier_risk_regression_v1") as run:
        if run_gridsearch:
            log.info("  [3-A] GridSearchCV for regressor…")
            gs_reg = GridSearchCV(
                base_reg, reg_param_grid,
                cv=5, scoring="neg_mean_absolute_error",
                n_jobs=-1, refit=True, verbose=0,
            )
            gs_reg.fit(X_train, y_reg_tr)
            reg_model  = gs_reg.best_estimator_
            reg_params = gs_reg.best_params_
        else:
            reg_params = {
                "n_estimators": 150, "learning_rate": 0.08,
                "max_depth": 4, "subsample": 0.9,
            }
            reg_model = GradientBoostingRegressor(**reg_params, random_state=42)
            reg_model.fit(X_train, y_reg_tr)

        run.log_params(reg_params)

        y_reg_pred = reg_model.predict(X_test).clip(0, 1)
        mae_reg    = mean_absolute_error(y_reg_te, y_reg_pred)
        rmse_reg   = np.sqrt(mean_squared_error(y_reg_te, y_reg_pred))
        mape_reg   = _mape(y_reg_te.values, y_reg_pred)

        reg_metrics = {
            "test_mae":  mae_reg,
            "test_rmse": rmse_reg,
            "test_mape": mape_reg,
        }
        run.log_metrics(reg_metrics)

        log.info(f"\n  ── 3-A REGRESSION RESULTS ────────────────────")
        log.info(f"  Best params: {reg_params}")
        log.info(f"  MAE  = {mae_reg:.4f}")
        log.info(f"  RMSE = {rmse_reg:.4f}")
        log.info(f"  MAPE = {mape_reg:.2f}%")

        # Regression scatter plot
        if MPL_AVAILABLE:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].scatter(y_reg_te, y_reg_pred,
                            alpha=0.7, color="#2563EB", edgecolors="white", s=60)
            lims = [min(y_reg_te.min(), y_reg_pred.min()) - 0.02,
                    max(y_reg_te.max(), y_reg_pred.max()) + 0.02]
            axes[0].plot(lims, lims, "k--", linewidth=1.0)
            axes[0].set_title("Supplier Risk — Actual vs Predicted Delay Prob.",
                               fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Actual delay probability")
            axes[0].set_ylabel("Predicted delay probability")

            residuals = y_reg_te.values - y_reg_pred
            axes[1].hist(residuals, bins=20, color="#7C3AED",
                         alpha=0.75, edgecolor="white")
            axes[1].axvline(0, color="black", linewidth=1.2)
            axes[1].set_title(f"Residuals  (RMSE={rmse_reg:.4f})")
            axes[1].set_xlabel("Residual")
            plt.tight_layout()
            _save_fig(fig, "supplier_risk_regression_eval")

        _save_model(reg_model, "supplier_risk_regression_v1")
        run.log_model(reg_model, "supplier_risk_regression_v1")

    # ──────────────────────────────────────────────────────────
    # 3-B  CLASSIFICATION
    # ──────────────────────────────────────────────────────────
    cls_param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.10],
        "max_depth":     [3, 5],
    }
    base_cls = GradientBoostingClassifier(random_state=42)

    with _MLflowRun("supplier_risk_classifier_v1") as run:
        if run_gridsearch:
            log.info("  [3-B] GridSearchCV for classifier…")
            gs_cls = GridSearchCV(
                base_cls, cls_param_grid,
                cv=5, scoring="f1_macro",
                n_jobs=-1, refit=True, verbose=0,
            )
            gs_cls.fit(X_train, y_cls_tr)
            cls_model  = gs_cls.best_estimator_
            cls_params = gs_cls.best_params_
        else:
            cls_params = {
                "n_estimators": 150, "learning_rate": 0.08, "max_depth": 4,
            }
            cls_model = GradientBoostingClassifier(**cls_params, random_state=42)
            cls_model.fit(X_train, y_cls_tr)

        run.log_params(cls_params)

        y_cls_pred  = cls_model.predict(X_test)
        y_cls_proba = cls_model.predict_proba(X_test)

        # Macro F1 + per-class F1
        f1_macro  = f1_score(y_cls_te, y_cls_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_cls_te, y_cls_pred, average="weighted", zero_division=0)

        cls_metrics = {
            "test_f1_macro":    f1_macro,
            "test_f1_weighted": f1_weighted,
        }
        run.log_metrics(cls_metrics)

        log.info(f"\n  ── 3-B CLASSIFIER RESULTS ────────────────────")
        log.info(f"  Best params: {cls_params}")
        log.info(f"  F1 (macro)   = {f1_macro:.4f}")
        log.info(f"  F1 (weighted)= {f1_weighted:.4f}")
        log.info(f"\n{classification_report(y_cls_te, y_cls_pred, zero_division=0, digits=4)}")

        # Confusion matrix
        if MPL_AVAILABLE:
            fig, ax = plt.subplots(figsize=(7, 6))
            tier_names = ["Low", "Medium", "High", "Critical"]
            present = sorted(y_class.unique())
            labels  = [tier_names[i] for i in present]
            cm = confusion_matrix(y_cls_te, y_cls_pred, labels=present)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix — Supplier Risk Tier",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            _save_fig(fig, "supplier_risk_confusion_matrix")

        _save_model(cls_model, "supplier_risk_classifier_v1")
        run.log_model(cls_model, "supplier_risk_classifier_v1")

    # ──────────────────────────────────────────────────────────
    # 3-C  SHAP for regression model
    # ──────────────────────────────────────────────────────────
    if SHAP_AVAILABLE:
        log.info("  Computing SHAP values for supplier risk models…")
        try:
            explainer_reg, sv_reg = compute_shap_values(
                reg_model, X_test, model_type="tree"
            )
            plot_shap_summary(
                sv_reg, X_test,
                "SHAP Summary — Supplier Delay Probability",
                "supplier_risk_shap_summary",
                plot_type="dot",
            )
            plot_shap_waterfall(
                explainer_reg, sv_reg, X_test,
                idx=int(np.argmax(y_reg_pred)),
                title="SHAP Waterfall — Highest Risk Supplier",
                fname="supplier_risk_shap_waterfall",
            )
            joblib.dump(explainer_reg,
                        MODELS_DIR / "supplier_risk_shap_explainer.joblib")
        except Exception as e:
            log.warning(f"  SHAP computation failed: {e}")
    else:
        log.info("  Computing permutation importance (SHAP fallback)…")
        fi_df = _permutation_feature_importance(
            reg_model, X_test, y_reg_te,
            scoring="neg_mean_absolute_error"
        )
        _plot_feature_importance(
            fi_df,
            "Feature Importance — Supplier Risk (Permutation)",
            "supplier_risk_feature_importance"
        )
        fi_df.to_csv(OUTPUTS_DIR / "supplier_risk_feature_importance.csv", index=False)

    # Consolidated eval summary
    eval_summary = {
        "model_regression":    "supplier_risk_regression_v1",
        "model_classifier":    "supplier_risk_classifier_v1",
        "metrics_regression":  reg_metrics,
        "metrics_classifier":  cls_metrics,
        "params_regression":   reg_params,
        "params_classifier":   cls_params,
        "features":            available_features,
    }
    with open(OUTPUTS_DIR / "supplier_risk_eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    log.info("  MODEL 3 complete.\n")
    return {
        "reg_model":  reg_model,
        "cls_model":  cls_model,
        "reg_metrics": reg_metrics,
        "cls_metrics": cls_metrics,
        "reg_params":  reg_params,
        "cls_params":  cls_params,
        "features":    available_features,
        "X_test":      X_test,
        "y_reg_test":  y_reg_te,
        "y_cls_test":  y_cls_te,
    }


# ═════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# Used by FastAPI endpoints in Section 4
# ═════════════════════════════════════════════════════════════

def predict_demand(
    product_id:  int,
    feature_row: dict,
    model_name:  str = "demand_forecasting_v1",
) -> dict:
    """
    Load the demand model and return a point prediction.

    Parameters
    ----------
    product_id  : product being forecast
    feature_row : dict of feature_name → value (must match DEMAND_FEATURES)
    model_name  : name of the saved .joblib file

    Returns
    -------
    {
        "product_id":       int,
        "predicted_demand": float,
        "model":            str,
    }
    """
    model = _load_model(model_name)
    X     = pd.DataFrame([feature_row])[DEMAND_FEATURES]
    pred  = float(model.predict(X)[0])
    return {
        "product_id":       product_id,
        "predicted_demand": round(max(0, pred), 2),
        "model":            model_name,
    }


def predict_stockout(
    product_id:  int,
    feature_row: dict,
    threshold:   float = 0.5,
    model_name:  str   = "stockout_prediction_v1",
) -> dict:
    """
    Load the stockout model and return probability + binary flag.

    Returns
    -------
    {
        "product_id":          int,
        "stockout_probability":float,
        "stockout_flag":       bool,
        "model":               str,
    }
    """
    model = _load_model(model_name)
    cols  = [f for f in STOCKOUT_FEATURES if f in feature_row]
    X     = pd.DataFrame([feature_row])[cols].fillna(0)
    prob  = float(model.predict_proba(X)[0][1])
    return {
        "product_id":           product_id,
        "stockout_probability": round(prob, 4),
        "stockout_flag":        bool(prob >= threshold),
        "model":                model_name,
    }


def predict_supplier_risk(
    supplier_id: int,
    feature_row: dict,
    reg_name:    str = "supplier_risk_regression_v1",
    cls_name:    str = "supplier_risk_classifier_v1",
) -> dict:
    """
    Load both supplier risk models and return delay probability
    + ordinal risk tier.

    Returns
    -------
    {
        "supplier_id":       int,
        "delay_probability": float,
        "risk_tier_encoded": int,
        "risk_tier_label":   str,
        "model_reg":         str,
        "model_cls":         str,
    }
    """
    TIER_LABELS = {0: "low", 1: "medium", 2: "high", 3: "critical"}

    reg_model  = _load_model(reg_name)
    cls_model  = _load_model(cls_name)
    cols       = [f for f in SUPPLIER_RISK_FEATURES if f in feature_row]
    X          = pd.DataFrame([feature_row])[cols].fillna(0)

    delay_prob  = float(np.clip(reg_model.predict(X)[0], 0, 1))
    tier_enc    = int(cls_model.predict(X)[0])
    tier_label  = TIER_LABELS.get(tier_enc, "medium")

    return {
        "supplier_id":       supplier_id,
        "delay_probability": round(delay_prob, 4),
        "risk_tier_encoded": tier_enc,
        "risk_tier_label":   tier_label,
        "model_reg":         reg_name,
        "model_cls":         cls_name,
    }


# ═════════════════════════════════════════════════════════════
# CONSOLIDATED EVALUATION REPORT
# ═════════════════════════════════════════════════════════════

def print_consolidated_report(results: dict) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print("  SECTION 3 — ML MODEL EVALUATION REPORT")
    print(sep)

    if "demand" in results:
        m = results["demand"]["metrics"]
        print(f"\n  MODEL 1 — Demand Forecasting")
        print(f"  {'Metric':<25} {'CV Mean':>10}  {'Hold-out':>10}")
        print(f"  {'-'*47}")
        print(f"  {'MAE':<25} {m['cv_mae_mean']:>10.4f}  {m['test_mae']:>10.4f}")
        print(f"  {'RMSE':<25} {m['cv_rmse_mean']:>10.4f}  {m['test_rmse']:>10.4f}")
        print(f"  {'MAPE (%)':<25} {m['cv_mape_mean']:>10.2f}  {m['test_mape']:>10.2f}")

    if "stockout" in results:
        m = results["stockout"]["metrics"]
        print(f"\n  MODEL 2 — Stockout Prediction")
        print(f"  {'Metric':<25} {'CV Mean':>10}  {'Hold-out':>10}")
        print(f"  {'-'*47}")
        print(f"  {'ROC-AUC':<25} {m['cv_auc_mean']:>10.4f}  {m['test_roc_auc']:>10.4f}")
        print(f"  {'F1 Score':<25} {m['cv_f1_mean']:>10.4f}  {m['test_f1']:>10.4f}")
        print(f"  {'Precision':<25} {'—':>10}  {m['test_precision']:>10.4f}")
        print(f"  {'Recall':<25} {'—':>10}  {m['test_recall']:>10.4f}")

    if "supplier_risk" in results:
        rm = results["supplier_risk"]["reg_metrics"]
        cm = results["supplier_risk"]["cls_metrics"]
        print(f"\n  MODEL 3-A — Supplier Delay Probability (Regression)")
        print(f"  {'MAE':<25} {'—':>10}  {rm['test_mae']:>10.4f}")
        print(f"  {'RMSE':<25} {'—':>10}  {rm['test_rmse']:>10.4f}")
        print(f"  {'MAPE (%)':<25} {'—':>10}  {rm['test_mape']:>10.2f}")
        print(f"\n  MODEL 3-B — Supplier Risk Tier (Classification)")
        print(f"  {'F1 (macro)':<25} {'—':>10}  {cm['test_f1_macro']:>10.4f}")
        print(f"  {'F1 (weighted)':<25} {'—':>10}  {cm['test_f1_weighted']:>10.4f}")

    print(f"\n  Artefacts saved to: {MODELS_DIR.resolve()}")
    print(f"  Plots saved to    : {OUTPUTS_DIR.resolve()}")
    print(sep + "\n")


# ═════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════

def run_all_models(run_gridsearch: bool = True) -> dict:
    """
    Train all three models end-to-end and return results dict.

    Parameters
    ----------
    run_gridsearch : set False to use preset hyperparameters and
                     skip GridSearchCV (much faster, use for CI/CD)

    Returns
    -------
    {
        "demand":        {model, metrics, params, features, X_test, y_test},
        "stockout":      {model, metrics, params, features, X_test, y_test},
        "supplier_risk": {reg_model, cls_model, reg_metrics, cls_metrics, ...},
    }
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  Section 3 — ML Training Pipeline")
    log.info(f"  GridSearch: {'ON' if run_gridsearch else 'OFF (fast mode)'}")
    log.info("═══════════════════════════════════════════════════════\n")

    results = {}

    results["demand"]        = train_demand_model(run_gridsearch=run_gridsearch)
    results["stockout"]      = train_stockout_model(run_gridsearch=run_gridsearch)
    results["supplier_risk"] = train_supplier_risk_model(run_gridsearch=run_gridsearch)

    print_consolidated_report(results)
    return results


if __name__ == "__main__":
    import sys
    # Pass --fast flag to skip GridSearchCV for a quicker run
    fast = "--fast" in sys.argv
    run_all_models(run_gridsearch=not fast)