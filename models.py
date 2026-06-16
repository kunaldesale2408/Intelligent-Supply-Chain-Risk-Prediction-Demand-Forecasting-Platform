"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
ML Models  |  Section 3
=============================================================

Three production-grade models + full SHAP explainability:

  MODEL 1  Demand Forecasting
           Algorithm : GradientBoostingRegressor (XGBoost-equivalent)
           Target    : daily_demand  (continuous)
           Metrics   : MAE, RMSE, MAPE, R²
           Split     : TimeSeriesSplit (no data leakage)

  MODEL 2  Stockout Prediction
           Algorithm : RandomForestClassifier  (binary)
           Target    : stockout_in_horizon  (0/1)
           Metrics   : ROC-AUC, F1, Precision, Recall
           Split     : TimeSeriesSplit, class-weight balanced

  MODEL 3  Supplier Risk
           Algorithm : GradientBoostingRegressor  (delay regression)
                     + GradientBoostingClassifier (risk tier, 4-class)
           Target    : delay_probability (reg) / risk_tier_encoded (clf)
           Metrics   : MAE, R² (regression); Macro-F1, ROC-AUC (clf)
           Split     : Leave-One-Supplier-Out cross-validation

  EXPLAINABILITY
           Permutation importance (sklearn native, no SHAP install needed)
           + Tree-SHAP approximation via marginal contributions
           Produces: feature importance bar charts, SHAP-style
           waterfall summary, top-N features per prediction

  EXPERIMENT TRACKING
           MLflowLogger — a lightweight drop-in that writes JSON run
           files to  mlruns/<experiment>/<run_id>/
           Compatible schema: replace with mlflow.start_run() in
           production once mlflow is pip-installed.

  ARTIFACT STORAGE
           Trained models → models/*.pkl
           Evaluation plots → models/plots/*.png
           Per-run metrics → mlruns/**/*.json

Requirements: pandas, numpy, scikit-learn, matplotlib
Production:   pip install xgboost mlflow shap
=============================================================
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

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
FEATURES_DIR = Path("./data/features")
MODELS_DIR   = Path("./models")
PLOTS_DIR    = MODELS_DIR / "plots"
MLRUNS_DIR   = Path("./mlruns")

for _d in (MODELS_DIR, PLOTS_DIR, MLRUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# LIGHTWEIGHT MLFLOW-COMPATIBLE EXPERIMENT TRACKER
# ─────────────────────────────────────────────────────────────

class MLflowLogger:
    """
    Minimal MLflow-compatible run logger.
    Persists params / metrics / tags / artifacts as JSON under
    mlruns/<experiment_name>/<run_id>/run_manifest.json

    ── Production swap ──────────────────────────────────────────
    Replace this class with:

        import mlflow
        mlflow.set_experiment("demand_forecasting")

        with mlflow.start_run(run_name="gbm_demand_v1"):
            mlflow.log_params({...})
            mlflow.log_metrics({...})
            mlflow.sklearn.log_model(model, "demand_forecaster")
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, experiment_name: str, run_name: str = ""):
        self.experiment_name = experiment_name
        self.run_id          = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_name        = run_name or self.run_id
        self.run_dir         = MLRUNS_DIR / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._params:    dict[str, Any]   = {}
        self._metrics:   dict[str, float] = {}
        self._tags:      dict[str, str]   = {}
        self._artifacts: list[str]        = []
        self._start_time = time.time()
        log.info(f"  [MLflow] Started  experiment={experiment_name}  "
                 f"run_id={self.run_id}")

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key] = round(float(value), 6)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_tag(self, key: str, value: str) -> None:
        self._tags[key] = str(value)

    def log_artifact(self, path: str) -> None:
        self._artifacts.append(str(path))

    def log_model(self, model: Any, model_name: str) -> Path:
        path = self.run_dir / f"{model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self.log_artifact(str(path))
        return path

    def end_run(self) -> dict:
        elapsed = round(time.time() - self._start_time, 2)
        manifest = {
            "run_id":       self.run_id,
            "run_name":     self.run_name,
            "experiment":   self.experiment_name,
            "status":       "FINISHED",
            "duration_s":   elapsed,
            "params":       self._params,
            "metrics":      self._metrics,
            "tags":         self._tags,
            "artifacts":    self._artifacts,
            "timestamp":    datetime.now().isoformat(),
        }
        manifest_path = self.run_dir / "run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"  [MLflow] Finished  duration={elapsed}s  "
                 f"metrics={self._metrics}")
        return manifest


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(
        np.minimum(100.0,
                   np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_roc_auc(y_true, y_score, multi_class: bool = False) -> float:
    try:
        if multi_class:
            return float(roc_auc_score(y_true, y_score,
                                        multi_class="ovr", average="macro"))
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def _save_plot(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_features(name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"features_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run feature_engineering.py first."
        )
    return pd.read_csv(path, low_memory=False)


def _save_model(model: Any, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> Any:
    """Load a persisted model by name (e.g. 'demand_forecaster')."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# EXPLAINABILITY ENGINE
# ─────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Model-agnostic explainability layer.

    Two modes
    ---------
    1. permutation_importance()
       Standard sklearn permutation importance on held-out data.
       Measures how much performance drops when each feature is shuffled.

    2. shap_approximation()
       Marginal contribution SHAP approximation — computes each
       feature's impact by comparing predictions with/without that
       feature set to its mean baseline value.

       ── Production swap ──────────────────────────────────────
       Replace shap_approximation() body with:

           import shap
           explainer   = shap.TreeExplainer(model)
           shap_values = explainer.shap_values(X_sample)
           # Then aggregate: mean_abs_shap = |shap_values|.mean(axis=0)
       ─────────────────────────────────────────────────────────
    """

    @staticmethod
    def permutation_importance(
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 10,
        task: str = "regression",
    ) -> pd.DataFrame:
        scoring = "r2" if task == "regression" else "roc_auc"
        result  = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring,
        )
        return pd.DataFrame({
            "feature":    feature_names,
            "importance": result.importances_mean,
            "std":        result.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    @staticmethod
    def shap_approximation(
        model: Any,
        X_sample: np.ndarray,
        feature_names: list[str],
        baseline: np.ndarray | None = None,
        n_samples: int = 50,
        task: str = "regression",
    ) -> pd.DataFrame:
        """
        For each feature i and sample row r:
            shap_i(r) ≈ f(r) − f(r with feature_i → baseline_i)

        Returns a DataFrame sorted by mean |SHAP| descending.
        """
        rng = np.random.default_rng(42)
        if X_sample.shape[0] > n_samples:
            idx      = rng.choice(X_sample.shape[0], n_samples, replace=False)
            X_sample = X_sample[idx]

        if baseline is None:
            if X_sample.shape[0] == 1:
                # Single-row inference (live API predictions): the mean of
                # one row IS that row, so "perturb feature_i -> baseline_i"
                # would be a no-op and every SHAP value would be exactly 0.
                # Use an all-zeros reference ("feature absent / neutral")
                # so each feature's marginal contribution is measurable.
                baseline = np.zeros_like(X_sample[0], dtype=float)
            else:
                baseline = np.nanmean(X_sample, axis=0)

        def _predict(X: np.ndarray) -> np.ndarray:
            if task == "classification" and hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)

        shap_vals = np.zeros((X_sample.shape[0], len(feature_names)))
        for i in range(len(feature_names)):
            X_pert        = X_sample.copy()
            X_pert[:, i]  = baseline[i]
            shap_vals[:, i] = _predict(X_sample) - _predict(X_pert)

        return pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            "mean_shap":     shap_vals.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    @staticmethod
    def plot_feature_importance(
        imp_df: pd.DataFrame,
        title: str,
        top_n: int = 20,
        importance_col: str = "importance",
    ) -> plt.Figure:
        top    = imp_df.head(top_n).copy()
        n_bars = len(top)
        fig, ax = plt.subplots(figsize=(9, max(4, n_bars * 0.37)))
        colors  = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_bars)[::-1])
        ax.barh(top["feature"][::-1], top[importance_col][::-1],
                color=colors[::-1], edgecolor="none", height=0.70)
        if "std" in top.columns and importance_col == "importance":
            ax.barh(top["feature"][::-1],
                    top["std"][::-1] * 2,
                    left=top[importance_col][::-1] - top["std"][::-1],
                    color="grey", alpha=0.22, height=0.70)
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_shap_waterfall(
        shap_df: pd.DataFrame,
        prediction: float,
        baseline: float,
        title: str,
        top_n: int = 12,
    ) -> plt.Figure:
        top = shap_df.head(top_n).copy()
        top["color"] = top["mean_shap"].apply(
            lambda v: "#d73027" if v > 0 else "#4575b4"
        )
        top = top.sort_values("mean_abs_shap")

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.44)))
        ax.barh(top["feature"], top["mean_shap"],
                color=top["color"], edgecolor="none", height=0.65)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("SHAP Value  (impact on model output)", fontsize=10)
        ax.set_title(
            f"{title}\nBaseline={baseline:.3f}  →  Prediction={prediction:.3f}",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        for _, row in top.iterrows():
            offset = 0.001 * (1 if row["mean_shap"] >= 0 else -1)
            ha     = "left" if row["mean_shap"] >= 0 else "right"
            ax.text(row["mean_shap"] + offset, row["feature"],
                    f"  {row['mean_shap']:+.3f}", va="center",
                    fontsize=7.5, color="black", ha=ha)
        fig.tight_layout()
        return fig


EX = ExplainabilityEngine()


# ═════════════════════════════════════════════════════════════
# MODEL 1 — DEMAND FORECASTING  (GBM Regressor)
# ═════════════════════════════════════════════════════════════

DEMAND_DROP = [
    "product_id", "sale_date",
    "daily_demand",           # target
    "daily_revenue",          # leakage: computed from demand × price
    "cat_Food & Beverage",    # reference category (avoid multicollinearity)
]

DEMAND_HYPERPARAMS = {
    "n_estimators":     200,
    "max_depth":        5,
    "learning_rate":    0.08,
    "subsample":        0.80,
    "min_samples_leaf": 10,
    "random_state":     42,
}


def train_demand_model(df: pd.DataFrame | None = None) -> dict:
    """
    Train an XGBoost-equivalent Gradient Boosting Regressor for
    daily product demand forecasting.

    Time-series aware split: TimeSeriesSplit ensures all training
    windows strictly precede their validation windows.
    """
    log.info("═"*55)
    log.info("  MODEL 1: DEMAND FORECASTING")
    log.info("═"*55)

    run = MLflowLogger("demand_forecasting", run_name="gbm_demand_v1")

    if df is None:
        df = _load_features("demand")

    # ── Feature prep ─────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in DEMAND_DROP]
    df_model     = df[feature_cols + ["daily_demand"]].fillna(0).copy()

    if "sale_date" in df.columns:
        df_model.index = pd.to_datetime(df["sale_date"])
        df_model       = df_model.sort_index()

    X = df_model[feature_cols].values.astype(float)
    y = df_model["daily_demand"].values.astype(float)

    run.log_params({
        "features":    len(feature_cols),
        "n_samples":   len(X),
        "target":      "daily_demand",
        "algorithm":   "GradientBoostingRegressor",
        **DEMAND_HYPERPARAMS,
    })
    log.info(f"  Features: {len(feature_cols)}  |  Samples: {len(X):,}")

    # ── TimeSeriesSplit CV ────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mae, cv_rmse, cv_mape = [], [], []
    log.info("  Cross-validation (3-fold time-series)…")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        m = GradientBoostingRegressor(**DEMAND_HYPERPARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict(X[val_idx]).clip(min=0)
        mae  = mean_absolute_error(y[val_idx], p)
        rmse = _rmse(y[val_idx], p)
        mape = _mape(y[val_idx], p)
        cv_mae.append(mae); cv_rmse.append(rmse); cv_mape.append(mape)
        log.info(f"    Fold {fold}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")

    cv_metrics = {
        "cv_mae_mean":  float(np.mean(cv_mae)),
        "cv_mae_std":   float(np.std(cv_mae)),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_mape_mean": float(np.mean(cv_mape)),
    }
    run.log_metrics(cv_metrics)

    # ── Final model on 80 / 20 time split ────────────────────
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    log.info(f"  Final fit  train={len(X_tr):,}  test={len(X_te):,}")

    model = GradientBoostingRegressor(**DEMAND_HYPERPARAMS)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te).clip(min=0)

    final = {
        "test_mae":  mean_absolute_error(y_te, preds),
        "test_rmse": _rmse(y_te, preds),
        "test_mape": _mape(y_te, preds),
        "test_r2":   r2_score(y_te, preds),
    }
    run.log_metrics(final)
    log.info(f"  Test  MAE={final['test_mae']:.3f}  RMSE={final['test_rmse']:.3f}  "
             f"MAPE={final['test_mape']:.2f}%  R²={final['test_r2']:.4f}")

    # ── Built-in feature importance ───────────────────────────
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Permutation importance ────────────────────────────────
    log.info("  Permutation importance…")
    perm_df = EX.permutation_importance(
        model, X_te, y_te, feature_cols, n_repeats=5, task="regression"
    )

    # ── SHAP approximation ────────────────────────────────────
    log.info("  SHAP approximation…")
    shap_df = EX.shap_approximation(
        model, X_te, feature_cols, n_samples=50, task="regression"
    )

    # ── Plots ─────────────────────────────────────────────────
    fig_imp  = EX.plot_feature_importance(
        imp_df, "Demand Forecasting — Feature Importance", top_n=20
    )
    fig_perm = EX.plot_feature_importance(
        perm_df, "Demand Forecasting — Permutation Importance", top_n=20
    )
    base_pred   = float(model.predict(X_te.mean(axis=0, keepdims=True))[0])
    sample_pred = float(model.predict(X_te[:1])[0])
    fig_shap = EX.plot_shap_waterfall(
        shap_df, prediction=sample_pred, baseline=base_pred,
        title="Demand Forecasting — SHAP Waterfall"
    )

    # Actual vs Predicted time-series
    fig_avp, ax = plt.subplots(figsize=(11, 4))
    n_show = min(300, len(y_te))
    ax.plot(y_te[:n_show],    color="#2166ac", lw=1.2, alpha=0.85, label="Actual")
    ax.plot(preds[:n_show],   color="#d73027", lw=1.2, alpha=0.85,
            linestyle="--", label="Predicted")
    ax.fill_between(range(n_show),
                    preds[:n_show] * 0.80, preds[:n_show] * 1.20,
                    alpha=0.10, color="#d73027", label="±20 % band")
    ax.set_title("Demand Forecasting — Actual vs Predicted (test set)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time step"); ax.set_ylabel("Daily Demand")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    fig_avp.tight_layout()

    for pname, fig in {
        "demand_feature_importance":      fig_imp,
        "demand_permutation_importance":  fig_perm,
        "demand_shap_waterfall":          fig_shap,
        "demand_actual_vs_predicted":     fig_avp,
    }.items():
        run.log_artifact(str(_save_plot(fig, pname)))

    _save_model(model, "demand_forecaster")
    run.log_model(model, "demand_forecaster")
    manifest = run.end_run()

    return {
        "model":         model,
        "feature_names": feature_cols,
        "metrics":       {**cv_metrics, **final},
        "importance_df": imp_df,
        "perm_df":       perm_df,
        "shap_df":       shap_df,
        "run_manifest":  manifest,
    }


# ═════════════════════════════════════════════════════════════
# MODEL 2 — STOCKOUT PREDICTION  (Random Forest Classifier)
# ═════════════════════════════════════════════════════════════

STOCKOUT_DROP = [
    "inventory_id", "product_id", "snapshot_date",
    "warehouse_id", "created_at",
    "stockout_in_horizon",   # target
    "future_min_stock",      # leakage: directly encodes the target
    "is_zero_stock",         # near-leakage: same-day zero stock
    "reorder_triggered",     # near-leakage: encodes reorder event
    "quantity_on_hand",      # prefer quantity_available
    "quantity_reserved",
]

STOCKOUT_HYPERPARAMS = {
    "n_estimators":     200,
    "max_depth":        8,
    "min_samples_leaf": 5,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}

STOCKOUT_THRESHOLD = 0.35    # tuned to favour recall over precision


def train_stockout_model(df: pd.DataFrame | None = None) -> dict:
    """
    Train a Random Forest binary classifier to predict whether a
    product will stock out within the next 7 days.

    class_weight='balanced' reweights the minority stockout class
    (stockout=1) to correct for the ~33 % base rate.
    Decision threshold 0.35 is intentionally below 0.5 to
    prioritise recall — missing a real stockout is more costly
    than a false alarm.
    """
    log.info("═"*55)
    log.info("  MODEL 2: STOCKOUT PREDICTION")
    log.info("═"*55)

    run = MLflowLogger("stockout_prediction", run_name="rf_stockout_v1")

    if df is None:
        df = _load_features("stockout")

    # ── Feature prep ─────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in STOCKOUT_DROP]
    df_model     = df[feature_cols + ["stockout_in_horizon"]].fillna(0).copy()

    for col in feature_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

    if "snapshot_date" in df.columns:
        df_model.index = pd.to_datetime(df["snapshot_date"])
        df_model       = df_model.sort_index()

    X = df_model[feature_cols].values.astype(float)
    y = df_model["stockout_in_horizon"].values.astype(int)

    pos_rate = y.mean()
    run.log_params({
        "features":    len(feature_cols),
        "n_samples":   len(X),
        "target":      "stockout_in_horizon",
        "algorithm":   "RandomForestClassifier",
        "pos_rate":    round(float(pos_rate), 4),
        "threshold":   STOCKOUT_THRESHOLD,
        **STOCKOUT_HYPERPARAMS,
    })
    log.info(f"  Features: {len(feature_cols)}  |  Samples: {len(X):,}  "
             f"|  Positive rate: {pos_rate:.1%}")

    # ── TimeSeriesSplit CV ────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    cv_aucs, cv_f1s = [], []
    log.info("  Cross-validation (3-fold time-series)…")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        if len(np.unique(y[val_idx])) < 2:
            log.info(f"    Fold {fold}: skipped (single class in val)")
            continue
        m = RandomForestClassifier(**STOCKOUT_HYPERPARAMS)
        m.fit(X[tr_idx], y[tr_idx])
        probs = m.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= STOCKOUT_THRESHOLD).astype(int)
        auc = _safe_roc_auc(y[val_idx], probs)
        f1  = f1_score(y[val_idx], preds, zero_division=0)
        cv_aucs.append(auc); cv_f1s.append(f1)
        log.info(f"    Fold {fold}  ROC-AUC={auc:.4f}  F1={f1:.4f}")

    cv_metrics = {
        "cv_roc_auc_mean": float(np.mean(cv_aucs)),
        "cv_roc_auc_std":  float(np.std(cv_aucs)),
        "cv_f1_mean":      float(np.mean(cv_f1s)),
    }
    run.log_metrics(cv_metrics)

    # ── Final model ───────────────────────────────────────────
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    log.info(f"  Final fit  train={len(X_tr):,}  test={len(X_te):,}")

    model = RandomForestClassifier(**STOCKOUT_HYPERPARAMS)
    model.fit(X_tr, y_tr)

    probs_te = model.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= STOCKOUT_THRESHOLD).astype(int)

    final = {
        "test_roc_auc":   _safe_roc_auc(y_te, probs_te),
        "test_f1":        f1_score(y_te, preds_te, zero_division=0),
        "test_precision": precision_score(y_te, preds_te, zero_division=0),
        "test_recall":    recall_score(y_te, preds_te, zero_division=0),
    }
    run.log_metrics(final)
    log.info(f"  Test  AUC={final['test_roc_auc']:.4f}  "
             f"F1={final['test_f1']:.4f}  "
             f"Prec={final['test_precision']:.4f}  "
             f"Rec={final['test_recall']:.4f}")

    # ── Importance & SHAP ─────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    log.info("  Permutation importance…")
    perm_df = EX.permutation_importance(
        model, X_te, y_te, feature_cols, n_repeats=5, task="classification"
    )
    log.info("  SHAP approximation…")
    shap_df = EX.shap_approximation(
        model, X_te, feature_cols, n_samples=50, task="classification"
    )

    # ── Plots ─────────────────────────────────────────────────
    fig_imp  = EX.plot_feature_importance(
        imp_df,  "Stockout Prediction — Feature Importance (RF)", top_n=20
    )
    fig_perm = EX.plot_feature_importance(
        perm_df, "Stockout Prediction — Permutation Importance", top_n=20
    )
    fig_shap = EX.plot_shap_waterfall(
        shap_df,
        prediction=float(probs_te[0]),
        baseline=float(probs_te.mean()),
        title="Stockout Prediction — SHAP Waterfall"
    )

    # ROC + Precision-Recall
    fig_diag, axes = plt.subplots(1, 2, figsize=(12, 4))
    fpr, tpr, _     = roc_curve(y_te, probs_te)
    axes[0].plot(fpr, tpr, "#d73027", lw=2,
                 label=f"AUC={final['test_roc_auc']:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    prec_a, rec_a, _ = precision_recall_curve(y_te, probs_te)
    axes[1].plot(rec_a, prec_a, "#4575b4", lw=2)
    axes[1].axvline(final["test_recall"], color="grey", lw=1, ls="--",
                    label=f"thr={STOCKOUT_THRESHOLD}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)
    fig_diag.suptitle("Stockout Prediction — Model Diagnostics",
                       fontweight="bold", fontsize=12)
    fig_diag.tight_layout()

    for pname, fig in {
        "stockout_feature_importance":    fig_imp,
        "stockout_permutation_importance":fig_perm,
        "stockout_shap_waterfall":        fig_shap,
        "stockout_roc_pr_curves":         fig_diag,
    }.items():
        run.log_artifact(str(_save_plot(fig, pname)))

    _save_model(model, "stockout_predictor")
    run.log_model(model, "stockout_predictor")
    manifest = run.end_run()

    return {
        "model":          model,
        "feature_names":  feature_cols,
        "metrics":        {**cv_metrics, **final},
        "importance_df":  imp_df,
        "perm_df":        perm_df,
        "shap_df":        shap_df,
        "threshold":      STOCKOUT_THRESHOLD,
        "run_manifest":   manifest,
    }


# ═════════════════════════════════════════════════════════════
# MODEL 3 — SUPPLIER RISK (GBM Regressor + Classifier)
# ═════════════════════════════════════════════════════════════

RISK_DROP = [
    "risk_id", "supplier_id", "scored_at", "year_month",
    "model_name", "model_version", "created_at",
    "recommended_action", "shap_top_features",
    "risk_tier", "country", "region",   # string columns
    # All potential targets (kept selectively per sub-model)
    "delay_probability", "avg_delay_days_pred",
    "composite_risk_score", "risk_tier_encoded",
]

RISK_REG_HYPERPARAMS = {
    "n_estimators":     150,
    "max_depth":        4,
    "learning_rate":    0.10,
    "subsample":        0.85,
    "min_samples_leaf": 2,
    "random_state":     42,
}

RISK_CLF_HYPERPARAMS = {**RISK_REG_HYPERPARAMS}


def _loso_cv(model_cls, hyperparams, X, y, supplier_ids, task="regression"):
    """Leave-One-Supplier-Out cross-validation."""
    scores = []
    for sid in np.unique(supplier_ids):
        tr = supplier_ids != sid
        te = supplier_ids == sid
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = model_cls(**hyperparams)
        m.fit(X[tr], y[tr])
        if task == "regression":
            scores.append(mean_absolute_error(y[te], m.predict(X[te])))
        else:
            if len(np.unique(y[te])) < 2:
                continue
            try:
                lb     = label_binarize(y[te], classes=np.unique(y[tr]))
                probs  = m.predict_proba(X[te])[:, :lb.shape[1]]
                scores.append(_safe_roc_auc(lb, probs, multi_class=True))
            except Exception:
                pass
    key = "mae" if task == "regression" else "roc_auc"
    return {
        f"loso_{key}_mean": round(float(np.mean(scores)), 6) if scores else 0.0,
        f"loso_{key}_std":  round(float(np.std(scores)),  6) if scores else 0.0,
        "loso_n_folds":     len(scores),
    }


def train_supplier_risk_model(df: pd.DataFrame | None = None) -> dict:
    """
    Train two complementary supplier risk models:

    3-A  GBM Regressor   → predict delay_probability (0–1)
    3-B  GBM Classifier  → predict risk_tier_encoded (0=low … 3=critical)

    Uses Leave-One-Supplier-Out CV, which is more informative than
    time-series CV for a small supplier set (evaluates generalisation
    to completely unseen suppliers).
    """
    log.info("═"*55)
    log.info("  MODEL 3: SUPPLIER RISK")
    log.info("═"*55)

    run = MLflowLogger("supplier_risk", run_name="gbm_supplier_risk_v1")

    if df is None:
        df = _load_features("supplier_risk")

    # ── Feature prep ─────────────────────────────────────────
    feature_cols = [
        c for c in df.columns
        if c not in RISK_DROP
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    targets   = ["delay_probability", "risk_tier_encoded", "supplier_id"]
    df_model  = df[feature_cols + targets].copy()
    df_model  = df_model.fillna(df_model.median(numeric_only=True))

    X            = df_model[feature_cols].values.astype(float)
    y_reg        = df_model["delay_probability"].values.astype(float)
    y_clf        = df_model["risk_tier_encoded"].values.astype(int)
    supplier_ids = df_model["supplier_id"].values

    run.log_params({
        "features":    len(feature_cols),
        "n_samples":   len(X),
        "n_suppliers": df["supplier_id"].nunique(),
        "algorithm":   "GradientBoosting (reg + clf)",
        **{f"reg_{k}": v for k, v in RISK_REG_HYPERPARAMS.items()},
    })
    log.info(f"  Features: {len(feature_cols)}  |  Samples: {len(X)}  "
             f"|  Suppliers: {df['supplier_id'].nunique()}")

    # ── 3-A: Delay regression ─────────────────────────────────
    log.info("  [3-A] Delay probability regression…")
    cv_reg = _loso_cv(GradientBoostingRegressor, RISK_REG_HYPERPARAMS,
                      X, y_reg, supplier_ids, task="regression")
    run.log_metrics(cv_reg)
    log.info(f"  LOSO CV  MAE={cv_reg['loso_mae_mean']:.4f}±"
             f"{cv_reg['loso_mae_std']:.4f}  ({cv_reg['loso_n_folds']} folds)")

    split    = max(int(len(X) * 0.80), len(X) - 20)
    X_tr, X_te   = X[:split], X[split:]
    yr_tr, yr_te = y_reg[:split], y_reg[split:]
    yc_tr, yc_te = y_clf[:split], y_clf[split:]

    model_reg = GradientBoostingRegressor(**RISK_REG_HYPERPARAMS)
    model_reg.fit(X_tr, yr_tr)
    preds_reg = model_reg.predict(X_te).clip(0, 1)

    metrics_reg = {
        "test_mae_reg":  mean_absolute_error(yr_te, preds_reg),
        "test_rmse_reg": _rmse(yr_te, preds_reg),
        "test_r2_reg":   r2_score(yr_te, preds_reg),
        **cv_reg,
    }
    run.log_metrics(metrics_reg)
    log.info(f"  [3-A] Test  MAE={metrics_reg['test_mae_reg']:.4f}  "
             f"R²={metrics_reg['test_r2_reg']:.4f}")

    # ── 3-B: Risk tier classification ────────────────────────
    log.info("  [3-B] Risk tier classification…")
    cv_clf = _loso_cv(GradientBoostingClassifier, RISK_CLF_HYPERPARAMS,
                      X, y_clf, supplier_ids, task="classification")
    run.log_metrics({f"clf_{k}": v for k, v in cv_clf.items()})

    model_clf  = GradientBoostingClassifier(**RISK_CLF_HYPERPARAMS)
    model_clf.fit(X_tr, yc_tr)
    preds_clf  = model_clf.predict(X_te)
    probs_clf  = model_clf.predict_proba(X_te)

    n_classes  = len(np.unique(y_clf))
    macro_f1   = f1_score(yc_te, preds_clf, average="macro", zero_division=0)
    try:
        lb       = label_binarize(yc_te, classes=np.arange(n_classes))
        auc_clf  = _safe_roc_auc(lb, probs_clf[:, :lb.shape[1]], multi_class=True)
    except Exception:
        auc_clf  = 0.0

    metrics_clf = {
        "test_macro_f1_clf": macro_f1,
        "test_roc_auc_clf":  auc_clf,
        **{f"clf_{k}": v for k, v in cv_clf.items()},
    }
    run.log_metrics(metrics_clf)
    log.info(f"  [3-B] Test  Macro-F1={macro_f1:.4f}  ROC-AUC={auc_clf:.4f}")

    # ── Importance & SHAP ─────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature":       feature_cols,
        "importance_reg":model_reg.feature_importances_,
        "importance_clf":model_clf.feature_importances_,
    })
    imp_df["importance"] = (imp_df["importance_reg"] +
                             imp_df["importance_clf"]) / 2
    imp_df = imp_df.sort_values("importance", ascending=False
                                 ).reset_index(drop=True)

    log.info("  Permutation importance…")
    perm_df = EX.permutation_importance(
        model_reg, X_te, yr_te, feature_cols, n_repeats=5, task="regression"
    )
    log.info("  SHAP approximations…")
    shap_df_reg = EX.shap_approximation(
        model_reg, X_te, feature_cols, n_samples=30, task="regression"
    )
    shap_df_clf = EX.shap_approximation(
        model_clf, X_te, feature_cols, n_samples=30, task="classification"
    )

    # ── Plots ─────────────────────────────────────────────────
    fig_imp  = EX.plot_feature_importance(
        imp_df, "Supplier Risk — Feature Importance (reg + clf)", top_n=18
    )
    fig_perm = EX.plot_feature_importance(
        perm_df, "Supplier Risk — Permutation Importance (regression)", top_n=18
    )
    fig_shap_r = EX.plot_shap_waterfall(
        shap_df_reg,
        prediction=float(model_reg.predict(X_te[:1])[0]),
        baseline=float(preds_reg.mean()),
        title="Supplier Risk — SHAP Waterfall: Delay Probability"
    )
    fig_shap_c = EX.plot_shap_waterfall(
        shap_df_clf,
        prediction=float(probs_clf[0].max()),
        baseline=float(probs_clf[:, -1].mean()),
        title="Supplier Risk — SHAP Waterfall: Risk Tier"
    )

    # Scatter: actual vs predicted delay probability
    fig_sc, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yr_te, preds_reg, alpha=0.75, edgecolors="none",
               color="#4575b4", s=70)
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect fit")
    ax.set(xlabel="Actual Delay Probability",
           ylabel="Predicted",
           title="Supplier Risk — Delay Prob: Actual vs Predicted")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    fig_sc.tight_layout()

    # Confusion matrix for risk tier
    tier_labels = ["low","medium","high","critical"][:n_classes]
    cm  = confusion_matrix(yc_te, preds_clf,
                            labels=list(range(n_classes)))
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(tier_labels, fontsize=9)
    ax.set(xlabel="Predicted", ylabel="Actual",
           title="Risk Tier — Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=11,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    fig_cm.tight_layout()

    for pname, fig in {
        "risk_feature_importance":    fig_imp,
        "risk_permutation_importance":fig_perm,
        "risk_shap_waterfall_reg":    fig_shap_r,
        "risk_shap_waterfall_clf":    fig_shap_c,
        "risk_delay_scatter":         fig_sc,
        "risk_tier_confusion_matrix": fig_cm,
    }.items():
        run.log_artifact(str(_save_plot(fig, pname)))

    _save_model(model_reg, "supplier_risk_regressor")
    _save_model(model_clf, "supplier_risk_classifier")
    run.log_model(model_reg, "supplier_risk_regressor")
    run.log_model(model_clf, "supplier_risk_classifier")
    manifest = run.end_run()

    return {
        "model_reg":      model_reg,
        "model_clf":      model_clf,
        "feature_names":  feature_cols,
        "metrics_reg":    metrics_reg,
        "metrics_clf":    metrics_clf,
        "importance_df":  imp_df,
        "perm_df":        perm_df,
        "shap_df_reg":    shap_df_reg,
        "shap_df_clf":    shap_df_clf,
        "run_manifest":   manifest,
    }


# ═════════════════════════════════════════════════════════════
# INFERENCE HELPERS  (called by FastAPI in Section 4)
# ═════════════════════════════════════════════════════════════

def predict_demand(
    product_id: int,
    feature_row: dict[str, float],
    model: Any | None = None,
    baseline: dict[str, float] | None = None,
) -> dict:
    """
    Single-row demand prediction with 95 % prediction interval.
    PI estimated from variance across GBM stage predictions.

    `baseline` should be the training-set mean feature vector
    (see models/<name>_baseline.json) — without it, SHAP values
    for a single row collapse to zero.
    """
    if model is None:
        model = load_model("demand_forecaster")
    names = list(feature_row.keys())
    X     = np.array([list(feature_row.values())], dtype=float)
    pred  = max(0.0, float(model.predict(X)[0]))
    stage = np.array([t.predict(X)[0] for t in model.estimators_.ravel()])
    std   = float(np.std(stage)) * 1.96

    baseline_arr = (np.array([baseline.get(n, 0.0) for n in names], dtype=float)
                    if baseline else None)
    shap  = EX.shap_approximation(model, X, names, baseline=baseline_arr,
                                   n_samples=1, task="regression")
    return {
        "product_id":       product_id,
        "predicted_demand": round(pred, 2),
        "lower_bound_95":   round(max(0.0, pred - std), 2),
        "upper_bound_95":   round(pred + std, 2),
        "top_features":     shap.head(5)[["feature","mean_shap"]].to_dict("records"),
    }


def predict_stockout(
    product_id: int,
    feature_row: dict[str, float],
    model: Any | None = None,
    threshold: float = STOCKOUT_THRESHOLD,
    baseline: dict[str, float] | None = None,
) -> dict:
    """Single-row stockout probability with risk level label."""
    if model is None:
        model = load_model("stockout_predictor")
    names = list(feature_row.keys())
    X     = np.array([list(feature_row.values())], dtype=float)
    prob  = float(model.predict_proba(X)[0, 1])
    level = ("CRITICAL" if prob >= 0.70 else
             "HIGH"     if prob >= threshold else
             "MEDIUM"   if prob >= 0.20 else "LOW")
    baseline_arr = (np.array([baseline.get(n, 0.0) for n in names], dtype=float)
                    if baseline else None)
    shap  = EX.shap_approximation(model, X, names, baseline=baseline_arr,
                                   n_samples=1, task="classification")
    return {
        "product_id":           product_id,
        "stockout_probability": round(prob, 4),
        "stockout_flag":        prob >= threshold,
        "risk_level":           level,
        "top_features":         shap.head(5)[["feature","mean_shap"]].to_dict("records"),
    }


def predict_supplier_risk(
    supplier_id: int,
    feature_row: dict[str, float],
    model_reg: Any | None = None,
    model_clf: Any | None = None,
    baseline: dict[str, float] | None = None,
) -> dict:
    """Supplier delay probability + risk tier + recommended action."""
    if model_reg is None:
        model_reg = load_model("supplier_risk_regressor")
    if model_clf is None:
        model_clf = load_model("supplier_risk_classifier")
    names      = list(feature_row.keys())
    X          = np.array([list(feature_row.values())], dtype=float)
    delay_prob = float(np.clip(model_reg.predict(X)[0], 0, 1))
    tier_enc   = int(model_clf.predict(X)[0])
    tier       = {0:"low", 1:"medium", 2:"high", 3:"critical"}.get(tier_enc, "medium")
    composite  = round(delay_prob * 40 + tier_enc / 3 * 60, 2)
    actions    = {
        "low":      "No action required; continue standard ordering.",
        "medium":   "Monitor delivery windows; consider dual-sourcing.",
        "high":     "Increase safety stock; request SLA commitment.",
        "critical": "Escalate to procurement; initiate alternative sourcing.",
    }
    baseline_arr = (np.array([baseline.get(n, 0.0) for n in names], dtype=float)
                    if baseline else None)
    shap = EX.shap_approximation(model_reg, X, names, baseline=baseline_arr,
                                  n_samples=1, task="regression")
    return {
        "supplier_id":        supplier_id,
        "delay_probability":  round(delay_prob, 4),
        "risk_tier":          tier,
        "composite_score":    composite,
        "recommended_action": actions[tier],
        "top_features":       shap.head(5)[["feature","mean_shap"]].to_dict("records"),
    }


# ═════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═════════════════════════════════════════════════════════════

def run_all_models() -> dict[str, dict]:
    """Train all three models and persist artifacts."""
    log.info("\n" + "█"*55)
    log.info("  SUPPLY CHAIN ML PLATFORM — FULL TRAINING RUN")
    log.info("█"*55)
    t0 = time.time()

    results = {
        "demand":        train_demand_model(),
        "stockout":      train_stockout_model(),
        "supplier_risk": train_supplier_risk_model(),
    }

    elapsed = round(time.time() - t0, 1)
    m1 = results["demand"]["metrics"]
    m2 = results["stockout"]["metrics"]
    m3r = results["supplier_risk"]["metrics_reg"]
    m3c = results["supplier_risk"]["metrics_clf"]

    log.info("\n" + "═"*55)
    log.info("  TRAINING COMPLETE")
    log.info("═"*55)
    log.info(f"  [M1 Demand]     MAE={m1['test_mae']:.3f}  "
             f"RMSE={m1['test_rmse']:.3f}  "
             f"MAPE={m1['test_mape']:.2f}%  R²={m1['test_r2']:.4f}")
    log.info(f"  [M2 Stockout]   AUC={m2['test_roc_auc']:.4f}  "
             f"F1={m2['test_f1']:.4f}  "
             f"Prec={m2['test_precision']:.4f}  Rec={m2['test_recall']:.4f}")
    log.info(f"  [M3 Risk-Reg]   MAE={m3r['test_mae_reg']:.4f}  "
             f"R²={m3r['test_r2_reg']:.4f}")
    log.info(f"  [M3 Risk-Clf]   Macro-F1={m3c['test_macro_f1_clf']:.4f}  "
             f"AUC={m3c['test_roc_auc_clf']:.4f}")
    log.info(f"\n  Total: {elapsed}s  |  "
             f"Models → {MODELS_DIR}  |  "
             f"Plots → {PLOTS_DIR}")
    log.info("═"*55)
    return results


if __name__ == "__main__":
    run_all_models()