"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
MLflow Tracking Integration  |  Section 4-A
=============================================================
This module shows the PRODUCTION MLflow integration: wrapping the
three training loops from models.py (Section 3) with real
mlflow.start_run() context managers, logging params/metrics/models,
and registering models in the MLflow Model Registry.

In Section 3, `MLflowLogger` is a lightweight JSON-based stand-in
that requires no extra dependencies. This file shows the exact
1:1 replacement once `pip install mlflow` is available.

─────────────────────────────────────────────────────────────
SETUP
─────────────────────────────────────────────────────────────
    pip install mlflow

    # Start a local tracking server (separate terminal):
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlartifacts \
        --host 0.0.0.0 --port 5000

    # Then point this script at it:
    export MLFLOW_TRACKING_URI=http://localhost:5000

─────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────
    python mlflow_tracking.py               # trains + logs all 3 models
    python mlflow_tracking.py demand        # only the demand model
    python mlflow_tracking.py stockout      # only the stockout model
    python mlflow_tracking.py supplier_risk # only the supplier risk models

Then browse results at http://localhost:5000
=============================================================
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize

# ── Reuse Section 3 building blocks ───────────────────────────
from models import (
    DEMAND_DROP,
    DEMAND_HYPERPARAMS,
    EX,                       # ExplainabilityEngine (SHAP + permutation)
    RISK_CLF_HYPERPARAMS,
    RISK_DROP,
    RISK_REG_HYPERPARAMS,
    STOCKOUT_DROP,
    STOCKOUT_HYPERPARAMS,
    STOCKOUT_THRESHOLD,
    _loso_cv,
    _mape,
    _rmse,
    _safe_roc_auc,
    _save_plot,
    _load_features,
    _save_model,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlflow_tracking")


# ─────────────────────────────────────────────────────────────
# MLFLOW SETUP
# ─────────────────────────────────────────────────────────────

def _init_mlflow():
    """
    Initialise MLflow and return the module.

    Raises a clear, actionable error if MLflow is not installed.
    Section 3 (models.py) provides a dependency-free MLflowLogger
    stand-in for environments where installing mlflow isn't possible.
    """
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            "MLflow is required for this module.\n"
            "Install it with:  pip install mlflow\n"
            "Section 3 (models.py) provides a dependency-free "
            "MLflowLogger stand-in if you prefer not to install MLflow."
        ) from e

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns_mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"MLflow tracking URI: {tracking_uri}")
    return mlflow


# ═════════════════════════════════════════════════════════════
# MODEL 1 — DEMAND FORECASTING (real mlflow.start_run)
# ═════════════════════════════════════════════════════════════

def train_demand_with_mlflow(df: pd.DataFrame | None = None) -> dict:
    """
    Trains the demand forecasting GBM and logs everything to MLflow:
      - params:    hyperparameters + dataset shape
      - metrics:   CV scores (per-fold + aggregate) and test metrics
      - artifacts: feature importance / SHAP plots
      - model:     registered as 'demand_forecaster' in the Model Registry
    """
    mlflow = _init_mlflow()
    mlflow.set_experiment("demand_forecasting")

    if df is None:
        df = _load_features("demand")

    feature_cols = [c for c in df.columns if c not in DEMAND_DROP]
    df_model     = df[feature_cols + ["daily_demand"]].fillna(0).copy()
    if "sale_date" in df.columns:
        df_model.index = pd.to_datetime(df["sale_date"])
        df_model       = df_model.sort_index()

    X = df_model[feature_cols].values.astype(float)
    y = df_model["daily_demand"].values.astype(float)

    with mlflow.start_run(run_name="gbm_demand_v1") as run:
        log.info(f"MLflow run_id: {run.info.run_id}")

        # ── 1. Log params ──────────────────────────────────────
        mlflow.log_params({
            "algorithm":  "GradientBoostingRegressor",
            "features":   len(feature_cols),
            "n_samples":  len(X),
            "target":     "daily_demand",
            **DEMAND_HYPERPARAMS,
        })

        # ── 2. Cross-validation, logged per fold via `step=` ───
        tscv = TimeSeriesSplit(n_splits=3)
        cv_mae, cv_rmse, cv_mape = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            m = GradientBoostingRegressor(**DEMAND_HYPERPARAMS)
            m.fit(X[tr_idx], y[tr_idx])
            p = m.predict(X[val_idx]).clip(min=0)

            mae, rmse, mape = (
                mean_absolute_error(y[val_idx], p),
                _rmse(y[val_idx], p),
                _mape(y[val_idx], p),
            )
            cv_mae.append(mae); cv_rmse.append(rmse); cv_mape.append(mape)

            mlflow.log_metrics(
                {"cv_mae": mae, "cv_rmse": rmse, "cv_mape": mape}, step=fold
            )

        mlflow.log_metrics({
            "cv_mae_mean":  float(np.mean(cv_mae)),
            "cv_rmse_mean": float(np.mean(cv_rmse)),
            "cv_mape_mean": float(np.mean(cv_mape)),
        })

        # ── 3. Final model on 80/20 split ──────────────────────
        split = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = GradientBoostingRegressor(**DEMAND_HYPERPARAMS)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te).clip(min=0)

        test_metrics = {
            "test_mae":  mean_absolute_error(y_te, preds),
            "test_rmse": _rmse(y_te, preds),
            "test_mape": _mape(y_te, preds),
            "test_r2":   r2_score(y_te, preds),
        }
        mlflow.log_metrics(test_metrics)
        log.info(f"Test metrics: {test_metrics}")

        # ── 4. Explainability artifacts ────────────────────────
        imp_df  = pd.DataFrame({
            "feature": feature_cols, "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        shap_df = EX.shap_approximation(model, X_te, feature_cols,
                                         n_samples=50, task="regression")

        fig_imp  = EX.plot_feature_importance(
            imp_df, "Demand Forecasting — Feature Importance", top_n=20
        )
        fig_shap = EX.plot_shap_waterfall(
            shap_df,
            prediction=float(model.predict(X_te[:1])[0]),
            baseline=float(model.predict(X_te.mean(axis=0, keepdims=True))[0]),
            title="Demand Forecasting — SHAP Waterfall",
        )
        mlflow.log_artifact(str(_save_plot(fig_imp,  "mlflow_demand_feature_importance")))
        mlflow.log_artifact(str(_save_plot(fig_shap, "mlflow_demand_shap_waterfall")))

        # ── 5. Log + register model ────────────────────────────
        # mlflow.sklearn.log_model handles serialisation, conda env,
        # and model signature inference automatically.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="demand_forecaster",
            registered_model_name="demand_forecaster",
        )

        # Also save locally for the FastAPI app (Section 4-B) to load directly
        _save_model(model, "demand_forecaster")

        mlflow.set_tags({"model_type": "demand_forecasting", "stage": "staging"})

        log.info(f"Demand model logged & registered  (run_id={run.info.run_id})")
        return {"model": model, "run_id": run.info.run_id, "metrics": test_metrics}


# ═════════════════════════════════════════════════════════════
# MODEL 2 — STOCKOUT PREDICTION (real mlflow.start_run)
# ═════════════════════════════════════════════════════════════

def train_stockout_with_mlflow(df: pd.DataFrame | None = None) -> dict:
    """
    Trains the stockout RandomForestClassifier and logs:
      - params:    hyperparameters, class balance, decision threshold
      - metrics:   ROC-AUC, F1, precision, recall (CV + test)
      - artifacts: feature importance, SHAP waterfall
      - model:     registered as 'stockout_predictor'
    """
    mlflow = _init_mlflow()
    mlflow.set_experiment("stockout_prediction")

    if df is None:
        df = _load_features("stockout")

    feature_cols = [c for c in df.columns if c not in STOCKOUT_DROP]
    df_model     = df[feature_cols + ["stockout_in_horizon"]].fillna(0).copy()
    for col in feature_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)
    if "snapshot_date" in df.columns:
        df_model.index = pd.to_datetime(df["snapshot_date"])
        df_model       = df_model.sort_index()

    X = df_model[feature_cols].values.astype(float)
    y = df_model["stockout_in_horizon"].values.astype(int)

    with mlflow.start_run(run_name="rf_stockout_v1") as run:
        log.info(f"MLflow run_id: {run.info.run_id}")

        mlflow.log_params({
            "algorithm":  "RandomForestClassifier",
            "features":   len(feature_cols),
            "n_samples":  len(X),
            "pos_rate":   round(float(y.mean()), 4),
            "threshold":  STOCKOUT_THRESHOLD,
            **STOCKOUT_HYPERPARAMS,
        })

        # ── Cross-validation, logged per fold ──────────────────
        tscv = TimeSeriesSplit(n_splits=3)
        cv_aucs, cv_f1s = [], []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            if len(np.unique(y[val_idx])) < 2:
                continue
            m = RandomForestClassifier(**STOCKOUT_HYPERPARAMS)
            m.fit(X[tr_idx], y[tr_idx])
            probs = m.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= STOCKOUT_THRESHOLD).astype(int)
            auc = _safe_roc_auc(y[val_idx], probs)
            f1  = f1_score(y[val_idx], preds, zero_division=0)
            cv_aucs.append(auc); cv_f1s.append(f1)
            mlflow.log_metrics({"cv_roc_auc": auc, "cv_f1": f1}, step=fold)

        mlflow.log_metrics({
            "cv_roc_auc_mean": float(np.mean(cv_aucs)),
            "cv_f1_mean":      float(np.mean(cv_f1s)),
        })

        # ── Final model ───────────────────────────────────────
        split = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = RandomForestClassifier(**STOCKOUT_HYPERPARAMS)
        model.fit(X_tr, y_tr)
        probs_te = model.predict_proba(X_te)[:, 1]
        preds_te = (probs_te >= STOCKOUT_THRESHOLD).astype(int)

        test_metrics = {
            "test_roc_auc":   _safe_roc_auc(y_te, probs_te),
            "test_f1":        f1_score(y_te, preds_te, zero_division=0),
            "test_precision": precision_score(y_te, preds_te, zero_division=0),
            "test_recall":    recall_score(y_te, preds_te, zero_division=0),
        }
        mlflow.log_metrics(test_metrics)
        log.info(f"Test metrics: {test_metrics}")

        # ── Explainability ───────────────────────────────────
        imp_df = pd.DataFrame({
            "feature": feature_cols, "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        shap_df = EX.shap_approximation(model, X_te, feature_cols,
                                         n_samples=50, task="classification")

        fig_imp  = EX.plot_feature_importance(
            imp_df, "Stockout Prediction — Feature Importance", top_n=20
        )
        fig_shap = EX.plot_shap_waterfall(
            shap_df,
            prediction=float(probs_te[0]),
            baseline=float(probs_te.mean()),
            title="Stockout Prediction — SHAP Waterfall",
        )
        mlflow.log_artifact(str(_save_plot(fig_imp,  "mlflow_stockout_feature_importance")))
        mlflow.log_artifact(str(_save_plot(fig_shap, "mlflow_stockout_shap_waterfall")))

        # ── Log + register model ─────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="stockout_predictor",
            registered_model_name="stockout_predictor",
        )
        _save_model(model, "stockout_predictor")

        mlflow.set_tags({
            "model_type": "stockout_classification",
            "stage":      "staging",
            "decision_threshold": str(STOCKOUT_THRESHOLD),
        })

        log.info(f"Stockout model logged & registered  (run_id={run.info.run_id})")
        return {"model": model, "run_id": run.info.run_id, "metrics": test_metrics}


# ═════════════════════════════════════════════════════════════
# MODEL 3 — SUPPLIER RISK (real mlflow.start_run, nested children)
# ═════════════════════════════════════════════════════════════

def train_supplier_risk_with_mlflow(df: pd.DataFrame | None = None) -> dict:
    """
    Trains both supplier risk sub-models inside a PARENT MLflow run,
    with each sub-model logged as a NESTED child run:

      Parent run: 'supplier_risk_v1'
        +-- child run: 'delay_regressor'  -> registers 'supplier_risk_regressor'
        +-- child run: 'risk_classifier'  -> registers 'supplier_risk_classifier'
    """
    mlflow = _init_mlflow()
    mlflow.set_experiment("supplier_risk")

    if df is None:
        df = _load_features("supplier_risk")

    feature_cols = [
        c for c in df.columns
        if c not in RISK_DROP and pd.api.types.is_numeric_dtype(df[c])
    ]
    df_model = df[feature_cols + ["delay_probability", "risk_tier_encoded",
                                   "supplier_id"]].copy()
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    X            = df_model[feature_cols].values.astype(float)
    y_reg        = df_model["delay_probability"].values.astype(float)
    y_clf        = df_model["risk_tier_encoded"].values.astype(int)
    supplier_ids = df_model["supplier_id"].values

    split = max(int(len(X) * 0.80), len(X) - 20)
    X_tr, X_te   = X[:split], X[split:]
    yr_tr, yr_te = y_reg[:split], y_reg[split:]
    yc_tr, yc_te = y_clf[:split], y_clf[split:]

    results = {}

    with mlflow.start_run(run_name="supplier_risk_v1") as parent_run:
        log.info(f"MLflow parent run_id: {parent_run.info.run_id}")
        mlflow.log_params({
            "features":    len(feature_cols),
            "n_samples":   len(X),
            "n_suppliers": df["supplier_id"].nunique(),
        })

        # ── CHILD RUN 1: Delay regression ───────────────────────
        with mlflow.start_run(run_name="delay_regressor", nested=True) as child:
            log.info(f"  MLflow child run_id: {child.info.run_id}")
            mlflow.log_params({
                "algorithm": "GradientBoostingRegressor",
                **RISK_REG_HYPERPARAMS,
            })

            cv_reg = _loso_cv(GradientBoostingRegressor, RISK_REG_HYPERPARAMS,
                              X, y_reg, supplier_ids, task="regression")
            mlflow.log_metrics(cv_reg)

            model_reg = GradientBoostingRegressor(**RISK_REG_HYPERPARAMS)
            model_reg.fit(X_tr, yr_tr)
            preds_reg = model_reg.predict(X_te).clip(0, 1)

            reg_metrics = {
                "test_mae":  mean_absolute_error(yr_te, preds_reg),
                "test_rmse": _rmse(yr_te, preds_reg),
                "test_r2":   r2_score(yr_te, preds_reg),
            }
            mlflow.log_metrics(reg_metrics)
            log.info(f"  Regression test metrics: {reg_metrics}")

            mlflow.sklearn.log_model(
                sk_model=model_reg,
                artifact_path="supplier_risk_regressor",
                registered_model_name="supplier_risk_regressor",
            )
            _save_model(model_reg, "supplier_risk_regressor")
            results["model_reg"]  = model_reg
            results["reg_run_id"] = child.info.run_id
            results["reg_metrics"]= reg_metrics

        # ── CHILD RUN 2: Risk tier classification ───────────────
        with mlflow.start_run(run_name="risk_classifier", nested=True) as child:
            log.info(f"  MLflow child run_id: {child.info.run_id}")
            mlflow.log_params({
                "algorithm": "GradientBoostingClassifier",
                "n_classes": int(len(np.unique(y_clf))),
                **RISK_CLF_HYPERPARAMS,
            })

            cv_clf = _loso_cv(GradientBoostingClassifier, RISK_CLF_HYPERPARAMS,
                              X, y_clf, supplier_ids, task="classification")
            mlflow.log_metrics(cv_clf)

            model_clf = GradientBoostingClassifier(**RISK_CLF_HYPERPARAMS)
            model_clf.fit(X_tr, yc_tr)
            preds_clf = model_clf.predict(X_te)
            probs_clf = model_clf.predict_proba(X_te)

            n_classes = len(np.unique(y_clf))
            macro_f1  = f1_score(yc_te, preds_clf, average="macro", zero_division=0)
            try:
                lb      = label_binarize(yc_te, classes=np.arange(n_classes))
                auc_clf = _safe_roc_auc(lb, probs_clf[:, :lb.shape[1]], multi_class=True)
            except Exception:
                auc_clf = 0.0

            clf_metrics = {"test_macro_f1": macro_f1, "test_roc_auc": auc_clf}
            mlflow.log_metrics(clf_metrics)
            log.info(f"  Classification test metrics: {clf_metrics}")

            mlflow.sklearn.log_model(
                sk_model=model_clf,
                artifact_path="supplier_risk_classifier",
                registered_model_name="supplier_risk_classifier",
            )
            _save_model(model_clf, "supplier_risk_classifier")
            results["model_clf"]  = model_clf
            results["clf_run_id"] = child.info.run_id
            results["clf_metrics"]= clf_metrics

        # ── SHAP artifact on the PARENT run ─────────────────────
        shap_df_reg = EX.shap_approximation(
            results["model_reg"], X_te, feature_cols, n_samples=30, task="regression"
        )
        fig_shap = EX.plot_shap_waterfall(
            shap_df_reg,
            prediction=float(results["model_reg"].predict(X_te[:1])[0]),
            baseline=float(preds_reg.mean()),
            title="Supplier Risk — SHAP Waterfall: Delay Probability",
        )
        mlflow.log_artifact(str(_save_plot(fig_shap, "mlflow_supplier_risk_shap")))

        mlflow.set_tags({"model_type": "supplier_risk", "stage": "staging"})
        log.info(f"Supplier risk models logged & registered "
                 f"(parent run_id={parent_run.info.run_id})")

    return results


# ═════════════════════════════════════════════════════════════
# MODEL PROMOTION — Staging → Production alias
# ═════════════════════════════════════════════════════════════

def promote_model_to_production(model_name: str, version: int | None = None) -> None:
    """
    Promote a registered model version to the 'Production' alias.

    Example
    -------
        promote_model_to_production("demand_forecaster")
        # promotes the LATEST version, or pass version=3 for a specific one

    In FastAPI (Section 4-B), load the production model with:

        import mlflow
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    """
    mlflow = _init_mlflow()
    client = mlflow.MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        version  = max(int(v.version) for v in versions)

    client.set_registered_model_alias(model_name, "Production", version)
    log.info(f"{model_name} v{version} promoted to 'Production' alias")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    log.info("="*55)
    log.info("  MLflow Tracking — Supply Chain ML Platform")
    log.info("="*55)

    if target in ("all", "demand"):
        train_demand_with_mlflow()
    if target in ("all", "stockout"):
        train_stockout_with_mlflow()
    if target in ("all", "supplier_risk"):
        train_supplier_risk_with_mlflow()

    log.info("="*55)
    log.info("  Done. Launch the MLflow UI to inspect runs:")
    log.info("    mlflow ui --backend-store-uri ./mlruns_mlflow")
    log.info("="*55)