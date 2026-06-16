"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
FastAPI Backend  |  Section 4-B
=============================================================
Production-style API exposing four endpoints:

    GET  /health                 — liveness / readiness probe
    POST /forecast                — demand forecast for a product
    POST /stockout                — stockout probability for a product
    POST /supplier-risk           — supplier delay risk score
    POST /ask-ai                  — AI Copilot natural-language Q&A

Design notes
------------
- Models are loaded ONCE at startup (lifespan handler) and kept
  in memory — avoids per-request pickle deserialisation cost.
- A lightweight in-memory "feature store" is built at startup by
  taking the MOST RECENT feature row per product_id / supplier_id
  from the Section-2 feature CSV files. In production this would
  instead be a low-latency feature store (Feast, Redis, or a
  materialised PostgreSQL view).
- All prediction logic (model.predict, SHAP) is delegated to
  models.py (Section 3) — this file is purely the API/transport layer.
- /ask-ai is a structured placeholder: it routes natural-language
  prompts to the appropriate model + templated business narrative.
  Swap the `_generate_ai_response()` body for a real LLM call.

Run locally
-----------
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive OpenAPI docs.

Requirements: fastapi, uvicorn, pydantic>=2, pandas, numpy,
              scikit-learn, matplotlib (via models.py)
=============================================================
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from pytz import timezone

import models as ml   # Section 3 — train_*, predict_*, load_model, EX

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FEATURES_DIR = Path("./data/features")
API_KEY_NAME = "X-API-Key"

# In production, load from environment variables / secrets manager:
#   import os
#   VALID_API_KEYS = set(os.environ.get("SUPPLYCHAIN_API_KEYS", "").split(","))
VALID_API_KEYS = {"dev-local-key-12345", "demo-key-supplychain"}

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Simple header-based API key auth.
    Swap for OAuth2 / JWT in production (fastapi.security.OAuth2PasswordBearer).
    """
    if api_key is None or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it via the "
                   f"'{API_KEY_NAME}' header.",
        )
    return api_key


# ─────────────────────────────────────────────────────────────
# IN-MEMORY MODEL & FEATURE-STORE REGISTRY
# Populated once at startup via the lifespan handler below.
# ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """Holds loaded models + latest feature rows for fast inference."""

    def __init__(self) -> None:
        self.demand_model:   object | None = None
        self.stockout_model: object | None = None
        self.risk_reg_model: object | None = None
        self.risk_clf_model: object | None = None

        self.latest_demand_features:   dict[int, dict] = {}
        self.latest_stockout_features: dict[int, dict] = {}
        self.latest_risk_features:     dict[int, dict] = {}

        self.demand_feature_cols:   list[str] = []
        self.stockout_feature_cols: list[str] = []
        self.risk_feature_cols:     list[str] = []

        self.loaded_at: str | None = None

    def load_all(self) -> None:
        t0 = time.time()
        log.info("Loading models from disk…")
        self.demand_model   = ml.load_model("demand_forecaster")
        self.stockout_model = ml.load_model("stockout_predictor")
        self.risk_reg_model = ml.load_model("supplier_risk_regressor")
        self.risk_clf_model = ml.load_model("supplier_risk_classifier")
        log.info("  ✓ All 4 models loaded")

        log.info("Building in-memory feature store (latest row per entity)…")
        self._load_demand_features()
        self._load_stockout_features()
        self._load_risk_features()

        self.loaded_at = datetime.utcnow().isoformat() + "Z"
        log.info(f"Model registry ready in {time.time()-t0:.2f}s "
                 f"({len(self.latest_demand_features)} products / demand, "
                 f"{len(self.latest_stockout_features)} products / stockout, "
                 f"{len(self.latest_risk_features)} suppliers / risk)")

    # ── Feature loaders ────────────────────────────────────────
    def _load_demand_features(self) -> None:
        df = pd.read_csv(FEATURES_DIR / "features_demand.csv", low_memory=False)
        self.demand_feature_cols = [
            c for c in df.columns if c not in ml.DEMAND_DROP
        ]
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        latest = df.sort_values("sale_date").groupby("product_id").tail(1)
        for _, row in latest.iterrows():
            pid = int(row["product_id"])
            self.latest_demand_features[pid] = {
                c: float(row[c]) for c in self.demand_feature_cols
            }

    def _load_stockout_features(self) -> None:
        df = pd.read_csv(FEATURES_DIR / "features_stockout.csv", low_memory=False)
        self.stockout_feature_cols = [
            c for c in df.columns if c not in ml.STOCKOUT_DROP
        ]
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        latest = df.sort_values("snapshot_date").groupby("product_id").tail(1)
        for _, row in latest.iterrows():
            pid = int(row["product_id"])
            self.latest_stockout_features[pid] = {
                c: float(pd.to_numeric(row[c], errors="coerce") or 0.0)
                for c in self.stockout_feature_cols
            }

    def _load_risk_features(self) -> None:
        df = pd.read_csv(FEATURES_DIR / "features_supplier_risk.csv", low_memory=False)
        self.risk_feature_cols = [
            c for c in df.columns
            if c not in ml.RISK_DROP and pd.api.types.is_numeric_dtype(df[c])
        ]
        df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
        df_filled = df.fillna(df.median(numeric_only=True))
        latest_idx = df.sort_values("scored_at").groupby("supplier_id").tail(1).index
        latest = df_filled.loc[latest_idx]
        for _, row in latest.iterrows():
            sid = int(row["supplier_id"])
            self.latest_risk_features[sid] = {
                c: float(row[c]) for c in self.risk_feature_cols
            }


registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────
# LIFESPAN — load models once at startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═"*55)
    log.info("  Supply Chain API — starting up")
    log.info("═"*55)
    try:
        registry.load_all()
    except FileNotFoundError as e:
        log.warning(f"⚠ Model artifacts not found: {e}")
        log.warning("⚠ Run `python models.py` (Section 3) before starting the API.")
    yield
    log.info("Supply Chain API — shutting down")


# ─────────────────────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Intelligent Supply Chain Risk & Demand Forecasting API",
    description=(
        "Endpoints for demand forecasting, stockout prediction, "
        "supplier risk scoring, and an AI Copilot for natural-language "
        "supply-chain Q&A."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to dashboard origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# MIDDLEWARE — request logging + latency tracking
# ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"]   = request_id
    response.headers["X-Response-Time-ms"] = str(duration_ms)
    log.info(f"[{request_id}] {request.method} {request.url.path} "
             f"→ {response.status_code}  ({duration_ms} ms)")
    return response


# ═════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════

class FeatureContribution(BaseModel):
    """A single feature's SHAP-style contribution to a prediction."""
    feature: str = Field(..., description="Feature name")
    mean_shap: float = Field(..., description="Signed contribution to the prediction")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    loaded_at: str | None
    products_available: int
    suppliers_available: int


# ── /forecast ────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    product_id: int = Field(..., gt=0, description="Product ID to forecast", examples=[1])
    horizon_days: int = Field(
        default=7, ge=1, le=30,
        description="Forecast horizon in days (informational; model predicts next-day demand)"
    )


class ForecastResponse(BaseModel):
    product_id: int
    predicted_demand: float = Field(..., description="Point forecast for next-day demand")
    lower_bound_95: float = Field(..., description="Lower bound of 95% prediction interval")
    upper_bound_95: float = Field(..., description="Upper bound of 95% prediction interval")
    horizon_days: int
    model_name: str = "xgboost_demand_v1"
    model_version: str = "1.0.0"
    generated_at: str
    top_features: list[FeatureContribution]


# ── /stockout ────────────────────────────────────────────────

class StockoutRequest(BaseModel):
    product_id: int = Field(..., gt=0, description="Product ID to evaluate", examples=[1])


class StockoutResponse(BaseModel):
    product_id: int
    stockout_probability: float = Field(..., ge=0, le=1)
    stockout_flag: bool = Field(..., description="True if probability exceeds the decision threshold")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    decision_threshold: float
    horizon_days: int = 7
    model_name: str = "stockout_predictor_v1"
    model_version: str = "1.0.0"
    generated_at: str
    top_features: list[FeatureContribution]


# ── /supplier-risk ───────────────────────────────────────────

class SupplierRiskRequest(BaseModel):
    supplier_id: int = Field(..., gt=0, description="Supplier ID to evaluate", examples=[1])


class SupplierRiskResponse(BaseModel):
    supplier_id: int
    delay_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["low", "medium", "high", "critical"]
    composite_score: float = Field(..., ge=0, le=100)
    recommended_action: str
    model_name: str = "supplier_risk_v1"
    model_version: str = "1.0.0"
    generated_at: str
    top_features: list[FeatureContribution]


# ── /ask-ai ───────────────────────────────────────────────────

class AskAIRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=3, max_length=1000,
        description="Natural-language question for the AI Copilot",
        examples=["What is the demand outlook for product 1 next week?"],
    )
    product_id: int | None = Field(
        default=None, description="Optional product context for the query"
    )
    supplier_id: int | None = Field(
        default=None, description="Optional supplier context for the query"
    )


class AskAIResponse(BaseModel):
    prompt: str
    response: str = Field(..., description="Natural-language business recommendation")
    intent: Literal["demand", "stockout", "supplier_risk", "general"]
    sql_query: str | None = Field(
        default=None,
        description="Auto-generated SQL the Copilot would run against PostgreSQL"
    )
    data_context: dict | None = Field(
        default=None, description="Structured data backing the response"
    )
    generated_at: str


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness / readiness probe. Used by Docker healthchecks & load balancers."""
    models_loaded = all([
        registry.demand_model is not None,
        registry.stockout_model is not None,
        registry.risk_reg_model is not None,
        registry.risk_clf_model is not None,
    ])
    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
        loaded_at=registry.loaded_at,
        products_available=len(registry.latest_demand_features),
        suppliers_available=len(registry.latest_risk_features),
    )


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    tags=["Predictions"],
    summary="Forecast next-day demand for a product",
    dependencies=[Depends(require_api_key)],
)
async def forecast(req: ForecastRequest):
    """
    Returns a point forecast plus 95% prediction interval for the
    requested product's next-day demand, using the most recent
    engineered feature row (lags, rolling stats, seasonality, etc.)

    Raises 503 if models have not been loaded (run Section 3 training first).
    Raises 404 if the product has no feature history.
    """
    if registry.demand_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demand model not loaded. Run `python models.py` first.",
        )

    feature_row = registry.latest_demand_features.get(req.product_id)
    if feature_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feature history found for product_id={req.product_id}",
        )

    result = ml.predict_demand(
        product_id=req.product_id,
        feature_row=feature_row,
        model=registry.demand_model,
    )

    return ForecastResponse(
        product_id=result["product_id"],
        predicted_demand=result["predicted_demand"],
        lower_bound_95=result["lower_bound_95"],
        upper_bound_95=result["upper_bound_95"],
        horizon_days=req.horizon_days,
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        top_features=[
            FeatureContribution(**f) for f in result["top_features"]
        ],
    )


@app.post(
    "/stockout",
    response_model=StockoutResponse,
    tags=["Predictions"],
    summary="Predict stockout probability within 7 days",
    dependencies=[Depends(require_api_key)],
)
async def stockout(req: StockoutRequest):
    """
    Returns the probability that the requested product will stock
    out (quantity_on_hand = 0) at some point in the next 7 days,
    along with a categorical risk level and the top contributing
    features.
    """
    if registry.stockout_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stockout model not loaded. Run `python models.py` first.",
        )

    feature_row = registry.latest_stockout_features.get(req.product_id)
    if feature_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feature history found for product_id={req.product_id}",
        )

    result = ml.predict_stockout(
        product_id=req.product_id,
        feature_row=feature_row,
        model=registry.stockout_model,
    )

    return StockoutResponse(
        product_id=result["product_id"],
        stockout_probability=result["stockout_probability"],
        stockout_flag=result["stockout_flag"],
        risk_level=result["risk_level"],
        decision_threshold=ml.STOCKOUT_THRESHOLD,
        generated_at=datetime.utcnow().isoformat() + "Z",
        top_features=[
            FeatureContribution(**f) for f in result["top_features"]
        ],
    )


@app.post(
    "/supplier-risk",
    response_model=SupplierRiskResponse,
    tags=["Predictions"],
    summary="Score supplier delivery delay risk",
    dependencies=[Depends(require_api_key)],
)
async def supplier_risk(req: SupplierRiskRequest):
    """
    Returns the predicted delay probability, an ordinal risk tier
    (low/medium/high/critical), a composite 0-100 risk score, and a
    recommended procurement action for the given supplier.
    """
    if registry.risk_reg_model is None or registry.risk_clf_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supplier risk models not loaded. Run `python models.py` first.",
        )

    feature_row = registry.latest_risk_features.get(req.supplier_id)
    if feature_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feature history found for supplier_id={req.supplier_id}",
        )

    result = ml.predict_supplier_risk(
        supplier_id=req.supplier_id,
        feature_row=feature_row,
        model_reg=registry.risk_reg_model,
        model_clf=registry.risk_clf_model,
    )

    return SupplierRiskResponse(
        supplier_id=result["supplier_id"],
        delay_probability=result["delay_probability"],
        risk_tier=result["risk_tier"],
        composite_score=result["composite_score"],
        recommended_action=result["recommended_action"],
        generated_at=datetime.utcnow().isoformat() + "Z",
        top_features=[
            FeatureContribution(**f) for f in result["top_features"]
        ],
    )


# ─────────────────────────────────────────────────────────────
# /ask-ai  — AI Copilot (prompt-engineering placeholder)
# ─────────────────────────────────────────────────────────────

def _classify_intent(prompt: str) -> Literal["demand", "stockout", "supplier_risk", "general"]:
    """
    Simple keyword-based intent router.

    ── Production swap ──────────────────────────────────────────
    Replace with an LLM function-call / tool-use classification step,
    e.g. ask an LLM to choose one of
        ["demand", "stockout", "supplier_risk", "general"]
    based on the user's prompt, then route accordingly.
    ─────────────────────────────────────────────────────────────
    """
    p = prompt.lower()
    if any(k in p for k in ["forecast", "demand", "sales", "sell"]):
        return "demand"
    if any(k in p for k in ["stockout", "stock out", "out of stock", "inventory", "reorder"]):
        return "stockout"
    if any(k in p for k in ["supplier", "vendor", "delay", "risk", "procurement"]):
        return "supplier_risk"
    return "general"


def _generate_sql(intent: str, product_id: int | None, supplier_id: int | None) -> str | None:
    """
    Template-based SQL generation matching the intent + entities.

    ── Production swap ──────────────────────────────────────────
    Replace with an LLM text-to-SQL call grounded in the schema
    from Section 1 (schema.sql), e.g.:

        prompt = (f"Given this schema:\\n{SCHEMA_DDL}\\n\\n"
                  f"Write a PostgreSQL query to answer: {user_prompt}")
        sql = llm.generate(prompt)
    ─────────────────────────────────────────────────────────────
    """
    if intent == "demand" and product_id:
        return (
            "SELECT sale_date, SUM(quantity_sold) AS daily_demand\n"
            "FROM sales\n"
            f"WHERE product_id = {product_id}\n"
            "  AND sale_date >= CURRENT_DATE - INTERVAL '30 days'\n"
            "GROUP BY sale_date\n"
            "ORDER BY sale_date;"
        )
    if intent == "stockout" and product_id:
        return (
            "SELECT snapshot_date, quantity_on_hand, quantity_available, reorder_triggered\n"
            "FROM inventory\n"
            f"WHERE product_id = {product_id}\n"
            "ORDER BY snapshot_date DESC\n"
            "LIMIT 30;"
        )
    if intent == "supplier_risk" and supplier_id:
        return (
            "SELECT scored_at, delay_probability, risk_tier, composite_risk_score\n"
            "FROM risk_scores\n"
            f"WHERE supplier_id = {supplier_id}\n"
            "ORDER BY scored_at DESC\n"
            "LIMIT 12;"
        )
    if intent == "supplier_risk":
        return (
            "SELECT s.supplier_name, r.risk_tier, r.composite_risk_score\n"
            "FROM risk_scores r\n"
            "JOIN suppliers s ON s.supplier_id = r.supplier_id\n"
            "WHERE r.scored_at = (SELECT MAX(scored_at) FROM risk_scores)\n"
            "ORDER BY r.composite_risk_score DESC\n"
            "LIMIT 10;"
        )
    return None


def _generate_ai_response(
    prompt: str,
    intent: str,
    product_id: int | None,
    supplier_id: int | None,
) -> tuple[str, dict | None]:
    """
    Generates the natural-language business recommendation.

    ── Production swap ──────────────────────────────────────────
    This function currently builds a templated response from model
    outputs. To make this a real "AI Copilot", replace the return
    statement with an LLM call such as:

        context = {"intent": intent, "model_output": data_context}
        llm_prompt = (
            f"You are a supply chain analyst. A user asked: '{prompt}'.\\n"
            f"Here is the relevant model output: {context}\\n"
            f"Write a concise, actionable business recommendation."
        )
        narrative = llm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": llm_prompt}],
        ).content[0].text
        return narrative, context
    ─────────────────────────────────────────────────────────────
    """
    # DEMAND intent
    if intent == "demand" and product_id is not None:
        feature_row = registry.latest_demand_features.get(product_id)
        if feature_row is None:
            return (f"I don't have demand history for product {product_id}.", None)
        result = ml.predict_demand(product_id, feature_row, registry.demand_model)
        narrative = (
            f"For product {product_id}, the model forecasts a next-day demand of "
            f"{result['predicted_demand']:.1f} units "
            f"(95% interval: {result['lower_bound_95']:.1f}\u2013{result['upper_bound_95']:.1f}). "
            f"The strongest drivers were "
            f"{', '.join(f['feature'] for f in result['top_features'][:3])}. "
            + ("Demand appears stable; current stock levels should be sufficient."
               if result['predicted_demand'] < 20
               else "Demand is elevated \u2014 consider reviewing reorder timing for this SKU.")
        )
        return narrative, result

    # STOCKOUT intent
    if intent == "stockout" and product_id is not None:
        feature_row = registry.latest_stockout_features.get(product_id)
        if feature_row is None:
            return (f"I don't have inventory history for product {product_id}.", None)
        result = ml.predict_stockout(product_id, feature_row, registry.stockout_model)
        narrative = (
            f"Product {product_id} has a {result['stockout_probability']:.0%} chance of "
            f"stocking out within the next 7 days "
            f"(risk level: {result['risk_level']}). "
            f"Key contributing factors: "
            f"{', '.join(f['feature'] for f in result['top_features'][:3])}. "
            + ("Recommend placing a reorder now to avoid a stockout."
               if result["stockout_flag"]
               else "Inventory position currently looks adequate.")
        )
        return narrative, result

    # SUPPLIER RISK intent
    if intent == "supplier_risk" and supplier_id is not None:
        feature_row = registry.latest_risk_features.get(supplier_id)
        if feature_row is None:
            return (f"I don't have risk history for supplier {supplier_id}.", None)
        result = ml.predict_supplier_risk(
            supplier_id, feature_row,
            registry.risk_reg_model, registry.risk_clf_model,
        )
        narrative = (
            f"Supplier {supplier_id} is rated '{result['risk_tier'].upper()}' risk "
            f"(composite score: {result['composite_score']:.1f}/100, "
            f"delay probability: {result['delay_probability']:.0%}). "
            f"Recommendation: {result['recommended_action']}"
        )
        return narrative, result

    # GENERAL supplier risk: rank all known suppliers
    if intent == "supplier_risk":
        ranked = []
        for sid, feats in registry.latest_risk_features.items():
            r = ml.predict_supplier_risk(
                sid, feats, registry.risk_reg_model, registry.risk_clf_model
            )
            ranked.append(r)
        ranked.sort(key=lambda r: r["composite_score"], reverse=True)
        top3 = ranked[:3]
        narrative = (
            "Here are the highest-risk suppliers right now: " +
            "; ".join(
                f"Supplier {r['supplier_id']} ({r['risk_tier']}, "
                f"score={r['composite_score']:.1f})"
                for r in top3
            ) + ". Consider reviewing contracts or dual-sourcing for these."
        )
        return narrative, {"ranked_suppliers": top3}

    return (
        "I can help with demand forecasts, stockout risk, and supplier risk "
        "scoring. Try asking, e.g., 'What is the stockout risk for product 5?' "
        "or 'Which suppliers are highest risk right now?'",
        None,
    )


@app.post(
    "/ask-ai",
    response_model=AskAIResponse,
    tags=["AI Copilot"],
    summary="Natural-language Q&A over supply chain data",
    dependencies=[Depends(require_api_key)],
)
async def ask_ai(req: AskAIRequest):
    """
    AI Copilot endpoint. Classifies the user's intent, retrieves the
    relevant model prediction(s), generates a templated SQL query for
    transparency, and returns a natural-language business
    recommendation.

    This is a PLACEHOLDER implementation suitable for demos. See the
    `_generate_ai_response()` docstring for how to wire in a real LLM
    for production-quality, free-form responses.
    """
    intent   = _classify_intent(req.prompt)
    sql      = _generate_sql(intent, req.product_id, req.supplier_id)
    response_text, data_context = _generate_ai_response(
        req.prompt, intent, req.product_id, req.supplier_id
    )

    return AskAIResponse(
        prompt=req.prompt,
        response=response_text,
        intent=intent,
        sql_query=sql,
        data_context=data_context,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Intelligent Supply Chain Risk & Demand Forecasting API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": ["/forecast", "/stockout", "/supplier-risk", "/ask-ai"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)