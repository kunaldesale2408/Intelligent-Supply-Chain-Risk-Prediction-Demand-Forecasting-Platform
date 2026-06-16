"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
Streamlit Dashboard  |  Section 5
=============================================================
Layout
  Sidebar  – Data source toggle, filters, live KPI cards
  Tab 1    – Overview:        revenue trend, channel/category mix
  Tab 2    – Demand Forecast: product forecast chart + CI band + SHAP
  Tab 3    – Stockout Monitor: alert table, inventory trend
  Tab 4    – Supplier Risk:   scorecard, delay scatter, top features
  Tab 5    – AI Copilot:     NL prompt → intent → insight + SQL

Data modes (toggle in sidebar)
  Direct   – loads CSVs from data/clean/ and data/features/,
             calls models.py predict_* helpers directly
  API      – calls FastAPI endpoints at API_BASE_URL

Run:
    streamlit run dashboard.py

With Docker (API mode):
    API_BASE_URL=http://api:8000 streamlit run dashboard.py
=============================================================
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── st.set_page_config must be the FIRST Streamlit call ──────────
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="⛓️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Intelligent Supply Chain Risk & Demand Forecasting Platform v1.0"},
)

warnings.filterwarnings("ignore")

# ── Plotly (graceful fallback if not installed) ───────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Direct model access (only needed in Direct mode) ─────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    import models as ml
    MODELS_OK = True
except Exception:
    MODELS_OK = False


# ═════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════

DATA_DIR     = Path(os.environ.get("DATA_DIR",     "./data/clean"))
FEATURES_DIR = Path(os.environ.get("FEATURES_DIR", "./data/features"))
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY      = os.environ.get("API_KEY",      "dev-local-key-12345")
API_HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

TIER_COLOR = {
    "low":      "#2ED573",
    "medium":   "#FFA502",
    "high":     "#FF6348",
    "critical": "#FF4757",
}

CHART_COLORS = ["#00C9A7", "#3742FA", "#FF4757", "#FFA502", "#2ED573", "#A29BFE"]


# ═════════════════════════════════════════════════════════════════
# CSS INJECTION
# ═════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

    /* ── KPI cards ────────────────────────────────────────────── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-left: 4px solid var(--accent, #00C9A7);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .kpi-label  { font-size: 11px; color: #6c757d; font-weight: 500;
                   text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
    .kpi-value  { font-size: 26px; font-weight: 700; color: #1a1a2e; line-height: 1; }
    .kpi-delta  { font-size: 12px; color: #6c757d; margin-top: 4px; }
    .kpi-pos    { color: #2ed573; }
    .kpi-neg    { color: #ff4757; }

    /* ── Risk tier badges ──────────────────────────────────────── */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600; letter-spacing: .04em;
        text-transform: uppercase;
    }
    .badge-low      { background: #d4f7e7; color: #1a8c50; }
    .badge-medium   { background: #fff3cd; color: #b45309; }
    .badge-high     { background: #ffe0d6; color: #c0392b; }
    .badge-critical { background: #ffd6d9; color: #891326; }

    /* ── Alert banner ──────────────────────────────────────────── */
    .alert-banner {
        background: #fff5f5; border: 1px solid #ff4757;
        border-left: 5px solid #ff4757; border-radius: 6px;
        padding: 12px 16px; margin-bottom: 16px;
    }
    .alert-banner strong { color: #c0392b; }

    /* ── AI Copilot response card ──────────────────────────────── */
    .ai-card {
        background: #f0fdf9; border: 1px solid #a7f3d0;
        border-radius: 8px; padding: 16px 20px; margin-top: 10px;
    }
    .ai-card p  { color: #1a1a2e; font-size: 14px; line-height: 1.65; margin: 0; }
    .intent-badge {
        display: inline-block; background: #00c9a7; color: white;
        font-size: 10px; font-weight: 600; padding: 2px 8px;
        border-radius: 12px; letter-spacing: .05em; margin-bottom: 8px;
    }

    /* ── Score progress bars ───────────────────────────────────── */
    .score-bar-wrap { width: 100%; background: #e9ecef; border-radius: 4px;
                      height: 8px; margin-top: 2px; }
    .score-bar      { height: 8px; border-radius: 4px; }

    /* ── Metric cards row (tabs) ───────────────────────────────── */
    .metric-mini { background: #f8f9fa; border-radius: 6px;
                   padding: 10px 14px; text-align: center; border: 1px solid #e9ecef; }
    .metric-mini .label { font-size: 11px; color: #6c757d;
                           text-transform: uppercase; letter-spacing: .04em; }
    .metric-mini .value { font-size: 20px; font-weight: 700; color: #1a1a2e; }

    div[data-testid="stTabs"] button { font-weight: 500; font-size: 14px; }
    div[data-testid="metric-container"] { background: #f8f9fa;
        border-radius: 8px; padding: 12px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# DATA LOADERS  (all cached — invalidate with st.cache_data.clear())
# ═════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sales.csv", low_memory=False)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["month"]     = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "inventory.csv", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv", low_memory=False)


@st.cache_data(ttl=3600, show_spinner=False)
def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "suppliers.csv", low_memory=False)


@st.cache_data(ttl=3600, show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "risk_scores.csv", low_memory=False)
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "forecasts.csv", low_memory=False)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_demand_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / "features_demand.csv", low_memory=False)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stockout_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_DIR / "features_stockout.csv", low_memory=False)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@st.cache_resource(show_spinner=False)
def load_models_direct():
    """Load pickle models once, shared across all sessions (Direct mode only)."""
    if not MODELS_OK:
        return None, None, None, None
    try:
        return (
            ml.load_model("demand_forecaster"),
            ml.load_model("stockout_predictor"),
            ml.load_model("supplier_risk_regressor"),
            ml.load_model("supplier_risk_classifier"),
        )
    except FileNotFoundError:
        return None, None, None, None


# ═════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════

def _api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE_URL}{endpoint}",
                          headers=API_HEADERS, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return None


def get_forecast(product_id: int, use_api: bool,
                 demand_feats: dict, model=None) -> dict | None:
    if use_api:
        return _api_post("/forecast", {"product_id": product_id})
    if model is None or product_id not in demand_feats:
        return None
    return ml.predict_demand(product_id, demand_feats[product_id], model=model)


def get_stockout(product_id: int, use_api: bool,
                 stockout_feats: dict, model=None) -> dict | None:
    if use_api:
        return _api_post("/stockout", {"product_id": product_id})
    if model is None or product_id not in stockout_feats:
        return None
    return ml.predict_stockout(product_id, stockout_feats[product_id], model=model)


def get_supplier_risk(supplier_id: int, use_api: bool,
                      risk_feats: dict, reg_model=None, clf_model=None) -> dict | None:
    if use_api:
        return _api_post("/supplier-risk", {"supplier_id": supplier_id})
    if reg_model is None or supplier_id not in risk_feats:
        return None
    return ml.predict_supplier_risk(supplier_id, risk_feats[supplier_id],
                                     model_reg=reg_model, model_clf=clf_model)


def get_ai_insight(prompt: str, product_id: int | None,
                   supplier_id: int | None, use_api: bool) -> dict | None:
    if use_api:
        return _api_post("/ask-ai", {"prompt": prompt,
                                      "product_id": product_id,
                                      "supplier_id": supplier_id})
    # Fallback: construct a minimal templated response
    return {"intent": "general",
            "response": "Connect to the API for AI Copilot responses. "
                        "Start the FastAPI service with: uvicorn main:app --port 8000",
            "sql_query": None,
            "data_context": None}


# ═════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════════

def _base_layout(**kwargs) -> dict:
    return dict(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1a1a2e"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,0.4)",
        margin=dict(l=20, r=20, t=36, b=20),
        #**kwargs,
    )
    base_config.update(kwargs)
    return base_config


def chart_revenue_trend(sales: pd.DataFrame) -> "go.Figure":
    monthly = sales.groupby("month")["revenue"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["revenue"],
        name="Revenue", mode="lines+markers",
        line=dict(color="#00C9A7", width=2.5),
        marker=dict(size=6, color="#00C9A7"),
        fill="tozeroy",
        fillcolor="rgba(0,201,167,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Monthly Revenue 2024", font=dict(size=15, weight=700)),
            xaxis=dict(showgrid=False, tickformat="%b"),
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#f0f0f0"),
            showlegend=False,
            height=280,
        )
    )
    return fig


def chart_channel_donut(sales: pd.DataFrame) -> "go.Figure":
    ch = sales.groupby("channel")["revenue"].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=ch["channel"].str.title(),
        values=ch["revenue"],
        hole=0.52,
        marker=dict(colors=CHART_COLORS),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Revenue by Channel", font=dict(size=14, weight=700)),
        showlegend=False, height=260,
        margin=dict(l=10, r=10, t=36, b=10),
    ))
    return fig


def chart_category_bar(sales: pd.DataFrame, products: pd.DataFrame) -> "go.Figure":
    merged = sales.merge(products[["product_id", "category"]], on="product_id")
    cat    = merged.groupby("category")["revenue"].sum().sort_values().reset_index()
    fig = go.Figure(go.Bar(
        x=cat["revenue"], y=cat["category"],
        orientation="h",
        marker=dict(color=CHART_COLORS[:len(cat)], opacity=0.85),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Revenue by Category", font=dict(size=14, weight=700)),
        xaxis=dict(tickprefix="$", tickformat=".2s", showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
        height=260,
        margin=dict(l=10, r=20, t=36, b=20),
    ))
    return fig


def chart_forecast(forecasts: pd.DataFrame, product_id: int) -> "go.Figure":
    df = forecasts[forecasts["product_id"] == product_id].sort_values("forecast_date")
    fig = go.Figure()

    # 95 % confidence band
    fig.add_trace(go.Scatter(
        x=df["forecast_date"], y=df["upper_bound_95"],
        name="Upper 95%", mode="lines",
        line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["forecast_date"], y=df["lower_bound_95"],
        name="95% PI", mode="lines",
        fill="tonexty", fillcolor="rgba(0,201,167,0.12)",
        line=dict(width=0),
        hoverinfo="skip",
    ))
    # Actual demand
    fig.add_trace(go.Scatter(
        x=df["forecast_date"], y=df["actual_demand"],
        name="Actual", mode="lines+markers",
        line=dict(color="#3742FA", width=1.8),
        marker=dict(size=5),
        hovertemplate="<b>Actual</b><br>%{x|%d %b}: %{y:.1f} units<extra></extra>",
    ))
    # Predicted demand
    fig.add_trace(go.Scatter(
        x=df["forecast_date"], y=df["predicted_demand"],
        name="Predicted", mode="lines+markers",
        line=dict(color="#00C9A7", width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="<b>Predicted</b><br>%{x|%d %b}: %{y:.1f} units<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"Demand Forecast — Product {product_id}",
                   font=dict(size=15, weight=700)),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="Units", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
    ))
    return fig


def chart_shap_bar(shap_data: list[dict], title: str = "Top Feature Drivers") -> "go.Figure":
    """Render SHAP feature contributions as a signed horizontal bar chart."""
    df = pd.DataFrame(shap_data).sort_values("mean_shap")
    colors = ["#FF6348" if v > 0 else "#3742FA" for v in df["mean_shap"]]
    fig = go.Figure(go.Bar(
        x=df["mean_shap"], y=df["feature"],
        orientation="h",
        marker=dict(color=colors, opacity=0.82),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#adb5bd", width=1, dash="dot"))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=13, weight=700)),
        xaxis=dict(title="Impact on prediction (SHAP)", showgrid=False),
        yaxis=dict(showgrid=False, automargin=True),
        showlegend=False, height=280,
        margin=dict(l=10, r=10, t=36, b=20),
    ))
    return fig


def chart_inventory_trend(inventory: pd.DataFrame, product_id: int,
                           products: pd.DataFrame) -> "go.Figure":
    df  = inventory[inventory["product_id"] == product_id].copy()
    rp  = products.loc[products["product_id"] == product_id, "reorder_point"].values
    rp  = float(rp[0]) if len(rp) else 50.0

    fig = go.Figure()
    fig.add_hline(y=rp, line=dict(color="#FFA502", width=1.5, dash="dash"),
                  annotation_text=f"Reorder point ({rp:.0f})",
                  annotation_position="top left",
                  annotation_font=dict(color="#FFA502", size=11))
    fig.add_trace(go.Scatter(
        x=df["snapshot_date"], y=df["quantity_on_hand"],
        name="Qty on Hand", mode="lines",
        line=dict(color="#3742FA", width=2),
        fill="tozeroy", fillcolor="rgba(55,66,250,0.07)",
        hovertemplate="%{x|%d %b}: %{y:.0f} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["snapshot_date"], y=df["quantity_available"],
        name="Available", mode="lines",
        line=dict(color="#00C9A7", width=1.5, dash="dot"),
        hovertemplate="Available %{x|%d %b}: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"Inventory Level — Product {product_id}",
                   font=dict(size=14, weight=700)),
        xaxis=dict(showgrid=False, tickformat="%b"),
        yaxis=dict(title="Units", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=300,
    ))
    return fig


def chart_risk_scatter(scorecard: pd.DataFrame) -> "go.Figure":
    tier_map = {"low": "#2ED573", "medium": "#FFA502",
                "high": "#FF6348", "critical": "#FF4757"}
    colors   = scorecard["risk_tier"].map(tier_map).fillna("#adb5bd")
    fig = go.Figure(go.Scatter(
        x=scorecard["hist_on_time_rate"] * 100,
        y=scorecard["delay_probability"] * 100,
        mode="markers+text",
        text=scorecard["supplier_name"].str.split().str[0],
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=scorecard["composite_risk_score"] / 3 + 12,
            color=colors,
            line=dict(width=1.5, color="white"),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "On-time rate: %{x:.1f}%<br>"
            "Delay probability: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Supplier Risk — On-time Rate vs Delay Probability",
                   font=dict(size=14, weight=700)),
        xaxis=dict(title="Historical On-Time Rate (%)", range=[0, 100],
                   gridcolor="#f0f0f0"),
        yaxis=dict(title="Delay Probability (%)", range=[0, 105],
                   gridcolor="#f0f0f0"),
        showlegend=False, height=340,
    ))
    # Add quadrant dividers
    fig.add_hline(y=40, line=dict(color="#adb5bd", width=1, dash="dot"))
    fig.add_vline(x=50, line=dict(color="#adb5bd", width=1, dash="dot"))
    return fig


# ═════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════

def render_sidebar(sales, inventory, products, forecasts):
    with st.sidebar:
        st.markdown("### ⛓️ Supply Chain Intelligence")
        st.caption("Risk & Demand Forecasting Platform v1.0")
        st.divider()

        # ── Data source ───────────────────────────────────────────
        data_mode = st.radio(
            "Data source",
            ["📂 Direct (CSV)", "🌐 API (FastAPI)"],
            index=0,
            help="Direct: reads local CSVs + runs models in-process.\n"
                 "API: calls FastAPI at " + API_BASE_URL,
        )
        use_api = data_mode.startswith("🌐")

        if use_api:
            try:
                r = requests.get(f"{API_BASE_URL}/health", timeout=3)
                health = r.json()
                if health.get("status") == "ok":
                    st.success(f"API online · {health.get('products_available', 0)} products")
                else:
                    st.warning("API degraded — models may not be loaded")
            except Exception:
                st.error(f"API unreachable at {API_BASE_URL}")

        st.divider()

        # ── Filters ───────────────────────────────────────────────
        st.markdown("**Filters**")
        categories = ["All"] + sorted(products["category"].unique().tolist())
        sel_cat    = st.selectbox("Category", categories, index=0)

        prod_opts = products.copy()
        if sel_cat != "All":
            prod_opts = prod_opts[prod_opts["category"] == sel_cat]
        prod_list  = ["All"] + [
            f"{r['product_id']}: {r['product_name']}"
            for _, r in prod_opts.sort_values("product_id").iterrows()
        ]
        sel_prod   = st.selectbox("Product", prod_list, index=0)
        sel_pid    = None if sel_prod == "All" else int(sel_prod.split(":")[0])

        sup_list   = ["All"] + [
            f"{r['supplier_id']}: {r['supplier_name']}"
            for _, r in load_suppliers().sort_values("supplier_id").iterrows()
        ]
        sel_sup    = st.selectbox("Supplier", sup_list, index=0)
        sel_sid    = None if sel_sup == "All" else int(sel_sup.split(":")[0])

        st.divider()

        # ── KPI cards ─────────────────────────────────────────────
        st.markdown("**Platform KPIs**")

        # Filter sales by category if needed
        s = sales.copy()
        if sel_cat != "All":
            cat_pids = products[products["category"] == sel_cat]["product_id"]
            s = s[s["product_id"].isin(cat_pids)]
        if sel_pid:
            s = s[s["product_id"] == sel_pid]

        total_rev = s["revenue"].sum()
        stockouts = (
            inventory[inventory["quantity_on_hand"] == 0]["product_id"].nunique()
        )
        reorders  = int(inventory["reorder_triggered"].sum())
        avg_mae   = forecasts["mae"].mean()

        def kpi(label, value, delta="", accent="#00C9A7"):
            st.markdown(
                f'<div class="kpi-card" style="--accent:{accent}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-delta">{delta}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        kpi("Total Revenue", f"${total_rev:,.0f}",
            "Jan–Dec 2024", "#00C9A7")
        kpi("Stockout Alerts", str(stockouts),
            "products with 0 stock", "#FF4757")
        kpi("Reorder Triggers", f"{reorders:,}",
            "replenishment events", "#FFA502")
        kpi("Forecast MAE", f"{avg_mae:.2f} units",
            "mean absolute error", "#3742FA")

    return use_api, sel_cat, sel_pid, sel_sid


# ═════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════

def tab_overview(sales, products):
    st.subheader("Revenue Overview")

    if not PLOTLY_OK:
        st.info("Install plotly to see charts: `pip install plotly`")
        return

    # Revenue trend full-width
    st.plotly_chart(
        chart_revenue_trend(sales),
        use_container_width=True, config={"displayModeBar": False},
    )

    # Channel donut  |  Category bar
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            chart_channel_donut(sales),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c2:
        st.plotly_chart(
            chart_category_bar(sales, products),
            use_container_width=True, config={"displayModeBar": False},
        )

    # Top-10 products by revenue
    st.markdown("#### Top 10 Products by Revenue")
    top10 = (
        sales.groupby("product_id")["revenue"].sum()
        .reset_index().nlargest(10, "revenue")
        .merge(products[["product_id", "product_name", "category"]], on="product_id")
        .rename(columns={"product_id": "ID", "product_name": "Product",
                          "category": "Category", "revenue": "Revenue"})
    )
    top10["Revenue"] = top10["Revenue"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top10.reset_index(drop=True), use_container_width=True,
                 hide_index=True)


# ═════════════════════════════════════════════════════════════════
# TAB 2 — DEMAND FORECAST
# ═════════════════════════════════════════════════════════════════

def tab_forecast(forecasts, products, use_api, sel_pid,
                 demand_feats, demand_model):
    st.subheader("Demand Forecast")

    # Product selector
    all_pids = sorted(forecasts["product_id"].unique().tolist())
    default  = all_pids.index(sel_pid) if sel_pid in all_pids else 0
    pid      = st.selectbox(
        "Select product",
        options=all_pids,
        index=default,
        format_func=lambda p: (
            products.loc[products["product_id"] == p, "product_name"]
            .values[0] if not products.empty else str(p)
        ),
    )

    # CV metrics row
    prod_fore = forecasts[forecasts["product_id"] == pid]
    mae_val   = prod_fore["mae"].mean()
    mape_val  = prod_fore["mape"].mean()
    conf_val  = prod_fore["confidence_score"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error",   f"{mae_val:.2f} units")
    m2.metric("MAPE",                  f"{mape_val:.1f}%")
    m3.metric("Confidence Score",      f"{conf_val:.1%}")

    # Forecast chart
    if PLOTLY_OK:
        st.plotly_chart(
            chart_forecast(forecasts, pid),
            use_container_width=True, config={"displayModeBar": False},
        )

    # Live prediction + SHAP
    st.markdown("#### Live Prediction (latest feature row)")
    result = get_forecast(pid, use_api, demand_feats, demand_model)

    if result:
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Demand",  f"{result['predicted_demand']:.1f} units")
        r2.metric("Lower Bound (95%)", f"{result['lower_bound_95']:.1f}")
        r3.metric("Upper Bound (95%)", f"{result['upper_bound_95']:.1f}")

        if result.get("top_features") and PLOTLY_OK:
            st.markdown("**SHAP Feature Drivers** — why this prediction?")
            st.plotly_chart(
                chart_shap_bar(result["top_features"],
                               "Demand SHAP — Top Feature Contributions"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    else:
        st.info("Prediction unavailable — train models or start the API.")


# ═════════════════════════════════════════════════════════════════
# TAB 3 — STOCKOUT MONITOR
# ═════════════════════════════════════════════════════════════════

def tab_stockout(inventory, products, use_api, sel_pid,
                 stockout_feats, stockout_model):
    st.subheader("Stockout Monitor")

    # Alert banner
    stockout_pids = inventory[inventory["quantity_on_hand"] == 0]["product_id"].nunique()
    st.markdown(
        f'<div class="alert-banner">'
        f"<strong>⚠ {stockout_pids} products</strong> currently have zero stock "
        f"and are actively stockedout."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Alert table
    st.markdown("#### Products Currently Stocked Out")
    alert_df = (
        inventory[inventory["quantity_on_hand"] == 0]
        .merge(products[["product_id", "product_name", "category"]], on="product_id")
        .groupby(["product_id", "product_name", "category"])
        .agg(stockout_days=("snapshot_date", "count"))
        .reset_index()
        .sort_values("stockout_days", ascending=False)
        .rename(columns={"product_id": "ID", "product_name": "Product",
                          "category": "Category",
                          "stockout_days": "Days Out of Stock"})
    )
    st.dataframe(alert_df.reset_index(drop=True), use_container_width=True,
                 hide_index=True,
                 column_config={"Days Out of Stock": st.column_config.ProgressColumn(
                     min_value=0, max_value=int(alert_df["Days Out of Stock"].max()),
                 )})

    # Inventory trend for selected / first product
    pid = sel_pid or int(alert_df["ID"].iloc[0]) if not alert_df.empty else 1
    st.markdown(f"#### Inventory Trend — Product {pid}")

    col_chart, col_live = st.columns([2, 1])

    with col_chart:
        if PLOTLY_OK:
            st.plotly_chart(
                chart_inventory_trend(inventory, pid, products),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col_live:
        st.markdown("**Live Stockout Risk**")
        result = get_stockout(pid, use_api, stockout_feats, stockout_model)
        if result:
            prob  = result["stockout_probability"]
            level = result["risk_level"]
            color = {"LOW": "#2ED573", "MEDIUM": "#FFA502",
                     "HIGH": "#FF6348", "CRITICAL": "#FF4757"}.get(level, "#adb5bd")
            st.metric("Stockout Probability (7d)", f"{prob:.1%}")
            st.markdown(
                f'<div style="margin:8px 0">'
                f'<span class="badge badge-{level.lower()}">{level}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            # Probability gauge bar
            st.markdown(
                f'<div class="score-bar-wrap">'
                f'<div class="score-bar" style="width:{prob*100:.1f}%;'
                f'background:{color}"></div></div>',
                unsafe_allow_html=True,
            )
            if result.get("top_features") and PLOTLY_OK:
                st.markdown("**Key Drivers**")
                st.plotly_chart(
                    chart_shap_bar(result["top_features"],
                                   "Stockout Risk Drivers"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
        else:
            st.info("Start the API or ensure models are trained.")


# ═════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLIER RISK
# ═════════════════════════════════════════════════════════════════

def tab_supplier_risk(risk_scores, suppliers, use_api, sel_sid,
                      risk_feats, risk_reg_model, risk_clf_model):
    st.subheader("Supplier Risk Scorecard")

    # Build latest-score-per-supplier table
    latest = risk_scores.sort_values("scored_at").groupby("supplier_id").tail(1)
    scorecard = latest.merge(
        suppliers[["supplier_id", "supplier_name", "country", "reliability_score"]],
        on="supplier_id"
    ).sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

    # ── Scorecard HTML table ──────────────────────────────────────
    st.markdown("#### Risk Scorecard — Latest Assessment")
    rows_html = ""
    for _, r in scorecard.iterrows():
        tier    = r["risk_tier"]
        score   = r["composite_risk_score"]
        prob    = r["delay_probability"]
        ontime  = r["hist_on_time_rate"] * 100
        bar_clr = TIER_COLOR.get(tier, "#adb5bd")
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px;font-weight:500">{r['supplier_name']}</td>
          <td style="padding:8px 12px;color:#6c757d">{r['country']}</td>
          <td style="padding:8px 6px">
            <span class="badge badge-{tier}">{tier.upper()}</span>
          </td>
          <td style="padding:8px 12px;min-width:120px">
            <div style="font-size:12px;font-weight:600;margin-bottom:3px">{score:.1f}/100</div>
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{min(score,100):.1f}%;background:{bar_clr}"></div>
            </div>
          </td>
          <td style="padding:8px 12px">{prob:.0%}</td>
          <td style="padding:8px 12px">{ontime:.0f}%</td>
          <td style="padding:8px 12px;font-size:12px;color:#6c757d;max-width:240px">
            {r['recommended_action']}
          </td>
        </tr>
        """

    st.markdown(
        f"""<table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead><tr style="background:#f8f9fa;border-bottom:2px solid #e9ecef">
          <th style="padding:10px 12px;text-align:left">Supplier</th>
          <th style="padding:10px 12px;text-align:left">Country</th>
          <th style="padding:10px 12px;text-align:left">Tier</th>
          <th style="padding:10px 12px;text-align:left">Composite Score</th>
          <th style="padding:10px 12px;text-align:left">Delay Prob.</th>
          <th style="padding:10px 12px;text-align:left">On-Time Rate</th>
          <th style="padding:10px 12px;text-align:left">Recommendation</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
        </table>""",
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")

    # ── Scatter + live prediction ─────────────────────────────────
    col_scatter, col_live = st.columns([2, 1])

    with col_scatter:
        if PLOTLY_OK:
            st.plotly_chart(
                chart_risk_scatter(scorecard),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col_live:
        sid = sel_sid or int(scorecard["supplier_id"].iloc[0])
        sname = suppliers.loc[
            suppliers["supplier_id"] == sid, "supplier_name"
        ].values[0]
        st.markdown(f"**Live Risk — {sname}**")
        result = get_supplier_risk(sid, use_api, risk_feats,
                                    risk_reg_model, risk_clf_model)
        if result:
            tier    = result["risk_tier"]
            score   = result["composite_score"]
            prob    = result["delay_probability"]
            bar_clr = TIER_COLOR.get(tier, "#adb5bd")
            st.metric("Composite Risk Score", f"{score:.1f} / 100")
            st.metric("Delay Probability",    f"{prob:.0%}")
            st.markdown(
                f'<span class="badge badge-{tier}">{tier.upper()} RISK</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="score-bar-wrap" style="margin:8px 0">'
                f'<div class="score-bar" style="width:{min(score,100):.1f}%;'
                f'background:{bar_clr}"></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(result.get("recommended_action", ""))

            if result.get("top_features") and PLOTLY_OK:
                st.plotly_chart(
                    chart_shap_bar(result["top_features"],
                                   "Risk Drivers (SHAP)"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
        else:
            st.info("Train models or start the API.")


# ═════════════════════════════════════════════════════════════════
# TAB 5 — AI COPILOT
# ═════════════════════════════════════════════════════════════════

def tab_ai_copilot(use_api, sel_pid, sel_sid):
    st.subheader("AI Supply Chain Copilot")
    st.caption(
        "Ask any question about demand, inventory risk, or supplier performance. "
        "The copilot classifies your intent, retrieves the relevant model output, "
        "and returns a business recommendation plus the SQL it would run."
    )

    # ── Example prompt chips ──────────────────────────────────────
    st.markdown("**Try an example prompt:**")
    ex_cols = st.columns(4)
    examples = [
        ("📦 Demand outlook",     f"What is the demand forecast for product {sel_pid or 1}?"),
        ("⚠️ Stockout risk",      f"Is product {sel_pid or 5} at risk of stocking out?"),
        ("🚚 Supplier risk",      f"Evaluate the risk for supplier {sel_sid or 6}."),
        ("🔝 Riskiest suppliers", "Which suppliers are highest risk right now?"),
    ]
    chosen_prompt = None
    for col, (label, prompt) in zip(ex_cols, examples):
        if col.button(label, use_container_width=True):
            chosen_prompt = prompt

    # ── Custom prompt input ───────────────────────────────────────
    prompt = st.text_area(
        "Or type your own question:",
        value=chosen_prompt or "",
        height=90,
        placeholder="e.g. Which product is most at risk of running out this week?",
    )

    send_col, _ = st.columns([1, 4])
    run = send_col.button("Ask Copilot", type="primary",
                           use_container_width=True)

    if not run and not chosen_prompt:
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:8px;padding:20px;
                    text-align:center;color:#6c757d;margin-top:12px">
        Enter a question above to get an AI-powered supply chain insight.
        </div>
        """, unsafe_allow_html=True)
        return

    if not prompt:
        st.warning("Please enter a question before submitting.")
        return

    with st.spinner("Thinking…"):
        result = get_ai_insight(
            prompt=prompt,
            product_id=sel_pid,
            supplier_id=sel_sid,
            use_api=use_api,
        )

    if not result:
        st.error("No response — is the API running?")
        return

    # ── Response card ──────────────────────────────────────────────
    intent = result.get("intent", "general")
    intent_labels = {
        "demand":        "📈 Demand Forecast",
        "stockout":      "⚠️ Stockout Risk",
        "supplier_risk": "🚚 Supplier Risk",
        "general":       "💬 General",
    }
    st.markdown(
        f'<div class="ai-card">'
        f'<div class="intent-badge">{intent_labels.get(intent, intent)}</div>'
        f'<p>{result.get("response", "")}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── SQL query expander ─────────────────────────────────────────
    if result.get("sql_query"):
        with st.expander("📋 Auto-generated SQL query", expanded=False):
            st.code(result["sql_query"], language="sql")
            st.caption(
                "This query would be executed against PostgreSQL to retrieve "
                "the live data backing this insight."
            )

    # ── Data context ───────────────────────────────────────────────
    if result.get("data_context"):
        with st.expander("📊 Model output data", expanded=False):
            ctx = result["data_context"]
            # Remove verbose feature lists for cleaner display
            ctx_clean = {k: v for k, v in ctx.items() if k != "top_features"}
            st.json(ctx_clean)

    # ── Production upgrade note ────────────────────────────────────
    if not use_api:
        st.info(
            "💡 **Connect to the API** for real AI Copilot responses powered by "
            "live model inference. Switch the data source to *API* in the sidebar, "
            "or wire `_generate_ai_response()` in `main.py` to an LLM."
        )


# ═════════════════════════════════════════════════════════════════
# FEATURE STORE BUILDER  (Direct mode only)
# ═════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def build_demand_feature_store(_demand_df: pd.DataFrame,
                                demand_cols: list[str]) -> dict:
    df = _demand_df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    latest = df.sort_values("sale_date").groupby("product_id").tail(1)
    return {
        int(r["product_id"]): {c: float(r[c]) for c in demand_cols}
        for _, r in latest.iterrows()
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_stockout_feature_store(_so_df: pd.DataFrame,
                                  so_cols: list[str]) -> dict:
    df = _so_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    latest = df.sort_values("snapshot_date").groupby("product_id").tail(1)
    return {
        int(r["product_id"]): {
            c: float(pd.to_numeric(r[c], errors="coerce") or 0.0)
            for c in so_cols
        }
        for _, r in latest.iterrows()
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_risk_feature_store(_risk_df: pd.DataFrame,
                              risk_cols: list[str]) -> dict:
    df = _risk_df.copy()
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    df_filled = df.fillna(df.median(numeric_only=True))
    idx    = df.sort_values("scored_at").groupby("supplier_id").tail(1).index
    latest = df_filled.loc[idx]
    return {
        int(r["supplier_id"]): {c: float(r[c]) for c in risk_cols}
        for _, r in latest.iterrows()
    }


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    _inject_css()

    # ── Load core data ────────────────────────────────────────────
    with st.spinner("Loading data…"):
        try:
            sales      = load_sales()
            inventory  = load_inventory()
            products   = load_products()
            suppliers  = load_suppliers()
            risk_scores= load_risk_scores()
            forecasts  = load_forecasts()
        except FileNotFoundError as e:
            st.error(f"Data not found: {e}\n\nRun `python generate_mock_data.py` then `python etl_pipeline.py` first.")
            st.stop()

    # ── Sidebar ───────────────────────────────────────────────────
    use_api, sel_cat, sel_pid, sel_sid = render_sidebar(
        sales, inventory, products, forecasts
    )

    # ── Feature stores + models (Direct mode) ────────────────────
    demand_feats   = {}
    stockout_feats = {}
    risk_feats     = {}
    demand_model   = stockout_model = risk_reg = risk_clf = None

    if not use_api and MODELS_OK:
        try:
            demand_df_raw   = load_demand_features()
            stockout_df_raw = load_stockout_features()
            risk_df_raw     = pd.read_csv(FEATURES_DIR / "features_supplier_risk.csv", low_memory=False)

            from models import DEMAND_DROP, STOCKOUT_DROP, RISK_DROP
            d_cols  = [c for c in demand_df_raw.columns if c not in DEMAND_DROP]
            so_cols = [c for c in stockout_df_raw.columns if c not in STOCKOUT_DROP]
            r_cols  = [c for c in risk_df_raw.columns
                       if c not in RISK_DROP and pd.api.types.is_numeric_dtype(risk_df_raw[c])]

            demand_feats   = build_demand_feature_store(demand_df_raw, d_cols)
            stockout_feats = build_stockout_feature_store(stockout_df_raw, so_cols)
            risk_feats     = build_risk_feature_store(risk_df_raw, r_cols)

            demand_model, stockout_model, risk_reg, risk_clf = load_models_direct()
        except Exception as e:
            st.sidebar.warning(f"Direct inference unavailable: {e}")

    # ── Tabs ──────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "📈 Demand Forecast",
        "⚠️ Stockout Monitor",
        "🚚 Supplier Risk",
        "🤖 AI Copilot",
    ])

    with tabs[0]:
        tab_overview(sales, products)

    with tabs[1]:
        tab_forecast(forecasts, products, use_api, sel_pid,
                     demand_feats, demand_model)

    with tabs[2]:
        tab_stockout(inventory, products, use_api, sel_pid,
                     stockout_feats, stockout_model)

    with tabs[3]:
        tab_supplier_risk(risk_scores, suppliers, use_api, sel_sid,
                          risk_feats, risk_reg, risk_clf)

    with tabs[4]:
        tab_ai_copilot(use_api, sel_pid, sel_sid)


if __name__ == "__main__":
    main()