"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
Feature Engineering  |  Section 2-B
=============================================================
Three feature sets, each consumed by a dedicated ML model:

  FE-1  DEMAND FORECASTING
        Input : daily aggregated sales per product
        Output: lag features (7, 14, 21, 30 days),
                rolling averages (7, 14, 30 days),
                rolling std-dev (7, 30 days),
                calendar / seasonality flags,
                price and discount aggregates

  FE-2  STOCKOUT PREDICTION
        Input : inventory snapshots + sales aggregates
        Output: days_of_inventory_remaining,
                inventory_turnover_ratio,
                stockout_rate_30d,
                demand_acceleration,
                is_below_reorder, consecutive_low_stock_days

  FE-3  SUPPLIER RISK
        Input : risk_scores table (historical supplier stats)
        Output: hist_mean_delay_days, hist_delay_variance,
                hist_on_time_rate, delay_trend_30d,
                composite_risk_score, orders_evaluated,
                rolling_delay_mean_3m, delay_spike_flag

All three functions return fully-featured DataFrames ready to
be split into X / y by the modelling layer.

Requirements: pandas >= 1.4, numpy >= 1.20
=============================================================
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger("feature_engineering")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

CLEAN_DIR      = Path("./data/clean")
FEATURES_DIR   = Path("./data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════

def load_parquet(table: str, directory: Path = CLEAN_DIR) -> pd.DataFrame:
    path = directory / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Clean data not found for '{table}'. Run etl_pipeline.py first."
        )
    return pd.read_csv(path, low_memory=False)


def save_features(df: pd.DataFrame, name: str) -> Path:
    path = FEATURES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    kb = path.stat().st_size / 1024
    log.info(f"  ✓  {name:<35} {len(df):>8,} rows → {path.name}  ({kb:.1f} KB)")
    return path


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Attach standard calendar / seasonality columns to any DataFrame
    that has a datetime column.
    """
    dt = pd.to_datetime(df[date_col])
    df["dow"]            = dt.dt.dayofweek          # 0 = Monday
    df["month"]          = dt.dt.month
    df["quarter"]        = dt.dt.quarter
    df["week_of_year"]   = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"]    = dt.dt.dayofyear
    df["is_weekend"]     = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.dt.is_month_end.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # Cyclical encoding — lets models understand "December is near January"
    df["month_sin"]  = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"]          / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"]          / 7)
    df["woy_sin"]    = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]    = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Holiday proximity flags (simple rule-based; replace with country calendar)
    df["is_holiday_period"] = (
        ((df["month"] == 11) & (dt.dt.day >= 25)) |  # Black Friday window
        ((df["month"] == 12) & (dt.dt.day >= 15)) |  # Christmas run-up
        ((df["month"] == 1)  & (dt.dt.day <= 7))     # New Year carry-over
    ).astype(int)

    df["is_back_to_school"] = (
        (df["month"] == 8) | ((df["month"] == 9) & (dt.dt.day <= 15))
    ).astype(int)

    return df


# ═════════════════════════════════════════════════════════════
# FE-1  DEMAND FORECASTING FEATURES
# ═════════════════════════════════════════════════════════════

def build_demand_features(
    sales_df:    pd.DataFrame | None = None,
    products_df: pd.DataFrame | None = None,
    min_history_days: int = 30,
) -> pd.DataFrame:
    """
    Build a product × day feature table for the XGBoost demand
    forecasting model.

    Key feature groups
    ------------------
    1. Target          : daily_demand (quantity_sold net of returns)
    2. Lag features    : demand at t-7, t-14, t-21, t-30
    3. Rolling stats   : mean & std over 7, 14, 30-day windows
    4. Price/discount  : rolling avg price and discount depth
    5. Calendar        : DOW, month, cyclical encodings, holidays
    6. Product metadata: category dummies, unit_price tier

    Parameters
    ----------
    sales_df    : clean sales DataFrame (or loaded from parquet)
    products_df : clean products DataFrame (or loaded from parquet)
    min_history_days : products with fewer days of history are dropped

    Returns
    -------
    DataFrame with one row per (product_id, sale_date), fully featured,
    no nulls in core features (forward-filled where needed).
    """
    log.info("── FE-1: DEMAND FORECASTING FEATURES ────────────────")

    if sales_df is None:
        sales_df = load_parquet("sales")
    if products_df is None:
        products_df = load_parquet("products")

    # ── 1. Daily aggregation ──────────────────────────────────
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
    daily = (
        sales_df
        .groupby(["product_id", "sale_date"])
        .agg(
            daily_demand      = ("net_quantity",   "sum"),
            daily_revenue     = ("revenue",        "sum"),
            avg_unit_price    = ("unit_price",      "mean"),
            avg_discount_pct  = ("discount_pct",    "mean"),
            num_transactions  = ("sale_id",         "count"),
            return_count      = ("is_returned",     "sum"),
        )
        .reset_index()
    )

    # ── 2. Complete date spine (no gaps) ─────────────────────
    date_min = daily["sale_date"].min()
    date_max = daily["sale_date"].max()
    all_dates   = pd.date_range(date_min, date_max, freq="D")
    all_products= daily["product_id"].unique()

    spine = pd.MultiIndex.from_product(
        [all_products, all_dates], names=["product_id", "sale_date"]
    ).to_frame(index=False)

    daily = spine.merge(daily, on=["product_id", "sale_date"], how="left")
    daily["daily_demand"] = daily["daily_demand"].fillna(0)

    # Forward-fill price / discount from last known value per product
    daily = daily.sort_values(["product_id", "sale_date"])
    for col in ["avg_unit_price", "avg_discount_pct"]:
        daily[col] = (
            daily.groupby("product_id")[col]
                 .transform(lambda s: s.ffill().bfill())
        )
    daily["num_transactions"] = daily["num_transactions"].fillna(0)
    daily["return_count"]     = daily["return_count"].fillna(0)

    # ── 3. Lag features ───────────────────────────────────────
    for lag in [7, 14, 21, 30]:
        daily[f"lag_{lag}"] = (
            daily.groupby("product_id")["daily_demand"]
                 .shift(lag)
        )

    # ── 4. Rolling statistics ─────────────────────────────────
    grp = daily.groupby("product_id")["daily_demand"]
    for window in [7, 14, 30]:
        daily[f"rolling_mean_{window}d"] = (
            grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        daily[f"rolling_std_{window}d"] = (
            grp.transform(lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # Rolling price and discount
    for col, new_prefix in [("avg_unit_price","roll_price"),
                              ("avg_discount_pct","roll_discount")]:
        daily[f"{new_prefix}_7d"] = (
            daily.groupby("product_id")[col]
                 .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

    # ── 5. Demand momentum & acceleration ────────────────────
    daily["demand_delta_7d"] = (
        daily.groupby("product_id")["daily_demand"]
             .transform(lambda s: s - s.shift(7))
    )
    daily["demand_accel"] = (
        daily.groupby("product_id")["daily_demand"]
             .transform(lambda s: s.diff().diff())
    )

    # ── 6. Calendar features ──────────────────────────────────
    daily = _add_calendar_features(daily, "sale_date")

    # ── 7. Merge product metadata ─────────────────────────────
    prod_meta = products_df[
        ["product_id","category","unit_cost","unit_price",
         "lead_time_days","reorder_point"]
    ].copy()

    # Price tier (1=budget … 4=premium)
    prod_meta["price_tier"] = pd.cut(
        prod_meta["unit_price"],
        bins=[0, 20, 100, 400, np.inf],
        labels=[1, 2, 3, 4]
    ).astype(int)

    # Category dummies
    cat_dummies = pd.get_dummies(prod_meta["category"], prefix="cat").astype(int)
    prod_meta   = pd.concat([prod_meta, cat_dummies], axis=1).drop(columns="category")

    daily = daily.merge(prod_meta, on="product_id", how="left")

    # ── 8. Drop products with insufficient history ────────────
    history_len = daily.groupby("product_id")["daily_demand"].count()
    valid_prods = history_len[history_len >= min_history_days].index
    before      = daily["product_id"].nunique()
    daily       = daily[daily["product_id"].isin(valid_prods)]

    log.info(f"  Products retained: {daily['product_id'].nunique()} / {before}"
             f" (min_history={min_history_days}d)")
    log.info(f"  Date span: {daily['sale_date'].min().date()} → "
             f"{daily['sale_date'].max().date()}")
    log.info(f"  Feature columns: {daily.shape[1]}  |  Rows: {len(daily):,}")

    return daily


# ═════════════════════════════════════════════════════════════
# FE-2  STOCKOUT PREDICTION FEATURES
# ═════════════════════════════════════════════════════════════

def build_stockout_features(
    inventory_df: pd.DataFrame | None = None,
    sales_df:     pd.DataFrame | None = None,
    products_df:  pd.DataFrame | None = None,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Build a product × day feature table for the binary stockout
    classification model.

    Target
    ------
    stockout_in_horizon : 1 if quantity_on_hand = 0 at any point
                          in the next `horizon_days`, else 0

    Key feature groups
    ------------------
    - Inventory position : qty_on_hand, qty_available, reorder flags
    - Turnover metrics   : inventory_turnover_ratio, days_of_inventory_remaining
    - Recent demand      : rolling mean / std of daily demand (7, 14, 30 d)
    - Demand vs supply   : demand_to_stock_ratio, coverage_ratio
    - Risk indicators    : consecutive_low_stock_days, stockout_rate_30d
    - Calendar           : same seasonality features as FE-1

    Parameters
    ----------
    horizon_days : how many days ahead to look for a stockout event

    Returns
    -------
    DataFrame with one row per (product_id, snapshot_date), labelled.
    """
    log.info("── FE-2: STOCKOUT PREDICTION FEATURES ───────────────")

    if inventory_df is None:
        inventory_df = load_parquet("inventory")
    if sales_df is None:
        sales_df = load_parquet("sales")
    if products_df is None:
        products_df = load_parquet("products")

    inventory_df["snapshot_date"] = pd.to_datetime(inventory_df["snapshot_date"])
    sales_df["sale_date"]         = pd.to_datetime(sales_df["sale_date"])

    # ── 1. Daily demand aggregation ───────────────────────────
    daily_demand = (
        sales_df.groupby(["product_id", "sale_date"])["net_quantity"]
        .sum().reset_index()
        .rename(columns={"sale_date": "snapshot_date",
                          "net_quantity": "daily_demand"})
    )

    # ── 2. Join inventory + demand ────────────────────────────
    df = inventory_df.merge(
        daily_demand,
        on=["product_id", "snapshot_date"],
        how="left"
    )
    df["daily_demand"] = df["daily_demand"].fillna(0)

    # ── 3. Sort for rolling calcs ─────────────────────────────
    df = df.sort_values(["product_id", "snapshot_date"]).reset_index(drop=True)
    grp_demand = df.groupby("product_id")["daily_demand"]

    # ── 4. Rolling demand statistics ─────────────────────────
    for window in [7, 14, 30]:
        df[f"demand_mean_{window}d"] = (
            grp_demand.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
        )
        df[f"demand_std_{window}d"] = (
            grp_demand.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )
        )

    # ── 5. Inventory turnover ratio ───────────────────────────
    # turnover = (demand_mean_30d × 30) / avg_qty_on_hand_30d
    avg_stock_30d = (
        df.groupby("product_id")["quantity_on_hand"]
          .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )
    df["avg_stock_30d"] = avg_stock_30d
    df["inventory_turnover_30d"] = (
        (df["demand_mean_30d"] * 30)
        / (df["avg_stock_30d"] + 1e-9)
    ).clip(upper=999)

    # ── 6. Days of inventory remaining ───────────────────────
    # DoI = quantity_available / avg_daily_demand_7d
    df["days_of_inventory"] = (
        df["quantity_available"]
        / (df["demand_mean_7d"] + 1e-9)
    ).clip(upper=365)

    # ── 7. Coverage ratio ─────────────────────────────────────
    # How many days of lead time does current stock cover?
    # Drop product cols already present in inventory (from ETL transform)
    inv_cols_to_skip = {"reorder_point", "reorder_quantity"}
    prod_cols = [c for c in ["product_id", "lead_time_days", "reorder_point",
                              "reorder_quantity", "unit_cost"]
                 if c not in inv_cols_to_skip or c not in df.columns]
    # Only bring in columns not already in df (avoid _x/_y suffixes)
    new_prod_cols = ["product_id"] + [c for c in prod_cols
                                        if c != "product_id" and c not in df.columns]
    df = df.merge(products_df[new_prod_cols], on="product_id", how="left")
    df["lead_time_coverage"] = (
        df["quantity_available"]
        / (df["demand_mean_7d"] * df["lead_time_days"] + 1e-9)
    ).clip(upper=10)

    # ── 8. Demand-to-stock ratio ──────────────────────────────
    df["demand_to_stock_ratio"] = (
        df["demand_mean_7d"] / (df["quantity_on_hand"] + 1e-9)
    ).clip(upper=10)

    # ── 9. Stockout-rate in last 30 days ─────────────────────
    df["is_zero_stock"] = (df["quantity_on_hand"] == 0).astype(int)
    df["stockout_rate_30d"] = (
        df.groupby("product_id")["is_zero_stock"]
          .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )

    # ── 10. Consecutive low-stock days ────────────────────────
    # "Low" = below reorder point
    df["is_low_stock"] = (df["quantity_available"] <= df["reorder_point"]).astype(int)

    def _consec_count(s: pd.Series) -> pd.Series:
        """Running count of consecutive 1s, resets on 0."""
        result = np.zeros(len(s), dtype=float)
        count  = 0
        for i, v in enumerate(s):
            count = count + 1 if v == 1 else 0
            result[i] = count
        return pd.Series(result, index=s.index)

    df["consec_low_stock_days"] = (
        df.groupby("product_id")["is_low_stock"]
          .transform(_consec_count)
    )

    # ── 11. Demand acceleration ───────────────────────────────
    df["demand_accel_7d"] = (
        df.groupby("product_id")["daily_demand"]
          .transform(lambda s: s.diff(7))
    )

    # ── 12. TARGET: will there be a stockout in next N days? ──
    # For each (product, date), look forward horizon_days.
    # Easier to compute as: future_min_stock in [t+1 … t+horizon]
    df_pivot = df.pivot_table(
        index="snapshot_date", columns="product_id",
        values="quantity_on_hand", aggfunc="min"
    )
    df_pivot_future = df_pivot.copy()
    for step in range(1, horizon_days + 1):
        shifted = df_pivot.shift(-step)
        df_pivot_future = df_pivot_future.combine(shifted, lambda a, b: np.minimum(a, b))

    future_min = (
        df_pivot_future
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={0: "future_min_stock"})
    )
    df = df.merge(future_min, on=["snapshot_date", "product_id"], how="left")
    df["stockout_in_horizon"] = (
        df["future_min_stock"].fillna(0) == 0
    ).astype(int)

    log.info(f"  Target balance: "
             f"{df['stockout_in_horizon'].mean()*100:.1f}% positive "
             f"(horizon={horizon_days}d)")

    # ── 13. Calendar features ──────────────────────────────────
    df = _add_calendar_features(df, "snapshot_date")

    # ── 14. Drop rows with no meaningful feature data ─────────
    df = df.dropna(subset=["demand_mean_7d", "days_of_inventory"]).reset_index(drop=True)

    log.info(f"  Feature columns: {df.shape[1]}  |  Rows: {len(df):,}")
    return df


# ═════════════════════════════════════════════════════════════
# FE-3  SUPPLIER RISK FEATURES
# ═════════════════════════════════════════════════════════════

def build_supplier_risk_features(
    risk_scores_df: pd.DataFrame | None = None,
    suppliers_df:   pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a supplier × time-window feature table for the supplier
    risk regression / classification model.

    Key feature groups
    ------------------
    - Historical delay stats  : mean, variance, on-time rate
    - Rolling 3-month delay   : rolling_delay_mean_3m,
                                rolling_on_time_rate_3m
    - Trend signals           : delay_trend (slope of 3m window),
                                delay_spike_flag
    - Supplier metadata       : country_risk_tier, reliability_score,
                                payment_terms_days, years_active
    - Target options          : delay_probability (regression),
                                risk_tier_encoded (ordinal classification)

    Parameters
    ----------
    risk_scores_df : clean risk_scores DataFrame
    suppliers_df   : clean suppliers DataFrame

    Returns
    -------
    DataFrame with one row per (supplier_id, month), fully featured.
    """
    log.info("── FE-3: SUPPLIER RISK FEATURES ──────────────────────")

    if risk_scores_df is None:
        risk_scores_df = load_parquet("risk_scores")
    if suppliers_df is None:
        suppliers_df = load_parquet("suppliers")

    # ── 1. Parse timestamps ───────────────────────────────────
    risk_scores_df["scored_at"] = pd.to_datetime(
        risk_scores_df["scored_at"], utc=True
    ).dt.tz_localize(None)

    # Derive year-month period
    risk_scores_df["year_month"] = (
        risk_scores_df["scored_at"].dt.to_period("M").dt.to_timestamp()
    )

    # ── 2. Sort for rolling calcs ─────────────────────────────
    df = risk_scores_df.sort_values(["supplier_id", "year_month"]).copy()
    grp = df.groupby("supplier_id")

    # ── 3. Rolling 3-month delay statistics ──────────────────
    df["rolling_delay_mean_3m"] = (
        grp["hist_mean_delay_days"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_delay_std_3m"] = (
        grp["hist_mean_delay_days"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).std().fillna(0))
    )
    df["rolling_on_time_rate_3m"] = (
        grp["hist_on_time_rate"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_delay_prob_3m"] = (
        grp["delay_probability"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # ── 4. Delay trend (slope over 3 months) ─────────────────
    def _slope(s: pd.Series) -> pd.Series:
        """Compute rolling OLS slope over a 3-period window."""
        result = np.full(len(s), np.nan)
        arr    = s.values
        for i in range(len(arr)):
            window_vals = arr[max(0, i-2): i+1]
            if len(window_vals) >= 2:
                x = np.arange(len(window_vals), dtype=float)
                coeffs     = np.polyfit(x, window_vals.astype(float), 1)
                result[i]  = coeffs[0]   # slope
        return pd.Series(result, index=s.index)

    df["delay_trend_3m"] = grp["hist_mean_delay_days"].transform(_slope)

    # ── 5. Spike flag (current delay >> rolling average) ─────
    df["delay_spike_flag"] = (
        df["hist_mean_delay_days"] > (df["rolling_delay_mean_3m"] + 2 * df["rolling_delay_std_3m"])
    ).astype(int)

    # ── 6. Lag features (previous month's delay / probability) ─
    df["lag_1m_delay_mean"]   = grp["hist_mean_delay_days"].transform(lambda s: s.shift(1))
    df["lag_1m_delay_prob"]   = grp["delay_probability"].transform(lambda s: s.shift(1))
    df["lag_2m_delay_mean"]   = grp["hist_mean_delay_days"].transform(lambda s: s.shift(2))

    # ── 7. Merge supplier metadata ────────────────────────────
    suppliers_df["onboarded_at"] = pd.to_datetime(
        suppliers_df["onboarded_at"], errors="coerce"
    )
    # Reference date for tenure calculation
    ref_date = pd.Timestamp("2024-12-31")
    suppliers_df["years_active"] = (
        (ref_date - suppliers_df["onboarded_at"]).dt.days / 365.25
    ).round(2)

    # Country risk tier: simple rule-based mapping
    high_risk_countries  = {"Bangladesh", "Indonesia", "Pakistan", "Nigeria"}
    medium_risk_countries= {"India", "Vietnam", "China", "Mexico", "Brazil"}
    def _country_risk(country: str) -> int:
        if country in high_risk_countries:
            return 3
        elif country in medium_risk_countries:
            return 2
        return 1

    suppliers_df["country_risk_tier"] = suppliers_df["country"].apply(_country_risk)

    df = df.merge(
        suppliers_df[[
            "supplier_id", "reliability_score", "payment_terms_days",
            "country_risk_tier", "years_active", "country", "region"
        ]],
        on="supplier_id", how="left"
    )

    # ── 8. Ordinal risk tier encoding ─────────────────────────
    tier_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    df["risk_tier_encoded"] = df["risk_tier"].map(tier_map).fillna(1).astype(int)

    # ── 9. Interaction features ───────────────────────────────
    df["reliability_x_ontime"] = df["reliability_score"] * df["hist_on_time_rate"]
    df["delay_var_x_mean"]     = df["hist_delay_variance"] * df["hist_mean_delay_days"]

    # ── 10. Country / region dummies ─────────────────────────
    region_dummies = pd.get_dummies(df["region"], prefix="region").astype(int)
    df = pd.concat([df, region_dummies], axis=1)

    # ── 11. Drop nulls only for critical feature columns ─────
    core_cols = [
        "hist_mean_delay_days", "hist_on_time_rate",
        "delay_probability", "composite_risk_score",
    ]
    df = df.dropna(subset=core_cols).reset_index(drop=True)

    log.info(f"  Suppliers: {df['supplier_id'].nunique()}")
    log.info(f"  Months covered: {df['year_month'].nunique()}")
    log.info(f"  Feature columns: {df.shape[1]}  |  Rows: {len(df):,}")
    return df


# ═════════════════════════════════════════════════════════════
# FEATURE SUMMARY REPORT
# ═════════════════════════════════════════════════════════════

def feature_summary(dfs: dict[str, pd.DataFrame]) -> None:
    """Print a quick quality summary for each feature set."""
    print("\n" + "="*62)
    print("  FEATURE ENGINEERING SUMMARY")
    print("="*62)

    targets = {
        "demand":        "daily_demand",
        "stockout":      "stockout_in_horizon",
        "supplier_risk": "delay_probability",
    }

    for name, df in dfs.items():
        target = targets.get(name, "—")
        null_pct = df.isnull().mean().mean() * 100
        print(f"\n[{name}]")
        print(f"  Rows          : {len(df):,}")
        print(f"  Columns       : {df.shape[1]}")
        print(f"  Null rate     : {null_pct:.2f}%  (across all columns)")
        if target in df.columns:
            if df[target].dtype in [float, int, "float64", "int64"]:
                print(f"  Target '{target}':  "
                      f"mean={df[target].mean():.3f}  "
                      f"std={df[target].std():.3f}  "
                      f"min={df[target].min():.2f}  "
                      f"max={df[target].max():.2f}")
    print("="*62 + "\n")


# ═════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════

def run_feature_engineering(
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build all three feature sets and optionally persist to Parquet.

    Returns
    -------
    {
        "demand":        demand_features_df,
        "stockout":      stockout_features_df,
        "supplier_risk": supplier_risk_features_df,
    }
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  Feature Engineering Pipeline — starting")
    log.info("═══════════════════════════════════════════════════════")

    demand_df   = build_demand_features()
    stockout_df = build_stockout_features()
    risk_df     = build_supplier_risk_features()

    feature_sets = {
        "demand":        demand_df,
        "stockout":      stockout_df,
        "supplier_risk": risk_df,
    }

    if save:
        log.info("── Saving feature sets ───────────────────────────────")
        for name, df in feature_sets.items():
            save_features(df, f"features_{name}")

    feature_summary(feature_sets)

    log.info("═══════════════════════════════════════════════════════")
    log.info("  Feature engineering complete.")
    log.info("═══════════════════════════════════════════════════════")

    return feature_sets


if __name__ == "__main__":
    run_feature_engineering()