"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
Mock Data Generator  |  Section 1
=============================================================
Generates 12 months of realistic synthetic data for all six
database tables, complete with:
  - Demand seasonality (weekly + annual cycles)
  - Product-specific trend lines (growth / decline / flat)
  - Deliberate edge cases: stockouts, supplier delays, returns,
    demand spikes (holidays), and slow-moving SKUs
  - Coherent inventory snapshots that react to sales volume
  - Supplier reliability profiles (excellent → unreliable)
  - Risk scores derived from simulated supplier behaviour

Output: one CSV per table in ./data/raw/
        optionally loads directly into PostgreSQL

Requirements: pandas >= 1.4, numpy >= 1.20
Optional:     psycopg2-binary (for direct DB load)
=============================================================
"""

import os
import math
import random
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

START_DATE  = date(2024, 1, 1)
END_DATE    = date(2024, 12, 31)
DATE_RANGE  = pd.date_range(START_DATE, END_DATE, freq="D")

N_PRODUCTS  = 40
N_SUPPLIERS = 10
WAREHOUSES  = ["WH-001", "WH-002", "WH-003"]

OUTPUT_DIR  = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────

def to_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  ✓  {name:30s}  {len(df):>7,} rows  →  {path}")
    return path


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def business_day_factor(d: date) -> float:
    """Sales are ~40 % lower on weekends."""
    return 0.60 if d.weekday() >= 5 else 1.0


def seasonality_multiplier(d: date, profile: str) -> float:
    """
    Annual seasonality shaped by product profile:
      'general'     – mild Q4 uplift
      'fashion'     – spring + autumn peaks
      'electronics' – back-to-school (Aug) + holiday (Nov-Dec) spikes
      'food'        – summer peak, quiet winter
    """
    month = d.month
    if profile == "electronics":
        base = {1:0.80, 2:0.75, 3:0.85, 4:0.85, 5:0.90, 6:0.90,
                7:0.95, 8:1.20, 9:1.10, 10:1.10, 11:1.40, 12:1.60}
    elif profile == "fashion":
        base = {1:0.70, 2:0.80, 3:1.20, 4:1.30, 5:1.10, 6:0.90,
                7:0.85, 8:1.00, 9:1.30, 10:1.20, 11:1.00, 12:0.90}
    elif profile == "food":
        base = {1:0.90, 2:0.85, 3:0.95, 4:1.00, 5:1.05, 6:1.20,
                7:1.30, 8:1.25, 9:1.10, 10:1.00, 11:0.95, 12:1.00}
    else:  # general
        base = {1:0.85, 2:0.85, 3:0.90, 4:0.95, 5:1.00, 6:0.95,
                7:0.95, 8:1.00, 9:1.00, 10:1.00, 11:1.10, 12:1.20}
    return base[month]


def is_holiday_spike(d: date) -> float:
    """Known retail event multipliers layered on top of seasonality."""
    spikes = {
        # (month, day): multiplier
        (1,  1):  0.40,   # New Year – very quiet
        (2, 14):  1.30,   # Valentine's Day
        (7,  4):  1.25,   # Independence Day (US)
        (11, 29): 2.50,   # Black Friday
        (11, 30): 2.20,   # Saturday after Black Friday
        (12,  1): 1.80,   # Cyber Monday adjacent
        (12, 24): 1.90,   # Christmas Eve
        (12, 25): 0.30,   # Christmas – store closures
        (12, 31): 1.20,   # New Year's Eve
    }
    return spikes.get((d.month, d.day), 1.0)


# ─────────────────────────────────────────────
# SECTION 1-A : PRODUCTS
# ─────────────────────────────────────────────

CATEGORIES = {
    "Electronics": {
        "sub": ["Smartphones", "Laptops", "Accessories", "Audio", "Wearables"],
        "cost_range": (50, 800),
        "margin": (1.20, 1.60),
        "lead_time": (5, 21),
        "reorder_point": (20, 80),
        "profile": "electronics",
    },
    "Apparel": {
        "sub": ["T-Shirts", "Jeans", "Jackets", "Footwear", "Accessories"],
        "cost_range": (8, 120),
        "margin": (1.80, 3.50),
        "lead_time": (10, 45),
        "reorder_point": (30, 150),
        "profile": "fashion",
    },
    "Food & Beverage": {
        "sub": ["Snacks", "Beverages", "Canned Goods", "Dairy", "Confectionery"],
        "cost_range": (1, 30),
        "margin": (1.30, 2.00),
        "lead_time": (2, 10),
        "reorder_point": (100, 500),
        "profile": "food",
    },
    "Home & Garden": {
        "sub": ["Furniture", "Lighting", "Cleaning", "Tools", "Decor"],
        "cost_range": (10, 300),
        "margin": (1.40, 2.50),
        "lead_time": (7, 30),
        "reorder_point": (15, 80),
        "profile": "general",
    },
}


def generate_products() -> pd.DataFrame:
    records = []
    cat_list = list(CATEGORIES.keys())

    for pid in range(1, N_PRODUCTS + 1):
        cat_name = cat_list[(pid - 1) % len(cat_list)]
        cfg      = CATEGORIES[cat_name]
        sub      = cfg["sub"][(pid - 1) // len(cat_list) % len(cfg["sub"])]
        cost     = round(rng.uniform(*cfg["cost_range"]), 2)
        price    = round(cost * rng.uniform(*cfg["margin"]), 2)
        lt       = int(rng.integers(*cfg["lead_time"]))
        rp       = int(rng.integers(*cfg["reorder_point"]))
        rq       = rp * int(rng.integers(2, 5))

        records.append({
            "product_id":       pid,
            "sku":              f"SKU-{pid:04d}",
            "product_name":     f"{sub} Product {pid:02d}",
            "category":         cat_name,
            "sub_category":     sub,
            "unit_cost":        cost,
            "unit_price":       price,
            "lead_time_days":   lt,
            "reorder_point":    rp,
            "reorder_quantity": rq,
            "is_active":        pid <= N_PRODUCTS - 2,   # last 2 are inactive/EOL
            "created_at":       "2023-01-01",
            "updated_at":       "2024-01-01",
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SECTION 1-B : SUPPLIERS
# ─────────────────────────────────────────────

SUPPLIER_PROFILES = [
    # (name_template, country, reliability_score, payment_terms)
    ("AlphaSource Global",     "China",       9.2,  30),
    ("BetaTrade Partners",     "India",       7.8,  45),
    ("Gamma Logistics Ltd",    "USA",         8.5,  15),
    ("Delta Supply Co",        "Germany",     9.5,  30),
    ("Epsilon Vendors",        "Vietnam",     6.5,  60),
    ("Zeta Manufacturing",     "Bangladesh",  5.8,  90),  # unreliable
    ("Eta Components Inc",     "Mexico",      8.0,  30),
    ("Theta Global Trade",     "UK",          9.0,  30),
    ("Iota Fast Supply",       "South Korea", 8.8,  15),
    ("Kappa Budget Source",    "Indonesia",   4.5,  90),  # very unreliable / cheap
]


def generate_suppliers() -> pd.DataFrame:
    records = []
    for sid, (name, country, score, terms) in enumerate(SUPPLIER_PROFILES, start=1):
        records.append({
            "supplier_id":        sid,
            "supplier_name":      name,
            "contact_email":      f"orders@{name.lower().replace(' ', '')[:12]}.com",
            "country":            country,
            "region":             "Asia" if country in ("China","India","Vietnam","Bangladesh","South Korea","Indonesia") else
                                  "Americas" if country in ("USA","Mexico") else "Europe",
            "reliability_score":  score,
            "payment_terms_days": terms,
            "is_active":          True,
            "onboarded_at":       "2022-06-01",
            "created_at":         "2022-06-01",
            "updated_at":         "2024-01-01",
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SECTION 1-C : SALES  (largest, most complex)
# ─────────────────────────────────────────────

# Assign each product a demand profile
PRODUCT_DEMAND_CONFIG: dict[int, dict] = {}

def _init_demand_config(products_df: pd.DataFrame):
    for _, row in products_df.iterrows():
        pid    = row["product_id"]
        cfg    = CATEGORIES[row["category"]]
        # Base daily demand scaled by price tier
        price  = row["unit_price"]
        if price < 20:
            base_demand = rng.integers(30, 120)
        elif price < 100:
            base_demand = rng.integers(10, 50)
        elif price < 500:
            base_demand = rng.integers(3, 20)
        else:
            base_demand = rng.integers(1, 8)

        # Long-term trend: -1 (declining), 0 (flat), +1 (growing)
        trend = rng.choice([-1, 0, 1], p=[0.2, 0.5, 0.3])

        # Noise level
        noise_std = max(1.0, base_demand * rng.uniform(0.10, 0.30))

        PRODUCT_DEMAND_CONFIG[pid] = {
            "base":    int(base_demand),
            "trend":   int(trend),
            "noise":   float(noise_std),
            "profile": cfg["profile"],
            "price":   float(price),
        }


def _stockout_windows(pid: int) -> list[tuple[date, date]]:
    """Randomly assign 0-3 stockout windows per product per year."""
    n_events = rng.integers(0, 4)
    windows  = []
    for _ in range(n_events):
        start_offset = int(rng.integers(0, 330))
        duration     = int(rng.integers(2, 10))
        ws = START_DATE + timedelta(days=start_offset)
        we = ws + timedelta(days=duration)
        windows.append((ws, we))
    return windows


def generate_sales(products_df: pd.DataFrame) -> pd.DataFrame:
    _init_demand_config(products_df)

    channels = ["retail", "online", "wholesale"]
    ch_wts   = [0.50, 0.35, 0.15]
    regions  = ["North", "South", "East", "West", "Central"]

    rows = []
    sale_id = 1

    for pid in products_df["product_id"].tolist():
        cfg        = PRODUCT_DEMAND_CONFIG[pid]
        stockouts  = _stockout_windows(pid)
        total_days = (END_DATE - START_DATE).days + 1

        for day_idx, ts in enumerate(DATE_RANGE):
            d = ts.date()

            # Are we in a stockout window? → no sales
            in_stockout = any(ws <= d <= we for ws, we in stockouts)
            if in_stockout:
                continue

            # Build demand signal
            trend_factor = 1.0 + cfg["trend"] * (day_idx / total_days) * 0.30
            season_mult  = seasonality_multiplier(d, cfg["profile"])
            holiday_mult = is_holiday_spike(d)
            bday_factor  = business_day_factor(d)
            noise        = rng.normal(0, cfg["noise"])

            qty_float = (cfg["base"] * trend_factor * season_mult
                         * holiday_mult * bday_factor + noise)
            qty = max(0, int(round(qty_float)))

            if qty == 0:
                continue

            # Micro-level: split across 1-4 transactions per day
            n_tx = int(rng.integers(1, 5)) if qty > 5 else 1
            splits = np.diff(np.sort(rng.integers(0, qty + 1, n_tx - 1)))
            splits = np.concatenate([[rng.integers(1, max(2, qty // n_tx + 1))], splits])
            splits = splits[splits > 0]
            if splits.sum() == 0:
                splits = np.array([qty])

            for q in splits:
                q = int(q)
                if q <= 0:
                    continue
                channel   = rng.choice(channels, p=ch_wts)
                discount  = round(float(rng.choice([0, 5, 10, 15, 20],
                                                   p=[0.60, 0.15, 0.12, 0.08, 0.05])), 2)
                eff_price = round(cfg["price"] * (1 - discount / 100), 2)
                is_ret    = bool(rng.random() < 0.02)   # 2 % return rate

                rows.append({
                    "sale_id":       sale_id,
                    "product_id":    pid,
                    "sale_date":     d.isoformat(),
                    "quantity_sold": q,
                    "unit_price":    eff_price,
                    "channel":       channel,
                    "region":        rng.choice(regions),
                    "discount_pct":  discount,
                    "is_returned":   is_ret,
                    "created_at":    d.isoformat(),
                })
                sale_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["sale_date", "product_id"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# SECTION 1-D : INVENTORY  (daily snapshots)
# ─────────────────────────────────────────────

def generate_inventory(products_df: pd.DataFrame,
                        sales_df:    pd.DataFrame) -> pd.DataFrame:
    """
    Simulate daily stock-on-hand per product per warehouse.
    Algorithm:
      - Start with random opening stock (above reorder point)
      - Subtract daily sales proportionally across warehouses
      - Trigger replenishment when stock falls below reorder point
        (replenishment arrives after lead_time_days, adding reorder_qty)
      - Clamp qty_on_hand to >= 0 (creates true stockout records)
    """
    # Pre-aggregate daily sales per product
    daily_sales = (
        sales_df.groupby(["product_id", "sale_date"])["quantity_sold"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_sold": "total_sold"})
    )
    daily_sales["sale_date"] = pd.to_datetime(daily_sales["sale_date"]).dt.date

    rows = []
    inv_id = 1

    for _, prod in products_df.iterrows():
        pid  = prod["product_id"]
        rp   = int(prod["reorder_point"])
        rq   = int(prod["reorder_quantity"])
        cost = float(prod["unit_cost"])
        lt   = int(prod["lead_time_days"])
        wh   = rng.choice(WAREHOUSES)

        # Opening stock: 1.5–4× reorder point
        qty   = int(rp * rng.uniform(1.5, 4.0))
        pending_replenishments: list[tuple[date, int]] = []

        for ts in DATE_RANGE:
            d = ts.date()

            # Deliver any pending replenishments
            arrived = [qty_r for (arr_d, qty_r) in pending_replenishments if arr_d == d]
            for a in arrived:
                qty += a
            pending_replenishments = [(arr_d, qty_r)
                                      for (arr_d, qty_r) in pending_replenishments
                                      if arr_d != d]

            # Subtract today's sales (primary warehouse absorbs 70-100%)
            sold_today = int(
                daily_sales.loc[
                    (daily_sales["product_id"] == pid) &
                    (daily_sales["sale_date"]   == d),
                    "total_sold"
                ].sum()
            )
            qty = max(0, qty - sold_today)

            # Reorder logic
            reorder_triggered = False
            if qty <= rp and not any(True for (arr_d, _) in pending_replenishments):
                reorder_triggered = True
                arrival = d + timedelta(days=lt)
                # Supplier delay: unreliable suppliers add 0-15 extra days
                delay = int(rng.integers(0, 16)) if rng.random() < 0.25 else 0
                pending_replenishments.append((arrival + timedelta(days=delay), rq))

            reserved = min(qty, int(rng.integers(0, max(1, qty // 4 + 1))))

            rows.append({
                "inventory_id":       inv_id,
                "product_id":         pid,
                "snapshot_date":      d.isoformat(),
                "warehouse_id":       wh,
                "quantity_on_hand":   qty,
                "quantity_reserved":  reserved,
                "reorder_triggered":  reorder_triggered,
                "unit_cost_snapshot": round(cost * rng.uniform(0.97, 1.03), 2),
                "created_at":         d.isoformat(),
            })
            inv_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SECTION 1-E : FORECASTS
# ─────────────────────────────────────────────

def generate_forecasts(products_df: pd.DataFrame,
                        sales_df:    pd.DataFrame) -> pd.DataFrame:
    """
    Simulate model-generated forecasts for the last 60 days of the year
    (Oct–Dec), with realistic errors and prediction intervals.
    Actual demand is backfilled for evaluation metrics.
    """
    # Aggregate real daily demand
    daily_actual = (
        sales_df.groupby(["product_id", "sale_date"])["quantity_sold"]
        .sum()
        .reset_index()
    )
    daily_actual["sale_date"] = pd.to_datetime(daily_actual["sale_date"]).dt.date

    forecast_start = date(2024, 10, 1)
    forecast_dates = pd.date_range(forecast_start, END_DATE, freq="D")

    rows = []
    fc_id = 1

    for _, prod in products_df.iterrows():
        pid = prod["product_id"]
        if pid not in PRODUCT_DEMAND_CONFIG:
            continue
        cfg = PRODUCT_DEMAND_CONFIG[pid]

        for ts in forecast_dates:
            fd = ts.date()

            # Simulate model prediction (true + noise)
            true_demand = (
                cfg["base"]
                * seasonality_multiplier(fd, cfg["profile"])
                * is_holiday_spike(fd)
                * business_day_factor(fd)
            )
            # Model error: ±15 % systematic + random noise
            pred_error  = rng.normal(0, true_demand * 0.12)
            predicted   = max(0, round(true_demand + pred_error, 2))

            # 95 % prediction interval (±20 %)
            margin      = predicted * 0.20
            lower       = max(0, round(predicted - margin, 2))
            upper       = round(predicted + margin, 2)

            # Actual demand (from sales data)
            actual_val = float(
                daily_actual.loc[
                    (daily_actual["product_id"] == pid) &
                    (daily_actual["sale_date"]   == fd),
                    "quantity_sold"
                ].sum()
            )
            actual = round(actual_val, 2)

            mae  = round(abs(predicted - actual), 4)
            # Cap MAPE at 100 % when actual is very small (avoids astronomic values)
            mape = round(
                min(100.0, abs(predicted - actual) / (actual + 1.0) * 100), 4
            )
            conf = round(clamp(1 - (abs(pred_error) / (true_demand + 1e-9)), 0.50, 0.99), 4)

            rows.append({
                "forecast_id":       fc_id,
                "product_id":        pid,
                "forecast_date":     fd.isoformat(),
                "generated_at":      (fd - timedelta(days=7)).isoformat(),
                "model_name":        "xgboost_demand_v1",
                "model_version":     "1.0.0",
                "predicted_demand":  predicted,
                "lower_bound_95":    lower,
                "upper_bound_95":    upper,
                "actual_demand":     actual,
                "mae":               mae,
                "mape":              mape,
                "confidence_score":  conf,
                "horizon_days":      7,
            })
            fc_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SECTION 1-F : RISK_SCORES
# ─────────────────────────────────────────────

# Delay distributions per supplier (in days beyond promised)
# Keyed by supplier_id; higher values = worse supplier
SUPPLIER_DELAY_PROFILES = {
    1: {"mean": 0.5,  "std": 0.8,  "max_delay": 3},   # AlphaSource  – excellent
    2: {"mean": 2.0,  "std": 2.5,  "max_delay": 10},  # BetaTrade    – good
    3: {"mean": 1.0,  "std": 1.2,  "max_delay": 5},   # Gamma        – good
    4: {"mean": 0.2,  "std": 0.5,  "max_delay": 2},   # Delta        – excellent
    5: {"mean": 4.0,  "std": 4.0,  "max_delay": 18},  # Epsilon      – average
    6: {"mean": 7.0,  "std": 6.0,  "max_delay": 25},  # Zeta         – poor
    7: {"mean": 1.5,  "std": 1.8,  "max_delay": 8},   # Eta          – good
    8: {"mean": 0.8,  "std": 0.9,  "max_delay": 4},   # Theta        – excellent
    9: {"mean": 1.2,  "std": 1.0,  "max_delay": 5},   # Iota         – good
    10:{"mean": 10.0, "std": 8.0,  "max_delay": 35},  # Kappa        – very poor
}


def _simulate_supplier_orders(supplier_id: int, n_orders: int = 60
                               ) -> pd.DataFrame:
    """Simulate historical delivery records for one supplier."""
    dp = SUPPLIER_DELAY_PROFILES[supplier_id]
    delays = np.clip(
        rng.normal(dp["mean"], dp["std"], n_orders),
        0, dp["max_delay"]
    )
    return pd.DataFrame({
        "supplier_id": supplier_id,
        "delay_days":  delays,
        "on_time":     delays <= 0.5,
    })


def generate_risk_scores(suppliers_df: pd.DataFrame) -> pd.DataFrame:
    """
    One risk score row per supplier, scored monthly (12 snapshots).
    Derives metrics from simulated historical order records.
    """
    rows = []
    risk_id = 1

    scored_months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")

    for _, sup in suppliers_df.iterrows():
        sid = int(sup["supplier_id"])
        dp  = SUPPLIER_DELAY_PROFILES[sid]

        for scored_ts in scored_months:
            scored_at = scored_ts.date()

            # Simulate ~60 historical orders per supplier
            orders_df = _simulate_supplier_orders(sid, n_orders=60)

            hist_mean      = float(orders_df["delay_days"].mean())
            hist_var       = float(orders_df["delay_days"].var())
            on_time_rate   = float(orders_df["on_time"].mean())
            n_orders       = len(orders_df)

            # Predicted delay (model adds small random error to history)
            pred_delay = max(0, round(hist_mean + rng.normal(0, dp["std"] * 0.15), 2))
            # Binary: probability delay > 2 days
            delay_prob = round(clamp(float((orders_df["delay_days"] > 2).mean())
                                     + rng.normal(0, 0.03), 0.0, 1.0), 4)

            # Composite risk score (0 = low risk, 100 = max risk)
            composite = round(clamp(
                delay_prob * 40 + (hist_mean / dp["max_delay"]) * 40
                + (1 - on_time_rate) * 20, 0, 100
            ), 2)

            if composite < 25:
                tier   = "low"
                action = "No action required; continue standard ordering."
            elif composite < 50:
                tier   = "medium"
                action = "Monitor delivery windows; consider dual-sourcing."
            elif composite < 75:
                tier   = "high"
                action = "Increase safety stock; request SLA commitment."
            else:
                tier   = "critical"
                action = "Escalate to procurement; initiate alternative sourcing."

            # Top SHAP features (stored as JSON)
            shap = {
                "hist_mean_delay":   round(hist_mean, 3),
                "on_time_rate":      round(on_time_rate, 3),
                "hist_delay_var":    round(hist_var, 3),
                "order_frequency":   n_orders,
            }

            rows.append({
                "risk_id":               risk_id,
                "supplier_id":           sid,
                "scored_at":             f"{scored_at}T00:00:00Z",
                "model_name":            "supplier_risk_v1",
                "model_version":         "1.0.0",
                "delay_probability":     delay_prob,
                "avg_delay_days_pred":   pred_delay,
                "risk_tier":             tier,
                "composite_risk_score":  composite,
                "hist_mean_delay_days":  round(hist_mean, 4),
                "hist_delay_variance":   round(hist_var,  4),
                "hist_on_time_rate":     round(on_time_rate, 4),
                "orders_evaluated":      n_orders,
                "shap_top_features":     str(shap),   # JSON-serialisable in DB loader
                "recommended_action":    action,
                "created_at":            f"{scored_at}T00:00:00Z",
            })
            risk_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SECTION 1-G : PRODUCT_SUPPLIER (junction)
# ─────────────────────────────────────────────

def generate_product_supplier(products_df:  pd.DataFrame,
                               suppliers_df: pd.DataFrame) -> pd.DataFrame:
    """Each product gets 1 primary + 0-2 secondary suppliers."""
    rows = []
    row_id = 1
    supplier_ids = suppliers_df["supplier_id"].tolist()

    for _, prod in products_df.iterrows():
        pid = prod["product_id"]
        # Primary supplier (exclude the 2 most unreliable for expensive products)
        pool = supplier_ids[:8] if prod["unit_price"] > 200 else supplier_ids
        primary_sid = int(rng.choice(pool))

        rows.append({
            "id":                    row_id,
            "product_id":            pid,
            "supplier_id":           primary_sid,
            "is_primary":            True,
            "quoted_lead_time_days": int(prod["lead_time_days"]),
            "last_quoted_price":     round(float(prod["unit_cost"]) * rng.uniform(0.95, 1.05), 2),
            "contract_start":        "2023-01-01",
            "contract_end":          "2025-12-31",
            "created_at":            "2023-01-01",
        })
        row_id += 1

        # 0-2 secondary suppliers
        n_secondary = int(rng.integers(0, 3))
        alt_pool = [s for s in supplier_ids if s != primary_sid]
        for secondary_sid in rng.choice(alt_pool, size=min(n_secondary, len(alt_pool)),
                                         replace=False):
            rows.append({
                "id":                    row_id,
                "product_id":            pid,
                "supplier_id":           int(secondary_sid),
                "is_primary":            False,
                "quoted_lead_time_days": int(prod["lead_time_days"]) + int(rng.integers(0, 8)),
                "last_quoted_price":     round(float(prod["unit_cost"]) * rng.uniform(1.02, 1.15), 2),
                "contract_start":        "2023-06-01",
                "contract_end":          "2025-06-30",
                "created_at":            "2023-06-01",
            })
            row_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# OPTIONAL: DIRECT POSTGRES LOADER
# ─────────────────────────────────────────────

def load_to_postgres(dfs: dict[str, pd.DataFrame], db_url: str):
    """
    Bulk-load all DataFrames into PostgreSQL.
    Requires: pip install psycopg2-binary sqlalchemy

    Usage:
        load_to_postgres(tables, "postgresql://user:pass@localhost:5432/supplychain_db")
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("⚠  SQLAlchemy not installed. Run: pip install psycopg2-binary sqlalchemy")
        return

    engine = create_engine(db_url, echo=False)
    with engine.connect() as conn:
        # Truncate in FK-safe order (children first)
        order = [
            "risk_scores", "forecasts", "inventory",
            "sales", "product_supplier", "products", "suppliers"
        ]
        for table in order:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
            except Exception:
                pass
        conn.commit()

    insert_order = [
        "products", "suppliers", "product_supplier",
        "sales", "inventory", "forecasts", "risk_scores"
    ]
    for table in insert_order:
        if table in dfs:
            dfs[table].to_sql(table, engine, if_exists="append", index=False, chunksize=5000)
            print(f"  ✓  Loaded {len(dfs[table]):>7,} rows → {table}")

    print("\n✅ All tables loaded to PostgreSQL.")


# ─────────────────────────────────────────────
# VALIDATION REPORT
# ─────────────────────────────────────────────

def print_validation_report(tables: dict[str, pd.DataFrame]):
    print("\n" + "="*60)
    print("  DATA VALIDATION REPORT")
    print("="*60)

    # Products
    p = tables["products"]
    print(f"\n[Products] {len(p)} rows")
    print(f"  Categories      : {p['category'].value_counts().to_dict()}")
    print(f"  Active SKUs     : {p['is_active'].sum()}")
    print(f"  Price range     : ${p['unit_price'].min():.2f} – ${p['unit_price'].max():.2f}")

    # Suppliers
    s = tables["suppliers"]
    print(f"\n[Suppliers] {len(s)} rows")
    print(f"  Reliability range : {s['reliability_score'].min()} – {s['reliability_score'].max()}")
    print(f"  Countries         : {s['country'].nunique()} unique")

    # Sales
    sl = tables["sales"]
    print(f"\n[Sales] {len(sl):,} rows")
    print(f"  Date range      : {sl['sale_date'].min()} → {sl['sale_date'].max()}")
    print(f"  Total units sold: {sl['quantity_sold'].sum():,}")
    print(f"  Return rate     : {sl['is_returned'].mean()*100:.1f}%")
    print(f"  Channels        : {sl['channel'].value_counts().to_dict()}")

    # Inventory
    iv = tables["inventory"]
    stockout_days = (iv["quantity_on_hand"] == 0).sum()
    print(f"\n[Inventory] {len(iv):,} rows")
    print(f"  Stockout records    : {stockout_days:,}  ({stockout_days/len(iv)*100:.1f}%)")
    print(f"  Reorder triggers    : {iv['reorder_triggered'].sum():,}")
    print(f"  Avg qty on hand     : {iv['quantity_on_hand'].mean():.0f}")

    # Forecasts
    fc = tables["forecasts"]
    print(f"\n[Forecasts] {len(fc):,} rows")
    print(f"  Avg MAPE            : {fc['mape'].mean():.2f}%")
    print(f"  Avg MAE             : {fc['mae'].mean():.2f}")
    print(f"  Avg confidence      : {fc['confidence_score'].mean():.3f}")

    # Risk scores
    rs = tables["risk_scores"]
    print(f"\n[Risk Scores] {len(rs):,} rows")
    print(f"  Risk tier distribution : {rs['risk_tier'].value_counts().to_dict()}")
    print(f"  Avg composite score    : {rs['composite_risk_score'].mean():.1f}/100")
    print(f"  Avg delay probability  : {rs['delay_probability'].mean():.3f}")
    print("="*60)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main(db_url: str | None = None):
    print("\n🚀 Intelligent Supply Chain – Mock Data Generator")
    print(f"   Period : {START_DATE} → {END_DATE}")
    print(f"   Output : {OUTPUT_DIR.resolve()}\n")

    # ── 1. Core dimension tables ──────────────────────────────
    print("Generating dimension tables…")
    products_df  = generate_products()
    suppliers_df = generate_suppliers()
    ps_df        = generate_product_supplier(products_df, suppliers_df)

    # ── 2. Fact tables ────────────────────────────────────────
    print("Generating sales (may take ~20 s for 40 products × 365 days)…")
    sales_df     = generate_sales(products_df)

    print("Generating inventory snapshots…")
    inventory_df = generate_inventory(products_df, sales_df)

    print("Generating forecasts…")
    forecasts_df = generate_forecasts(products_df, sales_df)

    print("Generating supplier risk scores…")
    risk_df      = generate_risk_scores(suppliers_df)

    # ── 3. Collect ────────────────────────────────────────────
    tables = {
        "products":         products_df,
        "suppliers":        suppliers_df,
        "product_supplier": ps_df,
        "sales":            sales_df,
        "inventory":        inventory_df,
        "forecasts":        forecasts_df,
        "risk_scores":      risk_df,
    }

    # ── 4. Write CSVs ─────────────────────────────────────────
    print("\nWriting CSVs…")
    for name, df in tables.items():
        to_csv(df, name)

    # ── 5. Validation ─────────────────────────────────────────
    print_validation_report(tables)

    # ── 6. Optional: load into Postgres ───────────────────────
    if db_url:
        print(f"\nLoading into PostgreSQL: {db_url}")
        load_to_postgres(tables, db_url)

    print("\n✅ Done. All files written to ./data/raw/\n")
    return tables


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else None
    # Example with DB:
    #   python generate_mock_data.py postgresql://user:pass@localhost:5432/supplychain_db
    main(db_url=db)
    