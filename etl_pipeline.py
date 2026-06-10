"""
=============================================================
Intelligent Supply Chain Risk Prediction & Demand Forecasting
ETL Pipeline  |  Section 2-A
=============================================================
Stages
  1. EXTRACT   – read CSVs (or simulate API pull); detect schema
  2. VALIDATE  – row-count checks, schema assertions, PK uniqueness
  3. CLEAN     – null imputation, outlier capping, type coercion,
                 duplicate removal, referential-integrity enforcement
  4. TRANSFORM – derived business columns (revenue, days coverage…)
  5. LOAD      – write cleaned Parquet files ready for feature
                 engineering; optional direct-to-PostgreSQL path

All quality findings are collected into an AuditLog that is
written to data/audit/etl_audit_<timestamp>.csv at the end.

Requirements: pandas >= 1.4, numpy >= 1.20
Optional:     sqlalchemy + psycopg2-binary  (for DB load)
=============================================================
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
RAW_DIR   = Path("./data/raw")
CLEAN_DIR = Path("./data/clean")
AUDIT_DIR = Path("./data/audit")

for _d in (CLEAN_DIR, AUDIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────
@dataclass
class AuditEntry:
    table:    str
    stage:    str
    check:    str
    status:   str
    rows_in:  int = 0
    rows_out: int = 0
    detail:   str = ""


class AuditLog:
    def __init__(self):
        self._entries: list[AuditEntry] = []

    def add(self, **kwargs) -> None:
        e = AuditEntry(**kwargs)
        self._entries.append(e)
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(e.status, "·")
        log.info(f"[{e.table:<18}] {icon} {e.stage:<10} {e.check} — {e.detail}")

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._entries])

    def save(self) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = AUDIT_DIR / f"etl_audit_{ts}.csv"
        self.to_df().to_csv(path, index=False)
        return path

    def summary(self) -> dict[str, int]:
        return self.to_df()["status"].value_counts().to_dict()


AUDIT = AuditLog()


# ─────────────────────────────────────────────────────────────
# SCHEMA CONTRACTS
# ─────────────────────────────────────────────────────────────
SCHEMA: dict[str, dict[str, Any]] = {
    "products": {
        "required_cols": ["product_id","sku","product_name","category",
                          "unit_cost","unit_price","lead_time_days",
                          "reorder_point","reorder_quantity"],
        "not_null":      ["product_id","sku","category","unit_cost","unit_price"],
        "pk":            "product_id",
        "numeric_ge0":   ["unit_cost","unit_price","lead_time_days",
                          "reorder_point","reorder_quantity"],
        "date_cols":     ["created_at","updated_at"],
    },
    "suppliers": {
        "required_cols": ["supplier_id","supplier_name","country",
                          "reliability_score","payment_terms_days"],
        "not_null":      ["supplier_id","supplier_name","country"],
        "pk":            "supplier_id",
        "numeric_ge0":   ["payment_terms_days"],
        "range_checks":  {"reliability_score": (0, 10)},
        "date_cols":     ["onboarded_at","created_at","updated_at"],
    },
    "sales": {
        "required_cols": ["sale_id","product_id","sale_date",
                          "quantity_sold","unit_price"],
        "not_null":      ["sale_id","product_id","sale_date","quantity_sold"],
        "pk":            "sale_id",
        "numeric_ge0":   ["quantity_sold","unit_price","discount_pct"],
        "range_checks":  {"discount_pct": (0, 100)},
        "date_cols":     ["sale_date","created_at"],
        "fk":            {"product_id": "products"},
    },
    "inventory": {
        "required_cols": ["inventory_id","product_id","snapshot_date",
                          "warehouse_id","quantity_on_hand"],
        "not_null":      ["inventory_id","product_id","snapshot_date","warehouse_id"],
        "pk":            "inventory_id",
        "numeric_ge0":   ["quantity_on_hand","quantity_reserved"],
        "date_cols":     ["snapshot_date","created_at"],
        "fk":            {"product_id": "products"},
    },
    "forecasts": {
        "required_cols": ["forecast_id","product_id","forecast_date",
                          "predicted_demand","model_name"],
        "not_null":      ["forecast_id","product_id","forecast_date","predicted_demand"],
        "pk":            "forecast_id",
        "numeric_ge0":   ["predicted_demand"],
        "date_cols":     ["forecast_date","generated_at"],
        "fk":            {"product_id": "products"},
    },
    "risk_scores": {
        "required_cols": ["risk_id","supplier_id","scored_at",
                          "delay_probability","risk_tier"],
        "not_null":      ["risk_id","supplier_id","delay_probability","risk_tier"],
        "pk":            "risk_id",
        "range_checks":  {"delay_probability": (0, 1),
                          "composite_risk_score": (0, 100)},
        "date_cols":     ["scored_at","created_at"],
        "fk":            {"supplier_id": "suppliers"},
    },
}


# ─────────────────────────────────────────────────────────────
# STAGE 1 — EXTRACT
# ─────────────────────────────────────────────────────────────

def extract_csv(table: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / f"{table}.csv"
    if not path.exists():
        AUDIT.add(table=table, stage="extract", check="file_exists",
                  status="FAIL", detail=f"Not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    AUDIT.add(table=table, stage="extract", check="file_read",
              status="PASS", rows_in=0, rows_out=len(df),
              detail=f"{len(df):,} rows, {df.shape[1]} cols from {path.name}")
    return df


def extract_all(tables: list[str] | None = None) -> dict[str, pd.DataFrame]:
    tables = tables or list(SCHEMA.keys())
    log.info("── STAGE 1: EXTRACT ──────────────────────────────────")
    return {t: extract_csv(t) for t in tables}


# ─────────────────────────────────────────────────────────────
# STAGE 2 — VALIDATE
# ─────────────────────────────────────────────────────────────

def validate(table: str, df: pd.DataFrame,
             ref_tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    contract = SCHEMA.get(table, {})
    n = len(df)

    # Column presence
    missing = [c for c in contract.get("required_cols", []) if c not in df.columns]
    if missing:
        AUDIT.add(table=table, stage="validate", check="required_cols",
                  status="FAIL", rows_in=n, detail=f"Missing: {missing}")
        raise ValueError(f"[{table}] Missing columns: {missing}")
    AUDIT.add(table=table, stage="validate", check="required_cols",
              status="PASS", rows_in=n, rows_out=n, detail="All required columns present")

    # PK uniqueness
    pk = contract.get("pk")
    if pk and pk in df.columns:
        dupes = int(df[pk].duplicated().sum())
        AUDIT.add(table=table, stage="validate", check="pk_unique",
                  status=("WARN" if dupes else "PASS"), rows_in=n, rows_out=n,
                  detail=f"{dupes:,} duplicate PKs in '{pk}'")

    # Not-null critical fields
    for col in contract.get("not_null", []):
        if col not in df.columns:
            continue
        nulls = int(df[col].isnull().sum())
        pct   = nulls / n * 100
        status = "FAIL" if pct > 10 else ("WARN" if nulls else "PASS")
        AUDIT.add(table=table, stage="validate", check=f"not_null:{col}",
                  status=status, rows_in=n,
                  detail=f"{nulls:,} nulls ({pct:.1f}%)")

    # Range checks
    for col, (lo, hi) in contract.get("range_checks", {}).items():
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        oor = int(((ser < lo) | (ser > hi)).sum())
        AUDIT.add(table=table, stage="validate", check=f"range:{col}",
                  status=("WARN" if oor else "PASS"), rows_in=n,
                  detail=f"{oor:,} values outside [{lo},{hi}]")

    # Non-negative
    for col in contract.get("numeric_ge0", []):
        if col not in df.columns:
            continue
        neg = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
        AUDIT.add(table=table, stage="validate", check=f"non_neg:{col}",
                  status=("WARN" if neg else "PASS"), rows_in=n,
                  detail=f"{neg:,} negative values")

    # FK checks
    if ref_tables:
        for fk_col, ref_table in contract.get("fk", {}).items():
            if fk_col not in df.columns or ref_table not in ref_tables:
                continue
            ref_pk  = SCHEMA[ref_table]["pk"]
            valid   = set(ref_tables[ref_table][ref_pk].tolist())
            orphans = int((~df[fk_col].isin(valid)).sum())
            AUDIT.add(table=table, stage="validate", check=f"fk:{fk_col}",
                      status=("WARN" if orphans else "PASS"), rows_in=n,
                      detail=f"{orphans:,} orphaned FK values")

    # Row count
    status = "FAIL" if n == 0 else "PASS"
    AUDIT.add(table=table, stage="validate", check="row_count",
              status=status, rows_in=n, rows_out=n, detail=f"{n:,} rows")

    return df


# ─────────────────────────────────────────────────────────────
# STAGE 3 — CLEAN
# ─────────────────────────────────────────────────────────────

def _iqr_cap(series: pd.Series, factor: float = 3.0) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    return series.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    df = df.drop_duplicates(subset=["product_id"])
    for col in ["unit_cost","unit_price","lead_time_days","reorder_point","reorder_quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["unit_cost","unit_price"]:
        df[col] = df.groupby("category")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median()).clip(lower=0)
    df["lead_time_days"]   = df["lead_time_days"].fillna(7).clip(lower=0)
    df["reorder_point"]    = df["reorder_point"].fillna(50).clip(lower=0)
    df["reorder_quantity"] = df["reorder_quantity"].fillna(100).clip(lower=1)
    df["is_active"]        = df["is_active"].fillna(True).astype(bool)
    for col in ["created_at","updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    AUDIT.add(table="products", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Dropped {n_in-len(df)} duplicates; nulls imputed with category medians")
    return df


def clean_suppliers(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    df = df.drop_duplicates(subset=["supplier_id"])
    df["reliability_score"]  = pd.to_numeric(df["reliability_score"],  errors="coerce").fillna(5.0).clip(0,10)
    df["payment_terms_days"] = pd.to_numeric(df["payment_terms_days"], errors="coerce").fillna(30).clip(lower=0)
    df["is_active"] = df["is_active"].fillna(True).astype(bool)
    for col in ["onboarded_at","created_at","updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    AUDIT.add(table="suppliers", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Dropped {n_in-len(df)} duplicates")
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    df["sale_date"]     = pd.to_datetime(df["sale_date"],    errors="coerce")
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce")
    df["unit_price"]    = pd.to_numeric(df["unit_price"],    errors="coerce")
    df["discount_pct"]  = pd.to_numeric(df["discount_pct"],  errors="coerce").fillna(0)
    df["is_returned"]   = df["is_returned"].astype(str).str.lower().isin(["true","1","yes"])
    df = df.dropna(subset=["sale_date","product_id","quantity_sold"])
    df = df[df["quantity_sold"] > 0]
    df["quantity_sold"] = _iqr_cap(df["quantity_sold"]).clip(lower=1).round().astype(int)
    df["unit_price"]    = _iqr_cap(df["unit_price"]).clip(lower=0)
    df["discount_pct"]  = df["discount_pct"].clip(0, 100)
    df["channel"] = (df["channel"].str.strip().str.lower()
                       .replace({"e-commerce":"online","web":"online",
                                 "store":"retail","b2b":"wholesale"}))
    df = df.drop_duplicates(subset=["sale_id"])
    AUDIT.add(table="sales", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Removed {n_in-len(df):,} invalid/duplicate rows; qty+price capped at 3xIQR")
    return df.reset_index(drop=True)


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    df["snapshot_date"]     = pd.to_datetime(df["snapshot_date"],    errors="coerce")
    df["quantity_on_hand"]  = pd.to_numeric(df["quantity_on_hand"],  errors="coerce")
    df["quantity_reserved"] = pd.to_numeric(df["quantity_reserved"], errors="coerce").fillna(0)
    df = df.dropna(subset=["snapshot_date","product_id","quantity_on_hand"])
    df["quantity_on_hand"]  = df["quantity_on_hand"].clip(lower=0).round().astype(int)
    df["quantity_reserved"] = df["quantity_reserved"].clip(lower=0).round().astype(int)
    df["quantity_reserved"] = df[["quantity_on_hand","quantity_reserved"]].min(axis=1)
    df["reorder_triggered"] = df["reorder_triggered"].astype(str).str.lower().isin(["true","1","yes"])
    df["unit_cost_snapshot"]= _iqr_cap(pd.to_numeric(df["unit_cost_snapshot"], errors="coerce"))
    df = df.drop_duplicates(subset=["product_id","warehouse_id","snapshot_date"], keep="last")
    AUDIT.add(table="inventory", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Removed {n_in-len(df):,} rows; reserved clamped to on_hand")
    return df.reset_index(drop=True)


def clean_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    for col in ["forecast_date","generated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["predicted_demand","actual_demand","confidence_score",
                "lower_bound_95","upper_bound_95","mape","mae"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["forecast_date","product_id","predicted_demand"])
    df["predicted_demand"] = df["predicted_demand"].clip(lower=0)
    df["confidence_score"] = df["confidence_score"].clip(0, 1)
    if "lower_bound_95" in df.columns:
        df["lower_bound_95"] = df[["lower_bound_95","predicted_demand"]].min(axis=1)
    if "upper_bound_95" in df.columns:
        df["upper_bound_95"] = df[["upper_bound_95","predicted_demand"]].max(axis=1)
    df = df.drop_duplicates(subset=["forecast_id"])
    AUDIT.add(table="forecasts", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Removed {n_in-len(df):,} rows; prediction bounds sanitised")
    return df.reset_index(drop=True)


def clean_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    if "scored_at" in df.columns:
        df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce", utc=True)
    for col, lo, hi in [("delay_probability",0,1),("composite_risk_score",0,100),
                         ("hist_on_time_rate",0,1)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
    for col in ["hist_mean_delay_days","hist_delay_variance","avg_delay_days_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    valid_tiers = {"low","medium","high","critical"}
    df["risk_tier"] = df["risk_tier"].where(df["risk_tier"].isin(valid_tiers), "medium")
    df = df.drop_duplicates(subset=["risk_id"]).dropna(subset=["supplier_id","delay_probability"])
    AUDIT.add(table="risk_scores", stage="clean", check="complete",
              status="PASS", rows_in=n_in, rows_out=len(df),
              detail=f"Removed {n_in-len(df):,} rows; tiers validated")
    return df.reset_index(drop=True)


_CLEANERS = {
    "products":    clean_products,
    "suppliers":   clean_suppliers,
    "sales":       clean_sales,
    "inventory":   clean_inventory,
    "forecasts":   clean_forecasts,
    "risk_scores": clean_risk_scores,
}


def clean_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    log.info("── STAGE 3: CLEAN ────────────────────────────────────")
    return {t: (_CLEANERS[t](df) if t in _CLEANERS else df) for t, df in raw.items()}


# ─────────────────────────────────────────────────────────────
# STAGE 4 — TRANSFORM
# ─────────────────────────────────────────────────────────────

def transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"]      = (df["quantity_sold"] * df["unit_price"]
                          * (1 - df["discount_pct"] / 100)).round(2)
    df["net_quantity"] = np.where(df["is_returned"], 0, df["quantity_sold"])
    df["year"]  = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month
    df["week"]  = df["sale_date"].dt.isocalendar().week.astype(int)
    df["dow"]   = df["sale_date"].dt.dayofweek
    return df


def transform_inventory(df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        products_df[["product_id","reorder_point","reorder_quantity"]],
        on="product_id", how="left"
    )
    df["quantity_available"] = (df["quantity_on_hand"] - df["quantity_reserved"]).clip(lower=0)
    df["is_below_reorder"]   = df["quantity_available"] <= df["reorder_point"]
    return df


def transform_all(clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    log.info("── STAGE 4: TRANSFORM ────────────────────────────────")
    t = dict(clean)
    t["sales"]     = transform_sales(clean["sales"])
    t["inventory"] = transform_inventory(clean["inventory"], clean["products"])
    AUDIT.add(table="sales", stage="transform", check="derived_cols",
              status="PASS", rows_in=len(clean["sales"]), rows_out=len(t["sales"]),
              detail="Added: revenue, net_quantity, year, month, week, dow")
    AUDIT.add(table="inventory", stage="transform", check="derived_cols",
              status="PASS", rows_in=len(clean["inventory"]), rows_out=len(t["inventory"]),
              detail="Added: quantity_available, is_below_reorder")
    return t


# ─────────────────────────────────────────────────────────────
# STAGE 5 — LOAD
# ─────────────────────────────────────────────────────────────

def load_parquet(transformed: dict[str, pd.DataFrame],
                  out_dir: Path = CLEAN_DIR) -> dict[str, Path]:
    log.info("── STAGE 5: LOAD (CSV) ───────────────────────────────")
    paths = {}
    for table, df in transformed.items():
        path = out_dir / f"{table}.csv"
        df.to_csv(path, index=False)
        kb = path.stat().st_size / 1024
        paths[table] = path
        AUDIT.add(table=table, stage="load", check="csv_write",
                  status="PASS", rows_in=len(df), rows_out=len(df),
                  detail=f"→ {path.name}  ({kb:.1f} KB)")
    return paths


def load_postgres(transformed: dict[str, pd.DataFrame], db_url: str) -> None:
    """Load clean DataFrames into PostgreSQL. Requires sqlalchemy + psycopg2-binary."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        log.warning("SQLAlchemy not installed — skipping Postgres load.")
        return
    log.info("── STAGE 5: LOAD (PostgreSQL) ────────────────────────")
    engine = create_engine(db_url, echo=False)
    insert_order = ["products","suppliers","product_supplier",
                    "sales","inventory","forecasts","risk_scores"]
    with engine.connect() as conn:
        for table in insert_order:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
            except Exception:
                pass
        conn.commit()
    for table in insert_order:
        if table not in transformed:
            continue
        transformed[table].to_sql(table, engine, if_exists="append",
                                   index=False, chunksize=5000, method="multi")
        AUDIT.add(table=table, stage="load", check="postgres_insert",
                  status="PASS", rows_in=len(transformed[table]),
                  rows_out=len(transformed[table]),
                  detail=f"{len(transformed[table]):,} rows → PostgreSQL:{table}")
    log.info("PostgreSQL load complete.")


# ─────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_pipeline(db_url: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Execute the full ETL pipeline:
        Extract → Validate → Clean → Transform → Load

    Parameters
    ----------
    db_url : optional PostgreSQL connection string
             e.g. 'postgresql://user:pass@localhost:5432/supplychain_db'

    Returns
    -------
    Dictionary of fully cleaned, transformed DataFrames
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  Supply Chain ETL Pipeline — starting")
    log.info("═══════════════════════════════════════════════════════")
    start = datetime.now()

    raw         = extract_all()
    log.info("── STAGE 2: VALIDATE ─────────────────────────────────")
    for table, df in raw.items():
        validate(table, df, ref_tables=raw)
    cleaned     = clean_all(raw)
    transformed = transform_all(cleaned)
    load_parquet(transformed)

    if db_url:
        load_postgres(transformed, db_url)

    audit_path = AUDIT.save()
    summary    = AUDIT.summary()
    elapsed    = (datetime.now() - start).total_seconds()

    log.info("═══════════════════════════════════════════════════════")
    log.info(f"  Done in {elapsed:.1f}s  |  "
             f"PASS={summary.get('PASS',0)}  "
             f"WARN={summary.get('WARN',0)}  "
             f"FAIL={summary.get('FAIL',0)}")
    log.info(f"  Audit log → {audit_path}")
    log.info("═══════════════════════════════════════════════════════")

    return transformed


if __name__ == "__main__":
    import sys
    run_pipeline(db_url=sys.argv[1] if len(sys.argv) > 1 else None)