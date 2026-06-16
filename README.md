# 🌐 Intelligent Supply Chain Risk Prediction & Demand Forecasting Platform

## 🚀 Project Overview

This enterprise-grade data science and MLOps platform provides a robust, end-to-end predictive solution addressing critical supply chain vulnerabilities: high demand volatility, severe inventory stockouts, and unpredictable supplier reliability. The platform relies on a highly modular architecture that orchestrates deterministic mock data simulation, rigid ETL processing, advanced time-series feature engineering, machine learning algorithmic engines, production lifecycle tracking via MLflow, a secure FastAPI deployment layer, and a live Streamlit analytics frontend.

---

## 🏗️ Core Systems & Files Architecture

The repository is structured to separate data engineering, modeling, infrastructure, and presentation layers. Below is the operational mapping of the pipeline's execution flow.

| File | Operational Role |
| :--- | :--- |
| `generate_mock_data.py` | Simulates multi-table transactional database records (Sales, Inventory, Suppliers, Products, Orders) across complex logic structures to mimic enterprise scale. |
| `etl_pipeline.py` | Performs rigid type enforcement, primary key integrity validation, missing value imputation, and sequential transaction auditing prior to downstream consumption. |
| `feature_engineering.py` | Generates advanced analytical features including multi-day demand rolling means, lagging windows (`lag_7`), momentum indicators (`demand_accel`), and complex interaction terms. |
| `models.py` | Contains the core training loop engines for three separate models utilizing walk-forward TimeSeriesSplit and Leave-One-Supplier-Out (LOSO) cross-validation to explicitly prevent target/temporal leakage. Features a custom, standalone JSON-based experiment tracking system fallback. |
| `mlflow_tracking.py` | Migrates training loops into production MLOps, communicating with an active local SQLite-backed tracking server to log hyperparameters, metrics, artifacts, and manage versions inside the official Model Registry. |
| `main.py` | A production-style FastAPI backend exposing secure REST endpoints (`/forecast`, `/stockout`, `/supplier-risk`, `/ask-ai`) protected by header-based API key authentication. Implements an in-memory Model Registry and Feature Store loaded on lifespan startup for sub-50ms inference. |
| `dashboard.py` | A modern Streamlit analytics interface featuring live KPI cards, full-width Plotly trend visualizations, interactive risk scorecard tables, and an AI Copilot natural-language panel communicating with the backend API. |

---

## 🧠 The AI Reasoning Core (Machine Learning Specs)

The system deploys three distinct, highly optimized machine learning models to synthesize the overall supply chain health matrix. 

### **Model 1: Demand Forecasting**
* **Algorithm:** `GradientBoostingRegressor`
* **Objective:** Predict continuous daily demand volume.
* **Engineering:** Fitted on dynamic rolling lag windows. Target leakage is aggressively guarded against by deliberately dropping all computed revenue arrays prior to training.
* **Evaluation Metrics:** MAE, RMSE, MAPE, and R².

### **Model 2: Stockout Prediction**
* **Algorithm:** `RandomForestClassifier`
* **Objective:** Predict binary 7-day inventory depletions (Stockout vs. Safe).
* **Engineering:** Solves severe real-world data imbalance by applying `class_weight='balanced'` reweighting techniques.
* **Asymmetric Thresholding:** The decision threshold is intentionally tuned to an asymmetric **0.35** to heavily prioritize **Recall over Precision**. This minimizes the steep financial penalties associated with empty-shelf events, accepting higher false-positive alerts as an operational trade-off.
* **Evaluation Metrics:** ROC-AUC, F1-Score, Precision, and Recall.

### **Model 3: Supplier Risk Framework**
* **Algorithm:** Ensemble architecture chaining a `GradientBoostingRegressor` (predicting continuous `delay_probability`) with a `GradientBoostingClassifier` (categorizing ordinal `risk_tier`: Low, Medium, High, Critical).
* **Engineering:** Utilizes Leave-One-Supplier-Out (LOSO) cross-validation to rigorously verify generalization performance and out-of-distribution robustness on completely unseen third-party vendors.
* **Evaluation Metrics:** MAE, R², Macro-F1, and multi-class ROC-AUC.

### **Explainability (XAI) Layer**
* **Framework:** Custom model-agnostic explainability engine computing SHAP-style approximations via Marginal Contribution formulas.
* **UI Integration:** Exposes live horizontal bar charts mapping top positive and negative feature drivers in real time via the Streamlit dashboard, ensuring algorithmic transparency for supply chain managers.

---

## 💻 Installation & Step-by-Step Running Guide

Follow these sequential steps to initialize the environment, generate the datasets, train the models, and launch the distributed services.

**1. Create and activate a Python virtual environment**
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**2. Install platform dependencies**
```bash
pip install -r requirements.txt
```

**3. Execute the core pipeline sequentially**
```bash
# Generate the simulated multi-table enterprise database
python generate_mock_data.py

# Run rigid data validation and cleaning
python etl_pipeline.py

# Compute time-series features and lagging variables
python feature_engineering.py

# Train ML models and run initial validations
python models.py
```

**4. Configure and run MLflow Production Tracking**
```bash
# Set environment variables for local SQLite tracking
# Windows PowerShell:
$env:MLFLOW_ALLOW_FILE_STORE="true"
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# macOS/Linux:
export MLFLOW_ALLOW_FILE_STORE="true"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Execute MLflow integration script
python mlflow_tracking.py
```

**5. Launch Distributed Services (Run in separate terminal windows)**

*Terminal A: Launch the FastAPI Backend*
```bash
uvicorn main:app --reload --port 8000
```

*Terminal B: Launch the Streamlit Dashboard*
```bash
streamlit run dashboard.py
```

---

## 🔒 Production Security & Architecture Considerations

### **API Authentication**
The FastAPI routing layer operates under strict zero-trust principles. All forecasting and data extraction endpoints (`/forecast`, `/stockout`, `/supplier-risk`, `/ask-ai`) require secure header verification matching valid local environment tokens. Requests lacking the exact `X-API-Key: dev-local-key-12345` parameter are instantly rejected with an HTTP 401 Unauthorized response.

### **AI Copilot Scaling & Isolation**
The integrated natural language AI Copilot interacts via the `/ask-ai` endpoint. Currently, the platform routes intents via custom keyword matching to dynamic text-to-PostgreSQL SQL query template logging. 

**Production Migration Callout:** To scale this module to commercial Large Language Model (LLM) providers (e.g., OpenAI, Anthropic, or proprietary fine-tuned endpoints):
1. **Secret Management:** Hardcoded tokens must be migrated to an enterprise secrets manager (AWS Secrets Manager, Azure Key Vault, or HashiCorp Vault).
2. **Network Layer:** Establish a dedicated outbound VPC endpoint to ensure LLM telemetry and proprietary supply-chain schemas do not traverse the public internet.
3. **Execution Sandbox:** Any transition from SQL-templating to dynamic Agentic SQL execution (Text-to-SQL) must be strictly isolated to read-only database replicas with heavily restricted IAM execution roles to prevent prompt injection and data corruption.