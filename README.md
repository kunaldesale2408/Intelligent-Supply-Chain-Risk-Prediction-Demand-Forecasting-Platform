# ⛓️ Intelligent Supply Chain Risk Prediction & Demand Forecasting Platform

## 🌍 1. Project Overview
This project is an AI-powered platform designed to solve common supply chain problems: unpredictable customer demand, running out of stock, and unreliable suppliers[cite: 8]. It provides a complete workflow, from generating sample data to training machine learning models, and displays the results on an interactive web dashboard[cite: 8]. 

---

## 🏗️ 2. Core Systems & Files
The project is organized into separate scripts that each handle one specific job:

* **`generate_mock_data.py`**: Creates realistic simulated data for sales, inventory, products, and suppliers so you have data to work with[cite: 7].
* **`etl_pipeline.py`**: Cleans the raw data, fixes missing values, and checks for errors to make sure the data is high quality[cite: 6].
* **`feature_engineering.py`**: Calculates useful metrics (like 7-day rolling averages and sales trends) to help the AI models learn better[cite: 5].
* **`models.py`**: Trains three different machine learning models and safely tests them on unseen data[cite: 2].
* **`mlflow_tracking.py`**: Tracks the training process, logs performance scores, and saves the best model versions using a tool called MLflow[cite: 3].
* **`main.py`**: The FastAPI server that securely connects our trained models to the web, providing fast predictions.
* **`dashboard.py`**: The Streamlit web app that displays live charts, risk scorecards, and an AI chat assistant[cite: 1].

---

## 🧠 3. Machine Learning Models
The platform uses three main AI models to make supply chain predictions:

* **Model 1: Demand Forecasting**
  * **What it does:** Predicts how many units of a product will sell each day.
  * **How it works:** Uses a Gradient Boosting model[cite: 2].
* **Model 2: Stockout Prediction**
  * **What it does:** Warns you if a product is likely to run out of stock in the next 7 days[cite: 2].
  * **How it works:** Uses a Random Forest model[cite: 2]. It is specially tuned to prioritize catching every possible stockout over being perfectly precise, so you never end up with empty shelves[cite: 2].
* **Model 3: Supplier Risk**
  * **What it does:** Predicts if a supplier will be delayed and gives them a risk rating (Low, Medium, High, or Critical)[cite: 2].
  * **How it works:** Uses a combination of Gradient Boosting models tested strictly on unseen suppliers[cite: 2].
* **Explainable AI (XAI):** For every prediction, the platform generates a chart explaining exactly *why* the AI made that decision[cite: 2].

---

## 🚀 4. Installation & Running Guide
Follow these steps in your terminal to run the whole project from start to finish.

**Step 1: Set up the environment**
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate
# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

