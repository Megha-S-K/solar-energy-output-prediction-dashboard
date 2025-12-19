# ☀️ Industrial Solar Energy Output Prediction & Monitoring Dashboard

This project implements an **industry-grade machine learning system** to predict and monitor **solar power output** for an industrial solar plant using historical plant data and weather data (Solcast).

The system combines:
- Time-series ML forecasting
- Robust model validation
- Operational monitoring via an interactive dashboard

---

## 🔍 Problem Statement
Industrial solar plants experience efficiency loss due to factors like:
- Dust and soiling
- Weather variability
- Seasonal and operational changes

The goal of this project is to:
1. Accurately predict expected solar power output
2. Compare predicted vs actual behavior
3. Enable early detection of abnormal deviations

---

## 🧠 Solution Overview

### ✔ Data Sources
- **Industrial solar plant data** (multiple inverters)
- **Weather & irradiance data** from Solcast (GHI, GTI, DNI, cloud opacity, etc.)

### ✔ Machine Learning Model
- **Random Forest Regressor**
- Physics-aware features (irradiance, zenith angle)
- Temporal features (hour, day of year, seasonality)
- Lag features for time-series memory

### ✔ Model Performance (Daytime)
- MAE: ~4–5 kW
- RMSE: ~6–7 kW
- MAPE: ~12–13%
- Stable across seasons, cloud conditions, and time of day

---

## 📊 Monitoring Dashboard (Streamlit)

The Streamlit dashboard provides:
- Actual vs Predicted power comparison
- Deviation (%) over time
- Key KPIs (current power, deviation, efficiency proxy)
- Date range and time resolution controls

> **Note:** Dashboard v0.1 focuses on monitoring and interpretability.  
> Automation and alerting are planned as future extensions.

---

## 🗂 Project Structure

```bash
├── app.py # Streamlit dashboard
├── requirements.txt
├── model/
│ └── model.joblib
├── data/
│ └── data.csv
├── notebooks/
│ └── model_training_colab.ipynb
└── README.md
```
---
