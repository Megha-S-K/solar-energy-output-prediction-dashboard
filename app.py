import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Solar Plant Monitoring Dashboard",
    layout="wide"
)

st.title("☀️ Solar Plant Monitoring Dashboard")
st.caption("Dashboard v0.1 — Monitoring & Prediction")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/solar_data_clean_final.csv", parse_dates=["dt"])

df = load_data()

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("solar_power_rf_v1.joblib")

model = load_model()

# ----------------------------
# Basic sanity checks
# ----------------------------
st.success("✅ Data and model loaded successfully")

st.write("Data preview:")
st.dataframe(df.head())

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

# Date range selector
min_date = df["dt"].min().date()
max_date = df["dt"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(max_date - pd.Timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Time resolution
resolution = st.sidebar.selectbox(
    "Time resolution",
    options=["15 min", "Hourly"],
    index=0
)

# ----------------------------
# Filter data by date
# ----------------------------
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

df_filtered = df[
    (df["dt"] >= start_date) &
    (df["dt"] <= end_date + pd.Timedelta(days=1))
].copy()

# ----------------------------
# Time aggregation
# ----------------------------
if resolution == "Hourly":
    df_filtered = (
        df_filtered
        .set_index("dt")
        .resample("1H")
        .mean()
        .reset_index()
    )

st.info(f"Showing {len(df_filtered)} records from {start_date.date()} to {end_date.date()}")

# ----------------------------
# Generate predictions
# ----------------------------
FEATURES = [
    "ghi", "gti", "dni",
    "clearsky_ghi", "clearsky_gti", "clearsky_dni",
    "cloud_opacity", "zenith",
    "air_temp", "relative_humidity", "precipitable_water",
    "hour", "dayofyear", "month", "weekday",
    "power_lag_1", "power_lag_2"
]

# Predict only if data exists
if len(df_filtered) > 0:
    df_filtered["predicted_power"] = model.predict(df_filtered[FEATURES])
else:
    df_filtered["predicted_power"] = []

# ----------------------------
# Deviation calculation
# ----------------------------
df_filtered["deviation_pct"] = (
    (df_filtered["total_power"] - df_filtered["predicted_power"])
    / df_filtered["predicted_power"]
    * 100
)

# Avoid division issues
df_filtered["deviation_pct"] = df_filtered["deviation_pct"].replace([float("inf"), -float("inf")], 0)
df_filtered["deviation_pct"] = df_filtered["deviation_pct"].fillna(0)

# ----------------------------
# KPI computation
# ----------------------------
if len(df_filtered) > 0:
    latest_row = df_filtered.iloc[-1]

    actual_power = latest_row["total_power"]
    predicted_power = latest_row["predicted_power"]

    deviation_pct = (
        (actual_power - predicted_power) / predicted_power * 100
        if predicted_power > 0 else 0
    )

    efficiency_proxy = (
        actual_power / latest_row["gti"]
        if latest_row["gti"] > 0 else 0
    )
else:
    actual_power = predicted_power = deviation_pct = efficiency_proxy = 0

# ----------------------------
# KPI display
# ----------------------------
st.subheader("Current Plant Status")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Actual Power (kW)", f"{actual_power:.2f}")
col2.metric("Predicted Power (kW)", f"{predicted_power:.2f}")
col3.metric("Deviation (%)", f"{deviation_pct:.2f}")
col4.metric("Efficiency (Power / GTI)", f"{efficiency_proxy:.4f}")

# ----------------------------
# Main chart: Actual vs Predicted
# ----------------------------
st.subheader("Actual vs Predicted Solar Power")

chart_df = df_filtered.set_index("dt")[["total_power", "predicted_power"]]

st.line_chart(chart_df)

# ----------------------------
# Deviation over time chart
# ----------------------------
st.subheader("Deviation Over Time (%)")

deviation_df = df_filtered.set_index("dt")[["deviation_pct"]]

st.line_chart(deviation_df)

