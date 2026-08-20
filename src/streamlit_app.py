"""
TS_Forecast Streamlit Dashboard

Run:
    streamlit run src/streamlit_app.py
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecast_models import (
    prepare_series,
    time_split,
    forecast_arima,
    forecast_sarima,
    forecast_prophet,
    forecast_lstm,
    calculate_metrics,
    compare_models,
)

st.set_page_config(
    page_title="TS_Forecast",
    page_icon="📈",
    layout="wide",
)

st.title("📈 TS_Forecast — Time Series Forecasting")
st.caption("Compare statistical and deep-learning forecasting models.")

with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    default_path = ROOT / "data" / "GOOG_Dataset.csv"

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    elif default_path.exists():
        df = pd.read_csv(default_path)
        source_name = "GOOG_Dataset.csv"
    else:
        df = None
        source_name = None

if df is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

st.success(f"Data loaded: {source_name}")

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

date_candidates = [
    c for c in df.columns
    if any(x in c.lower() for x in ["date", "time", "timestamp"])
]
date_col = st.sidebar.selectbox(
    "Date / time column",
    df.columns,
    index=df.columns.get_loc(date_candidates[0]) if date_candidates else 0,
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns were found.")
    st.stop()

target_default = next(
    (c for c in ["Close", "close", "Adj Close", "Adj_Close"] if c in numeric_cols),
    numeric_cols[0],
)
target_col = st.sidebar.selectbox(
    "Forecast target",
    numeric_cols,
    index=numeric_cols.index(target_default),
)

horizon = st.sidebar.slider("Forecast horizon", 1, 90, 30)

model_options = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
selected_models = st.sidebar.multiselect(
    "Models to compare",
    model_options,
    default=["ARIMA", "SARIMA", "Prophet", "LSTM"],
)

test_size = st.sidebar.slider(
    "Test-set size",
    10,
    min(90, max(10, len(df) // 3)),
    min(30, max(10, len(df) // 5)),
)

st.header("1. Time Series Overview")

try:
    series = prepare_series(df, date_col, target_col)
except Exception as exc:
    st.error(str(exc))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Observations", len(series))
col2.metric("Start", series.index.min().strftime("%Y-%m-%d"))
col3.metric("End", series.index.max().strftime("%Y-%m-%d"))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(series.index, series.values)
ax.set_title(f"{target_col} over time")
ax.set_xlabel("Date")
ax.set_ylabel(target_col)
ax.grid(alpha=0.25)
st.pyplot(fig)

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

if st.button("🚀 Run Forecast", type="primary"):
    train, test = time_split(series, test_size)

    with st.spinner("Training models..."):
        records = []
        test_predictions = {}

        for name in selected_models:
            try:
                if name == "ARIMA":
                    pred, _ = forecast_arima(train, len(test))
                elif name == "SARIMA":
                    pred, _ = forecast_sarima(train, len(test))
                elif name == "Prophet":
                    pred, _ = forecast_prophet(train, len(test))
                else:
                    pred, _, _ = forecast_lstm(train, len(test))

                test_predictions[name] = pred
                records.append({"Model": name, **calculate_metrics(test.values, pred)})

            except Exception as exc:
                st.warning(f"{name} could not be trained: {exc}")

    if not records:
        st.error("No model completed successfully.")
        st.stop()

    results = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)

    st.header("2. Model Comparison")
    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "MAPE": "{:.2f}%",
        }),
        use_container_width=True,
    )

    best_model = results.iloc[0]["Model"]
    st.success(f"🏆 Best model on the test set by RMSE: **{best_model}**")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test.index, test.values, label="Actual", linewidth=2)
    for name, pred in test_predictions.items():
        ax.plot(test.index, pred, label=name)
    ax.set_title("Actual vs Model Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.header("3. Future Forecast")

    forecast_results = {}

    with st.spinner("Generating future forecasts..."):
        for name in selected_models:
            try:
                if name == "ARIMA":
                    pred, _ = forecast_arima(series, horizon)
                elif name == "SARIMA":
                    pred, _ = forecast_sarima(series, horizon)
                elif name == "Prophet":
                    pred, _ = forecast_prophet(series, horizon)
                else:
                    pred, _, _ = forecast_lstm(series, horizon)

                forecast_results[name] = pred
            except Exception as exc:
                st.warning(f"Future forecast failed for {name}: {exc}")

    future_index = pd.bdate_range(
        series.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
    )

    forecast_df = pd.DataFrame(
        {name: values for name, values in forecast_results.items()},
        index=future_index,
    )
    forecast_df.index.name = "Date"

    st.dataframe(forecast_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    recent = series.tail(min(120, len(series)))
    ax.plot(recent.index, recent.values, label="Historical")
    for name in forecast_df.columns:
        ax.plot(forecast_df.index, forecast_df[name], label=f"{name} forecast")
    ax.set_title("Future Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download forecast CSV",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Educational project only. Forecasts are uncertain and are not financial advice."
)
