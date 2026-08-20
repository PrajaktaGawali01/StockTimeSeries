"""
Forecasting utilities for TS_Forecast.

This module keeps model logic separate from the Streamlit UI.
Models:
- ARIMA
- SARIMA
- Prophet
- LSTM

Metrics:
- MAE
- RMSE
- MAPE
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    Sequential = None


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-9))) * 100

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def prepare_series(df, date_col, target_col):
    data = df[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna().sort_values(date_col).drop_duplicates(date_col)
    data = data.set_index(date_col)[target_col].astype(float)

    if len(data) < 30:
        raise ValueError("At least 30 valid observations are recommended.")

    return data


def time_split(series, test_size):
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be smaller than the number of observations.")
    return series.iloc[:-test_size], series.iloc[-test_size:]


def forecast_arima(train, horizon, order=(5, 1, 0)):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    pred = fitted.forecast(steps=horizon)
    return np.asarray(pred), fitted


def forecast_sarima(train, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=horizon)
    return np.asarray(pred), fitted


def forecast_prophet(train, horizon):
    if Prophet is None:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future)
    pred = forecast["yhat"].tail(horizon).to_numpy()

    return pred, model


def _make_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)


def forecast_lstm(train, horizon, window=30, units=64, dropout=0.2, epochs=40, batch_size=32):
    if Sequential is None:
        raise ImportError("TensorFlow/Keras is not installed.")

    values = np.asarray(train, dtype=float).reshape(-1, 1)

    if len(values) <= window + 5:
        raise ValueError("Not enough training observations for the selected LSTM window.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = _make_sequences(scaled, window)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(units, input_shape=(window, 1)),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    )

    validation_split = 0.1 if len(X) >= 50 else 0.0
    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False,
    )
    if validation_split > 0:
        fit_kwargs.update(validation_split=validation_split, callbacks=[early_stop])

    model.fit(X, y, **fit_kwargs)

    # Recursive multi-step forecast
    history = scaled.flatten().tolist()
    predictions_scaled = []

    for _ in range(horizon):
        window_values = np.asarray(history[-window:], dtype=float).reshape(1, window, 1)
        next_value = float(model.predict(window_values, verbose=0)[0, 0])
        predictions_scaled.append(next_value)
        history.append(next_value)

    predictions = scaler.inverse_transform(
        np.asarray(predictions_scaled).reshape(-1, 1)
    ).flatten()

    return predictions, model, scaler


def compare_models(train, test, selected_models):
    records = []
    predictions = {}

    for name in selected_models:
        if name == "ARIMA":
            pred, _ = forecast_arima(train, len(test))
        elif name == "SARIMA":
            pred, _ = forecast_sarima(train, len(test))
        elif name == "Prophet":
            pred, _ = forecast_prophet(train, len(test))
        elif name == "LSTM":
            pred, _, _ = forecast_lstm(train, len(test))
        else:
            continue

        predictions[name] = pred
        metrics = calculate_metrics(test.values, pred)
        records.append({"Model": name, **metrics})

    results = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    return results, predictions
