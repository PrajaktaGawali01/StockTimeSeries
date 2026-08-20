# TS_Forecast — Time Series Forecasting
Time series analysis and stock price forecasting using ARIMA, SARIMA, Prophet, and LSTM models. The project explores historical Google stock data, analyzes trends and patterns, and compares different forecasting techniques.


## Project highlights

- Data loading and cleaning
- Exploratory time-series analysis
- Train/test time-aware validation
- ARIMA forecasting
- SARIMA forecasting
- Prophet forecasting
- LSTM forecasting
- MAE / RMSE / MAPE model comparison
- Interactive Streamlit dashboard
- Downloadable forecast results
- GitHub Actions syntax check

## Repository structure

```text
TS_Forecast/
├── data/
│   ├── GOOG_Dataset.csv
│   └── DATA_DICTIONARY.md
├── notebooks/
│   ├── Time_Series_Analysis.ipynb
│   └── TS_Forecast_rewritten.ipynb
├── src/
│   ├── __init__.py
│   ├── forecast_models.py
│   └── streamlit_app.py
├── models/
├── outputs/
├── .github/workflows/checks.yml
├── requirements.txt
├── run_app.py
├── PROJECT_STATUS.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
└── README.md
```

## Setup

Recommended: Python 3.10 or 3.11.

```bash
git clone https://github.com/YOUR_USERNAME/TS_Forecast.git
cd TS_Forecast
python -m venv .venv
```

.venv\Scripts\activate




Install packages:


pip install -r requirements.txt


## Run the Streamlit dashboard


streamlit run src/streamlit_app.py


The app can use the included `data/GOOG_Dataset.csv` or an uploaded CSV.

### Dashboard features

- CSV upload
- Automatic date-column selection
- Forecast-target selection
- Forecast horizon selection
- Model selection
- Actual-vs-predicted visualization
- MAE / RMSE / MAPE comparison
- Best-model identification
- Future forecast visualization
- Download forecast CSV

## Run the notebook


jupyter notebook


Open:

`notebooks/TS_Forecast_rewritten.ipynb`

For the included dataset, use:

```python
DATA_PATH = "../data/GOOG_Dataset.csv"
```

## Models

### ARIMA

Uses autoregressive, differencing and moving-average components.

### SARIMA

Extends ARIMA with seasonal components.

### Prophet

Models trend and common seasonal patterns.

### LSTM

Uses a recurrent neural network for nonlinear sequence forecasting.

## Evaluation

Models are compared using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

Lower values generally indicate better test-set performance.

## Project improvements included in this version

The original Streamlit logic has been separated into:

`src/forecast_models.py`

while the interface is kept in:

`src/streamlit_app.py`

This makes the project easier to maintain, test, and extend.

## Future improvements

- Walk-forward cross-validation
- Automated hyperparameter search
- Technical indicators such as SMA, EMA and RSI
- Exogenous variables
- Prediction intervals
- Experiment tracking
- Unit tests
- Deployment

