# Stock Price Prediction Using Machine Learning

This project implements a stock price prediction pipeline aligned with the minor report in [stock_price_prediction_minor_report.md](/Users/suryanshx/Documents/New%20project/stock_price_prediction_mino[...])

It focuses on:

- historical stock data collection using `yfinance`
- technical indicator based feature engineering
- sliding window sequence generation
- LSTM based next-day close price prediction
- leakage-safe time-series train/validation/test split
- evaluation using `MAE`, `RMSE`, `MAPE`, and `R2`
- recruiter-friendly visual artifacts for model validation

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── stock_price_prediction_minor_report.md
└── src
    ├── __init__.py
    ├── config.py
    ├── data_pipeline.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train the project on a stock symbol such as `AAPL`:

```bash
python3 -m src.train --symbol AAPL --start 2020-01-01 --end 2025-01-01 --window-size 30
```

## Output

The training script saves outputs inside `artifacts/<symbol>/`:

- `metrics.json`
- `model.keras`
- `scaler.pkl`
- `predictions.csv`
- `actual_vs_predicted.png`
- `training_history.png`
- `residual_analysis.png`
- `feature_correlation_heatmap.png`
- `price_with_indicators.png`


