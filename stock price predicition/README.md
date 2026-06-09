# Stock Price Prediction & ML Comparative Analysis Dashboard

This project implements a stock price prediction pipeline aligned

It focuses on:
- **Historical stock data collection** using `yfinance`
- **Technical indicator-based feature engineering** (21 indicators including Bollinger Bands, Stochastic Oscillator, ATR, OBV, RSI, MACD, SMA, EMA)
- **Sliding window sequence generation** (converting time-series sequence into supervised samples)
- **Multi-model comparative forecasting** comparing sequential deep learning models (**LSTM**, **GRU**) against classical machine learning baselines (**Linear Regression**, **Random Forest**)
- **Leakage-safe time-series train/validation/test split** (chronological splitting)
- **Evaluation** using `MAE`, `RMSE`, `MAPE`, `R2`, and directional accuracy
- **Interactive Streamlit dashboard** to run training, view diagnostics (loss curves, residuals distribution, multicollinearity heatmaps), perform live next-day forecasts, and review interview concepts
- **Recruiter-friendly visual artifacts** for model validation

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── stock_price_prediction_minor_report.md
├── portfolio_showcase.md
├── resume_project_summary.md
├── app.py                     # Streamlit dashboard
└── src
    ├── __init__.py
    ├── config.py
    ├── data_pipeline.py       # Advanced indicators & preprocessing
    ├── model.py               # LSTM & GRU models
    ├── train.py               # Multi-model training pipeline
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

### 1. Training Pipeline
Train and compare all models on a stock symbol such as `AAPL`:

```bash
python3 -m src.train --symbol AAPL --start 2020-01-01 --end 2025-01-01 --window-size 30 --epochs 60
```

This script trains LSTM, GRU, Linear Regression, and Random Forest models sequentially and saves comparative outputs inside `artifacts/<symbol>/`:

- `metrics.json` (comparative error and accuracy values)
- `model_lstm.keras` & `model_gru.keras` (neural network binaries)
- `model_linear.pkl` & `model_random_forest.pkl` (baseline binary files)
- `scaler.pkl` & `target_scaler.pkl` (feature and target scaling configurations)
- `predictions.csv` (testing set comparison prediction table)
- `actual_vs_predicted.png` (visual forecasting trend chart)
- `model_comparison.png` (error vs accuracy metrics bar chart)
- `training_history.png` (LSTM vs GRU train and validation loss curves)
- `residual_analysis.png` (residual error distribution density curves)
- `feature_correlation_heatmap.png` (correlation heatmap to evaluate collinearity)
- `price_with_indicators.png` (stock price chart with Bollinger Bands and moving averages)

### 2. Streamlit Dashboard
Launch the interactive web dashboard:

```bash
streamlit run app.py
```

Features of the dashboard:
- **Comparative Overview**: Performance table comparing MAE, RMSE, MAPE, R², and Directional Accuracy.
- **Real-Time Forecasting**: Automatically pulls the latest market data to run inferences for the next trading day.
- **Interactive Indicators**: Explores Bollinger Bands, SMA, EMA, and volume flows.
- **Model Diagnostics**: Visualizes residuals distributions, loss histories, and collinearity heatmaps.
- **Recruiter Interview Cheat Sheet**: Explains temporal leakage prevention, LSTM vs GRU gating complexity, and why baseline validation is critical.

## Note
- This project predicts the next closing price, not long-range future prices.
- It is an academic project and not trading advice.
