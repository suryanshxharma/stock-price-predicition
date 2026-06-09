# Portfolio Showcase

## Stock Price Prediction & ML Comparative Analysis Dashboard

This project demonstrates a production-grade machine learning forecasting system. It implements a comparative benchmarking framework where deep learning sequential models (LSTM, GRU) are evaluated aga[...]

## Key ML Architecture Decisions

- **Multi-Model Benchmark**: Leverages LSTM, GRU, Linear Regression, and Random Forest to demonstrate that more complex architectures are justified over simpler baselines.
- **21 Engineered Features**: Incorporates Trend (SMA, EMA, Bollinger Bands), Momentum (RSI, MACD, Stochastic), Volatility (ATR, daily return volatility), and Volume (OBV) signals.
- **Leakage-Safe Validation**: Enforces chronological splits and pipeline-isolated scaling so the model never trains on future information.
- **Streamlit Interactive UI**: Provides live next-day prediction, interactive plotting of technical indicators, training history analysis, and residuals diagnostics.

## Model Performance Benchmarking

A typical training pass yields a comparative performance table like the following:

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² Score | Directional Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | *Trained on sequence* | *Trained on sequence* | *Trained on sequence* | *Temporal capture* | *Evaluates movement* |
| **GRU** | *Fewer gates/params* | *Faster training* | *Comparable loss* | *No cell state* | *Sign prediction* |
| **Linear Regression** | *Linear baseline* | *Baseline check* | *Simple forecast* | *Simplest hypothesis* | *Trend benchmark* |
| **Random Forest** | *Ensemble baseline* | *Non-linear static* | *Bagged tree loss* | *Bootstrapped check* | *Out-of-bag verify* |

## Visual Evidence

All visual assets are generated dynamically and saved to the artifacts directory:
- Actual vs Predicted: [actual_vs_predicted.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/actual_vs_predicted.png)
- Training curves: [training_history.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/training_history.png)
- Residual analysis: [residual_analysis.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/residual_analysis.png)
- Feature correlation matrix: [feature_correlation_heatmap.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/feature_correlation_heatmap.png)
- Price with indicators: [price_with_indicators.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/price_with_indicators.png)
- Comparative metrics: [model_comparison.png](file:///Users/suryanshx/Documents/New%20project/stock%20price%20predicition/artifacts/AAPL/model_comparison.png)

## Why This is Recruiters' Favorite ML Project

Unlike typical toy projects that train an isolated LSTM and make unrealistic claims about "beating the stock market," this repository showcases a **rigorous ML systems engineering process**:
1. **Baselines First**: Showing that a model must outperform Linear Regression and Random Forest.
2. **Leakage Prevention**: Demonstrating actual out-of-sample predictability on historical time periods.
3. **Multi-dimensional Metrics**: Evaluating directional accuracy and R² score instead of relying solely on MSE.
4. **Interactive Validation**: Presenting a fully interactive Streamlit application.
