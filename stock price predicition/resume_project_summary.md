# Resume Project Summary

## Project Title

Stock Price Prediction & ML Comparative Analysis Dashboard

## Resume-Ready Description

Designed and built an end-to-end stock price prediction and model validation dashboard using Streamlit. Engineered a 21-feature financial dataset incorporating Bollinger Bands, Stochastic Oscillator, RSI, MACD, ATR, and OBV. Implemented and benchmarked sequence models (LSTM, GRU) against traditional machine learning baselines (Linear Regression, Random Forest) on a 30-day sliding window. Mitigated temporal data leakage through chronological split structures and leak-safe preprocessing pipelines, generating interactive prediction metrics, residual distributions, and feature correlation heatmaps.

## Strong Resume Bullet Options

* **Architected an End-to-End ML Pipeline**: Constructed a stock price forecasting pipeline using `Python`, `TensorFlow`, `scikit-learn`, `pandas`, and `yfinance` to evaluate sequential deep learning (LSTM, GRU) against static baselines (Linear Regression, Random Forest).
* **Engineered 21 Advanced Financial Features**: Formulated technical indicators for trend, momentum, volatility, and volume (including Bollinger Bands, Stochastic Oscillator, RSI, MACD, ATR, and OBV) to represent temporal market state.
* **Eliminated Temporal Data Leakage**: Enforced chronological train/validation/test splits, leakage-safe sliding-window generation, and isolated fit-transform preprocessing steps to yield authentic out-of-sample evaluations.
* **Developed Interactive Model Diagnostic Dashboard**: Engineered a `Streamlit` application displaying live next-day inference using pre-trained models, interactive indicator charts, residuals density distribution, and an interview concept guide.
* **Analyzed Model Trade-offs**: Evaluated model performance across MAE, RMSE, MAPE, R², and Directional Accuracy, demonstrating that GRUs train faster with fewer parameters while baseline metrics quantify sequence dependency validity.

## Interview Talking Points

* **Avoiding Temporal Leakage**: Why time-series data cannot be randomly shuffled and how isolated feature scalers prevent future information from leaking into training history.
* **Baselines as Sanity Checks**: Why deep learning models are only justified if they outperform simple Linear Regression (linear correlation) and Random Forest (non-linear static patterns) on windowed features.
* **LSTM vs. GRU Gating tradeoffs**: Gated Recurrent Units merge hidden/cell states and use 2 gates instead of 3, leading to faster training times and lower risk of overfitting on noisy price data.
* **Understanding Model Drift & Shift**: How predicting tomorrow's price can degenerate into outputting today's price (shifted graph) and how Directional Accuracy identifies this issue when standard errors look misleadingly low.
* **Residual Analysis**: Using residual density plots to inspect if errors are normally distributed and center around zero, validating constant variance (homoscedasticity).
