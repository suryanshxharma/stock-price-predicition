from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model

# Import project utilities
from src.config import TrainingConfig
from src.data_pipeline import FEATURE_COLUMNS, add_technical_indicators, download_stock_data
from src.train import run_training

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(45deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .section-subheader {
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .recruiter-q {
        font-weight: 700;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Stock Price Prediction & ML Comparative Analysis Dashboard</div>', unsafe_allow_html=True)
st.write("This interactive application runs and compares sequential deep learning models (LSTM, GRU) against traditional ML baselines (Linear Regression, Random Forest) to forecast stock close prices. Constructed with time-series best practices to prevent future-data leakage.")

# Sidebar Controls
st.sidebar.header("⚙️ Model Configuration")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper().strip()

# Date range selection
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col_end:
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# Hyperparameters
window_size = st.sidebar.slider("Sliding Window Size (Days)", min_value=5, max_value=60, value=30)
epochs = st.sidebar.slider("Deep Learning Epochs", min_value=5, max_value=150, value=60)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
lstm_units = st.sidebar.slider("Hidden Units (LSTM/GRU)", min_value=16, max_value=128, value=64)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)

run_train_btn = st.sidebar.button("🚀 Train & Compare All Models", use_container_width=True)

# Helper functions
def load_saved_scaler_and_models(sym):
    artifact_dir = Path("artifacts") / sym.upper()
    if not (artifact_dir / "scaler.pkl").exists() or not (artifact_dir / "target_scaler.pkl").exists():
        return None
    try:
        with open(artifact_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(artifact_dir / "target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)
            
        lstm_model = load_model(artifact_dir / "model_lstm.keras")
        gru_model = load_model(artifact_dir / "model_gru.keras")
        
        with open(artifact_dir / "model_linear.pkl", "rb") as f:
            linear_model = pickle.load(f)
        with open(artifact_dir / "model_random_forest.pkl", "rb") as f:
            rf_model = pickle.load(f)
            
        return scaler, target_scaler, lstm_model, gru_model, linear_model, rf_model
    except Exception as e:
        st.sidebar.error(f"Could not load pre-trained models: {e}")
        return None

def predict_next_day(sym, win_size, scaler, target_scaler, lstm_m, gru_m, lr_m, rf_m):
    # Fetch recent data to compute features
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_str = (pd.Timestamp.now() - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    
    try:
        raw_df = download_stock_data(sym, start=start_str, end=today_str)
        df = add_technical_indicators(raw_df)
        
        if len(df) < win_size:
            st.warning(f"Not enough recent trading days for {sym} to build window of {win_size} days. Found {len(df)} rows.")
            return None
            
        latest_features = df[FEATURE_COLUMNS].iloc[-win_size:]
        latest_date = df["Date"].iloc[-1]
        latest_close = df["Close"].iloc[-1]
        
        # Scale features
        scaled_features = scaler.transform(latest_features)
        dl_input = np.expand_dims(scaled_features, axis=0) # (1, win_size, feature_count)
        flat_input = dl_input.reshape(1, -1) # (1, win_size * feature_count)
        
        # Models predictions
        lstm_s = lstm_m.predict(dl_input, verbose=0)
        gru_s = gru_m.predict(dl_input, verbose=0)
        lr_s = lr_m.predict(flat_input)
        rf_s = rf_m.predict(flat_input)
        
        # Inverse scaled targets
        lstm_p = target_scaler.inverse_transform(lstm_s.reshape(-1, 1))[0][0]
        gru_p = target_scaler.inverse_transform(gru_s.reshape(-1, 1))[0][0]
        lr_p = target_scaler.inverse_transform(lr_s.reshape(-1, 1))[0][0]
        rf_p = target_scaler.inverse_transform(rf_s.reshape(-1, 1))[0][0]
        
        return {
            "latest_date": latest_date,
            "latest_close": latest_close,
            "LSTM": lstm_p,
            "GRU": gru_p,
            "Linear Regression": lr_p,
            "Random Forest": rf_p
        }
    except Exception as e:
        st.error(f"Error executing live next-day prediction: {e}")
        return None

# Trigger training if requested
if run_train_btn:
    with st.spinner(f"Downloading data and training LSTM, GRU, Linear Regression, and Random Forest models for {symbol}..."):
        config = TrainingConfig(
            symbol=symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            window_size=window_size,
            epochs=epochs,
            batch_size=batch_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        try:
            metrics = run_training(config)
            st.success(f"Models successfully trained for {symbol}!")
        except Exception as e:
            st.error(f"Error during training pipeline: {e}")

# Check for existing models
artifact_path = Path("artifacts") / symbol
has_artifacts = artifact_path.exists() and (artifact_path / "metrics.json").exists()

if has_artifacts:
    # Load summary & metrics
    with open(artifact_path / "metrics.json") as f:
        metrics_dict = json.load(f)
    with open(artifact_path / "training_summary.json") as f:
        summary_dict = json.load(f)
        
    # Upgrade old flat metrics format to new nested format dynamically
    if "mae" in metrics_dict:
        metrics_dict = {"LSTM": metrics_dict}
        
    # Layout tabs
    tab_overview, tab_prediction, tab_indicators, tab_diagnostics, tab_guide = st.tabs([
        "📊 Comparative Overview", 
        "🔮 Next-Day Prediction",
        "📈 Technical Indicators", 
        "🔬 Model Diagnostics",
        "🎓 Recruiter ML Guide"
    ])
    
    # Tab 1: Comparative Overview
    with tab_overview:
        st.markdown('<div class="section-subheader">Performance Metrics Comparison</div>', unsafe_allow_html=True)
        
        # Convert metrics to dataframe with explicit column alignment
        formatted_metrics = {}
        for model_name, m in metrics_dict.items():
            formatted_metrics[model_name] = {
                "MAE ($)": m.get("mae", 0.0),
                "RMSE ($)": m.get("rmse", 0.0),
                "MAPE (%)": m.get("mape", 0.0),
                "R² Score": m.get("r2", 0.0),
                "Directional Accuracy (%)": m.get("directional_accuracy", 0.0)
            }
        metrics_df = pd.DataFrame(formatted_metrics).T
        
        st.dataframe(
            metrics_df.style.format({
                "MAE ($)": "{:.2f}",
                "RMSE ($)": "{:.2f}",
                "MAPE (%)": "{:.2f}%",
                "R² Score": "{:.4f}",
                "Directional Accuracy (%)": "{:.2f}%"
            }).highlight_max(subset=["R² Score", "Directional Accuracy (%)"], color="#d4edda")
              .highlight_min(subset=["MAE ($)", "RMSE ($)", "MAPE (%)"], color="#d4edda"),
            use_container_width=True
        )
        
        # Add quick commentary on metrics
        st.info("**Key Insight for Interviews:** Traditional baselines like Linear Regression and Random Forest serve as standard validation. If the deep learning sequential models (LSTM, GRU) do not outperform the baseline models significantly, it implies temporal dependencies are weak relative to noise, or the sequential model is overfitting.")

        # Show prediction vs actual & comparative plots
        st.markdown('<div class="section-subheader">Visual Pipeline Performance Plots</div>', unsafe_allow_html=True)
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if (artifact_path / "actual_vs_predicted.png").exists():
                st.image(str(artifact_path / "actual_vs_predicted.png"), caption="Actual Close vs Predictions (Test Set)", use_column_width=True)
        with col_img2:
            if (artifact_path / "model_comparison.png").exists():
                st.image(str(artifact_path / "model_comparison.png"), caption="Comparative Performance of Models Across Metrics", use_column_width=True)

    # Tab 2: Next-Day Prediction
    with tab_prediction:
        st.markdown('<div class="section-subheader">Real-Time Next-Day Forecasting</div>', unsafe_allow_html=True)
        
        models_data = load_saved_scaler_and_models(symbol)
        if models_data:
            scaler, target_scaler, lstm_model, gru_model, linear_model, rf_model = models_data
            
            with st.spinner("Fetching latest market data and running inference..."):
                pred_results = predict_next_day(
                    symbol, window_size, scaler, target_scaler, 
                    lstm_model, gru_model, linear_model, rf_model
                )
                
            if pred_results:
                latest_date_fmt = pd.to_datetime(pred_results["latest_date"]).strftime("%B %d, %Y")
                st.write(f"**Latest available trading day in Yahoo Finance:** {latest_date_fmt}")
                st.write(f"**Last Close Price:** ${pred_results['latest_close']:.2f}")
                st.write("Below are the predicted close prices for the next trading day:")
                
                col1, col2, col3, col4 = st.columns(4)
                
                models_list = ["LSTM", "GRU", "Linear Regression", "Random Forest"]
                cols = [col1, col2, col3, col4]
                
                for idx, model_name in enumerate(models_list):
                    pred_val = pred_results[model_name]
                    delta = pred_val - pred_results["latest_close"]
                    delta_pct = (delta / pred_results["latest_close"]) * 100
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{model_name} Forecast",
                            value=f"${pred_val:.2f}",
                            delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
                            delta_color="normal"
                        )
                
                # Add disclaimer
                st.caption("*Disclaimer: This prediction is for educational/resume presentation purposes and does not constitute trading advice.*")
            else:
                st.warning("Could not compute next-day forecast due to insufficient recent market data.")
        else:
            st.warning("Could not load scaler or models. Please run a model training pass first.")

    # Tab 3: Technical Indicators
    with tab_indicators:
        st.markdown('<div class="section-subheader">Engineered Technical Indicators</div>', unsafe_allow_html=True)
        st.write("We expanded the stock features beyond closing price to engineer **21 features** capturing trend, momentum, volatility, and volume signals. This provides deep feature representations to help models isolate market signals from noise.")
        
        if (artifact_path / "price_with_indicators.png").exists():
            st.image(str(artifact_path / "price_with_indicators.png"), caption="Close Price plotted with 20-Day SMA, 12-Day EMA, and Bollinger Bands", use_column_width=True)
            
        st.markdown("### Engineered Feature Details")
        st.markdown("""
        * **Trend Indicators**: Simple Moving Averages (`SMA_5`, `SMA_10`, `SMA_20`) and Exponential Moving Averages (`EMA_12`, `EMA_26`). Bollinger Bands (`BOL_UPPER`, `BOL_LOWER`) capture volatility-adjusted trend boundaries.
        * **Momentum & Oscillators**: Relative Strength Index (`RSI_14`), Moving Average Convergence Divergence (`MACD`, `MACD_SIGNAL`), and Stochastic Oscillator (`STOCH_K`, `STOCH_D`).
        * **Volatility & Returns**: Daily returns (`DAILY_RETURN`), and 10-day rolling standard deviation of daily returns (`VOLATILITY_10`) alongside Average True Range (`ATR_14`).
        * **Volume Indicators**: On-Balance Volume (`OBV`) captures cumulative volume flow to confirm price movements.
        """)

    # Tab 4: Model Diagnostics
    with tab_diagnostics:
        st.markdown('<div class="section-subheader">Neural Network & Residual Diagnostics</div>', unsafe_allow_html=True)
        
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            if (artifact_path / "training_history.png").exists():
                st.image(str(artifact_path / "training_history.png"), caption="LSTM and GRU Training vs Validation Loss curves (verifying no overfitting)", use_column_width=True)
        with col_diag2:
            if (artifact_path / "residual_analysis.png").exists():
                st.image(str(artifact_path / "residual_analysis.png"), caption="Residual Error Density Distributions across all 4 models", use_column_width=True)
                
        st.markdown('<div class="section-subheader">Feature Multicollinearity Assessment</div>', unsafe_allow_html=True)
        if (artifact_path / "feature_correlation_heatmap.png").exists():
            st.image(str(artifact_path / "feature_correlation_heatmap.png"), caption="Feature Correlation Heatmap (used to analyze redundant features)", use_column_width=True)

    # Tab 5: Recruiter ML Guide
    with tab_guide:
        st.markdown('<div class="section-subheader">Interview QA & Concept Cheat Sheet</div>', unsafe_allow_html=True)
        st.write("Prepare for technical review sessions by understanding the design decisions made in this pipeline.")
        
        with st.expander("Q1: Why can we NOT randomly split time-series data into train, validation, and test sets?"):
            st.markdown("""
            <div class="recruiter-q">Why it matters:</div>
            Random shuffling (standard `train_test_split`) causes <strong>temporal data leakage</strong>. In time series, data points are highly correlated across time. If you randomly shuffle, information from future dates (e.g., day $t+1$) is leaked into the training set, and we evaluate on past dates (e.g., day $t$). This results in artificially high validation performance that collapses during real-world future evaluation. 
            <br><br>
            <strong>Our Solution:</strong> We implement a strict <strong>chronological split</strong> (70% oldest data for training, 15% middle for validation, 15% newest for test) to evaluate the model on truly unseen future data, matching real-world walk-forward testing.
            """, unsafe_allow_html=True)
            
        with st.expander("Q2: Why did you implement Linear Regression and Random Forest as baseline models?"):
            st.markdown("""
            <div class="recruiter-q">Why it matters:</div>
            A complex neural network (LSTM/GRU) is useless if it cannot outperform a simple linear baseline or a tree-based model. 
            <ul>
                <li><strong>Linear Regression</strong> evaluates if stock movements can be captured via a simple linear combination of engineered features.</li>
                <li><strong>Random Forest</strong> captures static, non-linear relationships without temporal modeling.</li>
                <li><strong>LSTM/GRU</strong> capture long-range sequential, temporal dependencies across the 30-day window.</li>
            </ul>
            Comparing them proves whether the added complexity of Recurrent Neural Networks pays off in predictive power.
            """, unsafe_allow_html=True)
            
        with st.expander("Q3: What are the differences between LSTM and GRU architectures?"):
            st.markdown("""
            <div class="recruiter-q">Why it matters:</div>
            Both solve the vanishing gradient problem in standard RNNs using internal gates.
            <ul>
                <li><strong>LSTM (Long Short-Term Memory):</strong> Has three gates (input, forget, output) and maintains an explicit cell state ($c_t$) alongside the hidden state ($h_t$).</li>
                <li><strong>GRU (Gated Recurrent Unit):</strong> Merges the cell state and hidden state, and features only two gates (reset and update). It has fewer parameters, making it computationally faster and less prone to overfitting on smaller datasets.</li>
            </ul>
            In our experiments, we compare both architectures to identify if the added parameters of LSTM yield better representation or simply lead to overfitting.
            """, unsafe_allow_html=True)
            
        with st.expander("Q4: Why does a model sometimes forecast a 'shifted' version of the actual price?"):
            st.markdown("""
            <div class="recruiter-q">Why it matters:</div>
            Stock prices resemble a random walk. If the model is not trained properly or lacks signal, it discovers that the safest mathematical prediction to minimize MSE is: 
            $$\hat{y}_{t+1} \approx y_t$$
            This manifests visually as the predicted line looking exactly like the actual line shifted by one day. 
            <br><br>
            <strong>How to inspect this:</strong> Residual plots and **Directional Accuracy** (predicting the sign of return correctly) help diagnose this. If Directional Accuracy is around 50%, the model is just guessing up/down, even if its RMSE or MAE is low.
            """, unsafe_allow_html=True)

else:
    st.warning(f"No pre-trained model artifacts found for symbol {symbol}. Please configure parameters and click 'Train & Compare All Models' in the sidebar.")
