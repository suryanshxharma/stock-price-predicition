from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed as tf_set_seed

from .config import TrainingConfig
from .data_pipeline import flatten_windows, prepare_data
from .model import build_gru_model, build_lstm_model
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train stock price prediction models.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    return TrainingConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    directional_accuracy = float(np.mean(direction_true == direction_pred) * 100) if len(direction_true) else 0.0

    return {
        "mae": float(mae),
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
    }


def save_prediction_plot(dates: pd.Series, actual: np.ndarray, model_preds: dict[str, np.ndarray], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual Close Price", linewidth=2.5, color="black")
    colors = {"LSTM": "#1f77b4", "GRU": "#ff7f0e", "Linear Regression": "#2ca02c", "Random Forest": "#d62728"}
    for name, pred in model_preds.items():
        plt.plot(dates, pred, label=f"Predicted ({name})", linewidth=1.5, alpha=0.8, color=colors.get(name))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Model Comparison: Actual vs Predicted Stock Close Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_history_plot(lstm_history: dict, gru_history: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # LSTM
    axes[0].plot(lstm_history.get("loss", []), label="Train Loss", linewidth=2)
    axes[0].plot(lstm_history.get("val_loss", []), label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("LSTM: Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # GRU
    axes[1].plot(gru_history.get("loss", []), label="Train Loss", linewidth=2)
    axes[1].plot(gru_history.get("val_loss", []), label="Val Loss", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("GRU: Training vs Validation Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_residual_plot(actual: np.ndarray, model_preds: dict[str, np.ndarray], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    colors = {"LSTM": "#1f77b4", "GRU": "#ff7f0e", "Linear Regression": "#2ca02c", "Random Forest": "#d62728"}
    for name, pred in model_preds.items():
        residuals = actual - pred
        sns.kdeplot(residuals, label=f"{name} Residuals", color=colors.get(name), fill=True, alpha=0.1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Density")
    plt.title("Residual Error Distributions Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_comparison_bar_chart(all_metrics: dict[str, dict], output_path: Path) -> None:
    models = list(all_metrics.keys())
    metrics_list = ["mae", "rmse", "mape", "directional_accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    titles = {
        "mae": "Mean Absolute Error (Lower is Better)",
        "rmse": "Root Mean Squared Error (Lower is Better)",
        "mape": "Mean Absolute Percentage Error (Lower is Better)",
        "directional_accuracy": "Directional Accuracy % (Higher is Better)"
    }
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, metric in enumerate(metrics_list):
        values = [all_metrics[model][metric] for model in models]
        bars = axes[i].bar(models, values, color=colors[:len(models)], edgecolor='black', alpha=0.8)
        axes[i].set_title(titles[metric], fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    plt.suptitle("Model Evaluation Summary Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_correlation_heatmap(dataframe: pd.DataFrame, feature_columns: list[str], output_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    correlation = dataframe[feature_columns].corr()
    sns.heatmap(correlation, cmap="YlGnBu", linewidths=0.3)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_price_indicator_plot(dataframe: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe["Date"], dataframe["Close"], label="Close", linewidth=2)
    plt.plot(dataframe["Date"], dataframe["SMA_20"], label="SMA 20", linewidth=1.8)
    plt.plot(dataframe["Date"], dataframe["EMA_12"], label="EMA 12", linewidth=1.6)
    plt.plot(dataframe["Date"], dataframe["BOL_UPPER"], label="Bollinger Upper", linewidth=1.2, linestyle="--")
    plt.plot(dataframe["Date"], dataframe["BOL_LOWER"], label="Bollinger Lower", linewidth=1.2, linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Close Price with Trend Indicators & Bollinger Bands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_training(config: TrainingConfig) -> dict:
    set_seed(config.random_seed)
    tf_set_seed(config.random_seed)

    prepared = prepare_data(
        symbol=config.symbol,
        start=config.start,
        end=config.end,
        window_size=config.window_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    artifact_dir = ensure_dir(config.artifact_dir)

    # Save scalers for live predictions
    with (artifact_dir / "scaler.pkl").open("wb") as file_obj:
        pickle.dump(prepared.scaler, file_obj)
    with (artifact_dir / "target_scaler.pkl").open("wb") as file_obj:
        pickle.dump(prepared.target_scaler, file_obj)

    # Setup deep learning callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    actuals = prepared.target_scaler.inverse_transform(prepared.y_test.reshape(-1, 1)).reshape(-1)
    
    # Dicts to collect predictions & metrics
    all_preds = {}
    all_metrics = {}

    # 1. Train LSTM
    print("Training LSTM model...")
    lstm_model = build_lstm_model(
        window_size=config.window_size,
        feature_count=len(prepared.feature_columns),
        lstm_units=config.lstm_units,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
    )
    lstm_history = lstm_model.fit(
        prepared.X_train,
        prepared.y_train,
        validation_data=(prepared.X_val, prepared.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=[early_stopping],
        shuffle=False,
    )
    print("LSTM training finished.")
    lstm_model.save(artifact_dir / "model_lstm.keras")
    print("LSTM model saved.")
    
    scaled_lstm_preds = lstm_model.predict(prepared.X_test, verbose=0).reshape(-1, 1)
    lstm_preds = prepared.target_scaler.inverse_transform(scaled_lstm_preds).reshape(-1)
    all_preds["LSTM"] = lstm_preds
    all_metrics["LSTM"] = evaluate_predictions(actuals, lstm_preds)

    # 2. Train GRU
    print("Training GRU model...")
    gru_model = build_gru_model(
        window_size=config.window_size,
        feature_count=len(prepared.feature_columns),
        gru_units=config.lstm_units,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
    )
    gru_history = gru_model.fit(
        prepared.X_train,
        prepared.y_train,
        validation_data=(prepared.X_val, prepared.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=[early_stopping],
        shuffle=False,
    )
    print("GRU training finished.")
    gru_model.save(artifact_dir / "model_gru.keras")
    print("GRU model saved.")
    
    scaled_gru_preds = gru_model.predict(prepared.X_test, verbose=0).reshape(-1, 1)
    gru_preds = prepared.target_scaler.inverse_transform(scaled_gru_preds).reshape(-1)
    all_preds["GRU"] = gru_preds
    all_metrics["GRU"] = evaluate_predictions(actuals, gru_preds)

    # Prepare 2D flattened features for baseline models
    X_train_flat = flatten_windows(prepared.X_train)
    X_val_flat = flatten_windows(prepared.X_val)
    X_test_flat = flatten_windows(prepared.X_test)

    # 3. Train Linear Regression Baseline
    print("Training Linear Regression baseline...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_flat, prepared.y_train)
    with (artifact_dir / "model_linear.pkl").open("wb") as file_obj:
        pickle.dump(lr_model, file_obj)
    print("Linear Regression model saved.")
        
    scaled_lr_preds = lr_model.predict(X_test_flat).reshape(-1, 1)
    lr_preds = prepared.target_scaler.inverse_transform(scaled_lr_preds).reshape(-1)
    all_preds["Linear Regression"] = lr_preds
    all_metrics["Linear Regression"] = evaluate_predictions(actuals, lr_preds)

    # 4. Train Random Forest Baseline
    print("Training Random Forest baseline...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=config.random_seed
    )
    rf_model.fit(X_train_flat, prepared.y_train)
    with (artifact_dir / "model_random_forest.pkl").open("wb") as file_obj:
        pickle.dump(rf_model, file_obj)
    print("Random Forest model saved.")
        
    scaled_rf_preds = rf_model.predict(X_test_flat).reshape(-1, 1)
    rf_preds = prepared.target_scaler.inverse_transform(scaled_rf_preds).reshape(-1)
    all_preds["Random Forest"] = rf_preds
    all_metrics["Random Forest"] = evaluate_predictions(actuals, rf_preds)

    # Save comparative outputs
    print("Saving predictions CSV...")
    predictions_df = pd.DataFrame(
        {
            "Date": prepared.test_dates,
            "Actual_Close": actuals,
            "Predicted_Close_LSTM": all_preds["LSTM"],
            "Predicted_Close_GRU": all_preds["GRU"],
            "Predicted_Close_Linear": all_preds["Linear Regression"],
            "Predicted_Close_Random_Forest": all_preds["Random Forest"],
        }
    )
    predictions_df.to_csv(artifact_dir / "predictions.csv", index=False)

    # Save visual comparison files
    print("Saving prediction plot...")
    save_prediction_plot(
        dates=prepared.test_dates,
        actual=actuals,
        model_preds=all_preds,
        output_path=artifact_dir / "actual_vs_predicted.png",
    )
    print("Saving training history plot...")
    save_training_history_plot(lstm_history.history, gru_history.history, artifact_dir / "training_history.png")
    print("Saving residual plot...")
    save_residual_plot(actuals, all_preds, artifact_dir / "residual_analysis.png")
    print("Saving comparison bar chart...")
    save_comparison_bar_chart(all_metrics, artifact_dir / "model_comparison.png")
    print("Saving feature correlation heatmap...")
    save_feature_correlation_heatmap(
        prepared.dataframe,
        prepared.feature_columns,
        artifact_dir / "feature_correlation_heatmap.png",
    )
    print("Saving price indicator plot...")
    save_price_indicator_plot(prepared.dataframe, artifact_dir / "price_with_indicators.png")

    print("Saving metrics json...")
    save_json(artifact_dir / "metrics.json", all_metrics)
    print("Saving training summary json...")
    save_json(
        artifact_dir / "training_summary.json",
        {
            "symbol": config.symbol,
            "start": config.start,
            "end": config.end,
            "window_size": config.window_size,
            "feature_count": len(prepared.feature_columns),
            "epochs_requested": config.epochs,
            "epochs_trained_lstm": len(lstm_history.history.get("loss", [])),
            "epochs_trained_gru": len(gru_history.history.get("loss", [])),
            "features": prepared.feature_columns,
            "metrics": all_metrics,
        },
    )

    return all_metrics


def main() -> None:
    config = parse_args()
    all_metrics = run_training(config)
    print("Training completed.")
    for model_name, metrics in all_metrics.items():
        print(f"\n--- {model_name} Metrics ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
