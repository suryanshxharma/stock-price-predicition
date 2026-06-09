from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    window_size: int,
    feature_count: int,
    lstm_units: int,
    dropout_rate: float,
    learning_rate: float,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def build_gru_model(
    window_size: int,
    feature_count: int,
    gru_units: int,
    dropout_rate: float,
    learning_rate: float,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            GRU(gru_units, return_sequences=True),
            Dropout(dropout_rate),
            GRU(gru_units // 2),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
