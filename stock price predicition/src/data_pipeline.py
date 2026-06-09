from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_SIGNAL",
    "DAILY_RETURN",
    "VOLATILITY_10",
    "MOMENTUM_3",
    "MOMENTUM_5",
    "MOMENTUM_10",
    "BOL_UPPER",
    "BOL_LOWER",
    "STOCH_K",
    "STOCH_D",
    "ATR_14",
    "OBV",
]


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    scaler: MinMaxScaler
    target_scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.Series
    close_reference: np.ndarray


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for symbol={symbol!r} between {start} and {end}.")
    data = normalize_downloaded_columns(data)
    data = data.reset_index()
    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["SMA_5"] = close.rolling(window=5).mean()
    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["DAILY_RETURN"] = close.pct_change()
    df["VOLATILITY_10"] = df["DAILY_RETURN"].rolling(window=10).std()

    df["MOMENTUM_3"] = close - close.shift(3)
    df["MOMENTUM_5"] = close - close.shift(5)
    df["MOMENTUM_10"] = close - close.shift(10)

    # Bollinger Bands
    df["BOL_UPPER"] = df["SMA_20"] + 2 * close.rolling(window=20).std()
    df["BOL_LOWER"] = df["SMA_20"] - 2 * close.rolling(window=20).std()

    # Stochastic Oscillator
    high_14 = high.rolling(window=14).max()
    low_14 = low.rolling(window=14).min()
    df["STOCH_K"] = ((close - low_14) / (high_14 - low_14).replace(0, np.nan)) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(window=3).mean()

    # Average True Range (ATR_14)
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    df["OBV"] = (direction * volume).cumsum()

    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be valid fractions.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[MinMaxScaler, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_scaler = MinMaxScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(train_df[feature_columns])
    X_val = feature_scaler.transform(val_df[feature_columns])
    X_test = feature_scaler.transform(test_df[feature_columns])

    y_train = target_scaler.fit_transform(train_df[["Close"]]).reshape(-1)
    y_val = target_scaler.transform(val_df[["Close"]]).reshape(-1)
    y_test = target_scaler.transform(test_df[["Close"]]).reshape(-1)

    return feature_scaler, target_scaler, X_train, X_val, X_test, y_train, y_val, y_test


def create_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    X, y, date_index = [], [], []
    for idx in range(window_size, len(features)):
        X.append(features[idx - window_size : idx])
        y.append(targets[idx])
        date_index.append(dates.iloc[idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1), pd.Series(date_index)


def prepare_data(
    symbol: str,
    start: str,
    end: str,
    window_size: int,
    train_ratio: float,
    val_ratio: float,
) -> PreparedData:
    raw_df = download_stock_data(symbol=symbol, start=start, end=end)
    df = add_technical_indicators(raw_df)

    train_df, val_df, test_df = chronological_split(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    scaler, target_scaler, train_scaled, val_scaled, test_scaled, train_targets, val_targets, test_targets = scale_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=FEATURE_COLUMNS,
    )

    X_train, y_train, _ = create_sliding_windows(
        features=train_scaled,
        targets=train_targets,
        window_size=window_size,
        dates=train_df["Date"],
    )
    X_val, y_val, _ = create_sliding_windows(
        features=val_scaled,
        targets=val_targets,
        window_size=window_size,
        dates=val_df["Date"],
    )
    X_test, y_test, test_dates = create_sliding_windows(
        features=test_scaled,
        targets=test_targets,
        window_size=window_size,
        dates=test_df["Date"],
    )

    if not len(X_train) or not len(X_val) or not len(X_test):
        raise ValueError(
            "Not enough rows after preprocessing to create sliding windows. "
            "Try a larger date range or smaller window size."
        )

    return PreparedData(
        dataframe=df,
        feature_columns=FEATURE_COLUMNS,
        scaler=scaler,
        target_scaler=target_scaler,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_dates=test_dates,
        close_reference=df["Close"].to_numpy(),
    )


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flattens a 3D array of shape (samples, window_size, features)
    into a 2D array of shape (samples, window_size * features)
    for traditional ML algorithms like Linear Regression or XGBoost.
    """
    samples, window_size, features = X.shape
    return X.reshape(samples, window_size * features)
