# Stock Price Prediction Using Machine Learning

Minor project report submitted in partial fulfilment of the requirement for the degree of Bachelor of Technology in Computer Science and Engineering.

**Prepared by**

- AADITYA (221030039)
- SURYANSH SHARMA (221030426)
- PARAV THAKUR (221030250)

**Under the supervision of**

Ms. Nitika, Assistant Professor

Department of Computer Science and Engineering and Information Technology

Jaypee University of Information Technology, Waknaghat, Himachal Pradesh

---

## Table of Contents

1. Abstract
2. Chapter 1: Introduction
3. Chapter 2: Feasibility Study, Requirements Analysis and Design
4. Chapter 3: Implementation
5. Chapter 4: Results, Limitations and Future Scope
6. References

---

## Abstract

Stock price prediction is a challenging time-series forecasting problem because the market is influenced by volatility, trend shifts, investor behavior, and external events. This project presents a machine learning based framework for predicting future stock closing prices using historical market data and engineered technical indicators. The proposed system focuses on a sequence-aware learning pipeline built around sliding window based data preparation, feature scaling, and deep learning for temporal pattern extraction.

The project uses historical stock data such as Open, High, Low, Close, and Volume values collected over a fixed period. From this raw data, additional indicators such as moving averages, exponential moving averages, Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), rolling volatility, and momentum are computed. A sliding window mechanism is then used to convert the continuous time series into supervised learning sequences so that the model can learn short-term and medium-term dependencies.

For prediction, Long Short-Term Memory (LSTM) networks are used because they are well suited for sequential data and can preserve useful information over time steps better than traditional regression models. The model is trained and evaluated using time-series aware data splitting to avoid data leakage. Performance is measured using MAE, RMSE, MAPE, and R2 score. The report also discusses key interview-relevant concepts such as sequence modeling, leakage prevention, window size selection, walk-forward validation, regularization, and hyperparameter tuning.

This project demonstrates how time-series preprocessing and deep learning can be combined to build a practical stock price prediction system suitable for academic evaluation and technical interviews.

---

## Chapter 1: Introduction

### 1.1 Introduction

The stock market is one of the most dynamic financial systems, where price movement depends on supply and demand, investor sentiment, macroeconomic conditions, company performance, and global events. Predicting stock prices is therefore a complex problem involving non-linearity, temporal dependencies, and noise.

Traditional machine learning models such as Linear Regression, Decision Trees, and Random Forest can capture some useful relationships, but they usually rely on fixed tabular features and may not fully model the sequential nature of stock prices. Since stock data is fundamentally a time series, it is important to use a technique that can learn from historical sequences directly.

This project approaches stock price prediction as a supervised time-series forecasting problem. Historical data is transformed into sequential samples using a sliding window approach. These sequences are then used to train an LSTM-based neural network, which is capable of learning temporal dependencies across consecutive trading days.

The goal is to estimate the next closing price of a stock from recent historical patterns and technical indicators while following a reliable and leakage-safe machine learning workflow.

### 1.2 Objectives

The main objectives of the project are listed below:

- To collect and preprocess historical stock market data.
- To engineer informative technical indicators from raw OHLCV data.
- To transform time-series data into supervised learning sequences using a sliding window.
- To build an LSTM-based deep learning model for next-day stock price prediction.
- To evaluate model performance using standard regression metrics.
- To study interview-relevant concepts such as time-series validation, overfitting control, and hyperparameter tuning.

### 1.3 Motivation

Stock price prediction is an attractive domain because it combines finance, statistics, machine learning, and time-series analysis in a single problem statement. It is also a strong interview topic because it allows discussion of real-world data challenges such as:

- Non-stationary data
- Noisy signals
- Feature engineering
- Data leakage
- Sequence creation with sliding windows
- Model comparison between classical ML and deep learning
- Proper evaluation of temporal data

This project was selected to build both practical implementation skills and conceptual depth in applied machine learning.

### 1.4 Scope

The scope of this project includes:

- Predicting the next closing price from historical stock data
- Using technical indicators to improve signal quality
- Applying sequence models to time-series data
- Performing offline training and evaluation on historical records

### 1.5 Technical Requirements

**Hardware**

- Development machine with at least 8 GB RAM
- Multi-core processor recommended for faster training

**Software**

- Python 3.x
- Jupyter Notebook or VS Code
- Pandas
- NumPy
- scikit-learn
- TensorFlow or Keras
- Matplotlib and Seaborn
- yfinance

### 1.6 Expected Outcomes

- A cleaned and processed stock market dataset
- A feature-engineered time-series dataset
- Sliding window based training samples
- A trained LSTM model for price prediction
- Performance comparison using multiple evaluation metrics
- Plots showing actual vs predicted prices

---

## Chapter 2: Feasibility Study, Requirements Analysis and Design

### 2.1 Feasibility Study

#### 2.1.1 Problem Definition

The task is to predict the next closing price of a selected stock based on historical daily price data and derived technical indicators. Since price values depend on previous market behavior, the learning system must capture sequential dependencies instead of treating each row as fully independent.

#### 2.1.2 Technical Feasibility

The project is technically feasible because:

- Historical stock data is publicly accessible through Yahoo Finance.
- Python provides mature libraries for preprocessing, visualization, and modeling.
- LSTM models can be trained on academic-scale datasets using standard machines.
- Time-series forecasting pipelines can be implemented without specialized infrastructure.

#### 2.1.3 Economic Feasibility

The project has low development cost because it uses open-source libraries and public datasets. It can be completed using a standard academic development setup.

#### 2.1.4 Operational Feasibility

The workflow can be executed by students with knowledge of Python, machine learning, and basic neural networks. The output is easy to explain, visualize, and evaluate in an academic setting.

### 2.2 Requirements Analysis

#### 2.2.1 Functional Requirements

The system should:

- Download historical stock data for a selected symbol
- Clean missing and invalid values
- Generate technical indicators
- Normalize the data before training
- Create sequential samples using a sliding window
- Train the prediction model on historical sequences
- Predict the next-day closing price
- Evaluate predictions using regression metrics

#### 2.2.2 Non-Functional Requirements

The system should satisfy the following qualities:

- Accuracy: predictions should follow the underlying trend reasonably well
- Reliability: the model should produce stable outputs on unseen time periods
- Explainability: the pipeline should be easy to justify in viva and interviews
- Modularity: the code should support changing stock symbols and window sizes
- Reproducibility: the training process should be repeatable with fixed seeds

### 2.3 Dataset Description

The project uses historical stock data retrieved through the `yfinance` library. The dataset contains:

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

The data frequency is daily, and the analysis can be performed on one or more years of trading history depending on availability.

### 2.4 Feature Engineering

To improve predictive performance, the following features are generated:

- Simple Moving Average (SMA-5, SMA-10, SMA-20)
- Exponential Moving Average (EMA-12, EMA-26)
- Relative Strength Index (RSI)
- MACD and signal line
- Daily return
- Rolling standard deviation as volatility
- Momentum over 3-day, 5-day, and 10-day periods

These features help the model capture trend, momentum, volatility, and mean-reversion signals.

### 2.5 Design Concepts with Interview Value

#### 2.5.1 Sliding Window

Sliding window is the core concept used to convert time-series data into supervised learning data. Instead of predicting from a single row, the model observes a fixed number of previous time steps.

Example:

- If window size = 30, then the model uses the previous 30 trading days to predict day 31.
- The window then slides by one step and uses days 2 to 31 to predict day 32.

This approach is important because:

- It preserves temporal order
- It allows the model to learn short-term dependencies
- It transforms sequential data into trainable samples

#### 2.5.2 Data Leakage Prevention

Time-series projects are highly sensitive to leakage. Random shuffling must be avoided because future information should never influence past predictions. To prevent leakage:

- Train, validation, and test sets are split chronologically
- Scaling is fitted only on training data
- Future labels are never used in feature generation

#### 2.5.3 Walk-Forward Validation

Walk-forward validation is better than random cross-validation for time series. The model is repeatedly trained on past data and evaluated on the next unseen block. This gives a more realistic estimate of live performance.

#### 2.5.4 Why LSTM Instead of Standard Regression

LSTM is preferred because:

- It is designed for sequential dependencies
- It can retain information over multiple time steps
- It handles temporal context better than standard tabular models

#### 2.5.5 Regularization and Overfitting Control

Since deep learning models may overfit small datasets, the following controls are useful:

- Dropout layers
- Early stopping
- Validation monitoring
- Reduced model complexity

### 2.6 Data Flow

The data flow of the project is:

1. Fetch stock data
2. Clean and preprocess records
3. Generate technical indicators
4. Normalize features
5. Create sliding window sequences
6. Train LSTM model
7. Predict next-day closing price
8. Evaluate and visualize results

---

## Chapter 3: Implementation

### 3.1 Dataset Used in the Project

The project uses historical daily stock data collected using Yahoo Finance. A single stock such as AAPL, TSLA, or INFY may be used for demonstration. The dataset typically contains around 252 trading records per year.

### 3.2 Preprocessing Steps

The preprocessing pipeline consists of the following steps:

1. Remove null values created after rolling calculations.
2. Sort data by date in ascending order.
3. Select input features and target variable.
4. Scale the feature values using MinMaxScaler.
5. Convert the sequential data into input-output windows.

### 3.3 Sliding Window Formulation

Let the multivariate feature matrix be represented as `X = [x1, x2, x3, ..., xn]`.

For a window size `w`, each training sample is created as:

- Input: `[xt-w, xt-w+1, ..., xt-1]`
- Output: `yt`

Where `yt` is the closing price at time `t`.

If `w = 30`, then each input sample contains 30 consecutive trading days of features. This gives the network temporal context instead of isolated values.

### 3.4 Model Architecture

A typical LSTM architecture used in this project is:

1. Input layer receiving sequences of shape `(window_size, number_of_features)`
2. LSTM layer with 50 to 100 units
3. Dropout layer to reduce overfitting
4. Optional second LSTM layer for richer temporal modeling
5. Dense layer
6. Output layer predicting the next closing price

This design is simple enough to implement in a minor project while still being technically impressive in interviews.

### 3.5 Training Strategy

The training strategy includes:

- Chronological split into train, validation, and test sets
- Batch training over multiple epochs
- Early stopping based on validation loss
- Hyperparameter tuning for window size, LSTM units, batch size, and learning rate

### 3.6 Evaluation Metrics

The following regression metrics are used:

- MAE: average absolute prediction error
- RMSE: penalizes larger errors more strongly
- MAPE: percentage error relative to actual values
- R2 Score: proportion of variance explained by the model

Additional interview-friendly evaluation ideas:

- Directional accuracy: whether the model correctly predicts up/down movement
- Residual analysis: checking whether errors are biased
- Actual vs predicted trend visualization

### 3.7 Pseudocode

```text
Function stock_price_prediction(symbol, window_size):
    Step 1: Download historical stock data
    Step 2: Clean missing values and sort by date
    Step 3: Generate technical indicators
    Step 4: Select features and target close price
    Step 5: Split the dataset chronologically
    Step 6: Fit scaler on training features only
    Step 7: Transform train, validation, and test data
    Step 8: Create sliding window sequences
    Step 9: Build LSTM model
    Step 10: Train model with early stopping
    Step 11: Predict prices on the test set
    Step 12: Inverse transform predictions if needed
    Step 13: Compute MAE, RMSE, MAPE, and R2
    Step 14: Plot actual vs predicted values
    Return trained model, predictions, and evaluation metrics
```

### 3.8 Important Implementation Concepts for Viva and Interviews

- Why scaling is needed before LSTM training
- Why random shuffle is harmful in time-series forecasting
- How window size affects short-term vs long-term context
- Why LSTM handles vanishing gradient better than vanilla RNN
- Why technical indicators may help, but can also introduce redundancy
- Why evaluation must be done on future unseen data only

---

## Chapter 4: Results, Limitations and Future Scope

### 4.1 Discussion of Results

The LSTM-based stock price prediction model is expected to learn general trend movement and local sequential behavior from historical data. When trained with properly engineered features and a suitable window size, the predictions usually follow the real closing-price curve with moderate lag during sharp movements.

The results should be discussed using:

- Actual vs predicted price plots
- MAE and RMSE values
- R2 score
- Trend consistency
- Directional correctness during rising and falling phases

An ideal discussion should highlight that the model performs better during stable trend periods and may struggle during sudden market shocks.

### 4.2 Key Observations

- Sliding window based training improves temporal learning compared to single-row prediction.
- Feature engineering helps the model detect momentum and trend signals.
- Chronological validation gives more realistic results than random splitting.
- LSTM captures sequence dependencies better than traditional regression models.

### 4.3 Limitations

Despite encouraging results, the project has several limitations:

- Stock prices are affected by external events not present in historical price data.
- Technical indicators are derived from past prices, so they cannot capture sudden news-driven jumps.
- Small datasets can make deep learning models unstable.
- Model performance depends heavily on window size and selected features.
- Predictions may lag during abrupt reversals.

### 4.4 Future Scope & Comparative Extensions

The project has been extended to incorporate a multi-model comparative benchmarking framework:

* **GRU Neural Network Implementation**: A Gated Recurrent Unit architecture was introduced to compare with the LSTM. Since GRUs combine hidden and cell states and utilize only two gates (update and reset), they require fewer parameters, train faster, and offer a natural control against overfitting on noisy price series.
* **Classical Baselines (Linear Regression, Random Forest)**: Linear Regression and Random Forest Regressors were trained on flattened window sequences. Hitting these baselines verifies whether temporal deep learning architectures (LSTM/GRU) add statistically significant predictive value over simpler linear or non-linear static models.
* **Extended Feature Set**: Engineered 21 technical indicators including Bollinger Bands (volatility trend boundaries), Stochastic Oscillator (fast/slow momentum indicators), Average True Range (ATR, for price volatility scale), and On-Balance Volume (OBV, representing cumulative volume flows).
* **Interactive Dashboard Integration**: Developed a Streamlit web application enabling recruiters to interactively select stock tickers, dates, and window sizes, run comparative model training, inspect residual error distributions, and explore next-day predictions.

### 4.5 Interview Takeaways

This project gives strong talking points for interviews:

- Time-series forecasting vs standard supervised learning
- Sliding window sequence generation
- LSTM working principle
- Data leakage and chronological splitting
- Evaluation for regression and trend prediction
- Overfitting control using dropout and early stopping
- Feature engineering for financial data

---

## References

1. Sepp Hochreiter and Jurgen Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
2. S. Haykin, Neural Networks and Learning Machines, 3rd ed., Pearson, 2009.
3. Jason Brownlee, Deep Learning for Time Series Forecasting, Machine Learning Mastery, 2018.
4. A. Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd ed., O'Reilly, 2022.
5. Yahoo Finance historical market data accessed through `yfinance`.

---

## Suggested Viva Questions

1. Why is stock price prediction considered a time-series problem?
2. What is the purpose of a sliding window?
3. Why is LSTM preferred over simple RNN?
4. What is data leakage and how did you avoid it?
5. Why do we scale features before neural network training?
6. What happens if window size is too small or too large?
7. How is walk-forward validation different from train-test split?
8. Why are MAE and RMSE both reported?
9. What are the limitations of technical-indicator based forecasting?
10. How would you improve this project further?
