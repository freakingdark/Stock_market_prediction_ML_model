# 📈 Stock Market Prediction using LSTM (Multi-Feature Deep Learning Model)

Author: Deepali Pandey  

This project implements a multi-feature LSTM-based deep learning model to predict next-day log returns of the NIFTY Bank index (^NSEBANK) using technical indicators and historical price data.

The model leverages sequence modeling with Long Short-Term Memory (LSTM) networks and multiple derived financial indicators.

---

# 🔹 Project Overview

Objective:
Predict next-day log returns and reconstruct next-day closing price using a deep learning sequence model trained on historical stock data.

Asset Used:
^NSEBANK (NIFTY Bank Index)

Time Range:
2008-01-01 to 2024-01-03

Sequence Length:
60 trading days

Train / Validation / Test Split:
- Train: 70%
- Validation: 10%
- Test: 20%

---

# 🔹 Features Used

Raw Market Features:
- Open
- High
- Low
- Close
- Volume

Technical Indicators:
- RSI (14)
- MACD
- MACD Signal
- ATR (Average True Range)
- EMA(12)
- EMA(26)
- EMA difference
- High-Low percentage
- Percentage change
- 1-day momentum
- 5-day momentum

Target Variable:
Next-day log return

target_return = log(Close(t+1) / Close(t))

---

# 🔹 Data Pipeline

1. Download historical data using yfinance.
2. Compute technical indicators using ta library.
3. Handle missing values.
4. Split dataset chronologically (no leakage).
5. Scale features using MinMaxScaler.
6. Create rolling sequences of length 60.
7. Train LSTM model on multi-dimensional sequences.

---

# 🔹 Model Architecture

Sequential Deep Learning Model:

- LSTM (128 units, return_sequences=True)
- Dropout (0.2)
- LSTM (64 units)
- Batch Normalization
- Dropout (0.2)
- Dense (32, ReLU)
- Dense (1, Linear)

Loss Function:
Mean Squared Error (MSE)

Optimizer:
Adam

Callbacks:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

---

# 🔹 Evaluation Metrics

- RMSE (log-return prediction)
- Directional Accuracy (sign prediction)
- Price Reconstruction using:
  
  predicted_price = base_price * exp(predicted_log_return)

Validation and test performance are computed after masking NaN/Inf values.

---

# 🔹 Visualizations

The project includes:

- Historical price plot
- RSI and MACD visualization
- Feature correlation heatmap
- Target return distribution
- Training vs validation loss curve
- True vs predicted closing price plot

---

# 🔹 Training Process

Sequences are created as:

For each time step t:
Input → past 60 days of features  
Output → log return at time t

The model learns temporal patterns across multiple financial indicators.

---

# 🔹 Key Machine Learning Concepts Demonstrated

- Time-series sequence modeling
- Feature engineering with technical indicators
- Multivariate LSTM networks
- Chronological data splitting
- Overfitting control using dropout and early stopping
- Learning rate scheduling
- Log-return modeling for stability
- Reconstruction of predicted prices

---

# 🔹 Installation

Clone the repository:

```bash
git clone https://github.com/freakingdark/Stock_market_prediction_ML_model.git
cd Stock_market_prediction_ML_model
