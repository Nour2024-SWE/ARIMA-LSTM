
# Hybrid ARIMA-LSTM Model for Financial Time Series Forecasting

A powerful hybrid forecasting model combining **ARIMA** (linear component) and **Bidirectional LSTM** (nonlinear residual component) to predict daily log returns of financial assets. Built with Python, TensorFlow/Keras, and statsmodels.

## 📌 Overview

Financial time series often contain both linear and nonlinear patterns. Traditional ARIMA models capture linear dependencies well but struggle with nonlinearity, while LSTM networks excel at learning complex patterns but may overfit on small datasets. This hybrid approach:

1. Fits an ARIMA model to capture the linear structure
2. Uses a Bidirectional LSTM to model the nonlinear residuals from ARIMA
3. Combines both predictions for superior accuracy

## ✨ Features

- **End-to-end pipeline**: Data loading, feature engineering, model training, and evaluation
- **Bidirectional LSTM** with dropout for robust residual learning
- **Feature engineering** including:
  - Rolling moving averages (5, 10, 20 days)
  - Rolling volatility (5, 10 days)
- **Fama-French 5-factor model** features + Momentum, Amihud liquidity measure
- **Scikit-learn compatible** API (fit, predict, evaluate)
- **Comprehensive metrics**: RMSE, MAE, R²
- **Visualization** of predictions and training history

## 📊 Model Architecture

```
Input Sequences (n_steps features)
         ↓
  Bidirectional LSTM (100 units, return_sequences=True)
         ↓
       Dropout (0.2)
         ↓
       LSTM (100 units)
         ↓
       Dropout (0.2)
         ↓
     Dense (1 unit)
         ↓
   Residual Prediction
         ↓
   + ARIMA Prediction
         ↓
   Final Hybrid Output
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
git clone https://github.com/yourusername/hybrid-arima-lstm.git
cd hybrid-arima-lstm
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy
pandas
matplotlib
scikit-learn
tensorflow
statsmodels
```

### Data Format

Your CSV file must contain:
- A `Date` column (parsed as datetime index)
- A `Daily Log Return` column (target variable)
- Feature columns (see `features` list in code)

Example structure:
| Date       | Daily Log Return | (MKT-RF) Excess Return | (SMB) | (HML) | ... |
|------------|------------------|------------------------|-------|-------|-----|
| 2020-01-01 | 0.0123           | 0.0156                 | -0.002| 0.003 | ... |

### Quick Start

```python
from Hybrid import HybridARIMALSTM, load_data, add_features

# Load and prepare data
df = load_data('your_data.csv')
df = add_features(df)

# Define features and target
features = ['(MKT-Rf) Excess Return on the Market ', '(SMB) Small-Minus-Big Return',
            '(HML) High-Minus-Low Return', '(RMW) Robust Minus Weak Return',
            '(CMA) Conservative Minus Aggressive Return', '(MOM) Momentum',
            'Amihud LIQ', 'MA_5', 'MA_10', 'MA_20', 'Volatility_5', 'Volatility_10']
target = 'Daily Log Return'

# Split data
train_size = int(0.8 * len(df))
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Initialize and train model
model = HybridARIMALSTM(arima_order=(1,0,1), lstm_units=100, n_steps=60)
model.fit(train[features], train[target])

# Evaluate
model.evaluate(test[features], test[target])
model.plot_predictions(test[features], test[target], "Test Set Results")
```

## 📈 Example Output

```
Training Set Performance:
RMSE: 0.008234
MAE: 0.006123
R²: 0.8912

Test Set Performance:
RMSE: 0.009876
MAE: 0.007456
R²: 0.8456
```

## 🧠 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arima_order` | (1,0,1) | (p,d,q) for ARIMA model |
| `lstm_units` | 100 | Number of LSTM neurons |
| `n_steps` | 60 | Lookback window length |
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Batch size for LSTM |

## 📁 Repository Structure

```
.
├── Hybrid.py               # Main model implementation
├── sample12.csv            # Example dataset (optional)
├── requirements.txt        # Dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 📚 Key References

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model

## ⚠️ Limitations & Improvements

**Current limitations:**
- ARIMA residual calculation assumes stationary residuals
- Requires sufficient historical data (n_steps + training samples)
- LSTM training can be slow on large datasets

**Suggested improvements:**
- Add hyperparameter tuning (GridSearchCV)
- Implement early stopping for LSTM
- Add multivariate ARIMA (VARIMA) support
- Include attention mechanism for LSTM
- Add cross-validation for time series

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - nour.zawawi@gmail.com

Project Link: https://github.com/Nour2024-SWE/ARIMA-LSTM/blob/main/Hybrid.py

## 🙏 Acknowledgements

- [TensorFlow/Keras](https://www.tensorflow.org/)
- [statsmodels](https://www.statsmodels.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Fama-French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
```
