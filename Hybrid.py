import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Feature engineering
def add_features(df):
    df['MA_5'] = df['Daily Log Return'].rolling(window=5).mean()
    df['MA_10'] = df['Daily Log Return'].rolling(window=10).mean()
    df['MA_20'] = df['Daily Log Return'].rolling(window=20).mean()
    df['Volatility_5'] = df['Daily Log Return'].rolling(window=5).std()
    df['Volatility_10'] = df['Daily Log Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Create sequences for LSTM
def create_sequences(data, target, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(target[i+n_steps])
    return np.array(X), np.array(y)

# Hybrid ARIMA-LSTM Model
class HybridARIMALSTM(BaseEstimator, RegressorMixin):
    def __init__(self, arima_order=(1,0,0), lstm_units=100, n_steps=60, epochs=50, batch_size=32):
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        
    def fit(self, X, y):
        # Store original data for reference
        self.X = X
        self.y = y
        
        # Step 1: Fit ARIMA model
        print("Fitting ARIMA model...")
        self.arima = ARIMA(y, order=self.arima_order)
        self.arima_fit = self.arima.fit()
        arima_pred = self.arima_fit.predict(start=1, end=len(y))
        
        # Step 2: Calculate ARIMA residuals
        residuals = y - arima_pred
        
        # Step 3: Prepare LSTM data
        print("Preparing LSTM data...")
        scaled_X = self.scaler.fit_transform(X)
        scaled_residuals = self.scaler_target.fit_transform(residuals.values.reshape(-1, 1))
        
        # Create sequences for LSTM
        X_seq, y_seq = create_sequences(scaled_X, scaled_residuals, self.n_steps)
        
        # Step 4: Build and train LSTM model
        print("Building and training LSTM model...")
        self.lstm_model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True, 
                            input_shape=(X_seq.shape[1], X_seq.shape[2]))),
            Dropout(0.2),
            LSTM(self.lstm_units),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), 
                              loss='mse', 
                              metrics=['mae'])
        
        history = self.lstm_model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        # Store training history for plotting
        self.training_history = history.history
        
        return self
    
    def predict(self, X):
        # Need to ensure we have enough historical data for the LSTM
        if len(X) < self.n_steps:
            raise ValueError(f"Need at least {self.n_steps} samples for prediction")
        
        # Step 1: Get ARIMA predictions
        arima_pred = self.arima_fit.predict(start=1, end=len(X)+self.n_steps)[-len(X):]
        
        # Step 2: Prepare LSTM input
        scaled_X = self.scaler.transform(X)
        
        # Create sequences for LSTM prediction
        X_seq = []
        for i in range(len(scaled_X) - self.n_steps + 1):
            X_seq.append(scaled_X[i:i+self.n_steps])
        X_seq = np.array(X_seq)
        
        # Step 3: Get LSTM residuals predictions
        lstm_pred = self.lstm_model.predict(X_seq)
        lstm_pred = self.scaler_target.inverse_transform(lstm_pred).flatten()
        
        # Pad lstm_pred if necessary to match arima_pred length
        if len(lstm_pred) < len(arima_pred):
            lstm_pred = np.concatenate([np.zeros(len(arima_pred)-len(lstm_pred)), lstm_pred])
        
        # Step 4: Combine predictions
        hybrid_pred = arima_pred + lstm_pred
        
        return hybrid_pred
    
    def evaluate(self, X, y_true):
        preds = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        return rmse, mae, r2
    
    def plot_predictions(self, X, y_true, title="Predictions vs Actual"):
        preds = self.predict(X)
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true, label='Actual', color='blue')
        plt.plot(y_true.index, preds, label='Predicted', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Daily Log Return')
        plt.legend()
        plt.grid()
        plt.show()

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data('sample12.csv')
    df = add_features(df)
    
    # Select features and target
    features = ['(MKT-Rf) Excess Return on the Market ', '(SMB) Small-Minus-Big Return',
                '(HML) High-Minus-Low Return', '(RMW) Robust Minus Weak Return',
                '(CMA) Conservative Minus Aggressive Return', '(MOM) Momentum',
                'Amihud LIQ', 'MA_5', 'MA_10', 'MA_20', 'Volatility_5', 'Volatility_10']
    target = 'Daily Log Return'
    
    # Split data
    train_size = int(0.8 * len(df))
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # Create and fit hybrid model
    print("Initializing hybrid model...")
    hybrid_model = HybridARIMALSTM(
        arima_order=(1,0,1),  # ARIMA(p,d,q) parameters
        lstm_units=100,
        n_steps=60,
        epochs=50,
        batch_size=32
    )
    
    print("Training hybrid model...")
    hybrid_model.fit(train[features], train[target])
    
    # Evaluate on training set
    print("\nTraining Set Performance:")
    hybrid_model.evaluate(train[features], train[target])
    hybrid_model.plot_predictions(train[features], train[target], "Training Set Predictions")
    
    # Evaluate on test set
    print("\nTest Set Performance:")
    hybrid_model.evaluate(test[features], test[target])
    hybrid_model.plot_predictions(test[features], test[target], "Test Set Predictions")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(hybrid_model.training_history['loss'], label='Train Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()