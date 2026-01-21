import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("=== PREPARING DATA FOR RNN ===")
df = pd.read_csv("/Users/mac/Documents/GITHUB REPOSITORY/SectionB-Q2-Adobe_Data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Use Close price for prediction
data_close = df[['Close']].values

# Normalize the data (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Set sequence length (look back period)
SEQ_LENGTH = 60  # Using 60 days to predict next day
X, y = create_sequences(scaled_data, SEQ_LENGTH)

print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}")  # (samples, seq_length, features)
print(f"y shape: {y.shape}")

# Split data (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build LSTM model
print("=== BUILDING LSTM MODEL ===")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error',
              metrics=['mae'])

model.summary()

# Train the model
print("=== TRAINING MODEL ===")
history = model.fit(X_train, y_train, 
                    epochs=5, 
                    batch_size=32, 
                    validation_split=0.1,
                    verbose=1)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_title('Model MAE (Mean Absolute Error)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Make predictions
print("=== MAKING PREDICTIONS ===")
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"Model Performance Metrics:")
print(f"MAE: ${mae:.4f}")
print(f"MSE: ${mse:.4f}")
print(f"RMSE: ${rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions
plt.figure(figsize=(14, 6))
plt.plot(df['Date'].values[-len(y_test_actual):], y_test_actual, 
         label='Actual Price', color='blue', linewidth=2)
plt.plot(df['Date'].values[-len(y_pred):], y_pred, 
         label='Predicted Price', color='red', linewidth=2, alpha=0.7)
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted Prices')
plt.grid(True, alpha=0.3)
plt.show()

# Create future predictions
print("=== MAKING FUTURE PREDICTIONS ===")
def predict_future(model, last_sequence, future_days=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        # Reshape for prediction
        current_input = current_sequence.reshape(1, SEQ_LENGTH, 1)
        
        # Predict next day
        next_pred = model.predict(current_input, verbose=0)
        
        # Append to predictions
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return np.array(future_predictions)

# Get last sequence from data
last_sequence = scaled_data[-SEQ_LENGTH:]

# Predict next 30 days
future_days = 30
future_predictions_scaled = predict_future(model, last_sequence, future_days)
future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))

# Create future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                             periods=future_days, freq='D')

# Plot with future predictions
plt.figure(figsize=(14, 6))

# Plot historical data (last 200 days)
historical_days = 200
plt.plot(df['Date'].values[-historical_days:], 
         df['Close'].values[-historical_days:], 
         label='Historical', color='blue', linewidth=2)

# Plot future predictions
plt.plot(future_dates, future_predictions, 
         label='Future Predictions', color='green', linewidth=2, linestyle='--')

plt.title('Stock Price: Historical and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== FUTURE PREDICTIONS (Next 30 Days) ===")
for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
    print(f"Day {i+1:2d} ({date.date()}): ${price[0]:.4f}")

# Feature importance analysis using permutation
print("=== FEATURE IMPORTANCE ANALYSIS ===")
# Let's check if adding more features helps
# Create multi-feature dataset
df_features = df[['Close', 'High', 'Low', 'Volume']].copy()
df_features['Volume'] = np.log1p(df_features['Volume'])  # Log transform volume

scaler_multi = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_multi.fit_transform(df_features)

print("Multi-feature dataset created. Rerun with [Close, High, Low, Volume] for potentially better results.")

# Save model
model.save('stock_price_lstm.h5')
print("Model saved as 'stock_price_lstm.h5'")




