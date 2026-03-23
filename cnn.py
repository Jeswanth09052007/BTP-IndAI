# =====================================
# 1️⃣ IMPORT LIBRARIES
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =====================================
# 2️⃣ LOAD DATA
# =====================================

df = pd.read_csv("nifty50_historical_data.csv")

# Fix column names if needed
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sort by date (IMPORTANT)
df = df.sort_values('Date')

# Drop missing values
df = df.dropna()


# =====================================
# 3️⃣ SELECT FEATURES
# =====================================

# Reorder to standard OHLCV
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]


# =====================================
# 4️⃣ NORMALIZATION
# =====================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)


# =====================================
# 5️⃣ CREATE SEQUENCES
# =====================================

def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][3])  # Close price
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(scaled_data, window_size)


# =====================================
# 6️⃣ TRAIN-TEST SPLIT
# =====================================

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# =====================================
# 7️⃣ BUILD CNN MODEL
# =====================================

model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(window_size, X.shape[2])),
    Dropout(0.2),
    
    Conv1D(32, kernel_size=2, activation='relu'),
    Flatten(),
    
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


# =====================================
# 8️⃣ TRAIN MODEL
# =====================================

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


# =====================================
# 9️⃣ PREDICTIONS
# =====================================

y_pred = model.predict(X_test)


# =====================================
# 🔟 INVERSE SCALING
# =====================================

dummy = np.zeros((len(y_pred), scaled_data.shape[1]))

# Predicted
dummy[:, 3] = y_pred.flatten()
y_pred_actual = scaler.inverse_transform(dummy)[:, 3]

# Actual
dummy[:, 3] = y_test.flatten()
y_test_actual = scaler.inverse_transform(dummy)[:, 3]


# =====================================
# 1️⃣1️⃣ METRICS
# =====================================

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print("RMSE:", rmse)
print("MAE:", mae)

# Direction Accuracy
direction_acc = np.mean(
    (y_pred_actual[1:] > y_pred_actual[:-1]) ==
    (y_test_actual[1:] > y_test_actual[:-1])
)

print("Directional Accuracy:", direction_acc)


# =====================================
# 📊 1. LOSS GRAPH
# =====================================

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()


# =====================================
# 📈 2. ACTUAL vs PREDICTED (MAIN GRAPH)
# =====================================

plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred_actual, label='Predicted Price')
plt.title("Actual vs Predicted Prices (CNN)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# =====================================
# 📉 3. ERROR DISTRIBUTION
# =====================================

errors = y_test_actual - y_pred_actual

plt.figure()
plt.hist(errors, bins=50)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()


# =====================================
# 📊 4. SCATTER PLOT
# =====================================

plt.figure()
plt.scatter(y_test_actual, y_pred_actual)
plt.title("Actual vs Predicted Scatter")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()