# =====================================
# 1️⃣ IMPORT LIBRARIES
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =====================================
# 2️⃣ LOAD AND CLEAN DATA
# =====================================

df = pd.read_csv("nifty50_historical_data.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df = df[['Close']]

print("Data range:", df.index[0], "to", df.index[-1])


# =====================================
# 3️⃣ VISUALIZE ORIGINAL DATA
# =====================================

plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Closing Price")
plt.show()


# =====================================
# 4️⃣ NORMALIZATION
# =====================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# =====================================
# 5️⃣ CREATE SEQUENCES
# =====================================

def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(scaled_data, window_size)


# =====================================
# 6️⃣ TRAIN-TEST SPLIT (TIME BASED)
# =====================================

train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# =====================================
# 7️⃣ BUILD RNN MODEL
# =====================================

model = Sequential([
    SimpleRNN(64, activation='tanh', return_sequences=False,
              input_shape=(window_size, 1)),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


# =====================================
# 8️⃣ TRAIN MODEL
# =====================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

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
# 🔟 INVERSE TRANSFORMATION
# =====================================

y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)


# =====================================
# 1️⃣1️⃣ EVALUATION
# =====================================

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

print("\n--- RNN Model Performance ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Directional Accuracy
direction_acc = np.mean(
    (y_pred_actual[1:] > y_pred_actual[:-1]) ==
    (y_test_actual[1:] > y_test_actual[:-1])
)

print(f"Directional Accuracy: {direction_acc:.4f}")


# =====================================
# 1️⃣2️⃣ LOSS GRAPH
# =====================================

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss (RNN)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# =====================================
# 1️⃣3️⃣ FINAL PLOT (IMPORTANT 🔥)
# =====================================

plt.figure(figsize=(12,6))

plt.plot(df.index[:train_size+window_size], 
         df['Close'][:train_size+window_size],
         label='Train Price')

plt.plot(df.index[train_size+window_size:], 
         y_test_actual,
         label='Actual Price')

plt.plot(df.index[train_size+window_size:], 
         y_pred_actual,
         label='Predicted Price',
         color='red')

plt.title("RNN: NIFTY50 Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()


# =====================================
# 1️⃣4️⃣ ERROR DISTRIBUTION
# =====================================

errors = y_test_actual.flatten() - y_pred_actual.flatten()

plt.figure()
plt.hist(errors, bins=50)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()


# =====================================
# 1️⃣5️⃣ SCATTER PLOT
# =====================================

plt.figure()
plt.scatter(y_test_actual, y_pred_actual)
plt.title("Actual vs Predicted Scatter")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()