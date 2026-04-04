import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('Electric_Production.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(df.head())
print("\n")

# Function to create sequences for time series
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to calculate directional accuracy
def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy: percentage of times the direction 
    (increase or decrease) is correctly predicted
    """
    y_true_diff = np.diff(y_true.flatten())
    y_pred_diff = np.diff(y_pred.flatten())
    
    correct_directions = np.sum((y_true_diff * y_pred_diff) > 0)
    total_directions = len(y_true_diff)
    
    return (correct_directions / total_directions) * 100

# Function to evaluate model
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    da = directional_accuracy(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Directional Accuracy: {da:.2f}%")
    
    return {'Model': model_name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'DA': da}

# Prepare the data
data = df['IPG2211A2N'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences
sequence_length = 12  # Using 12 months of history to predict next month
X, y = create_sequences(data_scaled, sequence_length)

# Split into train and test sets (80-20 split)
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Sequence length: {sequence_length}")
print(f"Features: {X_train.shape[2]}")

# Reshape for different model types
# CNN expects (samples, timesteps, features)
# RNN/LSTM expect (samples, timesteps, features)
# Already in correct shape

# ==================== CNN MODEL ====================
print("\n" + "="*50)
print("Training CNN Model")
print("="*50)

cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
cnn_model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train CNN
cnn_history = cnn_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions with CNN
cnn_train_pred = cnn_model.predict(X_train)
cnn_test_pred = cnn_model.predict(X_test)

# Inverse transform predictions
cnn_train_pred_inv = scaler.inverse_transform(cnn_train_pred)
cnn_test_pred_inv = scaler.inverse_transform(cnn_test_pred)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate CNN
cnn_results = evaluate_model(y_test_inv, cnn_test_pred_inv, "CNN")

# ==================== RNN MODEL ====================
print("\n" + "="*50)
print("Training RNN Model")
print("="*50)

rnn_model = Sequential([
    SimpleRNN(50, activation='tanh', return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    SimpleRNN(50, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
rnn_model.summary()

# Train RNN
rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions with RNN
rnn_train_pred = rnn_model.predict(X_train)
rnn_test_pred = rnn_model.predict(X_test)

# Inverse transform predictions
rnn_test_pred_inv = scaler.inverse_transform(rnn_test_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate RNN
rnn_results = evaluate_model(y_test_inv, rnn_test_pred_inv, "RNN")

# ==================== LSTM MODEL ====================
print("\n" + "="*50)
print("Training LSTM Model")
print("="*50)

lstm_model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
lstm_model.summary()

# Train LSTM
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions with LSTM
lstm_train_pred = lstm_model.predict(X_train)
lstm_test_pred = lstm_model.predict(X_test)

# Inverse transform predictions
lstm_test_pred_inv = scaler.inverse_transform(lstm_test_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate LSTM
lstm_results = evaluate_model(y_test_inv, lstm_test_pred_inv, "LSTM")

# ==================== COMPARE ALL MODELS ====================
print("\n" + "="*60)
print("FINAL COMPARISON - DIRECTIONAL ACCURACY")
print("="*60)

results_df = pd.DataFrame([cnn_results, rnn_results, lstm_results])
print(results_df.to_string(index=False))

# ==================== VISUALIZATION ====================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training Loss Curves
axes[0, 0].plot(cnn_history.history['loss'], label='CNN Train', alpha=0.7)
axes[0, 0].plot(cnn_history.history['val_loss'], label='CNN Val', alpha=0.7)
axes[0, 0].plot(rnn_history.history['loss'], label='RNN Train', alpha=0.7)
axes[0, 0].plot(rnn_history.history['val_loss'], label='RNN Val', alpha=0.7)
axes[0, 0].plot(lstm_history.history['loss'], label='LSTM Train', alpha=0.7)
axes[0, 0].plot(lstm_history.history['val_loss'], label='LSTM Val', alpha=0.7)
axes[0, 0].set_title('Training Loss Curves')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')

# Plot 2: Actual vs Predicted (Test Set)
test_dates = df.index[-len(y_test_inv):]
axes[0, 1].plot(test_dates, y_test_inv, label='Actual', linewidth=2, color='black')
axes[0, 1].plot(test_dates, cnn_test_pred_inv, label='CNN Prediction', alpha=0.7)
axes[0, 1].plot(test_dates, rnn_test_pred_inv, label='RNN Prediction', alpha=0.7)
axes[0, 1].plot(test_dates, lstm_test_pred_inv, label='LSTM Prediction', alpha=0.7)
axes[0, 1].set_title('Actual vs Predicted (Test Set)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Electric Production')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Directional Accuracy Bar Chart
models = results_df['Model'].values
da_scores = results_df['DA'].values
bars = axes[1, 0].bar(models, da_scores, color=['red', 'green', 'blue'])
axes[1, 0].set_title('Directional Accuracy Comparison')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Directional Accuracy (%)')
axes[1, 0].set_ylim([0, 100])
# Add value labels on bars
for bar, da in zip(bars, da_scores):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{da:.2f}%', ha='center', va='bottom')

# Plot 4: Error Metrics Comparison
x = np.arange(len(models))
width = 0.25
axes[1, 1].bar(x - width, results_df['RMSE'], width, label='RMSE', color='orange')
axes[1, 1].bar(x, results_df['MAE'], width, label='MAE', color='purple')
axes[1, 1].set_title('Error Metrics Comparison')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Error Value')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== ADDITIONAL ANALYSIS ====================
print("\n" + "="*60)
print("ADDITIONAL ANALYSIS")
print("="*60)

# Detailed directional accuracy analysis
def detailed_directional_analysis(y_true, y_pred, model_name):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    actual_direction = np.diff(y_true_flat)
    pred_direction = np.diff(y_pred_flat)
    
    correct_up = np.sum((actual_direction > 0) & (pred_direction > 0))
    correct_down = np.sum((actual_direction < 0) & (pred_direction < 0))
    total_up = np.sum(actual_direction > 0)
    total_down = np.sum(actual_direction < 0)
    
    print(f"\n{model_name} Directional Breakdown:")
    print(f"  Upward movements correctly predicted: {correct_up}/{total_up} ({correct_up/total_up*100:.2f}%)")
    print(f"  Downward movements correctly predicted: {correct_down}/{total_down} ({correct_down/total_down*100:.2f}%)")

detailed_directional_analysis(y_test_inv, cnn_test_pred_inv, "CNN")
detailed_directional_analysis(y_test_inv, rnn_test_pred_inv, "RNN")
detailed_directional_analysis(y_test_inv, lstm_test_pred_inv, "LSTM")

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")
print("Comparison plot saved as 'model_comparison.png'")