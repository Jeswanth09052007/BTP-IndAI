import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Use 'Close' price for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler, df

# Create sequences for time series prediction
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Calculate directional accuracy
def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy: percentage of times the predicted direction
    (up/down) matches the actual direction
    """
    actual_direction = np.diff(y_true.flatten())
    pred_direction = np.diff(y_pred.flatten())
    
    # Compare directions
    correct_directions = np.sum((actual_direction * pred_direction) > 0)
    total_directions = len(actual_direction)
    
    return correct_directions / total_directions if total_directions > 0 else 0

# Build RNN model
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and evaluate model
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    
    # Calculate directional accuracy
    train_dir_acc = directional_accuracy(y_train, train_predictions)
    test_dir_acc = directional_accuracy(y_test, test_predictions)
    
    # Calculate MSE and MAE
    train_mse = np.mean((y_train - train_predictions.flatten()) ** 2)
    test_mse = np.mean((y_test - test_predictions.flatten()) ** 2)
    train_mae = np.mean(np.abs(y_train - train_predictions.flatten()))
    test_mae = np.mean(np.abs(y_test - test_predictions.flatten()))
    
    return {
        'model_name': model_name,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'history': history,
        'train_pred': train_predictions,
        'test_pred': test_predictions
    }

# Main execution
def main():
    print("Loading and preparing data...")
    # Load data
    scaled_data, scaler, df = load_and_prepare_data('Sensex Dataset.csv')
    
    # Create sequences
    seq_length = 60  # Use 60 days to predict next day
    X, y = create_sequences(scaled_data, seq_length)
    
    # Reshape X for models (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into train and test sets (80-20 split, maintaining temporal order)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sequence length: {seq_length}")
    
    # Build models
    input_shape = (X_train.shape[1], 1)
    rnn_model = build_rnn_model(input_shape)
    cnn_model = build_cnn_model(input_shape)
    lstm_model = build_lstm_model(input_shape)
    
    # Train and evaluate models
    results = []
    
    # RNN
    rnn_results = train_and_evaluate(rnn_model, "RNN", X_train, X_test, y_train, y_test)
    results.append(rnn_results)
    
    # CNN
    cnn_results = train_and_evaluate(cnn_model, "CNN", X_train, X_test, y_train, y_test)
    results.append(cnn_results)
    
    # LSTM
    lstm_results = train_and_evaluate(lstm_model, "LSTM", X_train, X_test, y_train, y_test)
    results.append(lstm_results)
    
    # Print comparison results
    print("\n" + "="*80)
    print("COMPARISON RESULTS - DIRECTIONAL ACCURACY")
    print("="*80)
    print(f"{'Model':<10} {'Train Dir Acc':<15} {'Test Dir Acc':<15} {'Train MSE':<15} {'Test MSE':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_name']:<10} {result['train_dir_acc']*100:<15.2f}% "
              f"{result['test_dir_acc']*100:<15.2f}% "
              f"{result['train_mse']:<15.6f} {result['test_mse']:<15.6f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Find best model based on test directional accuracy
    best_model = max(results, key=lambda x: x['test_dir_acc'])
    print(f"\n🏆 Best Model based on Test Directional Accuracy: {best_model['model_name']}")
    print(f"   Test Directional Accuracy: {best_model['test_dir_acc']*100:.2f}%")
    
    # Calculate improvement over random guess (50%)
    for result in results:
        improvement = (result['test_dir_acc'] - 0.5) * 100
        print(f"\n{result['model_name']}:")
        print(f"  - Directional Accuracy: {result['test_dir_acc']*100:.2f}%")
        print(f"  - Improvement over random guess: {improvement:+.2f}%")
        print(f"  - Test MSE: {result['test_mse']:.6f}")
        print(f"  - Test MAE: {result['test_mae']:.6f}")
    
    # Visualization
    plot_comparison_results(results, df, scaler, seq_length)
    
    return results

def plot_comparison_results(results, df, scaler, seq_length):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1-3: Training loss curves
    for idx, result in enumerate(results):
        ax = axes[0, idx]
        ax.plot(result['history'].history['loss'], label='Training Loss', linewidth=2)
        ax.plot(result['history'].history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_title(f"{result['model_name']} - Training History", fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Predictions vs Actual (test set only)
    test_size = len(df) - seq_length - int(len(df) * 0.2)
    
    for idx, result in enumerate(results):
        ax = axes[1, idx]
        
        # Inverse transform predictions and actual values
        # For test predictions
        test_pred_inv = scaler.inverse_transform(result['test_pred'])
        
        # Get corresponding actual values (the last test_size values from the original data)
        actual_values = df['Close'].values[-test_size:]
        
        # Align lengths (predictions might be slightly different length)
        min_len = min(len(test_pred_inv.flatten()), len(actual_values))
        
        # Plot
        ax.plot(actual_values[:min_len], label='Actual', color='blue', alpha=0.7, linewidth=1.5)
        ax.plot(test_pred_inv.flatten()[:min_len], label='Predicted', color='red', alpha=0.7, linewidth=1.5)
        ax.set_title(f"{result['model_name']} - Test Set Predictions\n(Dir Acc: {result['test_dir_acc']*100:.1f}%)", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Sensex Close Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create bar chart for directional accuracy comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['model_name'] for r in results]
    train_acc = [r['train_dir_acc'] * 100 for r in results]
    test_acc = [r['test_dir_acc'] * 100 for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_acc, width, label='Training', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_acc, width, label='Testing', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Directional Accuracy Comparison: RNN vs CNN vs LSTM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)', alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('directional_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = main()
    
    # Print final recommendation
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    best_test_acc = max(results, key=lambda x: x['test_dir_acc'])
    print(f"\nBased on the analysis of the Sensex dataset:")
    print(f"• RNN Test Directional Accuracy: {results[0]['test_dir_acc']*100:.2f}%")
    print(f"• CNN Test Directional Accuracy: {results[1]['test_dir_acc']*100:.2f}%")
    print(f"• LSTM Test Directional Accuracy: {results[2]['test_dir_acc']*100:.2f}%")
    print(f"\n✓ Best performing model: {best_test_acc['model_name']} with {best_test_acc['test_dir_acc']*100:.2f}% directional accuracy")
    
    if best_test_acc['test_dir_acc'] > 0.5:
        print(f"✓ The {best_test_acc['model_name']} model shows significant predictive capability for market direction")
    else:
        print(f"⚠ None of the models significantly outperform random guessing (50%)")