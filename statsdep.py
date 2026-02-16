# =====================================
# 1️⃣ Import Libraries
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import arch  # for volatility modeling


# =====================================
# 2️⃣ Load and Clean Data
# =====================================

df = pd.read_csv("nifty50_historical_data.csv")

# Fix date parsing (since yours is dd-mm-yyyy)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)  # Ensure chronological order

# Ensure Close column is numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Remove infinite and missing values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df = df[['Close']]

print("Missing values after cleaning:\n", df.isna().sum())
print(f"Data range: {df.index.min()} to {df.index.max()}")
print(f"Total observations: {len(df)}")


# =====================================
# 3️⃣ Visualize Original Data
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original series
axes[0, 0].plot(df['Close'])
axes[0, 0].set_title("Closing Price")
axes[0, 0].set_ylabel("Price")

# Log transformation (to stabilize variance)
df['Close_log'] = np.log(df['Close'])
axes[0, 1].plot(df['Close_log'])
axes[0, 1].set_title("Log of Closing Price")
axes[0, 1].set_ylabel("Log Price")

# Returns (percentage change)
df['Returns'] = df['Close'].pct_change() * 100
axes[1, 0].plot(df['Returns'])
axes[1, 0].set_title("Daily Returns (%)")
axes[1, 0].set_ylabel("Returns %")
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)

# Log returns
df['Log_Returns'] = df['Close_log'].diff() * 100
axes[1, 1].plot(df['Log_Returns'])
axes[1, 1].set_title("Log Returns (%)")
axes[1, 1].set_ylabel("Log Returns %")
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================
# 4️⃣ Comprehensive Stationarity Tests
# =====================================

def check_stationarity(series, series_name):
    """Perform multiple stationarity tests"""
    print(f"\n{'='*50}")
    print(f"Stationarity Tests for: {series_name}")
    print('='*50)
    
    # Remove NaN values
    series_clean = series.dropna()
    
    # ADF Test
    adf_result = adfuller(series_clean, autolag='AIC')
    print(f"\nADF Test:")
    print(f"  Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Critical values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.4f}")
    
    # KPSS Test (complementary to ADF)
    try:
        kpss_result = kpss(series_clean, regression='c', nlags='auto')
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        print(f"  Critical values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.4f}")
    except:
        print("\nKPSS Test: Could not compute")
    
    # Conclusion
    if adf_result[1] < 0.05:
        print(f"\n✅ {series_name} is STATIONARY (reject H0 of ADF test)")
    else:
        print(f"\n❌ {series_name} is NOT STATIONARY (fail to reject H0 of ADF test)")

# Test different transformations
check_stationarity(df['Close'], "Original Close Price")
check_stationarity(df['Close_log'], "Log Close Price")
check_stationarity(df['Returns'].dropna(), "Returns (%)")
check_stationarity(df['Log_Returns'].dropna(), "Log Returns (%)")


# =====================================
# 5️⃣ Choose Best Transformation
# =====================================

# Based on tests, log returns are usually stationary
# Let's use log returns for modeling
df['Transformed'] = df['Log_Returns']

# Remove initial NaN from differencing
df_model = df.dropna().copy()


# =====================================
# 6️⃣ ACF and PACF for Transformed Series
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original transformed series
axes[0, 0].plot(df_model['Transformed'])
axes[0, 0].set_title("Log Returns")
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)

# Histogram
axes[0, 1].hist(df_model['Transformed'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title("Distribution of Log Returns")
axes[0, 1].set_xlabel("Log Returns %")

# ACF
plot_acf(df_model['Transformed'], ax=axes[1, 0], lags=40)
axes[1, 0].set_title("ACF - Log Returns")

# PACF
plot_pacf(df_model['Transformed'], ax=axes[1, 1], lags=40)
axes[1, 1].set_title("PACF - Log Returns")

plt.tight_layout()
plt.show()


# =====================================
# 7️⃣ Check for Seasonality
# =====================================

# Try to detect weekly pattern
df_model['DayOfWeek'] = df_model.index.dayofweek
weekly_avg = df_model.groupby('DayOfWeek')['Transformed'].mean()

plt.figure(figsize=(10, 5))
weekly_avg.plot(kind='bar')
plt.title('Average Log Returns by Day of Week')
plt.xlabel('Day (0=Monday, 4=Friday)')
plt.ylabel('Average Log Returns %')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.show()


# =====================================
# 8️⃣ Train-Test Split
# =====================================

train_size = int(len(df_model) * 0.8)
train = df_model['Transformed'][:train_size]
test = df_model['Transformed'][train_size:]

print(f"\nTrain size: {len(train)} observations")
print(f"Test size: {len(test)} observations")


# =====================================
# 9️⃣ Automated ARIMA Order Selection
# =====================================

def find_best_arima(train_series, p_range=range(0, 3), d_range=[0], q_range=range(0, 3)):
    """Simple grid search for ARIMA orders"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    print("\nSearching for best ARIMA order...")
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                    
                    print(f"ARIMA{p,d,q} - AIC: {model_fit.aic:.2f}")
                except:
                    continue
    
    print(f"\n✅ Best model: ARIMA{best_order} with AIC: {best_aic:.2f}")
    return best_model, best_order


# Find best model (for log returns, d should be 0 since it's already stationary)
best_model, best_order = find_best_arima(train, d_range=[0])


# =====================================
# 🔟 Fit Final ARIMA Model
# =====================================

print(f"\nFitting final model: ARIMA{best_order}")
final_model = ARIMA(train, order=best_order)
final_model_fit = final_model.fit()

print("\nModel Summary:")
print(final_model_fit.summary())


# =====================================
# 1️⃣1️⃣ Diagnostic Checks
# =====================================

# Residual analysis
residuals = final_model_fit.resid

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals over time
axes[0, 0].plot(residuals)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[0, 0].set_title("Residuals")

# Histogram of residuals
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title("Distribution of Residuals")
axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.3)

# ACF of residuals
plot_acf(residuals, ax=axes[1, 0], lags=40)
axes[1, 0].set_title("ACF of Residuals")

# QQ plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("QQ Plot")

plt.tight_layout()
plt.show()


# =====================================
# 1️⃣2️⃣ Forecast and Transform Back
# =====================================

# Forecast on transformed scale
forecast_transformed = final_model_fit.forecast(steps=len(test))
forecast_index = test.index

# Transform back to original price scale
# We need the last actual price to reconstruct
last_price = df['Close'].iloc[train_size]

# Reconstruct prices from log returns
forecast_prices = []
current_price = last_price

for ret in forecast_transformed:
    # Convert log return back to price
    # log_return = ln(P_t / P_{t-1}) * 100
    # So P_t = P_{t-1} * exp(log_return/100)
    current_price = current_price * np.exp(ret/100)
    forecast_prices.append(current_price)

# Actual test prices
actual_prices = df['Close'].iloc[train_size:train_size+len(test)]


# =====================================
# 1️⃣3️⃣ Evaluation
# =====================================

mae = mean_absolute_error(actual_prices, forecast_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices))
mape = np.mean(np.abs((actual_prices - forecast_prices) / actual_prices)) * 100

print("\n" + "="*50)
print("Model Performance Metrics")
print("="*50)
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")


# =====================================
# 1️⃣4️⃣ Plot Results
# =====================================

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: Transformed scale (log returns)
axes[0].plot(train.index, train, label='Train (Log Returns)')
axes[0].plot(test.index, test, label='Actual (Log Returns)')
axes[0].plot(forecast_index, forecast_transformed, label='Forecast (Log Returns)', color='red')
axes[0].set_title('Log Returns: Actual vs Forecast')
axes[0].set_ylabel('Log Returns %')
axes[0].legend()
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Plot 2: Original price scale
axes[1].plot(df.index[:train_size], df['Close'][:train_size], label='Train', alpha=0.7)
axes[1].plot(actual_prices.index, actual_prices, label='Actual Test', alpha=0.7)
axes[1].plot(forecast_index, forecast_prices, label='Forecast', color='red', linewidth=2)
axes[1].fill_between(forecast_index, 
                     np.array(forecast_prices) * 0.95, 
                     np.array(forecast_prices) * 1.05, 
                     color='red', alpha=0.2, label='95% Confidence Band')
axes[1].set_title('NIFTY50: Actual vs Forecast (Original Scale)')
axes[1].set_ylabel('Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================
# 1️⃣5️⃣ Try SARIMA if seasonality detected
# =====================================

# If you see weekly patterns, try SARIMA
if abs(weekly_avg.max()) > 0.1:  # Threshold for seasonality
    print("\n" + "="*50)
    print("Trying SARIMA model with weekly seasonality")
    print("="*50)
    
    try:
        sarima_model = SARIMAX(train, 
                               order=best_order, 
                               seasonal_order=(1, 0, 1, 5),  # Weekly seasonality (5 trading days)
                               enforce_stationarity=False,
                               enforce_invertibility=False)
        sarima_fit = sarima_model.fit(disp=False)
        print(sarima_fit.summary())
        
        # Compare with ARIMA
        print(f"\nARIMA AIC: {final_model_fit.aic:.2f}")
        print(f"SARIMA AIC: {sarima_fit.aic:.2f}")
        
        if sarima_fit.aic < final_model_fit.aic:
            print("✅ SARIMA performs better based on AIC")
        else:
            print("❌ ARIMA still performs better")
    except:
        print("SARIMA modeling failed - possibly too complex for this data")