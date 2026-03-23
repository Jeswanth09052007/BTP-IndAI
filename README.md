# Hybrid Statistical and Deep Learning Framework for Stock Price Forecasting

A robust hybrid stock forecasting model combining ARIMA (linear modeling) with hierarchical deep learning architectures (CNN, RNN, LSTM) to improve prediction accuracy and reduce forecasting errors.

## 📌 Project Overview

Stock price prediction is crucial for informed investment and financial decision-making. Accurate forecasts significantly impact portfolio management, risk assessment, and trading strategies. However, the real-world stock market presents substantial challenges due to its inherent complexity and volatility.

This project develops a **hierarchical hybrid framework** that integrates:
- **ARIMA** for capturing linear patterns and trends
- **CNN** for extracting local temporal patterns (short-term dependencies)
- **RNN** for intermediate sequential dependencies
- **LSTM** for long-term dependencies

## 👥 Team Members

| Name | Roll Number |
|------|------------|
| Jeswanth | UG2 CSES20240010202 |
| Purvaj | UG2 CSES20240010178 |
| Abhinav | UG2 ECES20240020337 |

**Guided by:** Dr. Jahnabi Chakravarty

## 🎯 Problem Statement

Stock market forecasting remains challenging due to:
- High volatility and non-linear patterns
- Influence from market sentiment, geopolitical events, economic indicators
- Complex temporal dependencies at multiple scales

Existing approaches either:
- Compare ARIMA and deep learning models independently
- Use shallow hybrid structures (direct ARIMA-LSTM integration)
- Lack hierarchical temporal learning distribution

## 🏗️ Project Objectives

### 1. Hierarchical Hybrid Framework
Develop a structured hybrid model integrating ARIMA with CNN, RNN, and LSTM to capture different temporal behaviors in stock price data.

### 2. Multi-Level Temporal Learning
Distribute learning across short-term (CNN), intermediate (RNN), and long-term (LSTM) dependencies to reduce over-reliance on a single deep learning model.

### 3. Improved Stability & Accuracy
Enhance prediction stability and reduce RMSE, MAE, and MAPE by combining linear statistical modeling with hierarchical deep learning.

## 📊 Dataset & Preprocessing

### Dataset
- **NIFTY50 historical data**
- Primary feature: Closing price
- Time-based data for sequential analysis
- Train-test split: 80% training, 20% testing

### Preprocessing Steps
- Data sorted chronologically to preserve temporal dependency
- Date parsing and indexing for time-series alignment
- Handling missing values (e.g., interpolation)
- Log transformation to stabilize variance
- Log returns calculation for stationarity
- Differencing for stationarity handling

## 🧠 Model Architecture

### 1. ARIMA Model
- Models linear trends in NIFTY50 log returns
- Parameter selection (p, d, q) using AIC and ACF/PACF
- Trained on 80% data, tested on 20% data

### 2. Hierarchical Deep Learning Components
- **CNN**: Captures short-term local patterns
- **RNN**: Handles intermediate sequential dependencies
- **LSTM**: Models long-term temporal relationships


### Training Strategy
1. Train ARIMA model on original data
2. Compute residuals (actual - ARIMA predictions)
3. Train hierarchical DL models on residual sequences
4. Final prediction = ARIMA forecast + DL residual predictions

## 🔬 Literature Review

### Statistical Models
**ARIMA (Autoregressive Integrated Moving Average)**
- Widely used for time-series forecasting
- Requires data stationarity (achieved through differencing)
- Parameter selection via AIC (Akaike Information Criterion)
- **Strength:** Excellent at modeling linear patterns and trends
- **Limitation:** Struggles with complex nonlinear relationships

### Deep Learning Models
| Model | Description |
|-------|-------------|
| **RNN** | Designed for sequential data, retains information from previous steps |
| **LSTM** | Specialized RNN with gate mechanisms, handles long-term dependencies effectively |
| **CNN** | Extracts hierarchical features, captures local patterns (WaveNet style) |

**Key Finding:** Previous studies show LSTM often achieves the lowest Mean Absolute Error (MAE) in stock price prediction.

## 🔍 Research Gap Identified

1. **Independent Modeling Limitation:** Most research either compares ARIMA and deep learning models independently or combines them without structured temporal decomposition.

2. **Shallow Hybrid Structures:** Existing hybrid approaches rely on direct ARIMA-LSTM integration, leading to over-dependence on LSTM for multi-scale pattern learning.

3. **Lack of Hierarchical Temporal Learning:** Limited exploration of architectures that explicitly distribute short-, medium-, and long-term dependency learning across different neural components.

## 📈 Evaluation Metrics

The framework will be benchmarked against individual models (ARIMA, RNN, LSTM) using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

## 📚 References

### Statistical Model Reference
Minhaj, N., Ahmed, R., Khalique, I. A., & Imran, M. (2023). A comparative research of stock price prediction of selected stock indexes and the stock market by using arima model. *Global Economics Science*, 1-19.

### Hybrid Model Reference
Alharbi, M. H. (2025). Prediction of the stock market using LSTM, ARIMA, and hybrid of LSTM-ARIMA models. *Journal of Knowledge Management Application and Practice*, 7(1), 15-22.

### Bibliometric Review
Vuong, P. H., Phu, L. H., Van Nguyen, T. H., Duy, L. N., Bao, P. T., & Trinh, T. D. (2024). A bibliometric literature review of stock price forecasting: From statistical model to deep learning approach. *Science Progress*, 107(1), 00368504241236557.

## 📅 Timeline

| Phase | Activity |
|-------|----------|
| 1 | Literature Review & Data Collection |
| 2 | Data Preprocessing & ARIMA Implementation |
| 3 | Hierarchical Deep Learning Model Development |
| 4 | Integration & Training |
| 5 | Evaluation & Benchmarking |
| 6 | Documentation & Final Presentation |

## 🚀 Key Contributions

- **Hierarchical Temporal Learning:** Distributes multi-scale dependency modeling across specialized neural components
- **Improved Generalization:** Reduces over-reliance on single recurrent layers
- **Enhanced Stability:** Combines linear statistical modeling with deep learning for robust predictions
- **Reduced Prediction Error:** Aims for lower RMSE, MAE, and MAPE compared to existing approaches


## 🛠️ Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow keras


