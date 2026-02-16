import yfinance as yf
import pandas as pd

# Download Nifty 50 data
nifty = yf.download('^NSEI', start='2007-09-17')

# Reset index to make Date a column (not index)
nifty.reset_index(inplace=True)

# Save to CSV with proper format
nifty.to_csv('nifty50_historical_fin.csv', index=False, encoding='utf-8-sig')

print(f"Downloaded {len(nifty)} records")
print(f"Date range: {nifty['Date'].iloc[0].date()} to {nifty['Date'].iloc[-1].date()}")
print("CSV saved with columns:", nifty.columns.tolist())