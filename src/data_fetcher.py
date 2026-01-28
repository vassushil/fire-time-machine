import yfinance as yf
import numpy as np
import json
import os

def fetch_and_analyze_market(ticker="^NSEI", period="20y"):
    print(f"Fetching data for {ticker}...")
    # Fetch historical data
    df = yf.download(ticker, period=period)

    print(df.describe())
    
    # Calculate daily log returns for better statistical properties
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # Annualize the parameters (Approx 252 trading days)
    mean_return = df['Returns'].mean() * 252
    volatility = df['Returns'].std() * np.sqrt(252)

    params = {
        "market_ticker": ticker,
        "annual_mean_return": round(float(mean_return), 4),
        "annual_volatility": round(float(volatility), 4),
        "last_updated": "2026-01-28"
    }

    # Save to JSON for the Engine to use
    os.makedirs('data', exist_ok=True)
    with open('data/market_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    print("Market Analysis Complete. Parameters saved to /data.")
    return params
    