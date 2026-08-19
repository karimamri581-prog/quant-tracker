import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta

SYMBOL = "BTCUSDT"
LOG_FILE = "paper_trading_log.csv"
INITIAL_CAPITAL = 100000.0
TRADE_COST_PCT = 0.0014 # 0.14% per side

def fetch_api(url, params):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_klines(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current = start_time
    while current < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": int(current.timestamp() * 1000), "endTime": int(end_time.timestamp() * 1000), "limit": 1000}
        data = fetch_api(url, params)
        if not data: break
        all_data.extend(data)
        current = pd.to_datetime(data[-1][0], unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
    df = pd.DataFrame(all_data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]

def get_funding(symbol, start_time, end_time):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    current = start_time
    while current < end_time:
        params = {"symbol": symbol, "startTime": int(current.timestamp() * 1000), "endTime": int(end_time.timestamp() * 1000), "limit": 1000}
        data = fetch_api(url, params)
        if not data: break
        all_data.extend(data)
        current = pd.to_datetime(data[-1]["fundingTime"], unit="ms", utc=True) + pd.Timedelta(milliseconds=1)
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]]

def run_logic():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=365*5) # 5 years of data
    
    print("Fetching historical klines...")
    klines = get_klines(SYMBOL, "8h", start_time, end_time)
    print("Fetching historical funding...")
    funding = get_funding(SYMBOL, start_time, end_time)
    
    # Merge
    df = pd.merge(funding, klines, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Backtest Logic (Candidate C)
    df["sma"] = df["close"].rolling(150).mean()
    df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
    df["signal"] = df["signal"].shift(1)
    
    df["strat_ret"] = df["signal"] * df["funding_rate"]
    df["trade"] = df["signal"].diff().abs()
    df["cost"] = df["trade"] * TRADE_COST_PCT
    df["net_ret"] = df["strat_ret"] - df["cost"]
    
    # Calculate Equity Curve
    df["equity"] = INITIAL_CAPITAL * (1 + df["net_ret"]).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["peak"] - df["equity"]) / df["peak"]
    
    # Get the most recent row
    last_row = df.iloc[-1]
    
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "funding_time": str(last_row["timestamp"]),
        "price": float(last_row["close"]),
        "sma_150_8h": float(last_row["sma"]),
        "funding_rate": float(last_row["funding_rate"]),
        "signal": "HOLD" if last_row["signal"] == 1.0 else "FLAT",
        "equity": float(last_row["equity"]),
        "cum_ret_pct": ((float(last_row["equity"]) / INITIAL_CAPITAL) - 1) * 100,
        "drawdown_pct": float(last_row["drawdown"] * 100)
    }
    
    # Append to CSV
    log_df = pd.DataFrame([log_data])
    write_header = not os.path.exists(LOG_FILE)
    log_df.to_csv(LOG_FILE, mode="a", header=write_header, index=False)
    print(f"Logged: {log_data}")

if __name__ == "__main__":
    run_logic()
