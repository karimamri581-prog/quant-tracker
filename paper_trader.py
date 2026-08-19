import requests
import pandas as pd
import numpy as np
import zipfile
import io
import os
from datetime import datetime, timezone, timedelta

SYMBOL = "BTCUSDT"
LOG_FILE = "paper_trading_log.csv"
INITIAL_CAPITAL = 100000.0
TRADE_COST_PCT = 0.0014 # 0.14% per side

def get_bulk_data(symbol, data_type, start_year):
    all_data = []
    current_date = datetime.now(timezone.utc)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"]
    
    for y in range(start_year, current_date.year + 1):
        for m in range(1, 13):
            if y == current_date.year and m > current_date.month: continue
            mm = f"{m:02d}"
            
            if data_type == "funding":
                url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{mm}.zip"
            else:
                url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{mm}.zip"
            
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    df = pd.read_csv(z.open(z.namelist()[0]))
                    
                    if data_type == "funding":
                        time_col = 'calcTime' if 'calcTime' in df.columns else 'calc_time'
                        rate_col = 'fundingRate' if 'fundingRate' in df.columns else 'last_funding_rate'
                        df["timestamp"] = pd.to_datetime(df[time_col], unit="ms", utc=True, errors="coerce")
                        df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
                        all_data.append(df.dropna(subset=["timestamp", "funding_rate"])[["timestamp", "funding_rate"]])
                    else:
                        df.columns = cols
                        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                        df["close"] = pd.to_numeric(df["close"], errors="coerce")
                        all_data.append(df.dropna(subset=["close"])[["timestamp", "close"]])
            except:
                pass

    if not all_data: return pd.DataFrame()
    return pd.concat(all_data).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def run_logic():
    print("Fetching historical funding...")
    funding = get_bulk_data(SYMBOL, "funding", 2020)
    print(f"  Funding rows: {len(funding)}")
    
    print("Fetching historical klines...")
    klines_1h = get_bulk_data(SYMBOL, "klines", 2020)
    if klines_1h.empty: 
        print("Error: Klines empty. Binance Vision might be down.")
        return
        
    # Resample 1h to 8h
    klines_8h = klines_1h.set_index("timestamp").resample("8h").last().dropna().reset_index()
    print(f"  8h Klines rows: {len(klines_8h)}")
    
    # Merge
    df = pd.merge(funding, klines_8h, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if df.empty:
        print("Error: Merged dataframe is empty.")
        return
    
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
