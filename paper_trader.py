import requests
import pandas as pd
import numpy as np
import zipfile
import io
import os
from datetime import datetime, timezone

INITIAL_CAPITAL = 100000.0
TRADE_COST_PCT = 0.0014
SLIPPAGE_ASSUMPTION = 0.0005
SYMBOL = "BTCUSDT"

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
    print(f"Fetching data for {SYMBOL}...")
    funding = get_bulk_data(SYMBOL, "funding", 2020)
    klines_1h = get_bulk_data(SYMBOL, "klines", 2020)
    if klines_1h.empty: return
        
    klines_8h = klines_1h.set_index("timestamp").resample("8h").last().dropna().reset_index()
    df = pd.merge(funding, klines_8h, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Strategy Logic
    df["sma"] = df["close"].rolling(150).mean()
    df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
    df["signal"] = df["signal"].shift(1)
    
    # Identify Trade Entries/Exits
    df["trade"] = df["signal"].diff().abs()
    trade_points = df[df["trade"] != 0].copy()
    
    trades = []
    current_equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    trade_id = 1
    
    # Simulate trades
    for i in range(0, len(trade_points) - 1, 2):
        entry = trade_points.iloc[i]
        exit = trade_points.iloc[i+1] if (i+1) < len(trade_points) else df.iloc[-1]
        
        entry_time = entry["timestamp"]
        exit_time = exit["timestamp"]
        entry_price = entry["close"]
        exit_price = exit["close"]
        side = "Long" if entry["signal"] == 1 else "Short"
        position_size = current_equity
        leverage = 1.0
        
        gross_pnl = (exit_price - entry_price) / entry_price * position_size if side == "Long" else (entry_price - exit_price) / entry_price * position_size
        
        hold_df = df[(df["timestamp"] > entry_time) & (df["timestamp"] <= exit_time)]
        funding_pnl = hold_df["funding_rate"].sum() * position_size
        
        fees = (entry_price * position_size * TRADE_COST_PCT) + (exit_price * position_size * TRADE_COST_PCT)
        slippage = (entry_price * position_size * SLIPPAGE_ASSUMPTION) + (exit_price * position_size * SLIPPAGE_ASSUMPTION)
        
        net_pnl = gross_pnl + funding_pnl - fees - slippage
        current_equity += net_pnl
        peak_equity = max(peak_equity, current_equity)
        drawdown = ((peak_equity - current_equity) / peak_equity) * 100
        
        trades.append({
            "Trade ID": trade_id,
            "Symbol": SYMBOL,
            "Entry timestamp": str(entry_time),
            "Exit timestamp": str(exit_time),
            "Side": side,
            "Entry price": float(entry_price),
            "Exit price": float(exit_price),
            "Position size": float(position_size),
            "Leverage": leverage,
            "Gross P&L": float(gross_pnl),
            "Funding P&L": float(funding_pnl),
            "Fees": float(fees),
            "Slippage": float(slippage),
            "Net P&L": float(net_pnl),
            "Equity after trade": float(current_equity),
            "Drawdown": float(drawdown),
            "Signal": "HOLD_CARRY" if side == "Long" else "FLAT",
            "API/execution status": "SUCCESS"
        })
        trade_id += 1

    log_df = pd.DataFrame(trades)
    log_file = "trade_journal.csv"
    
    if len(log_df) > 50:
        log_df = log_df.tail(50)
        
    write_header = not os.path.exists(log_file)
    log_df.to_csv(log_file, mode="w", header=True, index=False)
    print(f"Trade Journal updated with {len(log_df)} trades.")

if __name__ == "__main__":
    run_logic()
