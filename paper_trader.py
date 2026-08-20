import requests
import pandas as pd
import numpy as np
import zipfile
import io
import os
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
LOG_FILE = "trade_journal.csv"
INITIAL_CAPITAL = 100000.0
TRADE_COST_PCT = 0.0014
SLIPPAGE_ASSUMPTION = 0.0005
STRATEGY_VERSION = "Track_A_v2.0_Spot_Fix"
MAX_DRAWDOWN_LIMIT = 5.0

def get_funding_data(symbol, start_year):
    all_data = []
    current_date = datetime.now(timezone.utc)
    for y in range(start_year, current_date.year + 1):
        for m in range(1, 13):
            if y == current_date.year and m > current_date.month: continue
            mm = f"{m:02d}"
            url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{mm}.zip"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    df = pd.read_csv(z.open(z.namelist()[0]))
                    time_col = 'calcTime' if 'calcTime' in df.columns else 'calc_time'
                    rate_col = 'fundingRate' if 'fundingRate' in df.columns else 'last_funding_rate'
                    df["timestamp"] = pd.to_datetime(df[time_col], unit="ms", utc=True, errors="coerce")
                    df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
                    all_data.append(df.dropna(subset=["timestamp", "funding_rate"])[["timestamp", "funding_rate"]])
            except:
                pass
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def get_spot_data():
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance not installed.")
        return pd.DataFrame()
    
    ticker = "BTC-USD" if SYMBOL == "BTCUSDT" else "ETH-USD"
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Downloading spot data ({ticker})...")
    
    for attempt in range(3):
        try:
            df = yf.download(ticker, start="2020-01-01", end=end_date, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
    
    if df.empty:
        print("Error: Spot download failed.")
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(columns={"Close": "close"})
    return df[["timestamp", "close"]]

def run_logic():
    print("Fetching funding data...")
    funding = get_funding_data(SYMBOL, 2020)
    print(f"  Funding rows: {len(funding)}")
    
    spot = get_spot_data()
    if spot.empty:
        print("Error: No spot data.")
        return
    print(f"  Spot rows: {len(spot)}")
    
    spot_8h = spot.set_index("timestamp").resample("8h").last().dropna().reset_index()
    print(f"  Spot 8h rows: {len(spot_8h)}")
    
    df = pd.merge(funding, spot_8h, on="timestamp", how="left")
    df["close"] = df["close"].ffill()
    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
    print(f"  Merged rows: {len(df)}")
    
    if df.empty:
        print("Error: Merged data empty.")
        return
    
    df["sma"] = df["close"].rolling(150).mean()
    df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
    df["signal"] = df["signal"].shift(1)
    
    df["trade"] = df["signal"].diff().abs()
    df["funding_ret"] = df["signal"] * df["funding_rate"]
    df["cost"] = df["trade"] * (TRADE_COST_PCT + SLIPPAGE_ASSUMPTION) * 2
    df["net_ret"] = df["funding_ret"] - df["cost"]
    
    trade_points = df[df["trade"] != 0].copy()
    trades = []
    current_equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    trade_id = 1
    
    for i in range(0, len(trade_points) - 1, 2):
        entry = trade_points.iloc[i]
        exit_tp = trade_points.iloc[i+1] if (i+1) < len(trade_points) else df.iloc[-1]
        
        entry_time = entry["timestamp"]
        exit_time = exit_tp["timestamp"]
        entry_price = entry["close"]
        exit_price = exit_tp["close"]
        side = "Long Spot / Short Perp" if entry["signal"] == 1 else "Flat"
        notional = INITIAL_CAPITAL
        
        gross_pnl = 0.0
        hold_df = df[(df["timestamp"] > entry_time) & (df["timestamp"] <= exit_time)]
        funding_pnl = hold_df["funding_rate"].sum() * notional
        fees = notional * TRADE_COST_PCT * 2
        slippage = notional * SLIPPAGE_ASSUMPTION * 2
        net_pnl = gross_pnl + funding_pnl - fees - slippage
        
        current_equity += net_pnl
        peak_equity = max(peak_equity, current_equity)
        drawdown = ((peak_equity - current_equity) / peak_equity) * 100
        
        status = "SUCCESS"
        if drawdown > MAX_DRAWDOWN_LIMIT:
            status = "HALT_CIRCUIT_BREAKER"
        
        trades.append({
            "Trade ID": trade_id,
            "Symbol": SYMBOL,
            "Strategy Version": STRATEGY_VERSION,
            "Entry timestamp": str(entry_time),
            "Exit timestamp": str(exit_time),
            "Side": side,
            "Entry price": float(entry_price),
            "Exit price": float(exit_price),
            "Notional": float(notional),
            "Gross P&L": float(gross_pnl),
            "Funding P&L": float(funding_pnl),
            "Fees": float(fees),
            "Slippage": float(slippage),
            "Net P&L": float(net_pnl),
            "Equity after trade": float(current_equity),
            "Drawdown %": float(drawdown),
            "Signal": "HOLD_CARRY" if side == "Long Spot / Short Perp" else "FLAT",
            "Status": status
        })
        trade_id += 1
    
    cols = ["Trade ID", "Symbol", "Strategy Version", "Entry timestamp", "Exit timestamp",
            "Side", "Entry price", "Exit price", "Notional", "Gross P&L", "Funding P&L",
            "Fees", "Slippage", "Net P&L", "Equity after trade", "Drawdown %", "Signal", "Status"]
    
    if not trades:
        print("No trades generated.")
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)
        return
    
    log_df = pd.DataFrame(trades)
    if len(log_df) > 50:
        log_df = log_df.tail(50)
    log_df.to_csv(LOG_FILE, mode="w", header=True, index=False)
    print(f"Trade Journal: {len(log_df)} trades. Version: {STRATEGY_VERSION}")

if __name__ == "__main__":
    print(f"Starting {STRATEGY_VERSION}...")
    run_logic()
    print("Run complete.")
