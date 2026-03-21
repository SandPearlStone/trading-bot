#!/home/sandro/trading-venv/bin/python3
"""
trading_tools.py — Data fetcher functions for LLM analysis.
Integrates live exchange data + technical indicators + confluence scoring.
"""

import ccxt
import talib
import numpy as np
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime, timedelta

# ── DB Path & Functions ────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'market.db')

def load_from_db(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Load OHLCV from SQLite market.db. Returns DataFrame with columns: timestamp, open, high, low, close, volume."""
    conn = sqlite3.connect(DB_PATH)
    
    if interval == '15m':
        query = f"SELECT open_time as timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE symbol=? ORDER BY open_time DESC LIMIT ?"
    else:
        query = f"SELECT open_time as timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND interval=? ORDER BY open_time DESC LIMIT ?"
    
    if interval == '15m':
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
    else:
        df = pd.read_sql_query(query, conn, params=(symbol, interval, limit))
    
    conn.close()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_funding_from_db(symbol: str) -> pd.DataFrame:
    """Load funding rates from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, rate as funding_rate FROM funding_rates WHERE symbol=? ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    return df.sort_values('timestamp').reset_index(drop=True)

def load_oi_from_db(symbol: str) -> pd.DataFrame:
    """Load open interest from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, oi_value FROM open_interest WHERE symbol=? ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    return df.sort_values('timestamp').reset_index(drop=True)

# ── Function 1: check_symbol ──────────────────────────────────────────────────

def check_symbol(symbol: str, use_db: bool = False) -> dict:
    """Fetches live data + computes indicators for one symbol."""
    try:
        from confluence import score_setup
    except ImportError:
        return {
            'symbol': symbol,
            'error': 'confluence module not available',
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    result = {
        'symbol': symbol,
        'price': None,
        'change_1h': None,
        'change_4h': None,
        'change_24h': None,
        'volume_ratio': None,
        'rsi_15m': None,
        'rsi_1h': None,
        'rsi_4h': None,
        'macd_1h': None,
        'atr_1h': None,
        'atr_pct': None,
        'bb_width_1h': None,
        'adx_1h': None,
        'ema_trend': None,
        'regime': None,
        'confluence_score': None,
        'confluence_grade': None,
        'confluence_direction': None,
        'confluence_reasons': [],
        'confluence_missing': [],
        'funding_rate': None,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    try:
        if use_db:
            # Load from SQLite database
            df_1h = load_from_db(symbol, '1h', limit=100)
            df_4h = load_from_db(symbol, '4h', limit=100)
            df_15m = load_from_db(symbol, '15m', limit=100)
            
            if len(df_1h) == 0 or len(df_4h) == 0 or len(df_15m) == 0:
                result['error'] = 'Insufficient data in database'
                return result
        else:
            # Fetch from Binance
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Fetch OHLCV from Binance
            ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=100)
            ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=100)
            ohlcv_15m = exchange.fetch_ohlcv(symbol, '15m', limit=100)
            
            # Create DataFrames: [timestamp, open, high, low, close, volume]
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Current price
        price = float(df_1h['close'].iloc[-1])
        result['price'] = price
        
        # Price changes
        result['change_1h'] = float((df_1h['close'].iloc[-1] / df_1h['open'].iloc[-1] - 1) * 100)
        result['change_4h'] = float((df_4h['close'].iloc[-1] / df_4h['open'].iloc[0] - 1) * 100)
        
        # 24h change: use first 1h bar if available
        if len(df_1h) >= 24:
            result['change_24h'] = float((df_1h['close'].iloc[-1] / df_1h['close'].iloc[-24] - 1) * 100)
        else:
            result['change_24h'] = result['change_4h']
        
        # Volume ratio: current vol / 20-bar avg on 1h
        vol_20_avg = df_1h['volume'].tail(20).mean()
        result['volume_ratio'] = float(df_1h['volume'].iloc[-1] / vol_20_avg) if vol_20_avg > 0 else 1.0
        
        # RSI calculations (talib needs numpy arrays)
        result['rsi_15m'] = float(talib.RSI(df_15m['close'].values, timeperiod=14)[-1])
        result['rsi_1h'] = float(talib.RSI(df_1h['close'].values, timeperiod=14)[-1])
        result['rsi_4h'] = float(talib.RSI(df_4h['close'].values, timeperiod=14)[-1])
        
        # MACD on 1h
        macd, signal, hist = talib.MACD(df_1h['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        result['macd_1h'] = float(hist[-1])
        
        # ATR on 1h
        atr = talib.ATR(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, timeperiod=14)
        result['atr_1h'] = float(atr[-1])
        result['atr_pct'] = float((result['atr_1h'] / price * 100)) if price > 0 else None
        
        # Bollinger Bands width on 1h
        bb_up, bb_mid, bb_low = talib.BBANDS(df_1h['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        result['bb_width_1h'] = float(bb_up[-1] - bb_low[-1])
        
        # ADX on 1h
        adx = talib.ADX(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, timeperiod=14)
        result['adx_1h'] = float(adx[-1])
        
        # EMA trend: check if 8 > 21 > 55 (bullish) on 1h
        ema_8 = talib.EMA(df_1h['close'].values, timeperiod=8)[-1]
        ema_21 = talib.EMA(df_1h['close'].values, timeperiod=21)[-1]
        ema_55 = talib.EMA(df_1h['close'].values, timeperiod=55)[-1]
        
        if ema_8 > ema_21 > ema_55:
            result['ema_trend'] = 'BULLISH'
        elif ema_8 < ema_21 < ema_55:
            result['ema_trend'] = 'BEARISH'
        else:
            result['ema_trend'] = 'MIXED'
        
        # Get confluence score
        confluence_data = score_setup(symbol)
        result['confluence_score'] = confluence_data.get('score', 0)
        result['confluence_grade'] = confluence_data.get('grade', 'F')
        result['confluence_direction'] = confluence_data.get('direction', 'NO_TRADE')
        result['confluence_reasons'] = confluence_data.get('confluence_reasons', [])
        result['confluence_missing'] = confluence_data.get('missing', [])
        
        # Extract regime from details
        details = confluence_data.get('details', {})
        if isinstance(details, dict) and 'regime' in details:
            regime_info = details.get('regime', {})
            if isinstance(regime_info, dict):
                result['regime'] = regime_info.get('regime', 'UNKNOWN')
            else:
                result['regime'] = str(regime_info)
        
        # Get funding rate from SQLite
        funding_data = get_funding(symbol)
        result['funding_rate'] = funding_data.get('funding_rate')
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ── Function 2: check_btc ─────────────────────────────────────────────────────

def check_btc() -> dict:
    """Same as check_symbol('BTCUSDT') but always called."""
    return check_symbol('BTCUSDT')


# ── Function 3: get_levels ───────────────────────────────────────────────────

def get_levels(symbol: str) -> dict:
    """Extracts key levels from confluence.py output."""
    try:
        from confluence import score_setup
    except ImportError:
        return {'symbol': symbol, 'error': 'confluence module not available'}
    
    result = {
        'symbol': symbol,
        'price': None,
        'entry_zone': None,
        'optimal_entry': None,
        'stop_loss': None,
        'tp1': None,
        'tp2': None,
        'tp3': None,
        'nearest_fvg': None,
        'nearest_ob': None,
        'orderbook_imbalance': None,
    }
    
    try:
        # Get current price
        exchange = ccxt.binance({'enableRateLimit': True})
        ticker = exchange.fetch_ticker(symbol)
        result['price'] = ticker['last']
        
        # Get confluence data
        confluence_data = score_setup(symbol)
        
        result['entry_zone'] = confluence_data.get('entry_zone')
        result['optimal_entry'] = confluence_data.get('optimal_entry')
        result['stop_loss'] = confluence_data.get('stop_loss')
        result['tp1'] = confluence_data.get('tp1')
        result['tp2'] = confluence_data.get('tp2')
        result['tp3'] = confluence_data.get('tp3')
        
        # Extract nearest FVG and OB from confluence_reasons
        reasons = confluence_data.get('confluence_reasons', [])
        for reason in reasons:
            if 'FVG' in reason.upper() or 'fvg' in reason:
                result['nearest_fvg'] = reason
            if 'OB' in reason.upper() or 'order block' in reason.lower():
                result['nearest_ob'] = reason
        
        # Extract orderbook imbalance from details
        details = confluence_data.get('details', {})
        if isinstance(details, dict):
            result['orderbook_imbalance'] = details.get('orderbook_imbalance')
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ── Function 4: scan_market ──────────────────────────────────────────────────

def scan_market(symbols: list) -> list:
    """Calls check_symbol() for each symbol + check_btc(). Returns list sorted by confluence_score."""
    results = []
    
    # Always include BTC as first element
    btc_result = check_btc()
    results.append(btc_result)
    
    # Check each symbol
    for symbol in symbols:
        if symbol != 'BTCUSDT':
            result = check_symbol(symbol)
            results.append(result)
    
    # Sort by abs(confluence_score) descending, but keep BTC first
    btc = results.pop(0)
    results.sort(key=lambda x: abs(x.get('confluence_score', 0) or 0), reverse=True)
    results.insert(0, btc)
    
    return results


# ── Function 5: get_funding ───────────────────────────────────────────────────

def get_funding(symbol: str) -> dict:
    """Reads funding rate + OI from SQLite (data/market.db)."""
    result = {
        'symbol': symbol,
        'funding_rate': None,
        'funding_8h_ago': None,
        'oi_latest': None,
        'oi_change_pct': None,
    }
    
    try:
        # Load funding rates from DB
        funding_df = load_funding_from_db(symbol)
        if len(funding_df) > 0:
            result['funding_rate'] = float(funding_df['funding_rate'].iloc[-1])
            
            # Get funding rate from 8h ago (if available)
            if len(funding_df) > 1:
                eight_hours_ago_ms = datetime.utcnow().timestamp() * 1000 - (8 * 60 * 60 * 1000)
                older_funding = funding_df[funding_df['timestamp'] < eight_hours_ago_ms]
                if len(older_funding) > 0:
                    result['funding_8h_ago'] = float(older_funding['funding_rate'].iloc[-1])
        
        # Load open interest from DB
        oi_df = load_oi_from_db(symbol)
        if len(oi_df) > 0:
            result['oi_latest'] = float(oi_df['oi_value'].iloc[-1])
            
            # Get OI change percentage (24h)
            if len(oi_df) > 1:
                twenty_four_hours_ago_ms = datetime.utcnow().timestamp() * 1000 - (24 * 60 * 60 * 1000)
                older_oi = oi_df[oi_df['timestamp'] < twenty_four_hours_ago_ms]
                if len(older_oi) > 0:
                    oi_old = float(older_oi['oi_value'].iloc[-1])
                    result['oi_change_pct'] = ((result['oi_latest'] - oi_old) / oi_old * 100) if oi_old > 0 else None
        
    except Exception as e:
        result['error'] = str(e)
    
    return result
