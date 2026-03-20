#!/usr/bin/env python3
"""
db.py — SQLite operations for trades, positions, OHLCV cache

Manages:
- OHLCV caching (fetch once, reuse forever)
- Trade journal (auto-logged)
- Position tracking (live updates)
- Analytics queries (win rate, Sharpe, etc.)
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

DB_PATH = Path(__file__).parent / "data" / "trades.db"

def init_db():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # OHLCV cache
    c.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(symbol, timeframe, open_time)
        )
    ''')
    
    # Trades journal
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            entry_time TEXT,
            exit_time TEXT,
            pnl_pct REAL,
            grade TEXT,
            reason TEXT,
            status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Open positions
    c.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL,
            current_price REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            leverage INTEGER,
            opened_at TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'open'
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_PATH}")

def insert_ohlcv(symbol, timeframe, candles):
    """
    Insert OHLCV candles into cache.
    
    Args:
        symbol: 'BTCUSDT', 'ETHUSDT', etc.
        timeframe: '1h', '4h', '15m'
        candles: list of {time, open, high, low, close, volume}
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for candle in candles:
        try:
            c.execute('''
                INSERT OR IGNORE INTO ohlcv 
                (symbol, timeframe, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, timeframe,
                str(candle['time']),
                float(candle['open']),
                float(candle['high']),
                float(candle['low']),
                float(candle['close']),
                float(candle['volume'])
            ))
        except Exception as e:
            print(f"Warning inserting {symbol}: {e}")
    
    conn.commit()
    conn.close()

def get_ohlcv(symbol, timeframe, limit=100):
    """
    Fetch OHLCV from cache.
    
    Returns: list of candles, newest last
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timeframe = ?
        ORDER BY open_time ASC
        LIMIT ?
    ''', (symbol, timeframe, limit))
    
    rows = c.fetchall()
    conn.close()
    
    candles = []
    for row in rows:
        candles.append({
            'time': row[0],
            'open': row[1],
            'high': row[2],
            'low': row[3],
            'close': row[4],
            'volume': row[5]
        })
    
    return candles

def log_trade(symbol, direction, entry_price, exit_price, entry_time, exit_time, pnl_pct, grade, reason, status='closed'):
    """Log a trade to journal."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO trades 
        (symbol, direction, entry_price, exit_price, entry_time, exit_time, pnl_pct, grade, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, direction, entry_price, exit_price, entry_time, exit_time, pnl_pct, grade, reason, status))
    
    conn.commit()
    conn.close()
    print(f"✅ Logged: {symbol} {direction} @ {entry_price} → {exit_price} ({pnl_pct:+.2f}%)")

def open_position(symbol, direction, entry_price, sl, tp1, tp2, leverage, opened_at):
    """Track a new open position."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT OR REPLACE INTO positions
        (symbol, direction, entry_price, current_price, sl, tp1, tp2, leverage, opened_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    ''', (symbol, direction, entry_price, entry_price, sl, tp1, tp2, leverage, opened_at))
    
    conn.commit()
    conn.close()

def update_position(symbol, current_price):
    """Update current price for open position."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        UPDATE positions
        SET current_price = ?, updated_at = CURRENT_TIMESTAMP
        WHERE symbol = ? AND status = 'open'
    ''', (current_price, symbol))
    
    conn.commit()
    conn.close()

def close_position(symbol, exit_price, exit_time, pnl_pct):
    """Close an open position and log to trades."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get position details
    c.execute('''
        SELECT direction, entry_price, opened_at, leverage FROM positions
        WHERE symbol = ? AND status = 'open'
    ''', (symbol,))
    
    row = c.fetchone()
    if not row:
        print(f"❌ No open position for {symbol}")
        conn.close()
        return
    
    direction, entry_price, opened_at, leverage = row
    
    # Log to trades
    c.execute('''
        INSERT INTO trades
        (symbol, direction, entry_price, exit_price, entry_time, exit_time, pnl_pct, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'closed')
    ''', (symbol, direction, entry_price, exit_price, opened_at, exit_time, pnl_pct))
    
    # Mark position closed
    c.execute('''
        UPDATE positions SET status = 'closed', updated_at = CURRENT_TIMESTAMP
        WHERE symbol = ? AND status = 'open'
    ''', (symbol,))
    
    conn.commit()
    conn.close()
    print(f"✅ Closed: {symbol} {direction} ({pnl_pct:+.2f}%)")

def get_open_positions():
    """Fetch all open positions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT symbol, direction, entry_price, current_price, sl, tp1, tp2, leverage, opened_at
        FROM positions
        WHERE status = 'open'
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    positions = []
    for row in rows:
        positions.append({
            'symbol': row[0],
            'direction': row[1],
            'entry_price': row[2],
            'current_price': row[3],
            'sl': row[4],
            'tp1': row[5],
            'tp2': row[6],
            'leverage': row[7],
            'opened_at': row[8]
        })
    
    return positions

def get_trade_stats(symbol=None, days=30):
    """Calculate trade statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    where_clause = ""
    params = []
    
    if symbol:
        where_clause += "AND symbol = ? "
        params.append(symbol)
    
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    where_clause += "AND exit_time > ? "
    params.append(cutoff)
    
    c.execute(f'''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
            AVG(pnl_pct) as avg_pnl,
            MAX(pnl_pct) as best_trade,
            MIN(pnl_pct) as worst_trade
        FROM trades
        WHERE status = 'closed' {where_clause}
    ''', params)
    
    row = c.fetchone()
    conn.close()
    
    if not row or row[0] == 0:
        return None
    
    total, wins, avg, best, worst = row
    wr = (wins / total * 100) if total > 0 else 0
    
    return {
        'total_trades': total,
        'wins': wins,
        'win_rate': wr,
        'avg_pnl': avg,
        'best': best,
        'worst': worst
    }

if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
