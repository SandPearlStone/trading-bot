#!/usr/bin/env python3
"""
populate_ohlcv.py — One-time script to fetch all OHLCV and cache to SQLite
"""

from mexc import get_ohlcv
from db import insert_ohlcv
import sys

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 
           'AVAXUSDT', 'LINKUSDT', 'ARBUSDT', 'OPUSDT', 'PEPEUSDT', 'WIFUSDT']

TIMEFRAMES = ['1h', '4h']

print("📊 Fetching 90-day OHLCV for all symbols...")
print("=" * 60)

for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        try:
            print(f"Fetching {symbol} {tf}...", end=" ", flush=True)
            candles = get_ohlcv(symbol, tf, 500)  # ~20 days of 1h, ~80 days of 4h
            insert_ohlcv(symbol, tf, candles)
            print(f"✅ {len(candles)} candles")
        except Exception as e:
            print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ OHLCV cache populated. Never fetch again (use DB).")
