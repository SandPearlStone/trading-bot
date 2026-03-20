#!/usr/bin/env python3
"""
monitor_sol.py — Monitor active SOL long trade every 5 min

Tracks:
- Current price vs entry/SL/TP levels
- PnL % at leverage (40x)
- Structure status (bullish/bearish)
- Alert on: SL break, TP hit, major moves (±2%)
"""

from mexc import get_24h, get_ohlcv
from analysis import full_analysis
import json
from datetime import datetime

# Trade parameters
ENTRY = 88.04
SL = 87.35
TP1 = 90.58
TP2 = 92.90
TP3 = 91.24
LEVERAGE = 40

def monitor():
    """Check current SOL price and trade status."""
    try:
        stats = get_24h('SOLUSDT')
        price = stats['price']
        
        # Get latest candle
        c1h = get_ohlcv('SOLUSDT', '1h', 20)
        r1h = full_analysis(c1h, 'SOLUSDT')
        
        # Calculate metrics
        pnl_price = ((price - ENTRY) / ENTRY) * 100
        pnl_levered = pnl_price * LEVERAGE
        
        # Determine status
        if price <= SL:
            status = "🚨 SL HIT — STOP OUT"
            alert_level = "CRITICAL"
        elif price >= TP2:
            status = "🎯 TP2 HIT — TAKE PROFIT"
            alert_level = "WIN"
        elif price >= TP1:
            status = "✅ TP1 HIT — PARTIAL PROFIT"
            alert_level = "WIN"
        elif pnl_price <= -2:
            status = "⚠️ DOWN -2% — CHECK THESIS"
            alert_level = "WARNING"
        elif pnl_price >= 2:
            status = "📈 UP +2% — STRONG"
            alert_level = "GOOD"
        else:
            status = "⏳ HOLDING — IN CHOP"
            alert_level = "NORMAL"
        
        # 1H structure
        bias = r1h.get('bias', 'unknown')
        rsi = r1h.get('rsi', {}).get('value', 'N/A')
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "price": price,
            "entry": ENTRY,
            "sl": SL,
            "tp1": TP1,
            "tp2": TP2,
            "pnl_pct": round(pnl_price, 2),
            "pnl_levered": round(pnl_levered, 2),
            "status": status,
            "alert_level": alert_level,
            "bias": bias,
            "rsi": rsi,
            "distance_to_sl": round((price - SL) * 100, 2),
            "distance_to_tp1": round((TP1 - price) * 100, 2),
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    result = monitor()
    print(json.dumps(result, indent=2))
    
    # Print formatted alert
    print("\n" + "="*60)
    print(f"SOL MONITOR | {result.get('timestamp', 'N/A')}")
    print("="*60)
    print(f"Price:      ${result.get('price', 0):.2f}")
    print(f"Entry:      ${result.get('entry', 0):.2f}")
    print(f"PnL:        {result.get('pnl_pct', 0):+.2f}% ({result.get('pnl_levered', 0):+.2f}% @ {LEVERAGE}x)")
    print(f"1H Bias:    {result.get('bias', 'N/A')} | RSI {result.get('rsi', 'N/A')}")
    print(f"\nStatus:     {result.get('status', 'N/A')}")
    print(f"SL:         ${result.get('sl', 0):.2f} ({result.get('distance_to_sl', 0):+.2f} pts away)")
    print(f"TP1:        ${result.get('tp1', 0):.2f} ({result.get('distance_to_tp1', 0):+.2f} pts away)")
    print("="*60)
