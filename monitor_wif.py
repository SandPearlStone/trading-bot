#!/usr/bin/env python3
"""
monitor_wif.py — Monitor active WIF long trade every 1 min

Trade params: Entry $0.1705, SL $0.1689, TP1 $0.1724, TP2 $0.1745
"""

from mexc import get_24h, get_ohlcv
from analysis import full_analysis
import json
from datetime import datetime

# Trade parameters
ENTRY = 0.1705
SL = 0.1689
TP1 = 0.1724
TP2 = 0.1745
LEVERAGE = 30

def monitor():
    """Check current WIF price and trade status."""
    try:
        stats = get_24h('WIFUSDT')
        price = stats['price']
        
        # Get latest candle
        c1m = get_ohlcv('WIFUSDT', '1m', 20)
        c5m = get_ohlcv('WIFUSDT', '5m', 20)
        r5m = full_analysis(c5m, 'WIFUSDT')
        
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
        elif pnl_price <= -1:
            status = "⚠️ DOWN -1% — CHECK THESIS"
            alert_level = "WARNING"
        elif pnl_price >= 3:
            status = "📈 UP +3% — STRONG"
            alert_level = "GOOD"
        else:
            status = "⏳ HOLDING — IN RANGE"
            alert_level = "NORMAL"
        
        # 5m structure
        bias = r5m.get('bias', 'unknown')
        rsi = r5m.get('rsi', {}).get('value', 'N/A')
        
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
            "distance_to_tp1": round((TP1 - price) * 10000, 2),
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    result = monitor()
    print(json.dumps(result, indent=2))
    
    # Print formatted alert
    print("\n" + "="*60)
    print(f"WIF MONITOR | {result.get('timestamp', 'N/A')}")
    print("="*60)
    print(f"Price:      ${result.get('price', 0):.4f}")
    print(f"Entry:      ${result.get('entry', 0):.4f}")
    print(f"PnL:        {result.get('pnl_pct', 0):+.2f}% ({result.get('pnl_levered', 0):+.2f}% @ {LEVERAGE}x)")
    print(f"5m Bias:    {result.get('bias', 'N/A')} | RSI {result.get('rsi', 'N/A')}")
    print(f"\nStatus:     {result.get('status', 'N/A')}")
    print(f"TP1:        ${result.get('tp1', 0):.4f} ({result.get('distance_to_tp1', 0):+.2f} pips away)")
    print("="*60)
