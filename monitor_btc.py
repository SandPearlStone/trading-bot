#!/usr/bin/env python3
"""
monitor_btc.py — Monitor active BTC long trade every 1 min

Trade params: Entry $70,779.9, SL $69,696.56, TP1 $70,618.72, TP2 $71,172, R:R 1:4.18
"""

from mexc import get_24h, get_ohlcv
from analysis import full_analysis
import json
from datetime import datetime

# Trade parameters
ENTRY = 70779.9
SL = 70779.9  # MOVED TO BREAKEVEN
TP1 = 72404.90  # Adjusted for actual entry price
TP2 = 74029.90  # Adjusted for actual entry price
LEVERAGE = 10

def monitor():
    """Check current BTC price and trade status."""
    try:
        stats = get_24h('BTCUSDT')
        price = stats['price']
        
        # Get latest candles
        c1h = get_ohlcv('BTCUSDT', '1h', 20)
        r1h = full_analysis(c1h, 'BTCUSDT')
        
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
        elif pnl_price >= 1:
            status = "📈 UP +1% — STRONG"
            alert_level = "GOOD"
        else:
            status = "⏳ HOLDING — BOUNCE PHASE"
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
            "distance_to_tp1": round((TP1 - price), 2),
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    result = monitor()
    print(json.dumps(result, indent=2))
    
    # Print formatted alert
    print("\n" + "="*60)
    print(f"BTC MONITOR | {result.get('timestamp', 'N/A')}")
    print("="*60)
    print(f"Price:      ${result.get('price', 0):,.2f}")
    print(f"Entry:      ${result.get('entry', 0):,.2f}")
    print(f"PnL:        {result.get('pnl_pct', 0):+.2f}% ({result.get('pnl_levered', 0):+.2f}% @ {LEVERAGE}x)")
    print(f"1H Bias:    {result.get('bias', 'N/A')} | RSI {result.get('rsi', 'N/A')}")
    print(f"\nStatus:     {result.get('status', 'N/A')}")
    print(f"TP1:        ${result.get('tp1', 0):,.2f} ({result.get('distance_to_tp1', 0):+,.2f} pts away)")
    print("="*60)
