#!/usr/bin/env python3
"""
monitor_avax.py — Monitor AVAX long trade every 5 min

Trade params: Entry $9.453, SL $9.4285, TP1 $9.5585, TP2 $9.6365, 30x leverage
"""

from mexc import get_24h
import json
from datetime import datetime

# Trade parameters (CORRECTED)
ENTRY = 9.453
SL = 9.4285
TP1 = 9.5585
TP2 = 9.6365
LEVERAGE = 30

def monitor():
    """Check current AVAX price and trade status."""
    try:
        stats = get_24h('AVAXUSDT')
        price = stats['price']
        
        # Calculate metrics
        pnl_price = ((price - ENTRY) / ENTRY) * 100  # LONG: profit when price goes up
        pnl_levered = pnl_price * LEVERAGE
        
        # Determine status
        if price <= SL:
            status = "🚨 SL HIT — STOP OUT"
            alert_level = "CRITICAL"
        elif price >= TP2:
            status = "🎯 TP2 HIT — FULL PROFIT"
            alert_level = "WIN"
        elif price >= TP1:
            status = "✅ TP1 HIT — TAKE PARTIAL 50%"
            alert_level = "WIN"
        elif pnl_price >= 2:
            status = "📈 UP +2% — STRONG"
            alert_level = "GOOD"
        elif pnl_price <= -1:
            status = "⚠️ DOWN -1% — CHECK THESIS"
            alert_level = "WARNING"
        else:
            status = "⏳ HOLDING — GOOD ENTRY"
            alert_level = "NORMAL"
        
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
            "distance_to_tp1": round((TP1 - price), 4),
            "distance_to_tp2": round((TP2 - price), 4),
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    result = monitor()
    print(json.dumps(result, indent=2))
    
    # Print formatted alert
    print("\n" + "="*60)
    print(f"AVAX MONITOR | {result.get('timestamp', 'N/A')}")
    print("="*60)
    print(f"Price:      ${result.get('price', 0):.4f}")
    print(f"Entry:      ${result.get('entry', 0):.4f}")
    print(f"PnL:        {result.get('pnl_pct', 0):+.2f}% ({result.get('pnl_levered', 0):+.2f}% @ {LEVERAGE}x)")
    print(f"\nTP1:        ${result.get('tp1', 0):.4f} ({result.get('distance_to_tp1', 0):+.4f} away)")
    print(f"TP2:        ${result.get('tp2', 0):.4f} ({result.get('distance_to_tp2', 0):+.4f} away)")
    print(f"SL:         ${result.get('sl', 0):.4f}")
    print(f"\nStatus:     {result.get('status', 'N/A')}")
    print("="*60)
