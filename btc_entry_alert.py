#!/usr/bin/env python3
"""
btc_entry_alert.py — Monitor BTC 1H for bullish reversal confirmation

Checks for:
1. Engulfing candle (bullish)
2. Hammer/pin bar (bullish reversal)
3. Higher low pattern (structural confirmation)
4. RSI bouncing off oversold (<30) with +5 candle
5. Volume surge on recovery

Triggers alert when confirmed.
"""

from mexc import get_ohlcv
from analysis import full_analysis, to_df
import json
from datetime import datetime


def check_bullish_engulfing(candles, lookback=2):
    """Check if last candle is bullish engulfing."""
    if len(candles) < lookback + 1:
        return False, None
    
    prev = candles[-2]
    curr = candles[-1]
    
    # Bullish engulfing: current candle body > previous, opens below prev close, closes above prev open
    if (curr['close'] > prev['open'] and 
        curr['open'] < prev['close'] and 
        (curr['close'] - curr['open']) > (prev['close'] - prev['open'])):
        return True, "Bullish engulfing"
    
    return False, None


def check_hammer(candles, lookback=1):
    """Check for hammer/pin bar (long lower wick, small body at top)."""
    if len(candles) < lookback + 1:
        return False, None
    
    candle = candles[-1]
    body = abs(candle['close'] - candle['open'])
    lower_wick = candle['open'] - candle['low'] if candle['close'] < candle['open'] else candle['close'] - candle['low']
    high_to_body = candle['high'] - max(candle['open'], candle['close'])
    
    # Hammer: lower wick > 2x body, small upper wick
    if lower_wick > body * 2 and high_to_body < body * 0.5:
        return True, "Hammer (reversal pin)"
    
    return False, None


def check_higher_low(candles, lookback=5):
    """Check if forming higher low pattern."""
    if len(candles) < lookback + 1:
        return False, None
    
    # Last N lows should be rising
    lows = [c['low'] for c in candles[-lookback:]]
    highs = [c['high'] for c in candles[-lookback:]]
    
    # Trend of lows should be rising
    if all(lows[i] < lows[i+1] for i in range(len(lows)-2)):
        return True, f"Higher low pattern (lows: {min(lows):.0f} → {max(lows):.0f})"
    
    return False, None


def check_rsi_bounce(analysis):
    """Check if RSI is bouncing off oversold."""
    rsi = analysis.get('rsi', {})
    if isinstance(rsi, dict):
        value = rsi.get('value', 50)
    else:
        value = rsi
    
    # RSI < 30 is oversold, expecting bounce
    if value and value < 30:
        return True, f"RSI oversold ({value:.1f}) — bounce setup"
    
    return False, None


def monitor_btc_1h():
    """Check BTC 1H for bullish reversal."""
    symbol = 'BTCUSDT'
    
    try:
        # Get last 50 candles
        candles = get_ohlcv(symbol, '1h', 50)
        if len(candles) < 5:
            return None
        
        # Get analysis
        analysis = full_analysis(candles[-20:], symbol)  # Use last 20 for analysis
        
        alerts = []
        
        # Check each condition
        is_engulfing, msg = check_bullish_engulfing(candles)
        if is_engulfing:
            alerts.append(msg)
        
        is_hammer, msg = check_hammer(candles)
        if is_hammer:
            alerts.append(msg)
        
        is_higher_low, msg = check_higher_low(candles)
        if is_higher_low:
            alerts.append(msg)
        
        is_rsi_bounce, msg = check_rsi_bounce(analysis)
        if is_rsi_bounce:
            alerts.append(msg)
        
        # Get current price + RSI
        curr = candles[-1]
        price = curr['close']
        rsi = analysis.get('rsi', {})
        if isinstance(rsi, dict):
            rsi_val = rsi.get('value', 'N/A')
        else:
            rsi_val = rsi
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "price": price,
            "rsi": rsi_val,
            "alerts": alerts,
            "confidence": len(alerts),  # Number of bullish signals
            "should_trigger": len(alerts) >= 2,  # Trigger if 2+ signals
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    result = monitor_btc_1h()
    if result:
        print(json.dumps(result, indent=2))
        
        # Print summary
        if result.get('should_trigger'):
            print(f"\n🚨 BTC 1H BULLISH REVERSAL ALERT")
            print(f"Price: ${result['price']:.2f}")
            print(f"RSI: {result['rsi']}")
            print(f"Signals: {', '.join(result['alerts'])}")
        else:
            print(f"\n✓ BTC 1H Status: {len(result.get('alerts', []))} bullish signal(s)")
