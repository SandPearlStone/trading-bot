#!/usr/bin/env python3
"""
divergence_detector.py — Detect RSI & MACD divergences (regular + hidden)

Phase 2 addition: Hidden divergence detection (stronger in trending markets)
"""

import pandas as pd
import numpy as np
from analysis import rsi as calc_rsi, swing_points

def rsi_regular_divergence(df, rsi_period=14, lookback=5):
    """
    Detect RSI regular divergence (reversal signals).
    
    Bullish: Price LLow, RSI LLow (setup for bounce up)
    Bearish: Price HHigh, RSI LLow (setup for drop down)
    """
    
    df = df.copy()
    df['rsi'] = calc_rsi(df, rsi_period)
    df = swing_points(df, lookback)
    
    divergences = []
    
    # Bullish: price makes lower low, RSI makes higher low
    lows_idx = df['swing_low'].dropna().index.tolist()
    for i in range(1, len(lows_idx)):
        t_a, t_b = lows_idx[i-1], lows_idx[i]
        price_a, price_b = df.loc[t_a, 'low'], df.loc[t_b, 'low']
        rsi_a, rsi_b = df.loc[t_a, 'rsi'], df.loc[t_b, 'rsi']
        
        if price_a > price_b and rsi_a < rsi_b:  # LL, HL
            divergences.append({
                'type': 'bullish_regular',
                'price_a': price_a,
                'price_b': price_b,
                'rsi_a': rsi_a,
                'rsi_b': rsi_b,
                'strength': (rsi_b - rsi_a) / 100  # 0.1 = 10 RSI point diff
            })
    
    # Bearish: price makes higher high, RSI makes lower high
    highs_idx = df['swing_high'].dropna().index.tolist()
    for i in range(1, len(highs_idx)):
        t_a, t_b = highs_idx[i-1], highs_idx[i]
        price_a, price_b = df.loc[t_a, 'high'], df.loc[t_b, 'high']
        rsi_a, rsi_b = df.loc[t_a, 'rsi'], df.loc[t_b, 'rsi']
        
        if price_a < price_b and rsi_a > rsi_b:  # HH, LH
            divergences.append({
                'type': 'bearish_regular',
                'price_a': price_a,
                'price_b': price_b,
                'rsi_a': rsi_a,
                'rsi_b': rsi_b,
                'strength': (rsi_a - rsi_b) / 100
            })
    
    return divergences

def rsi_hidden_divergence(df, rsi_period=14, lookback=5):
    """
    Detect RSI hidden divergence (trend continuation signals).
    
    Bullish hidden: Price LLow, RSI HHigh (trend continues UP)
    Bearish hidden: Price HHigh, RSI LLow (trend continues DOWN)
    
    Often MORE RELIABLE than regular divergence in strong trends.
    """
    
    df = df.copy()
    df['rsi'] = calc_rsi(df, rsi_period)
    df = swing_points(df, lookback)
    
    divergences = []
    
    # Bullish hidden: price LL, RSI HH (strongest uptrend signal)
    lows_idx = df['swing_low'].dropna().index.tolist()
    for i in range(1, len(lows_idx)):
        t_a, t_b = lows_idx[i-1], lows_idx[i]
        price_a, price_b = df.loc[t_a, 'low'], df.loc[t_b, 'low']
        rsi_a, rsi_b = df.loc[t_a, 'rsi'], df.loc[t_b, 'rsi']
        
        if price_a > price_b and rsi_a < rsi_b:  # LL, HH
            divergences.append({
                'type': 'bullish_hidden',
                'price_a': price_a,
                'price_b': price_b,
                'rsi_a': rsi_a,
                'rsi_b': rsi_b,
                'strength': (rsi_b - rsi_a) / 100
            })
    
    # Bearish hidden: price HH, RSI LL (strongest downtrend signal)
    highs_idx = df['swing_high'].dropna().index.tolist()
    for i in range(1, len(highs_idx)):
        t_a, t_b = highs_idx[i-1], highs_idx[i]
        price_a, price_b = df.loc[t_a, 'high'], df.loc[t_b, 'high']
        rsi_a, rsi_b = df.loc[t_a, 'rsi'], df.loc[t_b, 'rsi']
        
        if price_a < price_b and rsi_a > rsi_b:  # HH, LL
            divergences.append({
                'type': 'bearish_hidden',
                'price_a': price_a,
                'price_b': price_b,
                'rsi_a': rsi_a,
                'rsi_b': rsi_b,
                'strength': (rsi_a - rsi_b) / 100
            })
    
    return divergences

def divergence_score(divs, div_type):
    """
    Score divergence strength (for confluence).
    
    Returns: 0-10 scale
    - Regular div = 5pts (reversal potential)
    - Hidden div = 8pts (trend continuation, stronger)
    - Strong (RSI diff > 15) = +2pts bonus
    """
    
    if not divs:
        return 0
    
    # Get most recent divergence
    latest = divs[-1]
    
    base_score = 5 if 'regular' in latest['type'] else 8
    strength_bonus = latest.get('strength', 0) * 10  # Convert to 0-10
    
    return min(10, base_score + strength_bonus * 0.2)

if __name__ == "__main__":
    from mexc import get_ohlcv
    from analysis import to_df
    
    # Test
    c1h = get_ohlcv("BTCUSDT", "1h", 100)
    df = to_df(c1h)
    
    reg_divs = rsi_regular_divergence(df)
    hid_divs = rsi_hidden_divergence(df)
    
    print(f"Regular divergences: {len(reg_divs)}")
    for div in reg_divs[-3:]:
        print(f"  {div['type']}: {div['rsi_a']:.0f} → {div['rsi_b']:.0f}")
    
    print(f"\nHidden divergences: {len(hid_divs)}")
    for div in hid_divs[-3:]:
        print(f"  {div['type']}: {div['rsi_a']:.0f} → {div['rsi_b']:.0f}")
