#!/usr/bin/env python3
"""
regime_detector.py — Classify market regime (TRENDING/NORMAL/HIGH_VOL/CHOPPY)

Phase 2 addition: Adaptive thresholds based on regime
"""

import pandas as pd
import numpy as np
from analysis import atr

def classify_regime(df_4h, df_1h):
    """
    Classify market regime using ATR volatility ratio + EMA slope.
    
    Regimes:
    - CHOPPY: Vol ratio < 0.8, ranging structure → SKIP entries
    - NORMAL: Vol ratio 0.8-1.2, mixed → trade with caution
    - TRENDING: Vol ratio > 1.2, moving, EMA steep → favor momentum
    
    Args:
        df_4h: 4H candles (for longer-term trend)
        df_1h: 1H candles (for entries)
    
    Returns: dict with regime, confidence, metrics
    """
    
    # ATR volatility ratio (current vs. historical)
    atr_20_4h = atr(df_4h, 20)
    atr_sma_4h = np.mean([atr(df_4h.iloc[max(0, i-60):i+1], 20) for i in range(len(df_4h))])
    
    vol_ratio = atr_20_4h[-1] / atr_sma_4h if atr_sma_4h > 0 else 1.0
    
    # EMA slope (trend strength)
    ema_21 = df_1h['close'].ewm(span=21).mean()
    ema_slope = (ema_21.iloc[-1] - ema_21.iloc[-5]) / ema_21.iloc[-5] * 100 if ema_21.iloc[-5] != 0 else 0
    
    # Range vs. ATR (structure quality)
    recent_high = df_1h['high'].iloc[-20:].max()
    recent_low = df_1h['low'].iloc[-20:].min()
    range_size = (recent_high - recent_low) / df_1h['close'].iloc[-1] * 100
    atr_pct = (atr_20_4h[-1] / df_1h['close'].iloc[-1]) * 100
    
    # Classify
    if vol_ratio < 0.8 and abs(ema_slope) < 0.5:
        regime = "CHOPPY"
        confidence = 0.9
    elif vol_ratio < 0.8:
        regime = "RANGING"
        confidence = 0.85
    elif vol_ratio > 1.2 and abs(ema_slope) > 1.0:
        regime = "TRENDING"
        confidence = 0.9
    elif vol_ratio > 1.2:
        regime = "VOLATILE"
        confidence = 0.8
    else:
        regime = "NORMAL"
        confidence = 0.7
    
    return {
        "regime": regime,
        "vol_ratio": round(vol_ratio, 2),
        "ema_slope": round(ema_slope, 2),
        "confidence": confidence,
        "recommendation": _regime_recommendation(regime)
    }

def adaptive_min_score(regime):
    """
    Adjust min_score threshold based on regime.
    
    CHOPPY:   min_score = 85 (skip most entries, only very clean setups)
    RANGING:  min_score = 75 (conservative entries)
    NORMAL:   min_score = 65 (standard threshold)
    TRENDING: min_score = 55 (favor momentum, lower threshold)
    VOLATILE: min_score = 70 (tight entries)
    """
    
    thresholds = {
        "CHOPPY": 85,
        "RANGING": 75,
        "NORMAL": 65,
        "TRENDING": 55,
        "VOLATILE": 70
    }
    
    return thresholds.get(regime, 65)

def _regime_recommendation(regime):
    """Return trading recommendation for regime."""
    
    recommendations = {
        "CHOPPY": "⚠️ SKIP entries. Wait for clearer direction.",
        "RANGING": "🟡 Reduced size. Take only clear mean-revert setups.",
        "NORMAL": "✓ Normal trading. Standard position sizing.",
        "TRENDING": "🟢 Favor momentum. Higher conviction on trend-following.",
        "VOLATILE": "⚠️ Tight stops. Wide move potential."
    }
    
    return recommendations.get(regime, "Unknown regime")

def get_regime_adjustment(regime):
    """
    Return position size multiplier based on regime.
    
    CHOPPY:   0.25x (quarter size, high risk)
    RANGING:  0.5x (half size)
    NORMAL:   1.0x (normal size)
    TRENDING: 1.2x (slight increase for momentum)
    VOLATILE: 0.8x (reduced, high volatility)
    """
    
    multipliers = {
        "CHOPPY": 0.25,
        "RANGING": 0.5,
        "NORMAL": 1.0,
        "TRENDING": 1.2,
        "VOLATILE": 0.8
    }
    
    return multipliers.get(regime, 1.0)

if __name__ == "__main__":
    from mexc import get_ohlcv
    from analysis import to_df
    
    # Test
    c4h = get_ohlcv("BTCUSDT", "4h", 100)
    c1h = get_ohlcv("BTCUSDT", "1h", 100)
    
    df4h = to_df(c4h)
    df1h = to_df(c1h)
    
    result = classify_regime(df4h, df1h)
    
    print(f"Regime:     {result['regime']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Vol Ratio:  {result['vol_ratio']}")
    print(f"EMA Slope:  {result['ema_slope']:+.2f}%")
    print(f"Min Score:  {adaptive_min_score(result['regime'])}")
    print(f"Position:   {get_regime_adjustment(result['regime']):.2f}x")
    print(f"\n{result['recommendation']}")
