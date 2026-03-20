#!/usr/bin/env python3
"""
Phase 4.4: ML Integration into Confluence Scoring

Create a new version of confluence.py that:
  1. Loads the trained Random Forest model
  2. Extracts ML features for each setup
  3. Predicts win probability from learned patterns
  4. Boosts/modulates confluence score based on model confidence

Input:
  - phase4_model.pkl (trained Random Forest)
  - confluence.py (original scoring engine)

Output:
  - confluence_with_ml.py (ML-enhanced scoring)

Integration Strategy:
  - Base score: Original confluence engine (unchanged)
  - ML boost: Random Forest predicts P(win) from features
  - Final score = base_score × (0.7 + 0.3 × p_win)
    * Minimum floor: 70% of original (prevents underscoring)
    * Maximum ceiling: 100% of original (prevents overscoring)
    * Gradual blend prevents shock from ML

Usage:
  from confluence_with_ml import score_setup_with_ml
  result = score_setup_with_ml(symbol, df_1h, df_4h, df_15m)
  # result["score"] = final ML-boosted score
  # result["ml_confidence"] = 0-1 probability of win
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

# Local imports
try:
    from confluence import score_setup
except ImportError:
    print("[phase4_ml_integration] WARNING: confluence not available", file=sys.stderr)
    score_setup = None


# ─── Configuration ─────────────────────────────────────────────────────────

MODEL_FILE = Path(__file__).parent / "phase4_model.pkl"
FEATURE_NAMES = [
    'vol_ratio',
    'rsi_strength',
    'rsi_div_regular',
    'rsi_div_hidden',
    'macd_div',
    'sentiment_fg',
    'mtf_bias',
    'confluence_score',
    'entry_to_sl',
    'direction_encoded',
    'duration_candles',
    'regime_CHOPPY',
    'regime_NORMAL',
    'regime_VOLATILE',
]


# ─── Global Model Cache ───────────────────────────────────────────────────────

_model = None
_model_loaded = False


def _load_model():
    """Load model on first use."""
    global _model, _model_loaded
    
    if _model_loaded:
        return _model
    
    try:
        with open(MODEL_FILE, "rb") as f:
            _model = pickle.load(f)
        _model_loaded = True
        print(f"[phase4_ml_integration] ✅ Loaded model from {MODEL_FILE}")
        return _model
    except Exception as e:
        print(f"[phase4_ml_integration] ❌ Failed to load model: {e}")
        _model_loaded = True
        return None


# ─── Feature Extraction (same as phase4_feature_engineering.py) ─────────────

def _extract_ml_features(
    setup_result: Dict,
    df_1h: pd.DataFrame,
    direction: str,
) -> Optional[pd.DataFrame]:
    """
    Extract ML features from setup result and 1h candles.

    Input: setup_result from confluence.score_setup()
    Output: DataFrame with one row of features
    """
    from analysis import atr as calc_atr, rsi, ema
    
    if len(df_1h) < 20:
        return None

    try:
        # 1. Regime (simplified)
        atr_series = calc_atr(df_1h, period=20)
        current_atr = atr_series.iloc[-1]
        hist_atr = atr_series.iloc[-50:-1].mean() if len(atr_series) > 50 else atr_series.mean()
        
        if np.isnan(current_atr) or np.isnan(hist_atr) or hist_atr == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = current_atr / hist_atr

        if vol_ratio < 0.8:
            regime = "CHOPPY"
        elif vol_ratio > 1.2:
            regime = "VOLATILE"
        else:
            regime = "NORMAL"

        # 2. RSI features
        rsi_vals = rsi(df_1h, period=14)
        last_rsi = rsi_vals.iloc[-1] if not np.isnan(rsi_vals.iloc[-1]) else 50
        rsi_strength = abs(last_rsi - 50) / 50

        rsi_div_regular = 0
        rsi_div_hidden = 0
        if len(rsi_vals) >= 5:
            recent_rsi = rsi_vals.iloc[-5:].values
            recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
            if len(recent_rsi) >= 2:
                if (df_1h["low"].iloc[-1] < df_1h["low"].iloc[-2]) and (recent_rsi[-1] > recent_rsi[-2]):
                    rsi_div_regular = 1
                if (df_1h["high"].iloc[-1] > df_1h["high"].iloc[-2]) and (recent_rsi[-1] < recent_rsi[-2]):
                    rsi_div_regular = 1

        # 3. MACD features
        macd_div = 0
        if len(df_1h) >= 26:
            ema12 = ema(df_1h, period=12)
            ema26 = ema(df_1h, period=26)
            macd = ema12 - ema26
            if len(macd) >= 5:
                recent_macd = macd.iloc[-5:].values
                if len(recent_macd) >= 2 and not np.isnan(recent_macd[-1]):
                    if (df_1h["high"].iloc[-1] > df_1h["high"].iloc[-2]) and (recent_macd[-1] < recent_macd[-2]):
                        macd_div = 1
                    elif (df_1h["low"].iloc[-1] < df_1h["low"].iloc[-2]) and (recent_macd[-1] > recent_macd[-2]):
                        macd_div = 1

        # 4. Sentiment (placeholder)
        sentiment_fg = 50

        # 5. MTF bias (placeholder)
        mtf_bias = 0

        # 6. Confluence score (normalized)
        confluence_score = setup_result.get("score", 0) / 100.0

        # 7. Entry-to-SL distance
        entry = setup_result.get("entry", 0)
        sl = setup_result.get("sl", entry)
        atr_val = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else 1.0
        if atr_val > 0:
            entry_to_sl = abs(entry - sl) / atr_val
        else:
            entry_to_sl = 1.5

        # 8. Direction encoded
        direction_encoded = 1.0 if direction == "LONG" else -1.0

        # 9. Duration (expected, use default)
        duration_candles = 5  # Placeholder

        # 10. Regime one-hot
        regime_CHOPPY = 1.0 if regime == "CHOPPY" else 0.0
        regime_NORMAL = 1.0 if regime == "NORMAL" else 0.0
        regime_VOLATILE = 1.0 if regime == "VOLATILE" else 0.0

        # Build feature dataframe
        features = {
            'vol_ratio': [vol_ratio],
            'rsi_strength': [rsi_strength],
            'rsi_div_regular': [rsi_div_regular],
            'rsi_div_hidden': [rsi_div_hidden],
            'macd_div': [macd_div],
            'sentiment_fg': [sentiment_fg],
            'mtf_bias': [mtf_bias],
            'confluence_score': [confluence_score],
            'entry_to_sl': [entry_to_sl],
            'direction_encoded': [direction_encoded],
            'duration_candles': [duration_candles],
            'regime_CHOPPY': [regime_CHOPPY],
            'regime_NORMAL': [regime_NORMAL],
            'regime_VOLATILE': [regime_VOLATILE],
        }

        df_features = pd.DataFrame(features)
        return df_features

    except Exception as e:
        print(f"[_extract_ml_features] WARNING: {e}")
        return None


# ─── ML-Enhanced Scoring ──────────────────────────────────────────────────────

def score_setup_with_ml(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    df_15m: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Score a setup with ML enhancement.

    Step 1: Get base confluence score
    Step 2: Extract ML features
    Step 3: Predict win probability with RF
    Step 4: Boost score based on confidence
    Step 5: Return enhanced result

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        df_1h: 1h candles (required, used only for feature extraction)
        df_4h: 4h candles (optional, not used in this version)
        df_15m: 15m candles (optional, not used in this version)

    Returns:
        Dict with:
          - score: Final ML-boosted score (0-100)
          - ml_confidence: P(win) from model (0-1)
          - grade: Grade (A/B/C/D based on score)
          - entry, sl, tp1, tp2, reason: Same as confluence.score_setup()
    """
    # 1. Base confluence score (score_setup fetches candles internally)
    if score_setup is None:
        return {"error": "confluence module not available"}

    base_result = score_setup(symbol)

    # 2. Load model
    model = _load_model()
    if model is None:
        # Fallback to base score if model unavailable
        return base_result

    # 3. Extract ML features
    direction = base_result.get("direction", "NEUTRAL")
    if direction == "NEUTRAL":
        # No trade signal
        return base_result

    features = _extract_ml_features(base_result, df_1h, direction)
    if features is None:
        return base_result

    # 4. Predict win probability
    try:
        y_pred_proba = model.predict_proba(features)[:, 1]
        ml_confidence = float(y_pred_proba[0])
    except Exception as e:
        print(f"[score_setup_with_ml] WARNING: Prediction failed: {e}")
        ml_confidence = 0.5  # Neutral if prediction fails

    # 5. Apply ML boost
    base_score = base_result.get("score", 0)

    # Boost formula: final_score = base_score × (0.7 + 0.3 × ml_confidence)
    # - If ml_confidence = 0 (predicted loss): final = base × 0.7 (30% penalty)
    # - If ml_confidence = 1 (predicted win): final = base × 1.0 (no boost)
    # - Prevents overconfidence, acts as confidence modulator
    boosted_score = base_score * (0.7 + 0.3 * ml_confidence)

    # 6. Update result
    result = base_result.copy()
    result["score"] = round(boosted_score, 1)
    result["ml_confidence"] = ml_confidence
    result["ml_boost"] = round(boosted_score - base_score, 1)

    # 7. Recalculate grade based on boosted score
    if boosted_score >= 80:
        result["grade"] = "A"
    elif boosted_score >= 65:
        result["grade"] = "B"
    elif boosted_score >= 50:
        result["grade"] = "C"
    else:
        result["grade"] = "D"

    return result


# ─── CLI Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mexc import get_ohlcv
    from analysis import to_df

    print("=" * 80)
    print("Phase 4.4: ML Integration Test")
    print("=" * 80)

    # Test on one symbol
    symbol = "BTCUSDT"
    print(f"\n[test] Scoring {symbol}...")

    try:
        candles_1h = get_ohlcv(symbol, "1h", 200)
        df_1h = to_df(candles_1h)

        result = score_setup_with_ml(symbol, df_1h)

        print(f"\nResult:")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Grade: {result.get('grade', 'N/A')}")
        print(f"  Direction: {result.get('direction', 'N/A')}")
        print(f"  ML Confidence: {result.get('ml_confidence', 'N/A'):.2%}")
        print(f"  ML Boost: {result.get('ml_boost', 'N/A')} pts")
        print(f"  Entry: {result.get('entry', 'N/A')}")
        print(f"  SL: {result.get('sl', 'N/A')}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
