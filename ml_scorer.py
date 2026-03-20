"""
ml_scorer.py — ML-backed confidence scoring for Phase 4

Loads a trained RandomForest model and scores setups with ML confidence.
Provides graceful fallback if model unavailable.
"""

import pickle
from pathlib import Path
import sys

MODEL_PATH = Path(__file__).parent / "phase4_model.pkl"
_model = None


def load_model():
    """Load trained RandomForest model at startup."""
    global _model
    try:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        print(f"✅ Phase 4 ML model loaded: {MODEL_PATH}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"⚠️  ML model load failed: {e}", file=sys.stderr)
        return False


def score_with_ml(features: dict) -> float:
    """
    Score setup with ML model. Returns confidence 0-1.
    
    Args:
        features: dict with keys like 'regime_encoded', 'vol_ratio', etc.
    
    Returns:
        float: ML confidence probability [0, 1]
    """
    if _model is None:
        return 0.5  # Graceful fallback
    
    try:
        # Extract base features in training order
        regime_encoded = features.get('regime_encoded', 1)  # 0=CHOPPY, 1=NORMAL, 2=TRENDING, 3=VOLATILE
        vol_ratio = features.get('vol_ratio', 1.0)
        rsi_strength = features.get('rsi_strength', 50)
        rsi_div_regular = features.get('rsi_div_regular', 0)
        rsi_div_hidden = features.get('rsi_div_hidden', 0)
        macd_div = features.get('macd_div', 0)
        sentiment_fg = features.get('sentiment_fg', 50)
        mtf_bias = features.get('mtf_bias', 0)
        confluence_score = features.get('confluence_score', 0.5)
        entry_to_sl = features.get('entry_to_sl', 1.5)
        direction_encoded = features.get('direction_encoded', 0)
        duration_candles = features.get('duration_candles', 4)
        
        # One-hot encode regime (4 features: CHOPPY, NORMAL, TRENDING, VOLATILE)
        regime_choppy = 1 if regime_encoded == 0 else 0
        regime_normal = 1 if regime_encoded == 1 else 0
        regime_trending = 1 if regime_encoded == 2 else 0
        regime_volatile = 1 if regime_encoded == 3 else 0
        
        # Build feature list in exact training order (14 features)
        feature_list = [
            vol_ratio,
            rsi_strength,
            rsi_div_regular,
            rsi_div_hidden,
            macd_div,
            sentiment_fg,
            mtf_bias,
            confluence_score,
            entry_to_sl,
            direction_encoded,
            duration_candles,
            regime_choppy,
            regime_normal,
            regime_trending,
            # regime_volatile is dropped (drop_first=True in training)
        ]
        
        # Handle NaN/inf: replace with 0 if non-numeric or NaN
        feature_list = [
            0 if not isinstance(v, (int, float)) or v != v
            else v for v in feature_list
        ]
        
        # Get probability for class 1 (win)
        prob = _model.predict_proba([feature_list])[0][1]
        return float(prob)
    except Exception as e:
        print(f"⚠️  ML scoring error: {e}", file=sys.stderr)
        return 0.5


def extract_features_from_setup(setup: dict, df_1h=None) -> dict:
    """
    Extract all 14 features from a confluence setup dict (matching model training).
    
    Args:
        setup: Result dict from score_setup()
        df_1h: 1h OHLCV dataframe (optional, for future enhancements)
    
    Returns:
        dict with all required keys for score_with_ml()
    """
    features = {}
    
    # 1. Regime (CHOPPY=0, NORMAL=1, TRENDING=2, VOLATILE=3)
    regime_info = setup.get('regime_info', {})
    regime = regime_info.get('regime', 'NORMAL')
    regime_map = {'CHOPPY': 0, 'NORMAL': 1, 'TRENDING': 2, 'VOLATILE': 3}
    features['regime_encoded'] = regime_map.get(regime, 1)
    
    # 2. Volatility ratio (vol_ratio)
    features['vol_ratio'] = regime_info.get('vol_ratio', 1.0)
    
    # 3. RSI strength (current RSI value, 0-100)
    features['rsi_strength'] = setup.get('rsi_current', 50)
    
    # 4. RSI divergence present (0 or 1)
    confluence_str = str(setup.get('confluence_reasons', []))
    features['rsi_div_regular'] = 1 if 'RSI divergence' in confluence_str else 0
    
    # 5. Hidden divergence present (0 or 1)
    features['rsi_div_hidden'] = 1 if 'Hidden divergence' in confluence_str else 0
    
    # 6. MACD divergence present (0 or 1)
    features['macd_div'] = 1 if 'MACD' in confluence_str else 0
    
    # 7. Sentiment adjustment (fear/greed index, 0-100)
    features['sentiment_fg'] = setup.get('sentiment_adjustment', 50)
    
    # 8. MTF bias (-1=SHORT, 0=NEUTRAL, 1=LONG)
    direction = setup.get('direction', 'NO_TRADE')
    if direction == 'LONG':
        features['mtf_bias'] = 1
    elif direction == 'SHORT':
        features['mtf_bias'] = -1
    else:
        features['mtf_bias'] = 0
    
    # 9. Confluence score (normalized, 0-1)
    features['confluence_score'] = min(setup.get('score', 50) / 100.0, 1.0)
    
    # 10. Entry to SL distance (in price units / ATR)
    try:
        entry = setup.get('optimal_entry', 0)
        sl = setup.get('stop_loss', entry)
        atr = setup.get('atr', 1)
        if atr > 0 and entry > 0:
            features['entry_to_sl'] = abs(entry - sl) / atr
        else:
            features['entry_to_sl'] = 1.5
    except:
        features['entry_to_sl'] = 1.5
    
    # 11. Direction encoded (-1=SHORT, 0=NO_TRADE, 1=LONG)
    if direction == 'LONG':
        features['direction_encoded'] = 1.0
    elif direction == 'SHORT':
        features['direction_encoded'] = -1.0
    else:
        features['direction_encoded'] = 0.0
    
    # 12. Duration (estimated candles in hold, default 4)
    features['duration_candles'] = 4
    
    return features


# Initialize on module import
load_model()
