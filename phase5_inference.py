"""
phase5_inference.py — Live inference for Phase 5 ML engine.

Load trained regressor + compute latest features → predict return & direction.

Usage:
    from phase5_inference import Phase5Inference
    inf = Phase5Inference()
    signal = inf.infer(symbol="BTCUSDT")  # → {direction, return, strength, ...}
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_feature_builder import Phase5FeatureBuilder
from phase5_regressor import Phase5Regressor

try:
    from confluence import score_symbol
except ImportError:
    score_symbol = None


class Phase5Inference:
    """
    Live inference wrapper.
    
    Steps:
      1. Load trained model & feature names
      2. Fetch latest OHLCV for symbol
      3. Compute features
      4. Predict forward return
      5. Gate: only signal if confluence > 40 AND |return| > threshold
    """
    
    def __init__(
        self,
        model_path: str = "models/phase5_reg.pkl",
        features_path: str = "models/phase5_reg_features.pkl",
    ):
        self.model = None
        self.feature_names = None
        self.feature_builder = Phase5FeatureBuilder()
        
        if os.path.exists(model_path) and os.path.exists(features_path):
            self.load_model(model_path, features_path)
        else:
            print(f"⚠️  Model not found at {model_path}. Train first with phase5_train.py")
    
    def load_model(self, model_path: str, features_path: str):
        """Load trained model and feature names."""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            
            with open(features_path, "rb") as f:
                self.feature_names = pickle.load(f)
            
            print(f"✅ Model loaded ({len(self.feature_names)} features)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def infer(
        self,
        df: pd.DataFrame,
        confluence_score: Optional[float] = None,
        min_confluence: float = 40.0,
        min_return_threshold: float = 0.002,
    ) -> dict:
        """
        Make live inference on latest bar.
        
        Args:
            df: DataFrame with latest OHLCV + features
            confluence_score: Existing Phase 1-3 confluence score (optional)
            min_confluence: Min confluence to allow signal
            min_return_threshold: Min predicted return to trade
        
        Returns: {
            "symbol": str,
            "direction": str,  # "LONG", "SHORT", "NONE"
            "predicted_return": float,  # %
            "signal_strength": float,  # abs(return)
            "confluence": float,  # Phase 1-3 score
            "should_trade": bool,  # confluence > threshold AND |return| > min
            "timestamp": datetime,
            "reason": str,  # Why signal was rejected (if any)
        }
        """
        if self.model is None:
            return {
                "direction": "NONE",
                "predicted_return": 0.0,
                "signal_strength": 0.0,
                "should_trade": False,
                "reason": "Model not loaded",
            }
        
        # Extract symbol
        symbol = df.get("symbol", [""])[0] if "symbol" in df.columns else ""
        
        # Get confluence if available
        if confluence_score is None and "confluence" in df.columns:
            confluence_score = df["confluence"].iloc[-1]
        
        confluence_score = confluence_score or 0.0
        
        # Predict
        try:
            # Get latest row
            df_latest = df.iloc[-1:].copy()
            
            # Select features
            feature_cols = [c for c in self.feature_names if c in df.columns]
            X = df[feature_cols].iloc[-1:].copy()
            
            # Fill NaN
            X = X.fillna(0)
            
            # Predict
            pred = self.model.predict(X)[0]
            signal_strength = abs(pred)
            
            # Determine direction
            if pred > min_return_threshold:
                direction = "LONG"
            elif pred < -min_return_threshold:
                direction = "SHORT"
            else:
                direction = "NONE"
            
            # Gate on confluence
            should_trade = (confluence_score >= min_confluence) and (signal_strength > min_return_threshold)
            
            # Rejection reason
            if not should_trade:
                if confluence_score < min_confluence:
                    reason = f"Confluence {confluence_score:.0f} < {min_confluence:.0f}"
                else:
                    reason = f"Return {signal_strength:.4%} < threshold {min_return_threshold:.4%}"
            else:
                reason = "Signal approved"
            
            return {
                "symbol": symbol,
                "direction": direction,
                "predicted_return": float(pred),
                "signal_strength": float(signal_strength),
                "confluence": float(confluence_score),
                "should_trade": should_trade,
                "timestamp": datetime.now(),
                "reason": reason,
            }
        
        except Exception as e:
            return {
                "direction": "NONE",
                "predicted_return": 0.0,
                "signal_strength": 0.0,
                "confluence": confluence_score,
                "should_trade": False,
                "timestamp": datetime.now(),
                "reason": f"Inference error: {e}",
            }
    
    def format_signal(self, result: dict) -> str:
        """Format inference result as human-readable string."""
        lines = [
            f"Symbol: {result.get('symbol', 'N/A')}",
            f"Direction: {result['direction']} ({result['predicted_return']:+.2%})",
            f"Signal Strength: {result['signal_strength']:.4%}",
            f"Confluence: {result['confluence']:.0f}",
            f"Should Trade: {'✅ YES' if result['should_trade'] else '❌ NO'}",
            f"Reason: {result['reason']}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("[Phase5 Inference] Testing on labeled_4h.csv...")
    
    csv_path = "data/processed/labeled_4h.csv"
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Build features
    fb = Phase5FeatureBuilder()
    df_features = fb.build(df)
    
    # Inference
    inf = Phase5Inference()
    
    if inf.model is not None:
        result = inf.infer(df_features)
        print("\n" + "=" * 60)
        print(inf.format_signal(result))
        print("=" * 60)
    else:
        print("⚠️  Model not trained. Run phase5_train.py first.")
