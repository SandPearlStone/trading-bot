#!/usr/bin/env python3
"""
Phase 4.1: Feature Engineering Pipeline

Extract features from backtest trades and build training dataset for Random Forest.

Input: backtest results (from backtest.py)
Output: features.csv (trades × features matrix for ML training)

Feature list:
  - regime: CHOPPY/NORMAL/VOLATILE (one-hot encoded: regime_choppy, regime_normal, regime_volatile)
  - vol_ratio: volatility ratio (ATR_current / ATR_MA)
  - rsi_strength: RSI distance from midline (50) normalized to 0-1
  - rsi_div_regular: 0/1 (RSI divergence present)
  - rsi_div_hidden: 0/1 (hidden RSI divergence present)
  - macd_div: 0/1 (MACD divergence present)
  - sentiment_fg: Fear & Greed index 0-100 (normalized to 0-1)
  - mtf_bias: -1 (bearish) / 0 (neutral) / 1 (bullish)
  - confluence_score: raw confluence score 0-100
  - entry_to_sl: distance from entry to stop loss (in ATR units)
  - direction_encoded: 1 (LONG) / -1 (SHORT)
  - r_multiple: actual R/R outcome
  - outcome_win: 1 (win) / 0 (loss) — TARGET VARIABLE

Usage:
  python3 phase4_feature_engineering.py
  # Reads trades from backtest(symbol) for each symbol in BACKTEST_SYMBOLS
  # Outputs features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import sys

# Local imports
try:
    from backtest import backtest
    from analysis import to_df, atr as calc_atr, ema, rsi
    from mexc import get_ohlcv
except ImportError as e:
    print(f"[feature_engineering] Import error: {e}", file=sys.stderr)
    sys.exit(1)


# ─── Configuration ─────────────────────────────────────────────────────────

BACKTEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ARBUSDT",
    "OPUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "DOGEUSDT",
]

LOOKBACK_DAYS = 90
INTERVAL = "1h"

OUTPUT_CSV = Path(__file__).parent / "phase4_features.csv"


# ─── Regime Detection ───────────────────────────────────────────────────────

def _classify_regime(df: pd.DataFrame, atr_period: int = 20) -> Tuple[str, float]:
    """
    Classify market regime as CHOPPY / NORMAL / VOLATILE based on ATR.

    Returns: (regime, vol_ratio)
    """
    try:
        atr_series = calc_atr(df, period=atr_period)
    except Exception as e:
        return "NORMAL", 1.0
    
    if atr_series.isna().all():
        return "NORMAL", 1.0
    
    current_atr = atr_series.iloc[-1]
    hist_atr = atr_series.iloc[-50:-1].mean() if len(atr_series) > 50 else atr_series.mean()

    if np.isnan(current_atr) or np.isnan(hist_atr) or hist_atr == 0:
        return "NORMAL", 1.0

    vol_ratio = current_atr / hist_atr

    if vol_ratio < 0.8:
        regime = "CHOPPY"
    elif vol_ratio > 1.2:
        regime = "VOLATILE"
    else:
        regime = "NORMAL"

    return regime, vol_ratio


# ─── RSI Features ──────────────────────────────────────────────────────────

def _extract_rsi_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract RSI-based features:
      - rsi_strength: normalized distance from midline (0-1)
      - rsi_div_regular: 0/1 (simple divergence detection)
      - rsi_div_hidden: 0/1 (hidden divergence detection)
    """
    rsi_vals = rsi(df, period=14)
    features = {
        "rsi_strength": 0.0,
        "rsi_div_regular": 0,
        "rsi_div_hidden": 0,
    }

    # RSI strength: distance from 50 (midline), normalized to 0-1
    if not rsi_vals.isna().all():
        last_rsi = rsi_vals.iloc[-1]
        if not np.isnan(last_rsi):
            # Distance from 50, normalized: 0 at midline, 0.5 at extremes (0 or 100)
            features["rsi_strength"] = abs(last_rsi - 50) / 50

    # Simple RSI divergence (last 5 candles)
    if len(rsi_vals) >= 5:
        recent_rsi = rsi_vals.iloc[-5:].values
        recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
        recent_highs = df["high"].iloc[-5:].values
        recent_lows = df["low"].iloc[-5:].values

        if len(recent_rsi) >= 2:
            # Bullish div: lower low in price, higher low in RSI
            if (recent_lows[-1] < recent_lows[-2]) and (recent_rsi[-1] > recent_rsi[-2]):
                features["rsi_div_regular"] = 1

            # Bearish div: higher high in price, lower high in RSI
            if (recent_highs[-1] > recent_highs[-2]) and (recent_rsi[-1] < recent_rsi[-2]):
                features["rsi_div_regular"] = 1

    # Hidden divergence (simplified: same direction as trend but RSI contracting)
    # For full version, integrate divergence_detector.py
    features["rsi_div_hidden"] = 0

    return features


# ─── MACD Divergence Feature ───────────────────────────────────────────────

def _extract_macd_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract MACD-based features:
      - macd_div: 0/1 (MACD divergence present)
    """
    # Simplified MACD: EMA(12) - EMA(26)
    if len(df) < 26:
        return {"macd_div": 0}

    ema12 = ema(df, period=12)
    ema26 = ema(df, period=26)
    macd = ema12 - ema26

    features = {"macd_div": 0}

    if len(macd) >= 5:
        recent_macd = macd.iloc[-5:].values
        recent_highs = df["high"].iloc[-5:].values
        recent_lows = df["low"].iloc[-5:].values

        # Simple divergence: price makes higher high but MACD makes lower high
        if len(recent_macd) >= 2 and not np.isnan(recent_macd[-1]):
            if (recent_highs[-1] > recent_highs[-2]) and (recent_macd[-1] < recent_macd[-2]):
                features["macd_div"] = 1
            elif (recent_lows[-1] < recent_lows[-2]) and (recent_macd[-1] > recent_macd[-2]):
                features["macd_div"] = 1

    return features


# ─── Sentiment Features (Placeholder) ───────────────────────────────────────

def _extract_sentiment_features() -> Dict[str, float]:
    """
    Extract sentiment features (Fear & Greed Index, BTC dominance).

    For now, use placeholder (random). In production, fetch from API.
    """
    # TODO: Integrate fear_greed_index and btc_dominance APIs
    return {
        "sentiment_fg": 50,  # 0-100, midpoint for now
    }


# ─── Multi-Timeframe Bias (Placeholder) ─────────────────────────────────────

def _extract_mtf_features() -> Dict[str, float]:
    """
    Extract multi-timeframe bias (4h, 1h, 15m agreement).

    For now, use placeholder. In production, fetch and analyze 4h/15m.
    """
    # TODO: Integrate mtf_analysis
    return {
        "mtf_bias": 0.0,  # -1 bearish, 0 neutral, 1 bullish
    }


# ─── Feature Extraction for One Trade ───────────────────────────────────────

def _extract_features_for_trade(
    trade_record: Dict,
    context_df: pd.DataFrame,
    symbol: str,
) -> Optional[Dict]:
    """
    Extract all features for a single trade.

    Input: trade_record from backtest (with entry, sl, tp, atr, outcome, r_multiple)
    Output: Dict with all features + outcome_win (target)
    """
    if len(context_df) < 20:
        return None

    features = {"symbol": symbol}

    # 1. Regime
    regime, vol_ratio = _classify_regime(context_df)
    features["regime"] = regime
    features["vol_ratio"] = vol_ratio

    # 2. RSI
    rsi_feats = _extract_rsi_features(context_df)
    features.update(rsi_feats)

    # 3. MACD
    macd_feats = _extract_macd_features(context_df)
    features.update(macd_feats)

    # 4. Sentiment (placeholder)
    sentiment_feats = _extract_sentiment_features()
    features.update(sentiment_feats)

    # 5. Multi-timeframe bias (placeholder)
    mtf_feats = _extract_mtf_features()
    features.update(mtf_feats)

    # 6. Confluence score (from backtest)
    features["confluence_score"] = trade_record.get("score", 0) / 100.0  # normalize 0-1

    # 7. Entry/SL distance (in ATR units)
    atr_val = trade_record.get("atr", 1.0)
    if atr_val > 0:
        features["entry_to_sl"] = abs(trade_record["entry"] - trade_record["sl"]) / atr_val
    else:
        features["entry_to_sl"] = 1.5  # default

    # 8. Direction encoded
    features["direction_encoded"] = 1.0 if trade_record["direction"] == "LONG" else -1.0

    # 9. Actual outcome (TARGET)
    features["outcome_win"] = 1 if trade_record["outcome"] == "win" else 0

    # 10. Duration (candles until close)
    features["duration_candles"] = trade_record.get("exit_candle", 1) - trade_record.get(
        "candle_idx", 0
    )

    # 11. Risk/reward multiple
    features["r_multiple"] = trade_record.get("r_multiple", 0.0)

    return features


# ─── Main Pipeline ─────────────────────────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    """
    Run backtest on all symbols, extract features, return combined DataFrame.
    """
    all_features = []

    for symbol in BACKTEST_SYMBOLS:
        print(f"[feature_engineering] Processing {symbol}...")

        try:
            # Run backtest
            backtest_result = backtest(symbol, interval=INTERVAL, lookback_days=LOOKBACK_DAYS, verbose=False)

            if "error" in backtest_result:
                print(f"  WARNING: {backtest_result['error']}")
                continue

            trades = backtest_result.get("trades", [])
            if not trades:
                print(f"  WARNING: No trades generated for {symbol}")
                continue

            # Fetch full candle history for context
            total_needed = math.ceil(LOOKBACK_DAYS * 24 * 60 / 60) + 200  # 90 days + context window
            all_candles = get_ohlcv(symbol, INTERVAL, min(total_needed, 1000))

            print(f"  Extracted {len(trades)} trades, {len(all_candles)} candles")

            # For each trade, extract features
            for trade in trades:
                candle_idx = trade["candle_idx"]

                # Get context window (200 candles before trade)
                context_start = max(0, candle_idx - 200)
                context_candles = all_candles[context_start:candle_idx]

                if len(context_candles) < 20:
                    continue

                # Convert to DataFrame for analysis
                df_ctx = to_df(context_candles)

                # Extract features
                features = _extract_features_for_trade(trade, df_ctx, symbol)
                if features:
                    all_features.append(features)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all features into DataFrame
    df = pd.DataFrame(all_features)
    print(f"\n[feature_engineering] Total trades processed: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDataFrame head:\n{df.head()}")
    print(f"\nTarget distribution (outcome_win):\n{df['outcome_win'].value_counts() if len(df) > 0 else 'N/A'}")

    return df


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("Phase 4.1: Feature Engineering")
    print("=" * 80)

    df_features = build_feature_matrix()

    if len(df_features) > 0:
        # Save to CSV
        df_features.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved {len(df_features)} trades × {len(df_features.columns)} features to {OUTPUT_CSV}")
    else:
        print("\n❌ No features extracted. Check backtest data.")
