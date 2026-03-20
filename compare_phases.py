#!/usr/bin/env python3
"""
compare_phases.py — Backtest Phase 1 vs Phase 2 trading systems (90-day historical)

Compares:
  Phase1Backend: Static min_score=65, NO regime gating, NO hidden divergence
  Phase2Backend: Adaptive min_score per regime, WITH hidden divergence bonus

Architecture:
  - Phase1Backend & Phase2Backend: Scoring engines
  - BacktestEngine: Walk-forward simulator (200-candle context)
  - StatsCalculator: Win rate, drawdown, Sharpe, avg R, profit factor
  - Output: CSV + formatted console table

Usage:
  python3 compare_phases.py
  # Output: compare_phases_results.csv
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

DB_PATH = Path(__file__).parent / "data" / "trades.db"

# ─── SIMPLIFIED ANALYSIS FUNCTIONS ──────────────────────────────────────────

def calculate_rsi(series, period=14):
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_ema(series, period=21):
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()

def detect_trend(df, ema_period=21):
    """Detect trend direction based on EMA."""
    ema = calculate_ema(df['close'], ema_period)
    close = df['close'].iloc[-1]
    ema_val = ema.iloc[-1]
    
    if close > ema_val:
        return "LONG"
    elif close < ema_val:
        return "SHORT"
    else:
        return "NEUTRAL"

def detect_rsi_divergence(df, rsi_period=14):
    """Simple RSI divergence detection."""
    rsi = calculate_rsi(df['close'], rsi_period)
    
    if len(df) < 10:
        return None, 0
    
    # Last 5 candles
    recent_highs = df['high'].iloc[-5:].values
    recent_lows = df['low'].iloc[-5:].values
    recent_rsi = rsi.iloc[-5:].values
    
    recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
    if len(recent_rsi) < 2:
        return None, 0
    
    # Check for bullish divergence (lower low in price, higher low in RSI)
    if recent_lows[-1] < recent_lows[-2] and recent_rsi[-1] > recent_rsi[-2]:
        return "bullish", 1
    
    # Check for bearish divergence (higher high in price, lower high in RSI)
    if recent_highs[-1] > recent_highs[-2] and recent_rsi[-1] < recent_rsi[-2]:
        return "bearish", 1
    
    return None, 0

def classify_volatility_regime(df, atr_period=20):
    """Classify market regime based on volatility."""
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    
    current_atr = atr.iloc[-1]
    hist_atr = atr.iloc[-50:-1].mean() if len(atr) > 50 else atr.mean()
    
    if np.isnan(current_atr) or np.isnan(hist_atr) or hist_atr == 0:
        return "NORMAL", 1.0
    
    vol_ratio = current_atr / hist_atr
    
    if vol_ratio < 0.8:
        return "CHOPPY", vol_ratio
    elif vol_ratio > 1.2:
        return "VOLATILE", vol_ratio
    else:
        return "NORMAL", vol_ratio

# ─── PHASE 1 BACKEND (Static, No Regime, No Hidden Divergence) ──────────────

class Phase1Backend:
    """Phase 1: Static min_score=65, confluence only, no adaptations."""
    
    def __init__(self):
        self.min_score = 65
        self.name = "Phase1"
    
    def score_setup(self, symbol: str, df_1h: pd.DataFrame) -> Dict:
        """
        Score a setup using Phase 1 logic (simplified).
        
        Returns: {score, direction, sl, tp1, tp2}
        """
        if len(df_1h) < 20:
            return {
                "score": 0,
                "direction": "NO_TRADE",
                "sl": None,
                "tp1": None,
                "regime": "STATIC"
            }
        
        # Phase 1 scoring components (simplified)
        score = 0
        
        # 1. Trend alignment (30 points)
        trend = detect_trend(df_1h, ema_period=21)
        if trend == "LONG":
            score += 30
            direction = "LONG"
        elif trend == "SHORT":
            score += 30
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        # 2. RSI position (20 points)
        rsi = calculate_rsi(df_1h['close'], 14)
        rsi_val = rsi.iloc[-1]
        
        if direction == "LONG" and not np.isnan(rsi_val):
            if 30 <= rsi_val <= 70:
                score += 20
            elif rsi_val < 30:
                score += 15  # Oversold, good for entry
        elif direction == "SHORT" and not np.isnan(rsi_val):
            if 30 <= rsi_val <= 70:
                score += 20
            elif rsi_val > 70:
                score += 15  # Overbought, good for entry
        
        # 3. Volatility structure (15 points)
        atr = calculate_atr(df_1h['high'], df_1h['low'], df_1h['close'], 14)
        atr_val = atr.iloc[-1]
        
        if not np.isnan(atr_val) and atr_val > 0:
            score += 15
        
        # 4. Price structure (10 points)
        if len(df_1h) >= 5:
            recent_highs = df_1h['high'].iloc[-5:].max()
            recent_lows = df_1h['low'].iloc[-5:].min()
            range_val = recent_highs - recent_lows
            if range_val > 0:
                score += 10
        
        # 5. Volume (10 points)
        if len(df_1h) >= 2:
            vol_ratio = df_1h['volume'].iloc[-1] / df_1h['volume'].iloc[-5:].mean()
            if vol_ratio > 0.8:
                score += 10
        
        # 6. NO hidden divergence in Phase 1 (score stays static)
        
        # Calculate SL and TP based on ATR
        current_price = df_1h['close'].iloc[-1]
        atr_val = atr.iloc[-1]
        
        if np.isnan(atr_val) or atr_val == 0:
            sl = None
            tp1 = None
        else:
            if direction == "LONG":
                sl = current_price - (2 * atr_val)
                tp1 = current_price + (3 * atr_val)
            elif direction == "SHORT":
                sl = current_price + (2 * atr_val)
                tp1 = current_price - (3 * atr_val)
            else:
                sl = None
                tp1 = None
        
        return {
            "score": min(100, score),
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "regime": "STATIC"
        }
    
    def should_enter(self, setup: Dict) -> bool:
        """Check if score meets Phase 1 threshold."""
        return setup["score"] >= self.min_score and setup["direction"] != "NO_TRADE"


# ─── PHASE 2 BACKEND (Adaptive, Regime-Gated, With Hidden Divergence) ────────

class Phase2Backend:
    """Phase 2: Adaptive min_score per regime, hidden divergence bonus."""
    
    def __init__(self):
        self.name = "Phase2"
    
    def score_setup(self, symbol: str, df_1h: pd.DataFrame) -> Dict:
        """
        Score a setup using Phase 2 logic.
        
        Returns: {score, direction, sl, tp1, regime, adaptive_threshold}
        """
        if len(df_1h) < 20:
            return {
                "score": 0,
                "direction": "NO_TRADE",
                "sl": None,
                "tp1": None,
                "regime": "UNKNOWN",
                "adaptive_threshold": 65
            }
        
        # Get base score (same as Phase 1)
        phase1 = Phase1Backend()
        base_setup = phase1.score_setup(symbol, df_1h)
        base_score = base_setup["score"]
        direction = base_setup["direction"]
        sl = base_setup["sl"]
        tp1 = base_setup["tp1"]
        
        # Phase 2 addition: Regime detection
        regime, vol_ratio = classify_volatility_regime(df_1h, atr_period=20)
        
        # Adaptive threshold
        adaptive_threshold = {
            "CHOPPY": 85,
            "NORMAL": 65,
            "VOLATILE": 70
        }.get(regime, 65)
        
        # Phase 2 addition: Hidden divergence bonus
        hidden_div_type, hidden_div_bonus = detect_rsi_divergence(df_1h, rsi_period=14)
        
        # Apply bonus if divergence aligns with direction
        phase2_score = base_score
        if hidden_div_type == "bullish" and direction == "LONG":
            phase2_score += hidden_div_bonus * 8  # +8 points max
        elif hidden_div_type == "bearish" and direction == "SHORT":
            phase2_score += hidden_div_bonus * 8
        
        return {
            "score": min(100, phase2_score),
            "direction": direction,
            "sl": sl,
            "tp1": tp1,
            "regime": regime,
            "adaptive_threshold": adaptive_threshold,
            "base_score": base_score,
            "hidden_div_bonus": hidden_div_bonus * 8
        }
    
    def should_enter(self, setup: Dict) -> bool:
        """Check if Phase 2 conditions are met."""
        # Skip choppy markets
        if setup.get("regime") == "CHOPPY":
            return False
        
        threshold = setup.get("adaptive_threshold", 65)
        return setup["score"] >= threshold and setup["direction"] != "NO_TRADE"


# ─── PHASE 4 BACKEND (Phase 2 + ML Confidence Scoring) ───────────────────────

class Phase4Backend:
    """Phase 4: Phase 2 + ML-enhanced confidence scoring."""
    
    def __init__(self):
        self.name = "Phase4"
    
    def score_setup(self, symbol: str, df_1h: pd.DataFrame) -> Dict:
        """
        Score a setup using Phase 4 logic (Phase 2 + ML confidence).
        
        Returns: Phase 2 result with added ml_confidence and blended score
        """
        try:
            # Import Phase 4 ML scorer (graceful fallback if unavailable)
            from confluence import score_setup_with_ml
            # Note: score_setup_with_ml expects symbol, tf, htf params
            # For backtest context, we use simplified scoring
            
            # Get Phase 2 baseline
            phase2 = Phase2Backend()
            base_setup = phase2.score_setup(symbol, df_1h)
            
            # Try to enhance with ML if available
            try:
                from ml_scorer import score_with_ml, extract_features_from_setup
                
                # Create minimal setup dict for feature extraction
                features = extract_features_from_setup(base_setup, df_1h)
                ml_confidence = score_with_ml(features)
                
                # Blend: final = base × (0.7 + 0.3 × ml_prob)
                raw_score = base_setup["score"]
                multiplier = 0.7 + (0.3 * ml_confidence)
                final_score = raw_score * multiplier
                
                base_setup["score"] = min(100, final_score)
                base_setup["ml_confidence"] = ml_confidence
                base_setup["raw_score"] = raw_score
                base_setup["ml_available"] = True
            except:
                # Fallback to Phase 2 if ML unavailable
                base_setup["ml_available"] = False
                base_setup["ml_confidence"] = 0.5
            
            return base_setup
        
        except Exception as e:
            # Graceful fallback to Phase 2
            phase2 = Phase2Backend()
            return phase2.score_setup(symbol, df_1h)
    
    def should_enter(self, setup: Dict) -> bool:
        """Check if Phase 4 conditions are met."""
        # Skip choppy markets (Phase 2 rule)
        if setup.get("regime") == "CHOPPY":
            return False
        
        # Phase 2 threshold check
        threshold = setup.get("adaptive_threshold", 65)
        if setup["score"] < threshold or setup["direction"] == "NO_TRADE":
            return False
        
        # Phase 4 addition: Skip if ML confidence is very low
        if setup.get("ml_available"):
            ml_conf = setup.get("ml_confidence", 0.5)
            if ml_conf < 0.35:  # Reject if ML confidence too low
                return False
        
        return True


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────

class BacktestEngine:
    """Walk-forward simulator (200-candle context window)."""
    
    def __init__(self, phase_backend, symbol: str, candles: List[Dict], lookback: int = 200):
        self.backend = phase_backend
        self.symbol = symbol
        self.candles = candles
        self.lookback = lookback
        self.trades = []
    
    def run(self) -> List[Dict]:
        """
        Simulate trades using walk-forward logic.
        """
        if len(self.candles) < self.lookback:
            return []
        
        # Convert to DataFrame for analysis functions
        df_full = pd.DataFrame(self.candles)
        for col in ["open", "high", "low", "close", "volume"]:
            df_full[col] = df_full[col].astype(float)
        df_full['time'] = pd.to_datetime(df_full['time'])
        
        # Walk forward
        for i in range(self.lookback, len(self.candles)):
            # Context: last 200 candles + current
            context_start = max(0, i - self.lookback)
            df_context = df_full.iloc[context_start:i+1].copy().reset_index(drop=True)
            
            current_price = float(self.candles[i]["close"])
            
            # Score setup
            setup = self.backend.score_setup(self.symbol, df_context)
            
            # Check entry conditions
            if not self.backend.should_enter(setup):
                continue
            
            # Create trade
            entry_price = current_price
            entry_time = self.candles[i]["time"]
            direction = setup["direction"]
            sl = setup["sl"]
            tp1 = setup["tp1"]
            
            if sl is None or tp1 is None:
                continue
            
            # Simulate trade until exit (next 50 candles max)
            trade = {
                "entry_price": entry_price,
                "entry_time": entry_time,
                "direction": direction,
                "sl": sl,
                "tp1": tp1,
                "exit_price": None,
                "exit_time": None,
                "pnl_pct": None,
                "exit_type": None,
                "setup_score": setup["score"]
            }
            
            # Track exit
            for j in range(i+1, min(i+50, len(self.candles))):
                candle = self.candles[j]
                high = float(candle["high"])
                low = float(candle["low"])
                close = float(candle["close"])
                
                if direction == "LONG":
                    # TP hit
                    if high >= tp1:
                        trade["exit_price"] = tp1
                        trade["exit_time"] = candle["time"]
                        trade["exit_type"] = "TP"
                        trade["pnl_pct"] = (tp1 - entry_price) / entry_price * 100
                        break
                    # SL hit
                    elif low <= sl:
                        trade["exit_price"] = sl
                        trade["exit_time"] = candle["time"]
                        trade["exit_type"] = "SL"
                        trade["pnl_pct"] = (sl - entry_price) / entry_price * 100
                        break
                else:  # SHORT
                    # TP hit
                    if low <= tp1:
                        trade["exit_price"] = tp1
                        trade["exit_time"] = candle["time"]
                        trade["exit_type"] = "TP"
                        trade["pnl_pct"] = (entry_price - tp1) / entry_price * 100
                        break
                    # SL hit
                    elif high >= sl:
                        trade["exit_price"] = sl
                        trade["exit_time"] = candle["time"]
                        trade["exit_type"] = "SL"
                        trade["pnl_pct"] = (entry_price - sl) / entry_price * 100
                        break
            
            # If no exit, mark as timeout
            if trade["exit_price"] is None:
                trade["exit_price"] = self.candles[min(i+50, len(self.candles)-1)]["close"]
                trade["exit_time"] = self.candles[min(i+50, len(self.candles)-1)]["time"]
                trade["exit_type"] = "TIMEOUT"
                if direction == "LONG":
                    trade["pnl_pct"] = (trade["exit_price"] - entry_price) / entry_price * 100
                else:
                    trade["pnl_pct"] = (entry_price - trade["exit_price"]) / entry_price * 100
            
            self.trades.append(trade)
        
        return self.trades


# ─── STATS CALCULATOR ────────────────────────────────────────────────────────

class StatsCalculator:
    """Calculate performance metrics."""
    
    @staticmethod
    def calculate(trades: List[Dict]) -> Dict:
        """Calculate win rate, max drawdown, Sharpe ratio, avg R, profit factor."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "avg_r": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "wins": 0,
                "losses": 0
            }
        
        pnls = [t["pnl_pct"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        
        # Win rate
        win_rate = wins / len(trades) * 100 if trades else 0
        
        # Max drawdown (cumulative)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if len(pnls) > 1:
            returns = np.array(pnls)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Average R (risk-reward per trade)
        avg_r = np.mean(pnls) / abs(np.min(pnls)) if losses > 0 and np.min(pnls) != 0 else 0
        
        # Profit factor
        gross_wins = sum(p for p in pnls if p > 0)
        gross_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # Total PnL
        total_pnl = sum(pnls)
        
        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_r": round(avg_r, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "losses": losses
        }


# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────

def load_ohlcv_from_db(symbol: str, timeframe: str = "1h", limit: int = 500) -> List[Dict]:
    """Load OHLCV candles from database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timeframe = ?
        ORDER BY open_time DESC
        LIMIT ?
    ''', (symbol, timeframe, limit))
    
    rows = c.fetchall()
    conn.close()
    
    # Reverse to get chronological order
    candles = []
    for row in reversed(rows):
        candles.append({
            "time": row[0],
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })
    
    return candles


def get_symbols_from_db() -> List[str]:
    """Get list of symbols in database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol')
    symbols = [row[0] for row in c.fetchall()]
    conn.close()
    
    return symbols


def main(with_ml=False):
    """
    Run comparison backtest.
    
    Args:
        with_ml: If True, include Phase 4 (with ML) in comparison
    """
    if with_ml:
        print("=" * 80)
        print("PHASE 1 vs PHASE 2 vs PHASE 4 (ML) COMPARISON BACKTEST")
        print("=" * 80)
    else:
        print("=" * 80)
        print("PHASE 1 vs PHASE 2 COMPARISON BACKTEST")
        print("=" * 80)
    print()
    
    # Get symbols
    symbols = get_symbols_from_db()
    print(f"📊 Testing {len(symbols)} symbols...")
    print()
    
    results = []
    
    # Test each symbol
    for symbol in symbols:
        print(f"🔄 {symbol:12s}", end=" ")
        
        # Load candles
        candles = load_ohlcv_from_db(symbol, timeframe="1h", limit=500)
        if len(candles) < 200:
            print(f"❌ Insufficient data ({len(candles)} candles)")
            continue
        
        # Phase 1 backtest
        phase1 = Phase1Backend()
        engine1 = BacktestEngine(phase1, symbol, candles, lookback=200)
        trades1 = engine1.run()
        stats1 = StatsCalculator.calculate(trades1)
        
        # Phase 2 backtest
        phase2 = Phase2Backend()
        engine2 = BacktestEngine(phase2, symbol, candles, lookback=200)
        trades2 = engine2.run()
        stats2 = StatsCalculator.calculate(trades2)
        
        # Phase 4 backtest (if requested)
        stats4 = {} if not with_ml else None
        if with_ml:
            try:
                phase4 = Phase4Backend()
                engine4 = BacktestEngine(phase4, symbol, candles, lookback=200)
                trades4 = engine4.run()
                stats4 = StatsCalculator.calculate(trades4)
            except Exception as e:
                print(f"\n  [Phase 4 error: {e}]", end=" ")
                stats4 = {}
        
        if with_ml and stats4:
            print(f"P1:{stats1['total_trades']:2d}T|{stats1['win_rate']:5.1f}%  P2:{stats2['total_trades']:2d}T|{stats2['win_rate']:5.1f}%  P4:{stats4.get('total_trades', 0):2d}T|{stats4.get('win_rate', 0):5.1f}%", end=" ")
        else:
            print(f"P1:{stats1['total_trades']:2d}T|{stats1['win_rate']:5.1f}%  P2:{stats2['total_trades']:2d}T|{stats2['win_rate']:5.1f}%", end=" ")
        
        # Compare metrics
        metrics = ["win_rate", "max_drawdown", "sharpe_ratio", "avg_r", "profit_factor", "total_pnl"]
        for metric in metrics:
            p1_val = stats1.get(metric, 0)
            p2_val = stats2.get(metric, 0)
            p4_val = stats4.get(metric, 0) if with_ml and stats4 else None
            
            # Skip if all zero
            if p1_val == 0 and p2_val == 0 and (p4_val is None or p4_val == 0):
                continue
            
            # Determine best performer
            if metric == "max_drawdown":
                # For drawdown, higher (less negative) is better
                values = [("Phase1", p1_val), ("Phase2", p2_val)]
                if with_ml and p4_val is not None:
                    values.append(("Phase4", p4_val))
                winner = max(values, key=lambda x: x[1])[0]
            else:
                # For all other metrics, higher is better
                values = [("Phase1", p1_val), ("Phase2", p2_val)]
                if with_ml and p4_val is not None:
                    values.append(("Phase4", p4_val))
                winner = max(values, key=lambda x: x[1])[0]
            
            # Calculate differences
            if p1_val != 0:
                pct_change_p2 = ((p2_val - p1_val) / abs(p1_val) * 100)
            else:
                pct_change_p2 = 100 if p2_val > 0 else 0
            
            diff_p2 = p2_val - p1_val
            
            result_row = {
                "Symbol": symbol,
                "Metric": metric,
                "Phase1": round(p1_val, 2),
                "Phase2": round(p2_val, 2),
                "Difference": round(diff_p2, 2),
                "%Change": round(pct_change_p2, 2),
                "Winner": winner
            }
            
            # Add Phase 4 if enabled
            if with_ml and p4_val is not None:
                if p1_val != 0:
                    pct_change_p4 = ((p4_val - p1_val) / abs(p1_val) * 100)
                else:
                    pct_change_p4 = 100 if p4_val > 0 else 0
                result_row["Phase4"] = round(p4_val, 2)
                result_row["P4vP1"] = round(p4_val - p1_val, 2)
                result_row["P4%Change"] = round(pct_change_p4, 2)
            
            results.append(result_row)
        
        print("✅")
    
    # Output CSV
    if results:
        df_results = pd.DataFrame(results)
        csv_path = Path(__file__).parent / "compare_phases_results.csv"
        df_results.to_csv(csv_path, index=False)
        print()
        print("=" * 80)
        print(f"📁 Results saved to: {csv_path}")
        print("=" * 80)
        print()
        
        # Print summary
        print("RESULTS BY SYMBOL:")
        print()
        for symbol in df_results['Symbol'].unique():
            symbol_data = df_results[df_results['Symbol'] == symbol]
            p2_wins = (symbol_data['Winner'] == 'Phase2').sum()
            p1_wins = (symbol_data['Winner'] == 'Phase1').sum()
            print(f"  {symbol:12s}: Phase2 wins {p2_wins}/{len(symbol_data)} metrics")
        
        print()
        phase2_total = (df_results["Winner"] == "Phase2").sum()
        phase1_total = (df_results["Winner"] == "Phase1").sum()
        print(f"🏆 OVERALL: Phase 2 wins {phase2_total}/{len(df_results)} comparisons")
        print()
        
        # Show detailed results
        print("DETAILED COMPARISON:")
        print(df_results.to_string(index=False))
    else:
        print("❌ No results to output.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare trading phases backtest")
    parser.add_argument("--with-ml", action="store_true", help="Include Phase 4 (ML-enhanced) in comparison")
    args = parser.parse_args()
    
    main(with_ml=args.with_ml)
