"""
phase5_train.py — Training orchestration for Phase 5 ML engine.

Steps:
  1. Load data from CSV (labeled_4h.csv)
  2. Build features with phase5_feature_builder
  3. Train regressor with walk-forward validation
  4. Compare to Phase 1-3 baseline (confluence score only)
  5. Save model & print results

Usage:
    python3 phase5_train.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_feature_builder import Phase5FeatureBuilder
from phase5_regressor import Phase5Regressor


def load_data(csv_path: str = "data/processed/labeled_4h.csv") -> pd.DataFrame:
    """Load raw data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    
    print(f"[Train] Loading {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def compute_baseline_metrics(df: pd.DataFrame) -> dict:
    """
    Compute Phase 1-3 baseline: confluence score only, no ML.
    
    Baseline signal: confluence > 2.0 (adapts to actual data scale)
    Returns: {signal_count, win_rate, sharpe, max_dd}
    """
    print("\n[Baseline] Computing Phase 1-3 confluence-only signals...")
    
    # Use existing 'confluence' column
    if "confluence" not in df.columns:
        print("  ⚠️  No 'confluence' column in data. Using default metric.")
        return {
            "signal_count": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    
    df = df.copy()
    
    # Label: forward return (with clipping)
    lookahead_bars = 10
    df["forward_return"] = df["close"].pct_change(-lookahead_bars) - 0.001
    df["forward_return"] = df["forward_return"].clip(-0.50, 0.50)
    
    # Baseline signal: confluence > 2.0 (actual scale of data)
    baseline_threshold = 2.0
    baseline_signals = df["confluence"] > baseline_threshold
    signal_count = baseline_signals.sum()
    
    if signal_count == 0:
        print(f"  ⚠️  No confluence > {baseline_threshold} signals found. Baseline metrics: 0")
        return {
            "signal_count": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    
    signal_returns = df[baseline_signals]["forward_return"].dropna()
    
    if len(signal_returns) == 0:
        return {
            "signal_count": signal_count,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    
    signal_returns = signal_returns.values
    win_rate = (signal_returns > 0).sum() / len(signal_returns)
    sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-9)) * np.sqrt(252 * 6)
    
    # Max drawdown
    cumsum = np.cumsum(signal_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = cumsum - running_max
    
    if running_max.max() > 1e-9:
        max_dd = drawdowns.min() / (running_max.max() + 1e-9)
    else:
        max_dd = drawdowns.min()
    
    print(f"  Signals (confluence > 40): {signal_count}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {max_dd:.2%}")
    
    return {
        "signal_count": signal_count,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def build_and_train(df: pd.DataFrame) -> dict:
    """
    Build features and train Phase 5 ML regressor.
    
    Returns: training results dict
    """
    print("\n[Features] Building phase5 feature set...")
    
    fb = Phase5FeatureBuilder()
    df_features = fb.build(df)
    
    print(f"  Shape after feature engineering: {df_features.shape}")
    print(f"  Feature count: {len(df_features.columns)}")
    print(f"  NaN count: {df_features.isna().sum().sum()}")
    
    print("\n[Regressor] Training Phase 5 ML model (walk-forward)...")
    
    regressor = Phase5Regressor()
    results = regressor.train(df_features, verbose=True)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    regressor.save()
    
    return results, regressor


def print_comparison(baseline: dict, ml_results: dict):
    """Print Phase 1-3 vs Phase 5 ML comparison."""
    print("\n" + "=" * 80)
    print("PHASE 1-3 vs PHASE 5 ML COMPARISON")
    print("=" * 80)
    
    print("\nPhase 1-3 (Confluence only):")
    print(f"  Signals: {baseline['signal_count']}")
    print(f"  Win rate: {baseline['win_rate']:.1%}")
    print(f"  Sharpe: {baseline['sharpe']:.2f}")
    print(f"  Max DD: {baseline['max_dd']:.2%}")
    
    avg = ml_results.get("avg_metrics", {})
    if avg:
        print("\nPhase 5 ML (Regressor):")
        print(f"  Avg signals (per fold): ~{avg.get('avg_signal_count', '?')}")
        print(f"  Avg win rate: {avg.get('avg_win_rate', 0):.1%}")
        print(f"  Avg Sharpe: {avg.get('avg_sharpe', 0):.2f}")
        print(f"  Avg Max DD: {avg.get('avg_max_dd', 0):.2%}")
        print(f"  Avg R²: {avg.get('avg_r2', 0):.4f}")
        
        # Improvement
        if baseline['sharpe'] > 0:
            sharpe_improvement = (avg.get('avg_sharpe', 0) - baseline['sharpe']) / baseline['sharpe'] * 100
        else:
            sharpe_improvement = 0
        
        if baseline['win_rate'] > 0:
            wr_improvement = (avg.get('avg_win_rate', 0) - baseline['win_rate']) / baseline['win_rate'] * 100
        else:
            wr_improvement = 0
        
        print(f"\nImprovement:")
        print(f"  Win rate: +{wr_improvement:+.1f}%")
        print(f"  Sharpe: +{sharpe_improvement:+.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print(f"PHASE 5 ML ENGINE — FREQAI REBUILD")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load
    df = load_data()
    
    # Baseline
    baseline = compute_baseline_metrics(df)
    
    # Build & train
    ml_results, regressor = build_and_train(df)
    
    # Comparison
    print_comparison(baseline, ml_results)
    
    print(f"\n✅ Training complete! Models saved to models/phase5_reg*.pkl")
    print(f"   Use phase5_inference.py to make live predictions.")
    
    return regressor


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
