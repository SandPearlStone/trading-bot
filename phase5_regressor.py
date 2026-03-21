"""
phase5_regressor.py — FreqAI-inspired regression model for Phase 5 ML engine.

Core features:
  - LightGBM Regressor (predict forward return %)
  - Rolling 30-day training window, retrain every 7 days
  - Walk-forward validation (temporal split, no lookahead)
  - Forward return label: (close[t+n] / close[t] - 1) - fees
  - Only trade when |predicted_return| > threshold (default 0.2%)
  - Save model + feature metadata for live inference

Usage:
    regressor = Phase5Regressor()
    regressor.train(df_features)  # Walk-forward training
    pred = regressor.predict(df_features.iloc[-1:])  # Latest bar
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 Regressor
# ──────────────────────────────────────────────────────────────────────────────

class Phase5Regressor:
    """
    LightGBM-based forward return predictor.
    
    Train on rolling 30-day windows, retrain every 7 days.
    Walk-forward validation with temporal splits.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        train_days: int = 30,
        retrain_every_days: int = 7,
        lookahead_bars: int = 10,
        min_return_threshold: float = 0.002,
        transaction_cost: float = 0.001,
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for gradient boosting
            max_depth: Max tree depth
            train_days: Training window in days
            retrain_every_days: Retrain frequency in days
            lookahead_bars: Bars ahead for label (10 × 4h = 10 days forward)
            min_return_threshold: Min predicted return to trade (0.2%)
            transaction_cost: Cost per round trip (0.1% each way = 0.001 total)
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.train_days = train_days
        self.retrain_every_days = retrain_every_days
        self.lookahead_bars = lookahead_bars
        self.min_return_threshold = min_return_threshold
        self.transaction_cost = transaction_cost
        self.random_state = random_state
        
        self.model: Optional[LGBMRegressor] = None
        self.feature_names: Optional[list[str]] = None
        self.scaler_mean: Optional[dict] = None
        self.scaler_std: Optional[dict] = None
        
    def _label_forward_return(self, df: pd.DataFrame) -> pd.Series:
        """
        Label: forward return N bars ahead, minus transaction cost.
        
        Forward return = (close[t+N] / close[t]) - 1
        Label = forward_return - transaction_cost
        
        Clip extreme outliers (from low-price tokens) to [-50%, +50%] range.
        This is unbounded (can be -1 to +inf), suitable for regression.
        """
        forward_return = df["close"].pct_change(-self.lookahead_bars)
        label = forward_return - self.transaction_cost
        
        # Clip extreme outliers (low-price tokens, delisted coins)
        label = label.clip(-0.50, 0.50)
        
        return label
    
    def _train_test_split_walk_forward(
        self,
        df: pd.DataFrame,
        n_folds: int = 4,
    ) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward temporal split: no lookahead, purge overlapping.
        
        Example (4 folds):
          Fold 1: train 2024-01 to 2024-07, test 2024-07 to 2024-09 (2 months)
          Fold 2: train 2024-04 to 2024-10, test 2024-10 to 2024-12
          ...
        
        Returns: list of (df_train, df_test) tuples
        """
        df = df.sort_index()
        n_bars = len(df)
        
        # Test size per fold: 2 months = ~14 trading bars per month × 2 = ~28 bars
        test_size = n_bars // (n_folds + 1)
        
        folds = []
        for i in range(n_folds):
            test_end_idx = (i + 1) * test_size
            test_start_idx = i * test_size
            train_end_idx = test_start_idx
            train_start_idx = max(0, train_end_idx - test_size * 2)  # 2x test window for training
            
            df_train = df.iloc[train_start_idx:train_end_idx].copy()
            df_test = df.iloc[test_start_idx:test_end_idx].copy()
            
            if len(df_train) > 100 and len(df_test) > 10:  # Minimum size
                folds.append((df_train, df_test))
        
        return folds
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """
        Prepare features for training/inference.
        
        Excludes: timestamp, index, label, close, symbol, etc.
        Returns: (feature_df, feature_names)
        """
        exclude_cols = {
            "close", "high", "low", "open", "volume",
            "&-target", "label", "symbol", "timestamp",
            "datetime", "time", "date", "is_signal", "entry", "sl", "tp",
        }
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("_")]
        df_features = df[feature_cols].copy()
        
        return df_features, feature_cols
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Train with walk-forward validation.
        
        Returns: dict with per-fold results (win_rate, sharpe, max_dd, r2, rmse)
        """
        df = df.sort_index()
        
        # Add label
        df["&-target"] = self._label_forward_return(df)
        
        # Walk-forward split
        folds = self._train_test_split_walk_forward(df, n_folds=4)
        
        results = {
            "fold_results": [],
            "best_model": None,
            "best_fold": 0,
            "avg_metrics": {},
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print("WALK-FORWARD TRAINING")
            print("=" * 80)
        
        fold_metrics = []
        
        for fold_idx, (df_train, df_test) in enumerate(folds):
            if verbose:
                train_start = df_train.index[0].strftime("%Y-%m-%d")
                train_end = df_train.index[-1].strftime("%Y-%m-%d")
                test_start = df_test.index[0].strftime("%Y-%m-%d")
                test_end = df_test.index[-1].strftime("%Y-%m-%d")
                print(f"\nFold {fold_idx + 1}/{len(folds)}")
                print(f"  Train: {train_start} → {train_end} ({len(df_train)} bars)")
                print(f"  Test:  {test_start} → {test_end} ({len(df_test)} bars)")
            
            # Prepare features
            X_train, feature_names = self._prepare_features(df_train)
            y_train = df_train["&-target"]
            
            X_test, _ = self._prepare_features(df_test)
            y_test = df_test["&-target"]
            
            # Remove NaN
            mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
            mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
            
            if len(X_train) < 50 or len(X_test) < 10:
                if verbose:
                    print(f"  ⚠️  Skipped (insufficient data after cleaning)")
                continue
            
            # Train model
            model = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_test = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            # Signal-based metrics (only top signals by confidence)
            # Use top 20% of predictions by absolute value as signals
            top_n = max(int(len(y_pred_test) * 0.20), 10)
            top_idx = np.argsort(np.abs(y_pred_test))[-top_n:]
            
            signal_returns = y_test.iloc[top_idx].values
            signal_count = len(top_idx)
            
            win_rate = (signal_returns > 0).sum() / signal_count
            
            # Sharpe ratio
            if len(signal_returns) > 1:
                sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-9)) * np.sqrt(252 * 6)
            else:
                sharpe = 0.0
            
            # Max drawdown (from cumulative return series)
            cumsum = np.cumsum(signal_returns)
            running_max = np.maximum.accumulate(cumsum)
            drawdowns = cumsum - running_max
            
            # Calculate max DD as percentage, handling edge cases
            if running_max.max() > 1e-9:
                max_dd = drawdowns.min() / (running_max.max() + 1e-9)
            else:
                max_dd = drawdowns.min()
            
            metrics = {
                "fold": fold_idx,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "signal_count": signal_count,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "max_dd": max_dd,
            }
            fold_metrics.append(metrics)
            results["fold_results"].append(metrics)
            
            if verbose:
                print(f"  RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")
                print(f"  Signals: {signal_count}/{len(X_test)} | Win: {win_rate:.1%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")
            
            # Keep best model
            if fold_idx == 0 or r2 > results["best_model"][0]:
                results["best_model"] = (r2, model, feature_names, X_train, y_train)
                results["best_fold"] = fold_idx
        
        # Average metrics
        if fold_metrics:
            avg_metrics = {
                "avg_rmse": np.mean([m["rmse"] for m in fold_metrics]),
                "avg_r2": np.mean([m["r2"] for m in fold_metrics]),
                "avg_win_rate": np.mean([m["win_rate"] for m in fold_metrics]),
                "avg_sharpe": np.mean([m["sharpe"] for m in fold_metrics]),
                "avg_max_dd": np.mean([m["max_dd"] for m in fold_metrics]),
            }
            results["avg_metrics"] = avg_metrics
            
            if verbose:
                print("\n" + "-" * 80)
                print("SUMMARY")
                print("-" * 80)
                print(f"Avg RMSE: {avg_metrics['avg_rmse']:.6f}")
                print(f"Avg R²: {avg_metrics['avg_r2']:.4f}")
                print(f"Avg Win Rate: {avg_metrics['avg_win_rate']:.1%}")
                print(f"Avg Sharpe: {avg_metrics['avg_sharpe']:.2f}")
                print(f"Avg Max DD: {avg_metrics['avg_max_dd']:.2%}")
        
        # Store best model
        if results["best_model"]:
            _, self.model, self.feature_names, _, _ = results["best_model"]
        
        return results
    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict forward return for latest bar(s).
        
        Returns: {
            "predicted_return": float,  # Predicted % return (can be negative)
            "direction": str,  # "LONG", "SHORT", or "NONE"
            "signal_strength": float,  # Abs(predicted_return)
            "should_trade": bool,  # |return| > threshold
        }
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        df = df.sort_index()
        X, _ = self._prepare_features(df)
        
        # Use only latest row
        X_latest = X.iloc[-1:].copy()
        
        # Fill NaN
        X_latest = X_latest.fillna(0)
        
        # Select features used in training
        if self.feature_names:
            X_latest = X_latest[[c for c in self.feature_names if c in X_latest.columns]]
        
        pred = self.model.predict(X_latest)[0]
        
        return {
            "predicted_return": float(pred),
            "direction": "LONG" if pred > self.min_return_threshold else ("SHORT" if pred < -self.min_return_threshold else "NONE"),
            "signal_strength": abs(pred),
            "should_trade": abs(pred) > self.min_return_threshold,
        }
    
    def save(self, model_path: str = "models/phase5_reg.pkl", features_path: str = "models/phase5_reg_features.pkl"):
        """Save model and feature names."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        with open(features_path, "wb") as f:
            pickle.dump(self.feature_names, f)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Features saved to {features_path}")
    
    def load(self, model_path: str = "models/phase5_reg.pkl", features_path: str = "models/phase5_reg_features.pkl"):
        """Load model and feature names."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        with open(features_path, "rb") as f:
            self.feature_names = pickle.load(f)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Features loaded from {features_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Test on labeled_4h.csv
    csv_path = "data/processed/labeled_4h.csv"
    
    if os.path.exists(csv_path):
        print(f"[Phase5 Regressor] Loading {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        print(f"  Shape: {df.shape}")
        
        # Train
        regressor = Phase5Regressor()
        results = regressor.train(df, verbose=True)
        
        print(f"\n✅ Training complete!")
        
        # Test inference
        if regressor.model is not None:
            pred = regressor.predict(df)
            print(f"\nLatest prediction:")
            print(f"  Predicted return: {pred['predicted_return']:.4%}")
            print(f"  Direction: {pred['direction']}")
            print(f"  Should trade: {pred['should_trade']}")
    else:
        print(f"❌ {csv_path} not found.")
