#!/home/sandro/trading-venv/bin/python3
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from lightgbm import LGBMRegressor


class TradingModel:
    """LightGBM regressor with walk-forward validation."""
    
    MODEL_PATH = 'models/freqai_model.pkl'
    META_PATH = 'models/freqai_meta.json'
    MIN_RETURN = 0.002
    
    def __init__(self):
        """Initialize. Try to load existing model from disk."""
        self.model = None
        self.feature_names = None
        self.train_timestamp = None
        self._try_load()
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> dict:
        """Train LGBMRegressor with walk-forward validation."""
        # Drop NaN labels
        valid_idx = labels.notna()
        X = features[valid_idx].reset_index(drop=True)
        y = labels[valid_idx].reset_index(drop=True)
        
        self.feature_names = X.columns.tolist()
        n = len(X)
        
        # Walk-forward folds: 4 temporal folds
        fold_splits = [
            (0, int(0.40*n), int(0.55*n)),  # train [0:40%], test [40:55%]
            (0, int(0.55*n), int(0.70*n)),  # train [0:55%], test [55:70%]
            (0, int(0.70*n), int(0.85*n)),  # train [0:70%], test [70:85%]
            (0, int(0.85*n), n),            # train [0:85%], test [85:100%]
        ]
        
        fold_results = []
        all_sharpes = []
        all_win_rates = []
        all_max_dds = []
        all_n_signals = []
        
        for fold_idx, (train_start, test_start, test_end) in enumerate(fold_splits, 1):
            X_train = X.iloc[train_start:test_start]
            y_train = y.iloc[train_start:test_start]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train
            model = LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=7,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, random_state=42, n_jobs=4, verbose=-1
            )
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            signals = np.abs(y_pred) > self.MIN_RETURN
            if signals.sum() > 0:
                y_sig = y_test[signals]
                y_pred_sig = y_pred[signals]
                
                # Win rate
                correct = ((y_pred_sig > 0) & (y_sig > 0)) | ((y_pred_sig < 0) & (y_sig < 0))
                win_rate = correct.mean()
                
                # Sharpe
                sharpe = (y_sig.mean() / y_sig.std() * np.sqrt(252)) if y_sig.std() > 0 else 0
                
                # Max DD
                cum_pnl = y_sig.cumsum()
                peak = cum_pnl.cummax()
                max_dd = (cum_pnl - peak).min() * 100
            else:
                win_rate = 0
                sharpe = 0
                max_dd = 0
            
            n_signals = signals.sum()
            
            fold_results.append({
                'sharpe': sharpe, 'win_rate': win_rate,
                'max_dd': max_dd, 'n_signals': n_signals
            })
            all_sharpes.append(sharpe)
            all_win_rates.append(win_rate)
            all_max_dds.append(max_dd)
            all_n_signals.append(n_signals)
            
            print(f"Fold {fold_idx}: Sharpe={sharpe:.2f} WR={win_rate:.1%} DD={max_dd:.1f}% Signals={n_signals}")
        
        # Train final model on ALL data
        self.model = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, n_jobs=4, verbose=-1
        )
        self.model.fit(X, y)
        self.train_timestamp = datetime.utcnow().isoformat()
        
        # Save
        self.save()
        
        avg_sharpe = np.mean(all_sharpes)
        avg_win_rate = np.mean(all_win_rates)
        avg_max_dd = np.mean(all_max_dds)
        avg_n_signals = int(np.mean(all_n_signals))
        
        print(f"Avg: Sharpe={avg_sharpe:.2f} WR={avg_win_rate:.1%} DD={avg_max_dd:.1f}%")
        
        return {
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'avg_max_dd': avg_max_dd,
            'avg_n_signals': avg_n_signals,
            'fold_results': fold_results
        }
    
    def predict(self, features: pd.DataFrame) -> dict:
        """Predict expected return for latest bar(s)."""
        if self.model is None:
            return {'expected_return': 0, 'direction': 'HOLD', 'signal_strength': 0,
                    'take_signal': False, 'top_features': []}
        
        # Use last row
        X = features.iloc[-1:].values
        ret = self.model.predict(X)[0]
        
        # Direction
        if ret > self.MIN_RETURN:
            direction = 'LONG'
        elif ret < -self.MIN_RETURN:
            direction = 'SHORT'
        else:
            direction = 'HOLD'
        
        # Signal strength
        signal_strength = abs(ret) / self.MIN_RETURN if self.MIN_RETURN > 0 else 0
        take_signal = abs(ret) > self.MIN_RETURN
        
        # Top features
        top_feats = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            'expected_return': ret,
            'direction': direction,
            'signal_strength': signal_strength,
            'take_signal': take_signal,
            'top_features': top_feats
        }
    
    def save(self):
        """Save model to MODEL_PATH and meta to META_PATH."""
        os.makedirs('models', exist_ok=True)
        
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        meta = {
            'feature_names': self.feature_names,
            'feature_version': 'v3',
            'train_timestamp': self.train_timestamp,
            'n_samples': self.model.n_features_in_ if self.model else 0,
            'avg_sharpe': 0,
            'avg_win_rate': 0
        }
        
        with open(self.META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def _try_load(self):
        """Try to load model from disk. Silent if not found."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                
                if os.path.exists(self.META_PATH):
                    with open(self.META_PATH, 'r') as f:
                        meta = json.load(f)
                        self.feature_names = meta.get('feature_names')
                        self.train_timestamp = meta.get('train_timestamp')
        except Exception:
            pass
