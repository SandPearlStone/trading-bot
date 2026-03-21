"""
phase5/model_trainer.py — LightGBM + XGBoost training with walk-forward CV.

Usage:
    python3 model_trainer.py --data data/processed/labeled_4h.csv
    python3 model_trainer.py --data ... --quick  # 2 splits for speed
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    LGB_PARAMS, XGB_PARAMS, WF_N_SPLITS, WF_TRAIN_FRAC, WF_EMBARGO,
    LGB_MODEL_PATH, XGB_MODEL_PATH, FEATURE_NAMES_PATH, LOGS_DIR,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "model_trainer.log")),
    ],
)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    log.warning("lightgbm not installed — LGB training disabled")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("xgboost not installed — XGB training disabled")


# ── Walk-forward split ────────────────────────────────────────────────────────

def walk_forward_splits(n: int, n_splits: int, train_frac: float, embargo: int):
    """
    Generate (train_idx, test_idx) index pairs.
    No shuffle. Embargo gap between train/test.
    """
    test_size = int(n * (1 - train_frac) / n_splits)
    for i in range(n_splits):
        test_end   = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end  = test_start - embargo
        if train_end < 10:
            continue
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_end)
        yield train_idx, test_idx


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    # Binary: treat -1 as 0, +1 as 1; ignore 0 (timeout) for win rate
    mask = y_true != 0
    if mask.sum() == 0:
        return {"accuracy": 0, "win_rate": 0, "f1": 0}

    yt = (y_true[mask] == 1).astype(int)
    yp = (y_pred[mask] == 1).astype(int)

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "win_rate":  round(accuracy_score(yt, yp) * 100, 2),
        "precision": round(precision_score(yt, yp, zero_division=0) * 100, 2),
        "recall":    round(recall_score(yt, yp, zero_division=0) * 100, 2),
        "f1":        round(f1_score(yt, yp, zero_division=0) * 100, 2),
    }
    if y_prob is not None and len(np.unique(yt)) > 1:
        try:
            metrics["auc"] = round(roc_auc_score(yt, y_prob[mask]), 4)
        except Exception:
            pass
    return metrics


# ── LightGBM trainer ──────────────────────────────────────────────────────────

class LGBTrainer:
    def __init__(self, params: dict = None):
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        self.params = params or LGB_PARAMS.copy()
        self.model = None
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str = LGB_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        log.info(f"LGB model saved → {path}")

    @classmethod
    def load(cls, path: str = LGB_MODEL_PATH) -> "LGBTrainer":
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ── XGBoost trainer ───────────────────────────────────────────────────────────

class XGBTrainer:
    def __init__(self, params: dict = None):
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        self.params = params or XGB_PARAMS.copy()
        self.model = None
        self.feature_importance: pd.DataFrame = pd.DataFrame()
        self._label_enc = LabelEncoder()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        from xgboost import XGBClassifier
        # XGBoost requires labels 0..N-1
        y_enc = self._label_enc.fit_transform(y_train)
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_enc)
        self.feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_enc = self.model.predict(X)
        return self._label_enc.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str = XGB_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump((self.model, self._label_enc), f)
        log.info(f"XGB model saved → {path}")

    @classmethod
    def load(cls, path: str = XGB_MODEL_PATH) -> "XGBTrainer":
        obj = cls()
        with open(path, "rb") as f:
            obj.model, obj._label_enc = pickle.load(f)
        return obj


# ── Cross-validate ────────────────────────────────────────────────────────────

def cross_validate(
    trainer_cls,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = WF_N_SPLITS,
    train_frac: float = WF_TRAIN_FRAC,
    embargo: int = WF_EMBARGO,
    trainer_params: dict = None,
) -> list[dict]:
    """Walk-forward CV. Returns list of per-fold metrics."""
    results = []
    splits = list(walk_forward_splits(len(X), n_splits, train_frac, embargo))

    for fold, (tr_idx, te_idx) in enumerate(splits):
        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_test,  y_test  = X.iloc[te_idx], y.iloc[te_idx]

        trainer = trainer_cls(params=trainer_params) if trainer_params else trainer_cls()
        trainer.fit(X_train, y_train)

        y_pred = trainer.predict(X_test)
        try:
            y_prob = trainer.predict_proba(X_test)[:, -1]  # prob of last class
        except Exception:
            y_prob = None

        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        metrics["fold"] = fold + 1
        metrics["train_size"] = len(tr_idx)
        metrics["test_size"]  = len(te_idx)
        log.info(f"Fold {fold+1}: {metrics}")
        results.append(metrics)

    return results


# ── Full training pipeline ────────────────────────────────────────────────────

def train_all(data_path: str, n_splits: int = WF_N_SPLITS) -> dict:
    """Load labeled data, cross-validate, train final models, save."""
    log.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"] if "timestamp" in open(data_path).readline() else [])

    if "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    # Drop rows with NaN label
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Feature columns: everything except label, symbol
    exclude = {"label", "symbol"}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, float, int]]
    X = df[feat_cols].fillna(0)
    y = df["label"]

    log.info(f"Data: {len(X)} rows, {len(feat_cols)} features, label dist={y.value_counts().to_dict()}")

    # Save feature names
    with open(FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(feat_cols, f)

    results = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if HAS_LGB:
        log.info("=== LightGBM Walk-Forward CV ===")
        lgb_cv = cross_validate(LGBTrainer, X, y, n_splits=n_splits)
        avg_lgb = {k: round(np.mean([r[k] for r in lgb_cv if k in r]), 2)
                   for k in ["accuracy", "win_rate", "f1"]}
        log.info(f"LGB CV avg: {avg_lgb}")

        log.info("Training final LGB on full data...")
        lgb_final = LGBTrainer()
        lgb_final.fit(X, y)
        lgb_final.save()

        results["lgb"] = {
            "cv": lgb_cv,
            "avg": avg_lgb,
            "feature_importance": lgb_final.feature_importance.head(20).to_dict(),
        }
        log.info(f"LGB top features:\n{lgb_final.feature_importance.head(10).to_string()}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if HAS_XGB:
        log.info("=== XGBoost Walk-Forward CV ===")
        xgb_cv = cross_validate(XGBTrainer, X, y, n_splits=n_splits)
        avg_xgb = {k: round(np.mean([r[k] for r in xgb_cv if k in r]), 2)
                   for k in ["accuracy", "win_rate", "f1"]}
        log.info(f"XGB CV avg: {avg_xgb}")

        log.info("Training final XGB on full data...")
        xgb_final = XGBTrainer()
        xgb_final.fit(X, y)
        xgb_final.save()

        results["xgb"] = {
            "cv": xgb_cv,
            "avg": avg_xgb,
            "feature_importance": xgb_final.feature_importance.head(20).to_dict(),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--splits", type=int, default=WF_N_SPLITS)
    parser.add_argument("--quick", action="store_true", help="Use 2 splits for speed")
    args = parser.parse_args()

    n = 2 if args.quick else args.splits
    results = train_all(args.data, n_splits=n)
    print("\n=== TRAINING COMPLETE ===")
    for model_name, info in results.items():
        print(f"\n{model_name.upper()} CV averages: {info['avg']}")
