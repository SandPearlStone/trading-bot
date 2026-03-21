"""
phase5/meta_labeler.py — Confidence meta-model (XGBoost).

Trains a secondary model that predicts how confident the ensemble is.
Input:  all features + ensemble direction prediction
Output: confidence score (0–1)
Gate:   only trade when confidence > CONFIDENCE_THRESHOLD (0.65)

Usage:
    python3 meta_labeler.py --data data/processed/labeled_4h.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    CONFIDENCE_THRESHOLD, META_MODEL_PATH, LOGS_DIR,
    WF_N_SPLITS, WF_TRAIN_FRAC, WF_EMBARGO,
)
from phase5.model_trainer import (
    LGBTrainer, XGBTrainer,
    walk_forward_splits, compute_metrics,
)
from phase5.ensemble import EnsemblePredictor

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "meta_labeler.log")),
    ],
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("xgboost not installed")


class MetaLabeler:
    """
    Confidence meta-model.

    Workflow:
    1. Train ensemble (LGB+XGB) on training split
    2. Get ensemble predictions on training data (in-fold predictions)
    3. Train XGBoost meta-model to predict whether ensemble was correct
    4. At inference: gate trades by meta-model confidence > threshold
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.meta_model = None
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        ensemble: Optional[EnsemblePredictor] = None,
    ):
        """
        Train the meta-model.

        If ensemble is provided, use it for predictions (avoids leakage).
        Otherwise trains a fresh ensemble on X_train.
        """
        if ensemble is None:
            ensemble = EnsemblePredictor()
            ensemble.fit(X_train, y_train)

        # Generate ensemble predictions on training data
        y_ensemble_pred = ensemble.predict(X_train)
        y_ensemble_proba = ensemble.predict_proba(X_train)

        # Meta-label: 1 if ensemble was correct, 0 if wrong
        meta_labels = (y_ensemble_pred == y_train.values).astype(int)

        # Build meta-features: original features + ensemble prediction/proba
        X_meta = X_train.copy()
        X_meta["ensemble_pred"] = y_ensemble_pred
        # Add probabilities for each class
        n_classes = y_ensemble_proba.shape[1]
        for c in range(n_classes):
            X_meta[f"ensemble_proba_{c}"] = y_ensemble_proba[:, c]
        X_meta["ensemble_max_proba"] = y_ensemble_proba.max(axis=1)

        # Train XGBoost meta-model (binary: correct vs incorrect)
        if HAS_XGB:
            from xgboost import XGBClassifier
            self.meta_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric="logloss",
            )
            self.meta_model.fit(X_meta.fillna(0), meta_labels)
        else:
            # Fallback: logistic regression
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_meta.fillna(0))
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            self.meta_model.fit(X_scaled, meta_labels)
            self._use_scaler = True

        self._fitted = True
        self._ensemble = ensemble
        self._meta_feature_cols = list(X_meta.columns)
        log.info(f"MetaLabeler trained. meta_labels dist: {np.bincount(meta_labels)}")
        return self

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Return confidence scores (0–1) for each row."""
        if not self._fitted:
            raise RuntimeError("MetaLabeler not fitted. Call .fit() first.")

        y_ensemble_pred = self._ensemble.predict(X)
        y_ensemble_proba = self._ensemble.predict_proba(X)

        X_meta = X.copy()
        X_meta["ensemble_pred"] = y_ensemble_pred
        n_classes = y_ensemble_proba.shape[1]
        for c in range(n_classes):
            X_meta[f"ensemble_proba_{c}"] = y_ensemble_proba[:, c]
        X_meta["ensemble_max_proba"] = y_ensemble_proba.max(axis=1)

        # Align columns
        for col in self._meta_feature_cols:
            if col not in X_meta.columns:
                X_meta[col] = 0
        X_meta = X_meta[self._meta_feature_cols]

        if hasattr(self, "_use_scaler") and self._use_scaler:
            X_scaled = self._scaler.transform(X_meta.fillna(0))
            return self.meta_model.predict_proba(X_scaled)[:, 1]

        return self.meta_model.predict_proba(X_meta.fillna(0))[:, 1]

    def predict_with_gate(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            directions  : ensemble prediction (-1/0/1)
            confidences : confidence scores (0–1)
            trade_mask  : bool array, True where confidence >= threshold
        """
        directions   = self._ensemble.predict(X)
        confidences  = self.predict_confidence(X)
        trade_mask   = confidences >= self.threshold
        return directions, confidences, trade_mask

    def save(self, path: str = META_MODEL_PATH):
        obj = {
            "meta_model": self.meta_model,
            "ensemble":   self._ensemble,
            "threshold":  self.threshold,
            "feature_cols": self._meta_feature_cols,
        }
        if hasattr(self, "_scaler"):
            obj["scaler"] = self._scaler
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        log.info(f"MetaLabeler saved → {path}")

    @classmethod
    def load(cls, path: str = META_MODEL_PATH) -> "MetaLabeler":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        meta = cls(threshold=obj["threshold"])
        meta.meta_model = obj["meta_model"]
        meta._ensemble  = obj["ensemble"]
        meta._meta_feature_cols = obj["feature_cols"]
        if "scaler" in obj:
            meta._scaler = obj["scaler"]
            meta._use_scaler = True
        meta._fitted = True
        return meta


# ── Walk-forward CV for meta-labeler ─────────────────────────────────────────

def cross_validate_meta(data_path: str, n_splits: int = WF_N_SPLITS) -> list[dict]:
    """Walk-forward evaluation of ensemble + meta-labeler gating."""
    df = pd.read_csv(data_path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    exclude = {"label", "symbol"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, float, int]]
    X = df[feat_cols].fillna(0)
    y = df["label"]

    results = []
    for fold, (tr_idx, te_idx) in enumerate(
        walk_forward_splits(len(X), n_splits, WF_TRAIN_FRAC, WF_EMBARGO)
    ):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        # Train ensemble
        ens = EnsemblePredictor()
        ens.fit(X_tr, y_tr)

        # Train meta
        meta = MetaLabeler()
        meta.fit(X_tr, y_tr, ensemble=ens)

        # Evaluate
        directions, confidences, mask = meta.predict_with_gate(X_te)
        y_te_arr = y_te.values

        # All signals (no gate)
        m_all = compute_metrics(y_te_arr, directions)

        # Gated signals (confidence >= threshold)
        if mask.sum() > 0:
            m_gated = compute_metrics(y_te_arr[mask], directions[mask])
        else:
            m_gated = {"win_rate": 0, "accuracy": 0}

        result = {
            "fold":              fold + 1,
            "all_win_rate":      m_all["win_rate"],
            "gated_win_rate":    m_gated["win_rate"],
            "gated_count":       int(mask.sum()),
            "total_count":       len(mask),
            "gate_pct":          round(mask.sum() / max(len(mask), 1) * 100, 1),
            "avg_confidence":    round(float(confidences.mean()), 3),
            "high_conf_accuracy":round(m_gated.get("accuracy", 0), 1),
        }
        log.info(f"Fold {fold+1}: all_wr={result['all_win_rate']}% gated_wr={result['gated_win_rate']}% gate%={result['gate_pct']}%")
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", type=int, default=WF_N_SPLITS)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--save", action="store_true", help="Train and save final meta-model")
    args = parser.parse_args()

    results = cross_validate_meta(args.data, args.splits)
    print("\n=== META-LABELER CV RESULTS ===")
    for r in results:
        print(
            f"  Fold {r['fold']}: "
            f"all_wr={r['all_win_rate']}% | "
            f"gated_wr={r['gated_win_rate']}% | "
            f"gate%={r['gate_pct']}% ({r['gated_count']}/{r['total_count']})"
        )

    avg_gated_wr = np.mean([r["gated_win_rate"] for r in results])
    avg_gate_pct = np.mean([r["gate_pct"] for r in results])
    print(f"\nAvg gated win rate: {avg_gated_wr:.1f}%")
    print(f"Avg gate%: {avg_gate_pct:.1f}% (trades taken)")

    if args.save:
        # Train final on all data
        df = pd.read_csv(args.data)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        exclude = {"label", "symbol"}
        feat_cols = [c for c in df.columns if c not in exclude
                     and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, float, int]]
        X = df[feat_cols].fillna(0)
        y = df["label"]

        ens = EnsemblePredictor()
        ens.fit(X, y)
        meta = MetaLabeler(threshold=args.threshold)
        meta.fit(X, y, ensemble=ens)
        meta.save()
        print(f"Final MetaLabeler saved → {META_MODEL_PATH}")
