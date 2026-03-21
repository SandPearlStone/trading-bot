"""
phase5/ensemble.py — Combine LGB + XGB predictions via weighted average.

Usage:
    python3 ensemble.py --data data/processed/labeled_4h.csv
    python3 ensemble.py --data ... --lgb-weight 0.6 --xgb-weight 0.4
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
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    ENSEMBLE_LGB_WEIGHT, ENSEMBLE_XGB_WEIGHT,
    LGB_MODEL_PATH, XGB_MODEL_PATH,
    LGB_OPT_MODEL_PATH, XGB_OPT_MODEL_PATH,
    FEATURE_NAMES_PATH, MODELS_DIR, LOGS_DIR,
)
from phase5.model_trainer import (
    LGBTrainer, XGBTrainer, compute_metrics,
    walk_forward_splits, WF_N_SPLITS, WF_TRAIN_FRAC, WF_EMBARGO,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "ensemble.log")),
    ],
)

try:
    import lightgbm; HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost; HAS_XGB = True
except ImportError:
    HAS_XGB = False


class EnsemblePredictor:
    """
    Weighted average ensemble of LGB + XGB.
    Predicts:
      - direction: majority vote
      - proba: weighted average of class probabilities
    """

    def __init__(
        self,
        lgb_weight: float = ENSEMBLE_LGB_WEIGHT,
        xgb_weight: float = ENSEMBLE_XGB_WEIGHT,
        use_optimized: bool = True,
    ):
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.use_optimized = use_optimized
        self.lgb_model: Optional[LGBTrainer] = None
        self.xgb_model: Optional[XGBTrainer] = None
        self.classes_: Optional[list] = None

    def load_models(self):
        """Load pre-trained LGB + XGB models from disk."""
        # Prefer optimized; fall back to baseline
        lgb_path = LGB_OPT_MODEL_PATH if (self.use_optimized and os.path.exists(LGB_OPT_MODEL_PATH)) else LGB_MODEL_PATH
        xgb_path = XGB_OPT_MODEL_PATH if (self.use_optimized and os.path.exists(XGB_OPT_MODEL_PATH)) else XGB_MODEL_PATH

        if HAS_LGB and os.path.exists(lgb_path):
            self.lgb_model = LGBTrainer.load(lgb_path)
            log.info(f"Loaded LGB from {lgb_path}")
        if HAS_XGB and os.path.exists(xgb_path):
            self.xgb_model = XGBTrainer.load(xgb_path)
            log.info(f"Loaded XGB from {xgb_path}")

        if not self.lgb_model and not self.xgb_model:
            raise RuntimeError("No models loaded. Run model_trainer.py first.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train both models from scratch (used in CV context)."""
        if HAS_LGB:
            self.lgb_model = LGBTrainer()
            self.lgb_model.fit(X, y)
        if HAS_XGB:
            self.xgb_model = XGBTrainer()
            self.xgb_model.fit(X, y)
        self.classes_ = sorted(y.unique().tolist())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average probabilities [n_samples, n_classes]."""
        probas = []
        weights = []

        if self.lgb_model is not None:
            p = self.lgb_model.predict_proba(X)
            probas.append(p)
            weights.append(self.lgb_weight)

        if self.xgb_model is not None:
            p = self.xgb_model.predict_proba(X)
            probas.append(p)
            weights.append(self.xgb_weight)

        if not probas:
            raise RuntimeError("No models available for prediction")

        total_w = sum(weights)
        ensemble_proba = sum(w / total_w * p for w, p in zip(weights, probas))
        return ensemble_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class (argmax of ensemble proba)."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        # Map back to original labels
        if self.lgb_model and hasattr(self.lgb_model.model, "classes_"):
            classes = self.lgb_model.model.classes_
        elif self.xgb_model and hasattr(self.xgb_model, "_label_enc"):
            classes = self.xgb_model._label_enc.classes_
        else:
            classes = np.array([-1, 0, 1])
        return classes[idx]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        try:
            proba = self.predict_proba(X)[:, -1]
        except Exception:
            proba = None
        return compute_metrics(y.values, y_pred, proba)


def walk_forward_ensemble_cv(data_path: str, n_splits: int = WF_N_SPLITS) -> list[dict]:
    """Walk-forward cross-validation of the ensemble."""
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
    for fold, (tr_idx, te_idx) in enumerate(walk_forward_splits(len(X), n_splits, WF_TRAIN_FRAC, WF_EMBARGO)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        ens = EnsemblePredictor()
        ens.fit(X_tr, y_tr)
        metrics = ens.evaluate(X_te, y_te)
        metrics["fold"] = fold + 1
        log.info(f"Ensemble Fold {fold+1}: {metrics}")
        results.append(metrics)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--lgb-weight", type=float, default=ENSEMBLE_LGB_WEIGHT)
    parser.add_argument("--xgb-weight", type=float, default=ENSEMBLE_XGB_WEIGHT)
    parser.add_argument("--splits", type=int, default=WF_N_SPLITS)
    args = parser.parse_args()

    results = walk_forward_ensemble_cv(args.data, args.splits)
    print("\n=== ENSEMBLE CV RESULTS ===")
    for r in results:
        print(f"  Fold {r['fold']}: win_rate={r.get('win_rate','?')}% acc={r.get('accuracy','?')}% f1={r.get('f1','?')}%")
    avg_wr = np.mean([r.get("win_rate", 0) for r in results])
    print(f"\nAverage win rate: {avg_wr:.1f}%")
