#!/usr/bin/env python3
"""
Phase 4.2: Random Forest Model Training

Train a Random Forest classifier to predict trade outcomes from features.

Input: phase4_features.csv (210 trades × 15 features)
Output:
  - phase4_model.pkl (trained Random Forest)
  - phase4_importance_scores.csv (feature importance ranking)
  - phase4_training_log.txt (CV results, metrics)

Model:
  - RandomForest(n_estimators=100, max_depth=10, random_state=42)
  - 80/20 train/test split
  - 5-fold cross-validation
  - Binary classification: outcome_win (0=loss, 1=win)

Metrics tracked:
  - Accuracy, Precision, Recall, F1, AUC
  - Cross-validation score (mean ± std)
  - Feature importance (Gini-based)
  - Confusion matrix, classification report

Usage:
  python3 phase4_model_training.py
  # Output: phase4_model.pkl, phase4_importance_scores.csv
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder


# ─── Configuration ─────────────────────────────────────────────────────────

INPUT_CSV = Path(__file__).parent / "phase4_features.csv"
MODEL_FILE = Path(__file__).parent / "phase4_model.pkl"
IMPORTANCE_CSV = Path(__file__).parent / "phase4_importance_scores.csv"
LOG_FILE = Path(__file__).parent / "phase4_training_log.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS = 100
MAX_DEPTH = 10


# ─── Feature Preparation ──────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for training:
    - Encode categorical features (regime)
    - Normalize numeric features (optional)
    - Return X, y

    Features used:
      - regime (categorical) → one-hot encode
      - vol_ratio (numeric)
      - rsi_strength (numeric)
      - rsi_div_regular, rsi_div_hidden, macd_div (binary)
      - sentiment_fg (numeric, 0-100)
      - mtf_bias (numeric, -1 to 1)
      - confluence_score (numeric, 0-1)
      - entry_to_sl (numeric)
      - direction_encoded (numeric, -1 or 1)
      - duration_candles (numeric)

    Target:
      - outcome_win (binary: 0=loss, 1=win)
    """
    df = df.copy()

    # Drop symbol (not a feature)
    if "symbol" in df.columns:
        df = df.drop("symbol", axis=1)

    # Drop r_multiple (outcome proxy, would leak)
    if "r_multiple" in df.columns:
        df = df.drop("r_multiple", axis=1)

    # One-hot encode regime
    regime_dummies = pd.get_dummies(df["regime"], prefix="regime", drop_first=False)
    df = df.drop("regime", axis=1)
    df = pd.concat([df, regime_dummies], axis=1)

    # Target variable
    y = df["outcome_win"]
    X = df.drop("outcome_win", axis=1)

    print(f"[prepare_features] Feature matrix shape: {X.shape}")
    print(f"[prepare_features] Feature columns: {X.columns.tolist()}")
    print(f"[prepare_features] Target distribution: \n{y.value_counts()}")

    return X, y


# ─── Model Training ───────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train Random Forest and evaluate.

    Returns: (model, metrics_dict)
    """
    print("\n[train_model] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    print("\n[train_model] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    print(f"✅ Model trained")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\n[train_model] Cross-validation ({CV_FOLDS}-fold)...")
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
    )
    print(f"  CV scores: {cv_scores}")
    print(f"  CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print(f"\n[train_model] Test set evaluation...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_score = 0.0

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    print(f"\n  Classification Report:")
    print(f"{classification_report(y_test, y_pred, zero_division=0)}")

    metrics = {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

    return model, metrics


# ─── Feature Importance ───────────────────────────────────────────────────────

def extract_importance(model: RandomForestClassifier, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained RF model.

    Returns DataFrame with columns: feature, importance, importance_norm
    Sorted by importance descending.
    """
    importance = model.feature_importances_
    importance_norm = importance / importance.sum()

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "importance_norm": importance_norm,
    })
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n[extract_importance] Feature Importance (top 10):")
    for idx, row in df_imp.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f} ({row['importance_norm']:.2%})")

    return df_imp


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Phase 4.2: Random Forest Model Training")
    print("=" * 80)

    # Load features
    print(f"\n[main] Loading features from {INPUT_CSV}...")
    if not INPUT_CSV.exists():
        print(f"❌ File not found: {INPUT_CSV}")
        return False

    df_features = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df_features)} trades × {len(df_features.columns)} features")

    # Prepare
    X, y = _prepare_features(df_features)

    # Train
    model, metrics = train_model(X, y)

    # Feature importance
    df_importance = extract_importance(model, X.columns.tolist())

    # Save model
    print(f"\n[main] Saving model to {MODEL_FILE}...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved")

    # Save importance
    print(f"[main] Saving feature importance to {IMPORTANCE_CSV}...")
    df_importance.to_csv(IMPORTANCE_CSV, index=False)
    print(f"✅ Importance saved")

    # Save log
    print(f"\n[main] Saving training log to {LOG_FILE}...")
    with open(LOG_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 4.2: Random Forest Model Training Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write(f"  Estimators: {N_ESTIMATORS}\n")
        f.write(f"  Max Depth: {MAX_DEPTH}\n")
        f.write(f"  Random State: {RANDOM_STATE}\n")
        f.write(f"  Test Size: {TEST_SIZE}\n")
        f.write(f"  CV Folds: {CV_FOLDS}\n\n")

        f.write("DATASET\n")
        f.write(f"  Total Samples: {len(df_features)}\n")
        f.write(f"  Train Samples: {metrics['train_size']}\n")
        f.write(f"  Test Samples: {metrics['test_size']}\n")
        f.write(f"  Features: {X.shape[1]}\n\n")

        f.write("CROSS-VALIDATION (5-fold)\n")
        f.write(f"  Scores: {metrics['cv_scores'].tolist()}\n")
        f.write(f"  Mean: {metrics['cv_mean']:.4f}\n")
        f.write(f"  Std: {metrics['cv_std']:.4f}\n\n")

        f.write("TEST SET PERFORMANCE\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1: {metrics['f1']:.4f}\n")
        f.write(f"  AUC: {metrics['auc']:.4f}\n\n")

        f.write("CONFUSION MATRIX\n")
        f.write(f"  [[TN, FP],\n")
        f.write(f"   [FN, TP]]\n")
        f.write(f"  {metrics['confusion_matrix'].tolist()}\n\n")

        f.write("FEATURE IMPORTANCE (Top 15)\n")
        for idx, row in df_importance.head(15).iterrows():
            f.write(f"  {row['feature']}: {row['importance_norm']:.2%}\n")
        f.write("\n")

        f.write("SUCCESS CRITERIA\n")
        if metrics['cv_mean'] > 0.60:
            f.write(f"  ✅ CV Score > 60%: {metrics['cv_mean']:.4f}\n")
        else:
            f.write(f"  ❌ CV Score > 60%: {metrics['cv_mean']:.4f}\n")

    print(f"✅ Log saved")

    print("\n" + "=" * 80)
    print("✅ Phase 4.2 Complete")
    print("=" * 80)
    print(f"\nDeliverables:")
    print(f"  - {MODEL_FILE}")
    print(f"  - {IMPORTANCE_CSV}")
    print(f"  - {LOG_FILE}")
    print(f"\nCV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
