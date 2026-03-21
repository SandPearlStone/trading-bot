"""
phase5/hyperopt.py — Optuna hyperparameter optimization for LGB + XGB.

Usage:
    python3 hyperopt.py --data data/processed/labeled_4h.csv --trials 200
    python3 hyperopt.py --data ... --trials 50 --model lgb   # LGB only
    python3 hyperopt.py --data ... --resume                  # continue study
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    OPTUNA_N_TRIALS, LGB_SEARCH_SPACE, XGB_SEARCH_SPACE,
    LGB_OPT_MODEL_PATH, XGB_OPT_MODEL_PATH, LOGS_DIR, WF_N_SPLITS,
    WF_TRAIN_FRAC, WF_EMBARGO,
)
from phase5.model_trainer import (
    LGBTrainer, XGBTrainer, walk_forward_splits, compute_metrics,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "hyperopt.log")),
    ],
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.error("optuna not installed. Run: pip install optuna")


# ── Objective helpers ─────────────────────────────────────────────────────────

def _suggest_params(trial: "optuna.Trial", search_space: dict, prefix: str = "") -> dict:
    params = {}
    for name, spec in search_space.items():
        full_name = f"{prefix}{name}"
        if isinstance(spec, tuple):
            if len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = trial.suggest_int(full_name, lo, hi)
                else:
                    params[name] = trial.suggest_float(full_name, lo, hi)
            elif len(spec) == 3 and spec[2] == "log":
                params[name] = trial.suggest_float(full_name, spec[0], spec[1], log=True)
    return params


def _objective_lgb(trial, X: pd.DataFrame, y: pd.Series) -> float:
    from lightgbm import LGBMClassifier
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 4, 10),
        **_suggest_params(trial, LGB_SEARCH_SPACE, prefix="lgb_"),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    scores = []
    for tr_idx, te_idx in walk_forward_splits(len(X), 2, WF_TRAIN_FRAC, WF_EMBARGO):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        m = compute_metrics(y_te.values, y_pred)
        scores.append(m["win_rate"])

    return float(np.mean(scores))


def _objective_xgb(trial, X: pd.DataFrame, y: pd.Series) -> float:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        **_suggest_params(trial, XGB_SEARCH_SPACE, prefix="xgb_"),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    }

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scores = []
    for tr_idx, te_idx in walk_forward_splits(len(X), 2, WF_TRAIN_FRAC, WF_EMBARGO):
        X_tr, y_tr = X.iloc[tr_idx], y_enc[tr_idx]
        X_te, y_te_enc = X.iloc[te_idx], y_enc[te_idx]
        y_te_orig = y.iloc[te_idx].values
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_pred_enc = model.predict(X_te)
        y_pred = le.inverse_transform(y_pred_enc)
        m = compute_metrics(y_te_orig, y_pred)
        scores.append(m["win_rate"])

    return float(np.mean(scores))


# ── Run optimization ──────────────────────────────────────────────────────────

def run_hyperopt(
    data_path: str,
    n_trials: int = OPTUNA_N_TRIALS,
    model: str = "both",
    study_name: str = "phase5",
    storage: str = None,
) -> dict:
    """Run Optuna optimization. Returns best params per model."""
    if not HAS_OPTUNA:
        raise ImportError("optuna not installed")

    log.info(f"Loading {data_path}")
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

    log.info(f"Data: {len(X)} rows × {len(feat_cols)} features")

    results = {}
    sampler = optuna.samplers.TPESampler(seed=42)

    # ── LGB ───────────────────────────────────────────────────────────────────
    if model in ("lgb", "both") and HAS_OPTUNA:
        try:
            from lightgbm import LGBMClassifier
            log.info(f"Starting LGB Optuna search ({n_trials} trials)...")

            storage_url = storage or f"sqlite:///{LOGS_DIR}/optuna_lgb.db"
            study = optuna.create_study(
                study_name=f"{study_name}_lgb",
                direction="maximize",
                sampler=sampler,
                storage=storage_url,
                load_if_exists=True,
            )
            study.optimize(
                lambda t: _objective_lgb(t, X, y),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best_lgb = study.best_params
            log.info(f"Best LGB: win_rate={study.best_value:.1f}% params={best_lgb}")

            # Retrain final model with best params
            lgb_params = {
                "n_estimators":      best_lgb.get("n_estimators", 200),
                "max_depth":         best_lgb.get("max_depth", 7),
                "num_leaves":        best_lgb.get("lgb_num_leaves", 63),
                "learning_rate":     best_lgb.get("lgb_learning_rate", 0.05),
                "min_child_samples": best_lgb.get("lgb_min_child_samples", 20),
                "subsample":         best_lgb.get("lgb_subsample", 0.8),
                "colsample_bytree":  best_lgb.get("lgb_colsample_bytree", 0.8),
                "reg_alpha":         best_lgb.get("lgb_reg_alpha", 0.1),
                "reg_lambda":        best_lgb.get("lgb_reg_lambda", 0.1),
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            }
            trainer = LGBTrainer(params=lgb_params)
            trainer.fit(X, y)
            trainer.save(LGB_OPT_MODEL_PATH)

            results["lgb"] = {
                "best_params": lgb_params,
                "best_win_rate": study.best_value,
                "n_trials": len(study.trials),
            }
        except Exception as e:
            log.error(f"LGB optimization failed: {e}")

    # ── XGB ───────────────────────────────────────────────────────────────────
    if model in ("xgb", "both") and HAS_OPTUNA:
        try:
            from xgboost import XGBClassifier
            log.info(f"Starting XGB Optuna search ({n_trials} trials)...")

            storage_url = storage or f"sqlite:///{LOGS_DIR}/optuna_xgb.db"
            study = optuna.create_study(
                study_name=f"{study_name}_xgb",
                direction="maximize",
                sampler=sampler,
                storage=storage_url,
                load_if_exists=True,
            )
            study.optimize(
                lambda t: _objective_xgb(t, X, y),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best_xgb = study.best_params
            log.info(f"Best XGB: win_rate={study.best_value:.1f}% params={best_xgb}")

            xgb_params = {
                "n_estimators":     best_xgb.get("n_estimators", 200),
                "max_depth":        best_xgb.get("xgb_max_depth", 6),
                "learning_rate":    best_xgb.get("xgb_learning_rate", 0.05),
                "min_child_weight": best_xgb.get("xgb_min_child_weight", 5),
                "subsample":        best_xgb.get("xgb_subsample", 0.8),
                "colsample_bytree": best_xgb.get("xgb_colsample_bytree", 0.8),
                "reg_alpha":        best_xgb.get("xgb_reg_alpha", 0.1),
                "reg_lambda":       best_xgb.get("xgb_reg_lambda", 1.0),
                "random_state": 42, "n_jobs": -1, "verbosity": 0,
                "eval_metric": "mlogloss",
            }
            trainer = XGBTrainer(params=xgb_params)
            trainer.fit(X, y)
            trainer.save(XGB_OPT_MODEL_PATH)

            results["xgb"] = {
                "best_params": xgb_params,
                "best_win_rate": study.best_value,
                "n_trials": len(study.trials),
            }
        except Exception as e:
            log.error(f"XGB optimization failed: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--trials", type=int, default=OPTUNA_N_TRIALS)
    parser.add_argument("--model", choices=["lgb", "xgb", "both"], default="both")
    parser.add_argument("--study-name", default="phase5")
    parser.add_argument("--storage", default=None, help="Optuna storage URL")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    args = parser.parse_args()

    results = run_hyperopt(args.data, args.trials, args.model, args.study_name, args.storage)
    print("\n=== HYPEROPT COMPLETE ===")
    for name, info in results.items():
        print(f"\n{name.upper()}: best_win_rate={info.get('best_win_rate', '?'):.1f}%")
        print(f"  Best params: {info.get('best_params', {})}")
