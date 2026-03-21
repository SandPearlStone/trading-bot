"""
phase5/pipeline.py — End-to-end pipeline runner.

Orchestrates: data → features → labels → train → validate → deploy

Usage:
    python3 pipeline.py --step all          # full pipeline
    python3 pipeline.py --step data         # download only
    python3 pipeline.py --step features     # build features only
    python3 pipeline.py --step labels       # label only
    python3 pipeline.py --step train        # train models
    python3 pipeline.py --step validate     # backtest
    python3 pipeline.py --step hyperopt     # background optuna
    python3 pipeline.py --quick             # fast test run
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    SYMBOLS, TIMEFRAMES, PRIMARY_TF, SECONDARY_TF,
    DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR,
    WF_N_SPLITS,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log")),
    ],
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def step_data(symbols: list[str], tfs: list[str], days: int = 730):
    """Download OHLCV data."""
    log.info("=== STEP: DATA DOWNLOAD ===")
    from phase5.data_fetcher import download_symbol, validate_alignment
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})

    total = len(symbols) * len(tfs)
    done = 0
    for sym in symbols:
        for tf in tfs:
            log.info(f"[{done+1}/{total}] {sym} {tf}")
            try:
                df = download_symbol(exchange, sym, tf, days)
                log.info(f"  ✓ {len(df):,} rows")
            except Exception as e:
                log.error(f"  ✗ {sym}/{tf}: {e}")
            done += 1

    validate_alignment(symbols, tfs)
    log.info("Data step complete.")


def step_features(symbols: list[str], primary_tf: str = PRIMARY_TF):
    """Build feature CSVs for all symbols."""
    log.info("=== STEP: FEATURE ENGINEERING ===")
    from phase5.feature_engineer import build_features_from_csv

    for sym in symbols:
        try:
            df = build_features_from_csv(sym, primary_tf, SECONDARY_TF)
            out = os.path.join(PROCESSED_DIR, f"{sym}_{primary_tf}_features.csv")
            df.to_csv(out)
            log.info(f"  ✓ {sym}: {len(df)} rows × {len(df.columns)} features → {out}")
        except FileNotFoundError:
            log.warning(f"  ✗ {sym}: CSV not found (run data step first)")
        except Exception as e:
            log.error(f"  ✗ {sym}: {e}")


def step_labels(symbols: list[str], tf: str = PRIMARY_TF) -> str:
    """Apply triple barrier labeling, return path to labeled CSV."""
    log.info("=== STEP: LABELING ===")
    from phase5.labeler import label_all_symbols, TripleBarrierLabeler

    df = label_all_symbols(symbols, tf)
    if df.empty:
        log.error("No data labeled. Check feature step.")
        return ""

    stats = TripleBarrierLabeler().validate_distribution(df)
    log.info(f"Label distribution: {stats['distribution']}")
    log.info(f"Win rate: {stats['win_rate']}% | Balanced: {stats['balanced']}")

    out = os.path.join(PROCESSED_DIR, f"labeled_{tf}.csv")
    df.to_csv(out)
    log.info(f"Labeled data → {out}")
    return out


def step_train(data_path: str, quick: bool = False) -> dict:
    """Train LGB + XGB with walk-forward CV."""
    log.info("=== STEP: MODEL TRAINING ===")
    from phase5.model_trainer import train_all

    n_splits = 2 if quick else WF_N_SPLITS
    results = train_all(data_path, n_splits=n_splits)
    for model_name, info in results.items():
        log.info(f"{model_name.upper()} CV avg: {info['avg']}")
    return results


def step_hyperopt(data_path: str, trials: int = 50, background: bool = True) -> None:
    """Launch Optuna hyperopt (optionally in background)."""
    log.info("=== STEP: HYPEROPT ===")
    cmd = [
        sys.executable, os.path.join(_HERE, "hyperopt.py"),
        "--data", data_path,
        "--trials", str(trials),
    ]
    if background:
        log.info(f"Launching hyperopt in background: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=open(os.path.join(LOGS_DIR, "hyperopt_bg.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        log.info(f"Hyperopt PID: {proc.pid}")
    else:
        subprocess.run(cmd, check=True)


def step_ensemble_cv(data_path: str, quick: bool = False):
    """Walk-forward ensemble CV."""
    log.info("=== STEP: ENSEMBLE CV ===")
    from phase5.ensemble import walk_forward_ensemble_cv
    import numpy as np

    n = 2 if quick else WF_N_SPLITS
    results = walk_forward_ensemble_cv(data_path, n_splits=n)
    avg_wr = round(float(np.mean([r.get("win_rate", 0) for r in results])), 2)
    log.info(f"Ensemble avg win rate: {avg_wr}%")
    return results


def step_meta(data_path: str, save: bool = True, quick: bool = False):
    """Train meta-labeler + cross-validate."""
    log.info("=== STEP: META-LABELING ===")
    from phase5.meta_labeler import cross_validate_meta, MetaLabeler, EnsemblePredictor
    import numpy as np
    import pandas as pd

    n = 2 if quick else WF_N_SPLITS
    results = cross_validate_meta(data_path, n_splits=n)
    avg_gated_wr = round(float(np.mean([r["gated_win_rate"] for r in results])), 2)
    avg_gate_pct = round(float(np.mean([r["gate_pct"] for r in results])), 1)
    log.info(f"Meta CV: gated_wr={avg_gated_wr}% gate%={avg_gate_pct}%")

    if save:
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

        ens = EnsemblePredictor()
        ens.fit(X, y)
        meta = MetaLabeler()
        meta.fit(X, y, ensemble=ens)
        meta.save()
        log.info("Final MetaLabeler saved.")

    return results


def step_validate(data_path: str, quick: bool = False):
    """Walk-forward validation + phase comparison."""
    log.info("=== STEP: VALIDATION ===")
    from phase5.validator import walk_forward_validate, compare_with_phases
    import numpy as np

    n = 2 if quick else WF_N_SPLITS
    results = walk_forward_validate(data_path, n_splits=n)

    compare_csv = os.path.join(os.path.dirname(_HERE), "compare_phases_results.csv")
    comparison = compare_with_phases(results, compare_csv)

    log.info("\n=== PHASE COMPARISON ===")
    log.info(f"\n{comparison.to_string(index=False)}")

    # Save
    import pandas as pd
    pd.DataFrame(results).to_csv(
        os.path.join(PROCESSED_DIR, "phase5_backtest_results.csv"), index=False
    )
    comparison.to_csv(
        os.path.join(PROCESSED_DIR, "phase5_vs_phases_comparison.csv"), index=False
    )

    avg_sharpe = round(float(np.mean([r.get("gated_sharpe", 0) for r in results])), 3)
    avg_wr     = round(float(np.mean([r.get("gated_win_rate", 0) for r in results])), 2)
    log.info(f"Phase 5: avg_sharpe={avg_sharpe} avg_win_rate={avg_wr}%")
    return results, comparison


def run_pipeline(
    symbols: list[str],
    tfs: list[str],
    steps: list[str],
    quick: bool = False,
    hyperopt_trials: int = 50,
    hyperopt_bg: bool = True,
):
    """Run pipeline steps in order."""
    start = time.time()
    labeled_path = os.path.join(PROCESSED_DIR, f"labeled_{PRIMARY_TF}.csv")

    for step in steps:
        t0 = time.time()
        try:
            if step == "data":
                step_data(symbols, tfs, days=730 if not quick else 90)
            elif step == "features":
                step_features(symbols, PRIMARY_TF)
            elif step == "labels":
                path = step_labels(symbols, PRIMARY_TF)
                if path:
                    labeled_path = path
            elif step == "train":
                if os.path.exists(labeled_path):
                    step_train(labeled_path, quick=quick)
                else:
                    log.error(f"Labeled data not found: {labeled_path}")
            elif step == "hyperopt":
                if os.path.exists(labeled_path):
                    step_hyperopt(labeled_path, trials=hyperopt_trials, background=hyperopt_bg)
            elif step == "ensemble":
                if os.path.exists(labeled_path):
                    step_ensemble_cv(labeled_path, quick=quick)
            elif step == "meta":
                if os.path.exists(labeled_path):
                    step_meta(labeled_path, save=True, quick=quick)
            elif step == "validate":
                if os.path.exists(labeled_path):
                    step_validate(labeled_path, quick=quick)
            else:
                log.warning(f"Unknown step: {step}")
        except Exception as e:
            log.error(f"Step '{step}' failed: {e}", exc_info=True)

        elapsed = time.time() - t0
        log.info(f"Step '{step}' done in {elapsed:.1f}s")

    total = time.time() - start
    log.info(f"\n=== PIPELINE COMPLETE in {total/60:.1f} min ===")


if __name__ == "__main__":
    ALL_STEPS = ["data", "features", "labels", "train", "hyperopt", "ensemble", "meta", "validate"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all", help=f"Step(s) to run: {ALL_STEPS} or 'all'")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--tfs", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--quick", action="store_true", help="Fast run: less data, 2 CV splits")
    parser.add_argument("--hyperopt-trials", type=int, default=50)
    parser.add_argument("--no-hyperopt-bg", action="store_true", help="Run hyperopt in foreground")
    args = parser.parse_args()

    if args.step == "all":
        steps = ALL_STEPS
    else:
        steps = args.step.split(",")

    run_pipeline(
        symbols=args.symbols,
        tfs=args.tfs,
        steps=steps,
        quick=args.quick,
        hyperopt_trials=args.hyperopt_trials,
        hyperopt_bg=not args.no_hyperopt_bg,
    )
