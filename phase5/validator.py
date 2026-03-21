"""
phase5/validator.py — Walk-forward backtesting + Phase 1-4 comparison.

Calculates: Sharpe, Win Rate, Max DD, Profit Factor
Generates: metrics CSV + comparison table

Usage:
    python3 validator.py --data data/processed/labeled_4h.csv
    python3 validator.py --data ... --compare ../compare_phases_results.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    CONFIDENCE_THRESHOLD, LOGS_DIR, PROCESSED_DIR,
    WF_N_SPLITS, WF_TRAIN_FRAC, WF_EMBARGO,
)
from phase5.model_trainer import walk_forward_splits
from phase5.ensemble import EnsemblePredictor
from phase5.meta_labeler import MetaLabeler

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "validator.log")),
    ],
)


# ── Financial metrics ─────────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 6) -> float:
    """Annualized Sharpe (4h = 6 candles/day × 252 trading days)."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as fraction."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def calc_backtest_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    use_gate: bool = True,
) -> dict:
    """
    Simulate backtest returns given true labels and predictions.

    Assumes:
      - Each trade has equal position size = 1R
      - Win  (+1 label correctly predicted) = +2R (TP = 2*ATR)
      - Loss (-1 label, or wrong direction) = -1.5R (SL = 1.5*ATR)
      - Timeout (0) = 0R
    """
    TP_R  = 2.0
    SL_R  = 1.5

    # Apply confidence gate
    if use_gate and confidences is not None:
        mask = confidences >= threshold
    else:
        mask = np.ones(len(labels), dtype=bool)

    returns = []
    for i in range(len(labels)):
        if not mask[i]:
            continue
        true_l = labels[i]
        pred_l = predictions[i]

        if true_l == 0:  # timeout
            returns.append(0.0)
        elif pred_l == true_l:  # correct direction
            returns.append(TP_R)
        else:  # wrong direction
            returns.append(-SL_R)

    returns = np.array(returns) if returns else np.array([0.0])
    equity  = np.cumsum(returns) + 100  # start at $100

    win_trades   = (returns > 0).sum()
    loss_trades  = (returns < 0).sum()
    total_trades = int(mask.sum())

    return {
        "win_rate":        round((win_trades / max(win_trades + loss_trades, 1)) * 100, 2),
        "sharpe":          round(sharpe_ratio(returns), 3),
        "max_drawdown":    round(max_drawdown(equity) * 100, 2),
        "profit_factor":   round(profit_factor(returns), 3),
        "total_trades":    total_trades,
        "gated_trades":    int(mask.sum()),
        "avg_return_r":    round(float(returns.mean()), 4),
        "total_pnl_r":     round(float(returns.sum()), 2),
    }


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validate(data_path: str, n_splits: int = WF_N_SPLITS) -> list[dict]:
    """Full walk-forward backtest: ensemble + meta-labeler gate."""
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

        # Train
        ens = EnsemblePredictor()
        ens.fit(X_tr, y_tr)

        meta = MetaLabeler()
        meta.fit(X_tr, y_tr, ensemble=ens)

        # Predict on test
        directions, confidences, trade_mask = meta.predict_with_gate(X_te)

        # Without gate
        m_nogated = calc_backtest_metrics(y_te.values, directions, use_gate=False)
        # With gate
        m_gated   = calc_backtest_metrics(y_te.values, directions, confidences, threshold=CONFIDENCE_THRESHOLD)

        result = {
            "fold":        fold + 1,
            "train_rows":  len(tr_idx),
            "test_rows":   len(te_idx),
            **{f"gated_{k}":   v for k, v in m_gated.items()},
            **{f"ungated_{k}": v for k, v in m_nogated.items()},
        }
        log.info(
            f"Fold {fold+1}: "
            f"gated_wr={m_gated['win_rate']}% sharpe={m_gated['sharpe']} "
            f"dd={m_gated['max_drawdown']}% pf={m_gated['profit_factor']}"
        )
        results.append(result)

    return results


# ── Compare vs Phase 1-4 ──────────────────────────────────────────────────────

def compare_with_phases(
    phase5_results: list[dict],
    compare_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build comparison table: Phase 5 vs Phase 1-4."""
    # Phase 5 aggregated
    p5 = {
        "model":        "Phase 5 (ML Ensemble)",
        "win_rate":     round(np.mean([r["gated_win_rate"] for r in phase5_results]), 2),
        "sharpe":       round(np.mean([r["gated_sharpe"] for r in phase5_results]), 3),
        "max_drawdown": round(np.mean([r["gated_max_drawdown"] for r in phase5_results]), 2),
        "profit_factor":round(np.mean([r["gated_profit_factor"] for r in phase5_results]), 3),
        "avg_trades":   round(np.mean([r["gated_gated_trades"] for r in phase5_results]), 0),
    }

    rows = [p5]

    if compare_csv and os.path.exists(compare_csv):
        df_compare = pd.read_csv(compare_csv)
        for phase_col in ["Phase1", "Phase2", "Phase4"]:
            if phase_col not in df_compare.columns:
                continue
            pivot = df_compare.pivot(index="Symbol", columns="Metric", values=phase_col) if "Symbol" in df_compare.columns else None
            if pivot is None:
                continue
            row = {"model": phase_col}
            for metric, col in [("win_rate", "win_rate"), ("sharpe", "sharpe_ratio"),
                                  ("max_drawdown", "max_drawdown"), ("profit_factor", "profit_factor")]:
                if col in pivot.columns:
                    row[metric] = round(pivot[col].mean(), 3)
            rows.append(row)

    comparison = pd.DataFrame(rows)
    return comparison


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", type=int, default=WF_N_SPLITS)
    parser.add_argument("--compare", default=None, help="Path to compare_phases_results.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log.info("Starting walk-forward validation...")
    results = walk_forward_validate(args.data, args.splits)

    print("\n=== PHASE 5 WALK-FORWARD BACKTEST ===")
    print(f"{'Fold':<6} {'WR%':<8} {'Sharpe':<8} {'Max DD%':<10} {'PF':<8} {'Trades'}")
    for r in results:
        print(
            f"  {r['fold']:<4} "
            f"{r.get('gated_win_rate','?'):<8} "
            f"{r.get('gated_sharpe','?'):<8} "
            f"{r.get('gated_max_drawdown','?'):<10} "
            f"{r.get('gated_profit_factor','?'):<8} "
            f"{r.get('gated_gated_trades','?')}"
        )

    # Comparison
    compare_path = args.compare or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "compare_phases_results.csv"
    )
    comparison = compare_with_phases(results, compare_path)
    print("\n=== PHASE COMPARISON ===")
    print(comparison.to_string(index=False))

    # Save results
    out_dir = args.out or PROCESSED_DIR
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "phase5_backtest_results.csv"), index=False)
    comparison.to_csv(os.path.join(out_dir, "phase5_vs_phases_comparison.csv"), index=False)
    print(f"\nSaved results → {out_dir}")
