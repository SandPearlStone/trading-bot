"""
phase5/labeler.py — Triple Barrier Method labeling.

For each row:
  - TP barrier = close + TP_ATR_MULTIPLE * ATR
  - SL barrier = close - SL_ATR_MULTIPLE * ATR
  - Timeout = TIMEOUT_CANDLES candles

Label:
  +1  → price hits TP before SL or timeout
  -1  → price hits SL before TP or timeout
   0  → timeout (neither hit within window)

Usage:
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.label(df_features)
    print(df_labeled["label"].value_counts())
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    TP_ATR_MULTIPLE, SL_ATR_MULTIPLE, TIMEOUT_CANDLES,
    PROCESSED_DIR, ATR_PERIOD,
)

log = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Apply triple barrier method to a feature DataFrame.
    Expects columns: close, atr_pct (or atr), and optionally high/low.
    """

    def __init__(
        self,
        tp_multiple: float = TP_ATR_MULTIPLE,
        sl_multiple: float = SL_ATR_MULTIPLE,
        timeout: int = TIMEOUT_CANDLES,
        direction: str = "long",  # "long" | "short" | "both"
    ):
        self.tp_multiple = tp_multiple
        self.sl_multiple = sl_multiple
        self.timeout = timeout
        self.direction = direction

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'label' column to df.
        Requires: close (in df or df.index named close), atr (numeric).
        If 'high'/'low' not in df, uses close as proxy.
        """
        df = df.copy()

        # Ensure we have close and atr
        if "close" not in df.columns:
            raise ValueError("DataFrame must have a 'close' column")
        if "atr" not in df.columns:
            # Recompute from atr_pct * close
            if "atr_pct" in df.columns:
                df["atr"] = df["atr_pct"] * df["close"]
            else:
                raise ValueError("DataFrame must have 'atr' or 'atr_pct' column")

        close = df["close"].values
        atr   = df["atr"].values
        high  = df["high"].values  if "high" in df.columns else close
        low   = df["low"].values   if "low"  in df.columns else close

        labels = self._vectorised_label(close, high, low, atr)
        df["label"] = labels
        return df

    def _vectorised_label(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
    ) -> np.ndarray:
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - self.timeout - 1):
            entry = close[i]
            atr_i = atr[i]
            if atr_i <= 0 or np.isnan(atr_i):
                labels[i] = 0
                continue

            tp = entry + self.tp_multiple * atr_i
            sl = entry - self.sl_multiple * atr_i

            label = 0
            for j in range(i + 1, min(i + self.timeout + 1, n)):
                hi_j = high[j]
                lo_j = low[j]

                if self.direction == "long":
                    if hi_j >= tp:
                        label = 1
                        break
                    if lo_j <= sl:
                        label = -1
                        break
                elif self.direction == "short":
                    if lo_j <= sl:
                        label = 1   # short hit TP (price went down)
                        break
                    if hi_j >= tp:
                        label = -1  # short hit SL (price went up)
                        break
                else:  # "both" — pure close-to-close
                    c_j = close[j]
                    if c_j >= tp:
                        label = 1
                        break
                    if c_j <= sl:
                        label = -1
                        break

            labels[i] = label

        # Last timeout rows get 0 (no future data)
        labels[n - self.timeout - 1:] = 0
        return labels

    def label_from_csv(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        """Load processed features CSV + raw OHLCV and apply labeling."""
        from phase5.config import DATA_DIR
        feat_path = os.path.join(PROCESSED_DIR, f"{symbol}_{tf}_features.csv")
        raw_path  = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")

        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Features CSV not found: {feat_path}")

        df_feat = pd.read_csv(feat_path, parse_dates=["timestamp"])
        df_feat = df_feat.set_index("timestamp")

        # Merge OHLCV back in for barrier calculation
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
            df_raw = df_raw.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df_feat.columns:
                    df_feat[col] = df_raw[col]

        return self.label(df_feat)

    def validate_distribution(self, df: pd.DataFrame) -> dict:
        """Return label distribution stats and balance check."""
        counts = df["label"].value_counts().to_dict()
        total = len(df)
        dist = {k: (v, round(v/total*100, 1)) for k, v in sorted(counts.items())}

        pos = counts.get(1, 0)
        neg = counts.get(-1, 0)
        timeout = counts.get(0, 0)
        ratio = pos / max(neg, 1)

        result = {
            "distribution": dist,
            "total": total,
            "win_rate": round(pos / max(pos + neg, 1) * 100, 1),
            "pos_neg_ratio": round(ratio, 2),
            "timeout_pct": round(timeout / total * 100, 1),
            "balanced": 0.8 <= ratio <= 1.5,
        }
        return result


def label_all_symbols(
    symbols: list[str],
    tf: str = "4h",
    direction: str = "long",
) -> pd.DataFrame:
    """Apply triple barrier labeling to all symbols, return combined DataFrame."""
    labeler = TripleBarrierLabeler(direction=direction)
    frames = []
    for sym in symbols:
        try:
            df = labeler.label_from_csv(sym, tf)
            df["symbol"] = sym
            stats = labeler.validate_distribution(df)
            log.info(
                f"{sym}: {stats['total']} rows | "
                f"WR={stats['win_rate']}% | "
                f"bal={stats['balanced']}"
            )
            frames.append(df)
        except FileNotFoundError:
            log.warning(f"{sym}: feature CSV not found, skipping")
        except Exception as e:
            log.error(f"{sym}: labeling error — {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, sort=True)
    combined = combined.dropna(subset=["label"])
    log.info(f"Combined: {len(combined)} labeled rows across {len(frames)} symbols")
    return combined


if __name__ == "__main__":
    import argparse
    from phase5.config import SYMBOLS, PROCESSED_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--tf", default="4h")
    parser.add_argument("--direction", default="long")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = label_all_symbols(args.symbols, args.tf, args.direction)
    if df.empty:
        print("No data labeled. Run data_fetcher + feature_engineer first.")
    else:
        stats = TripleBarrierLabeler().validate_distribution(df)
        print(f"\nLabel distribution:")
        for k, (cnt, pct) in stats["distribution"].items():
            name = {1: "WIN (+1)", -1: "LOSS (-1)", 0: "TIMEOUT (0)"}.get(k, str(k))
            print(f"  {name}: {cnt:,} ({pct}%)")
        print(f"Win rate: {stats['win_rate']}%")
        print(f"Balanced: {stats['balanced']}")

        out = args.out or os.path.join(PROCESSED_DIR, f"labeled_{args.tf}.csv")
        df.to_csv(out)
        print(f"Saved → {out}")
