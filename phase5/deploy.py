"""
phase5/deploy.py — Live inference engine.

Loads trained models, fetches real-time OHLCV, computes features,
runs ensemble + confidence gate, returns trade signals.

Integrates with existing Phase 1-3 scoring via `phase5_score()`.

Usage:
    python3 deploy.py --symbol BTCUSDT --tf 4h
    python3 deploy.py --live --symbols BTCUSDT ETHUSDT  # continuous loop
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import (
    SYMBOLS, PRIMARY_TF, SECONDARY_TF, CONFIDENCE_THRESHOLD,
    META_MODEL_PATH, LGB_MODEL_PATH, XGB_MODEL_PATH,
    FEATURE_NAMES_PATH, LOGS_DIR, LIVE_DELAY_SECS,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "deploy.log")),
    ],
)


# ── Signal dataclass ──────────────────────────────────────────────────────────

class Phase5Signal:
    """Result of Phase 5 inference for one symbol."""
    def __init__(
        self,
        symbol: str,
        direction: int,          # +1 LONG, -1 SHORT, 0 NEUTRAL
        confidence: float,       # 0–1
        gate_passed: bool,       # True if confidence >= threshold
        features: dict,          # key indicators
        timestamp: datetime = None,
    ):
        self.symbol     = symbol
        self.direction  = direction
        self.confidence = confidence
        self.gate_passed= gate_passed
        self.features   = features
        self.timestamp  = timestamp or datetime.now(timezone.utc)

    @property
    def direction_label(self) -> str:
        return {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(self.direction, "?")

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "direction":   self.direction_label,
            "confidence":  round(self.confidence, 3),
            "gate_passed": self.gate_passed,
            "timestamp":   self.timestamp.isoformat(),
            **self.features,
        }

    def __repr__(self) -> str:
        return (
            f"Phase5Signal({self.symbol} {self.direction_label} "
            f"conf={self.confidence:.2f} gate={'✓' if self.gate_passed else '✗'})"
        )


# ── Inference engine ──────────────────────────────────────────────────────────

class Phase5InferenceEngine:
    """
    Load models once, run inference for any symbol on demand.
    Thread-safe for concurrent symbol processing.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold
        self._meta_model = None
        self._feature_names: list[str] = []
        self._loaded = False

    def load(self):
        """Load all models from disk."""
        # Load feature names
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, "rb") as f:
                self._feature_names = pickle.load(f)
            log.info(f"Loaded {len(self._feature_names)} feature names")

        # Load meta-labeler (contains ensemble internally)
        if os.path.exists(META_MODEL_PATH):
            from phase5.meta_labeler import MetaLabeler
            self._meta_model = MetaLabeler.load(META_MODEL_PATH)
            log.info("Loaded MetaLabeler (ensemble + confidence model)")
        else:
            log.warning(f"MetaLabeler not found at {META_MODEL_PATH}. Run meta_labeler.py --save first.")
            # Fallback: load raw ensemble
            self._meta_model = None

        self._loaded = True

    def _fetch_ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        """Fetch recent OHLCV from Binance via ccxt."""
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        bars = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    def infer(self, symbol: str, tf: str = PRIMARY_TF) -> Optional[Phase5Signal]:
        """Run full inference pipeline for one symbol."""
        if not self._loaded:
            self.load()

        try:
            # Fetch data
            df_primary = self._fetch_ohlcv(symbol, tf, limit=300)
            df_secondary = None
            if SECONDARY_TF != tf:
                try:
                    df_secondary = self._fetch_ohlcv(symbol, SECONDARY_TF, limit=300)
                except Exception:
                    pass

            # Engineer features
            from phase5.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()
            df_feat = fe.engineer(df_primary, df_secondary)

            # Use latest candle (last row)
            latest = df_feat.iloc[[-1]].copy()

            # Align features to trained feature set
            if self._feature_names:
                for col in self._feature_names:
                    if col not in latest.columns:
                        latest[col] = 0
                X = latest[self._feature_names].fillna(0)
            else:
                exclude = {"label", "symbol"}
                feat_cols = [c for c in latest.columns if c not in exclude
                             and latest[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, float, int]]
                X = latest[feat_cols].fillna(0)

            # Inference
            if self._meta_model is not None:
                directions, confidences, mask = self._meta_model.predict_with_gate(X)
                direction   = int(directions[0])
                confidence  = float(confidences[0])
                gate_passed = bool(mask[0])
            else:
                direction   = 0
                confidence  = 0.0
                gate_passed = False

            # Extract key indicators for context
            features_summary = {}
            for col in ["rsi", "adx", "macd_hist", "bb_pct_b", "atr_pct", "regime", "confluence"]:
                if col in df_feat.columns:
                    features_summary[col] = round(float(df_feat[col].iloc[-1]), 4)

            return Phase5Signal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                gate_passed=gate_passed,
                features=features_summary,
            )

        except Exception as e:
            log.error(f"Inference failed for {symbol}: {e}")
            return None

    def scan_all(self, symbols: list[str] = SYMBOLS, tf: str = PRIMARY_TF) -> list[Phase5Signal]:
        """Scan all symbols and return gated signals."""
        signals = []
        for sym in symbols:
            sig = self.infer(sym, tf)
            if sig:
                signals.append(sig)
                if sig.gate_passed:
                    log.info(f"  ✓ SIGNAL: {sig}")
                else:
                    log.debug(f"  ✗ gated out: {sig.symbol} conf={sig.confidence:.2f}")
        return signals


# ── Integration with Phase 1-3 ────────────────────────────────────────────────

# Module-level engine (loaded once per process)
_engine: Optional[Phase5InferenceEngine] = None


def phase5_score(
    symbol: str,
    tf: str = PRIMARY_TF,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Public API for integration with find_trades.py.

    Returns:
        {
            "available": bool,
            "direction": "LONG" | "SHORT" | "NEUTRAL",
            "confidence": float (0-1),
            "gate_passed": bool,
            "features": dict,
        }
    """
    global _engine
    if _engine is None:
        _engine = Phase5InferenceEngine(confidence_threshold=threshold)
        try:
            _engine.load()
        except Exception as e:
            log.error(f"Phase5 engine load failed: {e}")
            return {"available": False, "error": str(e)}

    sig = _engine.infer(symbol, tf)
    if sig is None:
        return {"available": False}

    return {
        "available":   True,
        "direction":   sig.direction_label,
        "confidence":  sig.confidence,
        "gate_passed": sig.gate_passed,
        "features":    sig.features,
    }


# ── Live loop ─────────────────────────────────────────────────────────────────

def live_loop(symbols: list[str], tf: str, delay: int = LIVE_DELAY_SECS):
    """Continuous live scanning loop."""
    engine = Phase5InferenceEngine()
    engine.load()

    log.info(f"Live loop started: {symbols} @ {tf}, delay={delay}s")
    while True:
        signals = engine.scan_all(symbols, tf)
        actionable = [s for s in signals if s.gate_passed]

        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] Scanned {len(signals)} symbols")
        if actionable:
            print(f"  ✓ {len(actionable)} actionable signals:")
            for s in actionable:
                print(f"    {s}")
        else:
            print("  ✗ No signals above confidence threshold")

        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="Single symbol")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--tf", default=PRIMARY_TF)
    parser.add_argument("--live", action="store_true", help="Continuous loop")
    parser.add_argument("--delay", type=int, default=LIVE_DELAY_SECS)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    if args.live:
        syms = [args.symbol] if args.symbol else args.symbols
        live_loop(syms, args.tf, args.delay)
    else:
        engine = Phase5InferenceEngine(confidence_threshold=args.threshold)
        engine.load()

        syms = [args.symbol] if args.symbol else args.symbols
        signals = engine.scan_all(syms, args.tf)

        print(f"\n=== PHASE 5 SIGNALS ({args.tf}) ===")
        for s in signals:
            status = "✓ TRADE" if s.gate_passed else "✗ gated"
            print(f"  [{status}] {s.symbol} {s.direction_label} conf={s.confidence:.2f}")
            if s.features:
                print(f"           RSI={s.features.get('rsi','?')} ADX={s.features.get('adx','?')} regime={s.features.get('regime','?')}")
