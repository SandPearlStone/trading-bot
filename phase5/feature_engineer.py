"""
phase5/feature_engineer.py — 40+ feature engineering pipeline.

Features:
  - Technical: RSI, MACD, ADX, Stochastic, BB, Keltner, ROC
  - Volume: Volume ratio, MFI, OBV
  - Volatility: ATR, Historical vol, vol_ratio
  - Custom: Hidden divergence, regime classification, confluence score
  - Multi-timeframe: 1h RSI when trading on 4h
  - Temporal: hour of day, day of week

Usage:
    fe = FeatureEngineer()
    df_features = fe.engineer(df_4h, df_1h=df_1h)
    # df_features has 40+ columns, no NaN values
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
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, STOCH_K, STOCH_D, BB_PERIOD, BB_STD,
    KELTNER_PERIOD, KELTNER_ATR, ROC_PERIOD, ATR_PERIOD,
    HIST_VOL_PERIOD, OBV_SMOOTH, MFI_PERIOD,
    EMA_FAST, EMA_SLOW, EMA_TREND,
)

log = logging.getLogger(__name__)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    atr_val = _atr(high, low, close, period)
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # resolve ambiguity
    both = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[both & (plus_dm <= minus_dm)] = 0
    minus_dm[both & (minus_dm <= plus_dm)] = 0

    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-9)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    return dx.ewm(alpha=1/period, adjust=False).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    stoch_k = 100 * (close - lo) / (hi - lo).replace(0, 1e-9)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _bollinger(close: pd.Series, period: int, std_mult: float):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, 1e-9)
    pct_b = (close - lower) / (upper - lower).replace(0, 1e-9)
    return mid, upper, lower, width, pct_b


def _keltner(close: pd.Series, high: pd.Series, low: pd.Series, period: int, atr_mult: float):
    mid = close.ewm(span=period, adjust=False).mean()
    atr_val = _atr(high, low, close, period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return mid, upper, lower


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, 1e-9)
    return 100 - 100 / (1 + mfr)


def _roc(close: pd.Series, period: int) -> pd.Series:
    return 100 * (close - close.shift(period)) / close.shift(period).replace(0, 1e-9)


def _hist_vol(close: pd.Series, period: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252 * 24)  # annualized


def _regime(close: pd.Series, adx: pd.Series) -> pd.Series:
    """Simple regime: 0=Choppy, 1=Normal, 2=Trending"""
    regime = pd.Series(1, index=close.index)
    regime[adx < 20] = 0
    regime[adx > 30] = 2
    return regime


def _hidden_divergence(close: pd.Series, rsi_vals: pd.Series, period: int = 14) -> pd.Series:
    """Hidden divergence signal: +1 bullish, -1 bearish, 0 none"""
    signal = pd.Series(0, index=close.index)
    n = len(close)
    for i in range(period * 2, n):
        # Look back period for local extremes
        c_window = close.iloc[i-period:i+1]
        r_window = rsi_vals.iloc[i-period:i+1]

        # Bullish hidden divergence: higher lows in price, lower lows in RSI
        price_lo1 = c_window.iloc[0]
        price_lo2 = c_window.iloc[-1]
        rsi_lo1 = r_window.iloc[0]
        rsi_lo2 = r_window.iloc[-1]

        if price_lo2 > price_lo1 and rsi_lo2 < rsi_lo1:
            signal.iloc[i] = 1
        elif price_lo2 < price_lo1 and rsi_lo2 > rsi_lo1:
            signal.iloc[i] = -1

    return signal


def _confluence_score(df: pd.DataFrame) -> pd.Series:
    """Simple confluence: count bullish signals (0-10 scale)."""
    score = pd.Series(0.0, index=df.index)
    # RSI zone
    score += (df["rsi"] < 40).astype(float) * 1.0
    score += (df["rsi"] > 60).astype(float) * -1.0
    # MACD
    score += (df["macd_hist"] > 0).astype(float) * 1.0
    score += (df["macd_hist"] < 0).astype(float) * -1.0
    # ADX trend
    score += (df["adx"] > 25).astype(float) * 0.5
    # BB position
    score += (df["bb_pct_b"] < 0.2).astype(float) * 1.0
    score += (df["bb_pct_b"] > 0.8).astype(float) * -1.0
    # Stoch
    score += (df["stoch_k"] < 25).astype(float) * 1.0
    score += (df["stoch_k"] > 75).astype(float) * -1.0
    # EMA trend
    score += (df["ema_fast"] > df["ema_slow"]).astype(float) * 0.5
    score += (df["ema_fast"] < df["ema_slow"]).astype(float) * -0.5

    return score.clip(-5, 5)


# ── Main Class ────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Compute 40+ features from OHLCV data.
    Optionally merges a secondary (1h) timeframe for MTF features.
    """

    def __init__(self):
        self.feature_names: list[str] = []

    def engineer(
        self,
        df_primary: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_primary   : OHLCV DataFrame on primary timeframe (e.g. 4h).
                       Columns: timestamp, open, high, low, close, volume
        df_secondary : OHLCV DataFrame on secondary timeframe (e.g. 1h).
                       Used for MTF features. Optional.

        Returns
        -------
        DataFrame with all features, NaNs filled.
        """
        df = df_primary.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp") if not isinstance(df.index, pd.DatetimeIndex) else df

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── RSI ───────────────────────────────────────────────────────────
        df["rsi"] = _rsi(close, RSI_PERIOD)
        df["rsi_6"] = _rsi(close, 6)
        df["rsi_21"] = _rsi(close, 21)

        # ── MACD ──────────────────────────────────────────────────────────
        macd, signal, hist = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df["macd"]       = macd
        df["macd_signal"]= signal
        df["macd_hist"]  = hist
        df["macd_cross"] = (np.sign(hist) != np.sign(hist.shift(1))).astype(int)

        # ── ADX ───────────────────────────────────────────────────────────
        df["adx"] = _adx(high, low, close, ADX_PERIOD)

        # ── Stochastic ────────────────────────────────────────────────────
        df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, STOCH_K, STOCH_D)
        df["stoch_cross"] = ((df["stoch_k"] > df["stoch_d"]) &
                             (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))).astype(int)

        # ── Bollinger Bands ───────────────────────────────────────────────
        df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"], df["bb_pct_b"] = (
            _bollinger(close, BB_PERIOD, BB_STD)
        )
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)

        # ── Keltner Channel ───────────────────────────────────────────────
        df["kc_mid"], df["kc_upper"], df["kc_lower"] = (
            _keltner(close, high, low, KELTNER_PERIOD, KELTNER_ATR)
        )
        df["kc_pct"] = (close - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"]).replace(0, 1e-9)

        # ── ROC ───────────────────────────────────────────────────────────
        df["roc_10"] = _roc(close, ROC_PERIOD)
        df["roc_20"] = _roc(close, 20)

        # ── EMA ───────────────────────────────────────────────────────────
        df["ema_fast"]  = close.ewm(span=EMA_FAST, adjust=False).mean()
        df["ema_slow"]  = close.ewm(span=EMA_SLOW, adjust=False).mean()
        df["ema_trend"] = close.ewm(span=EMA_TREND, adjust=False).mean()
        df["ema_ratio"] = df["ema_fast"] / df["ema_slow"].replace(0, 1e-9)
        df["price_vs_ema_trend"] = (close - df["ema_trend"]) / df["ema_trend"].replace(0, 1e-9)

        # ── ATR & Volatility ──────────────────────────────────────────────
        df["atr"] = _atr(high, low, close, ATR_PERIOD)
        df["atr_pct"] = df["atr"] / close.replace(0, 1e-9)
        df["hist_vol"] = _hist_vol(close, HIST_VOL_PERIOD)
        df["vol_ratio"] = df["atr"] / df["atr"].rolling(50).mean().replace(0, 1e-9)

        # ── Volume ────────────────────────────────────────────────────────
        df["vol_ratio_20"] = volume / volume.rolling(20).mean().replace(0, 1e-9)
        df["vol_ratio_5"]  = volume / volume.rolling(5).mean().replace(0, 1e-9)
        df["obv"]          = _obv(close, volume)
        df["obv_signal"]   = df["obv"].ewm(span=OBV_SMOOTH, adjust=False).mean()
        df["obv_diff"]     = df["obv"] - df["obv_signal"]
        df["mfi"]          = _mfi(high, low, close, volume, MFI_PERIOD)

        # ── Price structure ────────────────────────────────────────────────
        df["candle_body"]  = (close - df["open"]) / df["atr"].replace(0, 1e-9)
        df["candle_wick_hi"]  = (high - close.clip(lower=df["open"])) / df["atr"].replace(0, 1e-9)
        df["candle_wick_lo"]  = (close.clip(upper=df["open"]) - low) / df["atr"].replace(0, 1e-9)
        df["range_pct"]    = (high - low) / close.replace(0, 1e-9)

        # ── Regime ────────────────────────────────────────────────────────
        df["regime"] = _regime(close, df["adx"])

        # ── Hidden divergence ─────────────────────────────────────────────
        df["hidden_div"] = _hidden_divergence(close, df["rsi"], RSI_PERIOD)

        # ── Confluence score ──────────────────────────────────────────────
        df["confluence"] = _confluence_score(df)

        # ── Temporal features ─────────────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour_of_day"]  = df.index.hour
            df["day_of_week"]  = df.index.dayofweek
            df["month"]        = df.index.month
            df["hour_sin"]     = np.sin(2 * np.pi * df["hour_of_day"] / 24)
            df["hour_cos"]     = np.cos(2 * np.pi * df["hour_of_day"] / 24)
            df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # ── Multi-Timeframe features (1h → 4h) ───────────────────────────
        if df_secondary is not None:
            df = self._merge_mtf(df, df_secondary)

        # ── Drop raw price columns not needed as features ──────────────────
        drop_cols = ["open", "high", "low", "close", "volume",
                     "bb_mid", "bb_upper", "bb_lower",
                     "kc_mid", "kc_upper", "kc_lower",
                     "ema_fast", "ema_slow", "ema_trend",
                     "obv", "obv_signal",
                     "macd", "macd_signal",
                     "symbol", "timeframe"]
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

        # ── Forward-fill then back-fill NaN ──────────────────────────────
        df = df.ffill().bfill()

        # Store feature names (everything except label if present)
        label_col = "label" if "label" in df.columns else None
        self.feature_names = [c for c in df.columns if c != label_col]

        log.info(f"FeatureEngineer: {len(self.feature_names)} features, {len(df)} rows")
        return df

    def _merge_mtf(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.DataFrame:
        """Merge secondary (1h) RSI/volume into primary (4h) rows."""
        sec = df_secondary.copy()
        if "timestamp" in sec.columns:
            sec = sec.set_index("timestamp")

        sec["rsi_1h"]      = _rsi(sec["close"], RSI_PERIOD)
        sec["rsi_1h_6"]    = _rsi(sec["close"], 6)
        sec["vol_1h"]      = sec["volume"] / sec["volume"].rolling(20).mean().replace(0, 1e-9)
        sec["atr_1h"]      = _atr(sec["high"], sec["low"], sec["close"], ATR_PERIOD)
        sec["macd_hist_1h"]= _macd(sec["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)[2]

        mtf_cols = ["rsi_1h", "rsi_1h_6", "vol_1h", "atr_1h", "macd_hist_1h"]
        sec_mtf = sec[mtf_cols]

        # Resample to match primary frequency
        freq_map = {"1h": "1h", "4h": "4h", "1d": "1D"}
        # Use forward reindex: for each primary timestamp, use latest 1h value
        df_primary = df_primary.copy()
        merged = pd.merge_asof(
            df_primary.sort_index(),
            sec_mtf.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        return merged

    def get_feature_names(self) -> list[str]:
        return self.feature_names


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_features_from_csv(symbol: str, primary_tf: str = "4h", secondary_tf: str = "1h") -> pd.DataFrame:
    """Load CSV files and build feature matrix for one symbol."""
    from phase5.config import DATA_DIR

    path_4h = os.path.join(DATA_DIR, f"{symbol}_{primary_tf}.csv")
    path_1h = os.path.join(DATA_DIR, f"{symbol}_{secondary_tf}.csv")

    if not os.path.exists(path_4h):
        raise FileNotFoundError(f"Missing: {path_4h}")

    df_4h = pd.read_csv(path_4h, parse_dates=["timestamp"])
    df_4h = df_4h.set_index("timestamp").sort_index()

    df_1h = None
    if os.path.exists(path_1h):
        df_1h = pd.read_csv(path_1h, parse_dates=["timestamp"])
        df_1h = df_1h.set_index("timestamp").sort_index()

    fe = FeatureEngineer()
    df_feat = fe.engineer(df_4h, df_secondary=df_1h)
    df_feat["symbol"] = symbol
    return df_feat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", default="4h")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from phase5.config import PROCESSED_DIR
    df = build_features_from_csv(args.symbol, args.tf)
    out = args.out or os.path.join(PROCESSED_DIR, f"{args.symbol}_{args.tf}_features.csv")
    df.to_csv(out)
    print(f"Saved {len(df)} rows × {len(df.columns)} cols → {out}")
    print(f"Features: {list(df.columns)}")
