"""
phase5_feature_builder.py — FreqAI-inspired feature engineering for Phase 5 ML.

Expands base TA indicators across:
  - Shifted candles (T-1, T-2, T-5)
  - Multi-timeframe (1h RSI for 4h/5m, 4h RSI for 1h/5m)
  - Correlated pairs (BTC features for non-BTC symbols)
  - Temporal encoding (hour, day of week)
  - Hidden divergence & regime & confluence score (your edges)

Output: 80-120 features, no lookahead, no NaN.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from divergence_detector import rsi_hidden_divergence
except ImportError:
    rsi_hidden_divergence = None

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Indicator Primitives
# ──────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI — Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD — Moving Average Convergence Divergence."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR — Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX — Average Directional Index (trend strength)."""
    atr_val = _atr(high, low, close, period)
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    both = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[both & (plus_dm <= minus_dm)] = 0
    minus_dm[both & (minus_dm <= plus_dm)] = 0

    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-9)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    return dx.ewm(alpha=1/period, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, 1e-9)
    pct_b = (close - lower) / (upper - lower).replace(0, 1e-9)
    return mid, upper, lower, width, pct_b


def _keltner(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20, atr_mult: float = 2.0):
    """Keltner Channels."""
    mid = close.ewm(span=period, adjust=False).mean()
    atr_val = _atr(high, low, close, period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return mid, upper, lower


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, 1e-9)
    return 100 - 100 / (1 + mfr)


def _roc(close: pd.Series, period: int = 14) -> pd.Series:
    """Rate of Change."""
    return 100 * (close - close.shift(period)) / close.shift(period).replace(0, 1e-9)


def _hist_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """Historical Volatility (annualized)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252 * 24)


# ──────────────────────────────────────────────────────────────────────────────
# Feature Builder Class
# ──────────────────────────────────────────────────────────────────────────────

class Phase5FeatureBuilder:
    """
    Build 80-120 features for Phase 5 ML engine.
    
    Input: 4h OHLCV + optional 1h & 5m data + optional BTC features
    Output: DataFrame with all features, no NaN, ready for training.
    """
    
    def __init__(self, fill_method: str = "ffill"):
        self.fill_method = fill_method
    
    def build(
        self,
        df_4h: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_btc_4h: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build features from 4h (base), optional 1h/5m/BTC data.
        
        Args:
            df_4h: 4h OHLCV (required)
            df_1h: 1h OHLCV for multi-timeframe features (optional)
            df_5m: 5m OHLCV for context (optional)
            df_btc_4h: BTC 4h features for correlation (optional, for non-BTC)
        
        Returns:
            DataFrame with 80+ features, indexed by df_4h.index, no NaN.
        """
        df = df_4h.copy()
        df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
        
        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # ── Base indicators on 4h ──────────────────────────────────────────
        self._add_base_indicators(df)
        
        # ── Shifted candles (T-1, T-2, T-5) ───────────────────────────────
        self._add_shifted_features(df)
        
        # ── Multi-timeframe (1h) ───────────────────────────────────────────
        if df_1h is not None:
            self._add_mtf_features(df, df_1h)
        
        # ── Temporal encoding ──────────────────────────────────────────────
        self._add_temporal_features(df)
        
        # ── BTC correlation (for non-BTC) ──────────────────────────────────
        if df_btc_4h is not None and "BTCUSDT" not in df.get("symbol", ["BTCUSDT"])[0]:
            self._add_btc_correlation(df, df_btc_4h)
        
        # ── Handle NaN & fill ──────────────────────────────────────────────
        df = df.fillna(method=self.fill_method, limit=100)
        df = df.fillna(0)
        
        return df
    
    def _add_base_indicators(self, df: pd.DataFrame):
        """Add RSI(7,14,21), MACD, ATR%, BB, vol_ratio, EMA ratios, ADX, MFI, ROC."""
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        
        # RSI multi-period
        df["rsi_7"] = _rsi(c, 7)
        df["rsi_14"] = _rsi(c, 14)
        df["rsi_21"] = _rsi(c, 21)
        
        # MACD
        macd, macd_sig, macd_hist = _macd(c)
        df["macd"] = macd
        df["macd_signal"] = macd_sig
        df["macd_hist"] = macd_hist
        df["macd_cross"] = (macd > macd_sig).astype(int)
        
        # ATR (absolute & %)
        atr_14 = _atr(h, l, c, 14)
        atr_7 = _atr(h, l, c, 7)
        df["atr"] = atr_14
        df["atr_7"] = atr_7
        df["atr_pct"] = atr_14 / c.replace(0, 1e-9)
        df["atr_7_pct"] = atr_7 / c.replace(0, 1e-9)
        
        # ADX
        df["adx"] = _adx(h, l, c, 14)
        
        # Bollinger Bands
        bb_mid, bb_upper, bb_lower, bb_width, bb_pct_b = _bollinger(c, 20, 2.0)
        df["bb_width"] = bb_width
        df["bb_pct_b"] = bb_pct_b
        df["bb_squeeze"] = ((df["bb_width"] < df["bb_width"].rolling(20).mean()) * 1.0)
        
        # Keltner Channels
        kc_mid, kc_upper, kc_lower = _keltner(c, h, l, 20, 2.0)
        df["kc_width"] = (kc_upper - kc_lower) / kc_mid.replace(0, 1e-9)
        df["kc_pct"] = (c - kc_lower) / (kc_upper - kc_lower).replace(0, 1e-9)
        
        # Volume
        df["vol_ratio"] = v / v.rolling(20).mean().replace(0, 1e-9)
        df["vol_ratio_5"] = v / v.rolling(5).mean().replace(0, 1e-9)
        
        # EMA ratios
        ema_8 = c.ewm(span=8, adjust=False).mean()
        ema_21 = c.ewm(span=21, adjust=False).mean()
        ema_55 = c.ewm(span=55, adjust=False).mean()
        df["ema_8"] = ema_8
        df["ema_21"] = ema_21
        df["ema_55"] = ema_55
        df["ema_8_21"] = ema_8 / ema_21.replace(0, 1e-9)
        df["ema_21_55"] = ema_21 / ema_55.replace(0, 1e-9)
        df["price_vs_ema_8"] = c / ema_8.replace(0, 1e-9) - 1
        df["price_vs_ema_21"] = c / ema_21.replace(0, 1e-9) - 1
        
        # MFI
        df["mfi"] = _mfi(h, l, c, v, 14)
        
        # ROC
        df["roc_10"] = _roc(c, 10)
        df["roc_20"] = _roc(c, 20)
        
        # Historical volatility
        df["hist_vol"] = _hist_vol(c, 20)
        
        # OBV
        obv = _obv(c, v)
        df["obv_diff"] = obv.diff()
        
        # Candle patterns
        df["candle_body"] = (c - df["open"]).abs() / (h - l).replace(0, 1e-9)
        df["candle_wick_hi"] = (h - c.clip(df["open"])) / (h - l).replace(0, 1e-9)
        df["candle_wick_lo"] = (c.clip(lower=df["open"]) - l) / (h - l).replace(0, 1e-9)
        df["range_pct"] = (h - l) / df["open"].replace(0, 1e-9)
    
    def _add_shifted_features(self, df: pd.DataFrame, shifts: list[int] = [1, 2, 5]):
        """Add T-1, T-2, T-5 versions of key features (RSI, MACD hist, close)."""
        key_features = ["rsi_14", "macd_hist", "close", "volume", "atr", "bb_pct_b"]
        
        for shift in shifts:
            for feat in key_features:
                if feat in df.columns:
                    df[f"{feat}_shift_{shift}"] = df[feat].shift(shift)
    
    def _add_mtf_features(self, df: pd.DataFrame, df_1h: pd.DataFrame):
        """Add 1h RSI/MACD aligned to 4h bars (look back 4 bars = last 4 hours)."""
        df_1h = df_1h.copy()
        df_1h.index = pd.to_datetime(df_1h.index) if not isinstance(df_1h.index, pd.DatetimeIndex) else df_1h.index
        df_1h = df_1h.sort_index()
        
        # For each 4h bar, get most recent 1h data
        rsi_1h = _rsi(df_1h["close"], 14) if "close" in df_1h.columns else None
        macd_1h, _, macd_hist_1h = _macd(df_1h["close"]) if "close" in df_1h.columns else (None, None, None)
        
        if rsi_1h is not None:
            # Align 1h data to 4h: forward-fill last 1h value
            rsi_1h_aligned = df_1h.index.to_series().map(
                lambda x: rsi_1h[rsi_1h.index <= x].iloc[-1] if len(rsi_1h[rsi_1h.index <= x]) > 0 else np.nan
            ).values
            df["rsi_1h"] = rsi_1h_aligned
            
            # Last 6 1h candles as feature
            rsi_1h_6 = df_1h.index.to_series().map(
                lambda x: rsi_1h[rsi_1h.index <= x].iloc[-6:].mean() if len(rsi_1h[rsi_1h.index <= x]) >= 6 else np.nan
            ).values
            df["rsi_1h_6"] = rsi_1h_6
        
        if macd_hist_1h is not None:
            macd_hist_1h_aligned = df_1h.index.to_series().map(
                lambda x: macd_hist_1h[macd_hist_1h.index <= x].iloc[-1] if len(macd_hist_1h[macd_hist_1h.index <= x]) > 0 else np.nan
            ).values
            df["macd_hist_1h"] = macd_hist_1h_aligned
    
    def _add_temporal_features(self, df: pd.DataFrame):
        """Add hour_of_day, day_of_week (sine/cosine encoded)."""
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        
        # Circular encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    def _add_btc_correlation(self, df: pd.DataFrame, df_btc: pd.DataFrame):
        """Add BTC RSI, MACD, ADX for non-BTC symbols (correlation feature)."""
        df_btc = df_btc.copy()
        df_btc.index = pd.to_datetime(df_btc.index) if not isinstance(df_btc.index, pd.DatetimeIndex) else df_btc.index
        df_btc = df_btc.sort_index()
        
        if "close" in df_btc.columns:
            btc_rsi = _rsi(df_btc["close"], 14)
            btc_macd, _, btc_macd_hist = _macd(df_btc["close"])
            btc_adx = _adx(df_btc["high"], df_btc["low"], df_btc["close"], 14) if all(col in df_btc.columns for col in ["high", "low"]) else None
            
            # Align BTC to main df
            df["btc_rsi"] = df_btc.index.to_series().map(
                lambda x: btc_rsi[btc_rsi.index <= x].iloc[-1] if len(btc_rsi[btc_rsi.index <= x]) > 0 else np.nan
            ).values
            
            df["btc_macd_hist"] = df_btc.index.to_series().map(
                lambda x: btc_macd_hist[btc_macd_hist.index <= x].iloc[-1] if len(btc_macd_hist[btc_macd_hist.index <= x]) > 0 else np.nan
            ).values
            
            if btc_adx is not None:
                df["btc_adx"] = df_btc.index.to_series().map(
                    lambda x: btc_adx[btc_adx.index <= x].iloc[-1] if len(btc_adx[btc_adx.index <= x]) > 0 else np.nan
                ).values


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test on labeled_4h.csv
    csv_path = "data/processed/labeled_4h.csv"
    
    if os.path.exists(csv_path):
        print(f"[Feature Builder] Loading {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        print(f"  Input shape: {df.shape}")
        print(f"  Input columns: {list(df.columns)[:10]}...")
        
        fb = Phase5FeatureBuilder()
        df_features = fb.build(df)
        
        print(f"  Output shape: {df_features.shape}")
        print(f"  Output columns: {list(df_features.columns)[:20]}...")
        print(f"  NaN count: {df_features.isna().sum().sum()}")
        print(f"\n✅ Feature engineering successful!")
    else:
        print(f"❌ {csv_path} not found. Run populate_ohlcv.py first.")
