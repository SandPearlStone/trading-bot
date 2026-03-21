#!/home/sandro/trading-venv/bin/python3
"""Feature contract for ML training and live inference. v3"""

import math
import numpy as np
import pandas as pd
import talib


class TradingFeatures:
    """Single feature contract — called identically at train and predict time."""

    FEATURE_VERSION = "v3"

    EXPECTED_COLUMNS = [
        # Group 1: Base TA (19)
        'rsi_7', 'rsi_14', 'rsi_21',
        'macd_hist', 'macd_signal',
        'atr_14', 'atr_pct',
        'bb_width', 'bb_pct_b',
        'adx_14',
        'ema_8_21_ratio', 'ema_21_55_ratio',
        'vol_ratio_20', 'vol_ratio_5',
        'roc_5', 'roc_10',
        'stoch_k', 'stoch_d',
        'mfi_14',
        # Group 2: Shifted (10)
        'rsi_14_lag1', 'rsi_14_lag2', 'rsi_14_lag5',
        'atr_pct_lag1', 'atr_pct_lag5',
        'vol_ratio_20_lag1', 'vol_ratio_20_lag5',
        'macd_hist_lag1', 'macd_hist_lag2',
        'roc_10_lag1',
        # Group 3: 15m context (5)
        'rsi_15m', 'ema_slope_15m', 'vol_spike_15m', 'momentum_15m', 'trend_15m',
        # Group 4: 4h context (4)
        'rsi_4h', 'trend_4h', 'adx_4h', 'atr_pct_4h',
        # Group 5: BTC context (4)
        'btc_rsi', 'btc_trend', 'btc_atr_pct', 'btc_roc_10',
        # Group 6: Temporal (4)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    def compute(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame = None,
                df_4h: pd.DataFrame = None, btc_1h: pd.DataFrame = None) -> pd.DataFrame:
        n = len(df_1h)
        close = df_1h['close'].values.astype(float)
        high = df_1h['high'].values.astype(float)
        low = df_1h['low'].values.astype(float)
        volume = df_1h['volume'].values.astype(float)
        out = {}

        # --- Group 1: Base TA ---
        out['rsi_7'] = talib.RSI(close, 7)
        out['rsi_14'] = talib.RSI(close, 14)
        out['rsi_21'] = talib.RSI(close, 21)
        _macd, _signal, _hist = talib.MACD(close)
        out['macd_hist'] = _hist
        out['macd_signal'] = _signal
        atr_14 = talib.ATR(high, low, close, 14)
        out['atr_14'] = atr_14
        out['atr_pct'] = atr_14 / close * 100
        bb_up, bb_mid, bb_lo = talib.BBANDS(close, 20)
        out['bb_width'] = (bb_up - bb_lo) / np.where(bb_mid == 0, 1, bb_mid)
        denom_bb = bb_up - bb_lo
        out['bb_pct_b'] = (close - bb_lo) / np.where(denom_bb == 0, 1, denom_bb)
        out['adx_14'] = talib.ADX(high, low, close, 14)
        ema_8 = talib.EMA(close, 8)
        ema_21 = talib.EMA(close, 21)
        ema_55 = talib.EMA(close, 55)
        out['ema_8_21_ratio'] = ema_8 / np.where(ema_21 == 0, 1, ema_21)
        out['ema_21_55_ratio'] = ema_21 / np.where(ema_55 == 0, 1, ema_55)
        sma_vol_20 = talib.SMA(volume, 20)
        sma_vol_5 = talib.SMA(volume, 5)
        out['vol_ratio_20'] = volume / np.where(sma_vol_20 == 0, 1, sma_vol_20)
        out['vol_ratio_5'] = volume / np.where(sma_vol_5 == 0, 1, sma_vol_5)
        out['roc_5'] = talib.ROC(close, 5)
        out['roc_10'] = talib.ROC(close, 10)
        out['stoch_k'], out['stoch_d'] = talib.STOCH(high, low, close)
        out['mfi_14'] = talib.MFI(high, low, close, volume, 14)

        # --- Group 2: Shifted ---
        rsi_14 = pd.Series(out['rsi_14'])
        atr_pct = pd.Series(out['atr_pct'])
        vol_r20 = pd.Series(out['vol_ratio_20'])
        macd_h = pd.Series(out['macd_hist'])
        roc10 = pd.Series(out['roc_10'])
        out['rsi_14_lag1'] = rsi_14.shift(1).values
        out['rsi_14_lag2'] = rsi_14.shift(2).values
        out['rsi_14_lag5'] = rsi_14.shift(5).values
        out['atr_pct_lag1'] = atr_pct.shift(1).values
        out['atr_pct_lag5'] = atr_pct.shift(5).values
        out['vol_ratio_20_lag1'] = vol_r20.shift(1).values
        out['vol_ratio_20_lag5'] = vol_r20.shift(5).values
        out['macd_hist_lag1'] = macd_h.shift(1).values
        out['macd_hist_lag2'] = macd_h.shift(2).values
        out['roc_10_lag1'] = roc10.shift(1).values

        # --- Group 3: 15m context ---
        if df_15m is not None and len(df_15m) >= 20:
            c15 = df_15m['close'].values.astype(float)
            v15 = df_15m['volume'].values.astype(float)
            rsi_15m_full = talib.RSI(c15, 14)
            ema8_15m = talib.EMA(c15, 8)
            ema21_15m = talib.EMA(c15, 21)
            sma_v15 = talib.SMA(v15, 20)
            vol_spike_full = v15 / np.where(sma_v15 == 0, 1, sma_v15)
            slope_full = np.where(ema8_15m == 0, 0,
                                  (ema8_15m - np.roll(ema8_15m, 4)) / np.where(np.roll(ema8_15m, 4) == 0, 1, np.roll(ema8_15m, 4)) * 100)
            slope_full[:4] = 0
            mom_full = np.where(np.roll(c15, 8) == 0, 0,
                                (c15 - np.roll(c15, 8)) / np.where(np.roll(c15, 8) == 0, 1, np.roll(c15, 8)) * 100)
            mom_full[:8] = 0
            trend_full = np.where(ema8_15m > ema21_15m, 1.0, -1.0)
            for name, arr in [('rsi_15m', rsi_15m_full), ('ema_slope_15m', slope_full),
                              ('vol_spike_15m', vol_spike_full), ('momentum_15m', mom_full),
                              ('trend_15m', trend_full)]:
                aligned = arr[3::4]
                if len(aligned) > n:
                    aligned = aligned[-n:]
                elif len(aligned) < n:
                    aligned = np.pad(aligned, (n - len(aligned), 0), constant_values=0)
                out[name] = aligned
        else:
            for name in ['rsi_15m', 'ema_slope_15m', 'vol_spike_15m', 'momentum_15m', 'trend_15m']:
                out[name] = np.zeros(n)

        # --- Group 4: 4h context ---
        if df_4h is not None and len(df_4h) >= 14:
            c4 = df_4h['close'].values.astype(float)
            h4 = df_4h['high'].values.astype(float)
            l4 = df_4h['low'].values.astype(float)
            rsi4 = talib.RSI(c4, 14)
            ema8_4 = talib.EMA(c4, 8)
            ema21_4 = talib.EMA(c4, 21)
            adx4 = talib.ADX(h4, l4, c4, 14)
            atr4 = talib.ATR(h4, l4, c4, 14)
            atr_pct4 = atr4 / np.where(c4 == 0, 1, c4) * 100
            trend4 = np.where(ema8_4 > ema21_4, 1.0, -1.0)
            for name, arr in [('rsi_4h', rsi4), ('trend_4h', trend4),
                              ('adx_4h', adx4), ('atr_pct_4h', atr_pct4)]:
                rep = np.repeat(arr, 4)
                if len(rep) >= n:
                    out[name] = rep[-n:]
                else:
                    out[name] = np.pad(rep, (n - len(rep), 0), constant_values=0)
        else:
            for name in ['rsi_4h', 'trend_4h', 'adx_4h', 'atr_pct_4h']:
                out[name] = np.zeros(n)

        # --- Group 5: BTC context ---
        if btc_1h is not None and len(btc_1h) >= 14:
            bc = btc_1h['close'].values.astype(float)
            bh = btc_1h['high'].values.astype(float)
            bl = btc_1h['low'].values.astype(float)
            out['btc_rsi'] = talib.RSI(bc, 14)[-n:]
            be8 = talib.EMA(bc, 8)
            be21 = talib.EMA(bc, 21)
            out['btc_trend'] = np.where(be8 > be21, 1.0, -1.0)[-n:]
            batr = talib.ATR(bh, bl, bc, 14)
            out['btc_atr_pct'] = (batr / np.where(bc == 0, 1, bc) * 100)[-n:]
            out['btc_roc_10'] = talib.ROC(bc, 10)[-n:]
        else:
            for name in ['btc_rsi', 'btc_trend', 'btc_atr_pct', 'btc_roc_10']:
                out[name] = np.zeros(n)

        # --- Group 6: Temporal ---
        if 'timestamp' in df_1h.columns:
            ts = pd.to_datetime(df_1h['timestamp'].values, unit='ms')
            hours = np.array([t.hour for t in ts], dtype=float)
            dows = np.array([t.dayofweek for t in ts], dtype=float)
            out['hour_sin'] = np.sin(2 * math.pi * hours / 24)
            out['hour_cos'] = np.cos(2 * math.pi * hours / 24)
            out['dow_sin'] = np.sin(2 * math.pi * dows / 7)
            out['dow_cos'] = np.cos(2 * math.pi * dows / 7)
        else:
            for name in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
                out[name] = np.zeros(n)

        # --- Assemble ---
        result = pd.DataFrame({col: out[col] for col in self.EXPECTED_COLUMNS})
        result = result.fillna(0)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        assert list(result.columns) == self.EXPECTED_COLUMNS, "Column contract violated"
        assert len(result) == n, f"Length mismatch: {len(result)} vs {n}"
        assert result.isnull().sum().sum() == 0, "NaN in output"
        return result
