"""Unit tests for src/indicators.py — pure math, no broker required."""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.indicators import compute_rsi, compute_atr, compute_ma, apply_all


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV where close follows a random walk."""
    np.random.seed(42)
    close  = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high   = close + np.abs(np.random.randn(n) * 0.3)
    low    = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({'open': close, 'high': high, 'low': low,
                         'close': close, 'volume': 1_000_000},
                        index=pd.date_range("2024-01-01", periods=n, freq='B'))


# ── RSI tests ─────────────────────────────────────────────────────────────────
class TestRSI:
    def test_output_length_matches_input(self):
        df = make_ohlcv()
        rsi = compute_rsi(df['close'])
        assert len(rsi) == len(df)

    def test_values_bounded_0_to_100(self):
        df  = make_ohlcv(300)
        rsi = compute_rsi(df['close']).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_monotone_rise_produces_high_rsi(self):
        close = pd.Series(range(1, 201), dtype=float)
        rsi   = compute_rsi(close, period=14)
        assert rsi.iloc[-1] > 90, f"Expected RSI>90 on monotone rise, got {rsi.iloc[-1]:.2f}"

    def test_monotone_fall_produces_low_rsi(self):
        close = pd.Series(range(200, 0, -1), dtype=float)
        rsi   = compute_rsi(close, period=14)
        assert rsi.iloc[-1] < 10, f"Expected RSI<10 on monotone fall, got {rsi.iloc[-1]:.2f}"


# ── ATR tests ─────────────────────────────────────────────────────────────────
class TestATR:
    def test_output_length_matches_input(self):
        df  = make_ohlcv()
        atr = compute_atr(df)
        assert len(atr) == len(df)

    def test_atr_non_negative(self):
        df  = make_ohlcv(300)
        atr = compute_atr(df).dropna()
        assert (atr >= 0).all()

    def test_wider_range_produces_higher_atr(self):
        df_narrow = make_ohlcv(100)
        df_wide   = df_narrow.copy()
        df_wide['high'] = df_wide['close'] + 10
        df_wide['low']  = df_wide['close'] - 10
        atr_n = compute_atr(df_narrow).iloc[-1]
        atr_w = compute_atr(df_wide).iloc[-1]
        assert atr_w > atr_n


# ── MA tests ──────────────────────────────────────────────────────────────────
class TestMA:
    def test_period_1_equals_series(self):
        s  = pd.Series([10.0, 20.0, 30.0])
        ma = compute_ma(s, 1)
        pd.testing.assert_series_equal(ma, s)

    def test_ma_period_1_equals_series(self):
        s  = pd.Series([10.0, 20.0, 30.0])
        ma = compute_ma(s, 1)
        pd.testing.assert_series_equal(ma, s)


# ── apply_all integration ─────────────────────────────────────────────────────
class TestApplyAll:
    def test_columns_added(self):
        df  = make_ohlcv()
        out = apply_all(df)
        expected = [
            'MA50', 'MA200', 'ATR', 'ATR_CHAND',
            'RSI', 'SMA200_SLOPE',
            'SMMA_FAST', 'SMMA_MED', 'SMMA_SLOW',
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_dead_columns(self):
        df  = make_ohlcv()
        out = apply_all(df)
        for dead in ('ADX', 'HIGH200', 'DONCH_UPPER', 'DONCH_LOWER'):
            assert dead not in out.columns, f"Dead column {dead!r} still present"

    def test_does_not_mutate_input(self):
        df   = make_ohlcv()
        orig = df.copy()
        apply_all(df)
        pd.testing.assert_frame_equal(df, orig)

    def test_values_are_finite_after_warmup(self):
        # 250 rows; slice the last 40 rows — well past the 205-bar warmup
        # needed for SMA200 (200) + SMA200_SLOPE_LOOKBACK (5).
        df  = make_ohlcv(250)
        out = apply_all(df)
        tail = out.iloc[-40:]
        non_negative_cols = ['MA50', 'MA200', 'ATR', 'ATR_CHAND', 'RSI']
        for col in non_negative_cols + ['SMA200_SLOPE']:
            assert tail[col].notna().all(), f"{col} has NaN in tail"
        for col in non_negative_cols:
            assert (tail[col] >= 0).all(), f"{col} has negative values in tail"
