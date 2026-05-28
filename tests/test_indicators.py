"""Unit tests for src/indicators.py — pure math, no IB required."""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.indicators import compute_rsi, compute_atr, compute_ma, compute_adx, apply_all


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

    def test_rsi_bounded_0_to_100(self):
        df  = make_ohlcv()
        rsi = compute_rsi(df['close']).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_only_first_row_is_nan(self):
        df  = make_ohlcv(50)
        rsi = compute_rsi(df['close'], period=14)
        # EWM-based RSI: diff() gives NaN at row 0; EWM produces values from row 1
        assert pd.isna(rsi.iloc[0])
        assert pd.notna(rsi.iloc[1])

    def test_constant_prices_yield_nan_or_50(self):
        # All-constant prices → gain and loss both 0 → division by zero
        s   = pd.Series([100.0] * 30)
        rsi = compute_rsi(s)
        # Should either be NaN or 50 — not raise
        valid = rsi.dropna()
        assert len(valid) == 0 or ((valid == 50).all() or valid.isna().all())

    def test_steadily_rising_prices_give_high_rsi(self):
        s   = pd.Series(range(1, 51, 1), dtype=float)
        rsi = compute_rsi(s, period=14).dropna()
        assert (rsi > 90).all(), "Monotone up-trend should produce RSI > 90"

    def test_steadily_falling_prices_give_low_rsi(self):
        s   = pd.Series(range(50, 0, -1), dtype=float)
        rsi = compute_rsi(s, period=14).dropna()
        assert (rsi < 10).all(), "Monotone down-trend should produce RSI < 10"


# ── ATR tests ─────────────────────────────────────────────────────────────────
class TestATR:
    def test_output_length_matches_input(self):
        df  = make_ohlcv()
        atr = compute_atr(df)
        assert len(atr) == len(df)

    def test_atr_is_positive(self):
        df  = make_ohlcv()
        atr = compute_atr(df).dropna()
        assert (atr > 0).all()

    def test_atr_finite_from_first_row(self):
        df  = make_ohlcv(30)
        atr = compute_atr(df, period=14)
        # EWM-based ATR: row 0 TR = high-low (prev_close NaN is skipna'd in max())
        # so ATR is finite and positive from bar 0 onwards
        assert pd.notna(atr.iloc[0])
        assert atr.iloc[0] > 0

    def test_wider_range_produces_larger_atr(self):
        n = 100
        np.random.seed(7)
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        df_tight = pd.DataFrame({
            'high':  close + 0.1, 'low': close - 0.1, 'close': close
        })
        df_wide = pd.DataFrame({
            'high':  close + 5.0, 'low': close - 5.0, 'close': close
        })
        atr_tight = compute_atr(df_tight).dropna().mean()
        atr_wide  = compute_atr(df_wide).dropna().mean()
        assert atr_wide > atr_tight


# ── MA tests ──────────────────────────────────────────────────────────────────
class TestMA:
    def test_ma_first_period_minus_one_is_nan(self):
        s  = pd.Series(range(1, 101, 1), dtype=float)
        ma = compute_ma(s, 50)
        assert ma.iloc[:49].isna().all()
        assert not pd.isna(ma.iloc[49])

    def test_ma_50th_value_equals_mean_of_first_50(self):
        s  = pd.Series(range(1, 101, 1), dtype=float)
        ma = compute_ma(s, 50)
        expected = s.iloc[:50].mean()
        assert abs(ma.iloc[49] - expected) < 1e-9

    def test_ma_period_1_equals_series(self):
        s  = pd.Series([10.0, 20.0, 30.0])
        ma = compute_ma(s, 1)
        pd.testing.assert_series_equal(ma, s)


# ── ADX tests ─────────────────────────────────────────────────────────────────
class TestADX:
    def test_returns_series_same_length(self):
        df  = make_ohlcv()
        adx = compute_adx(df)
        assert len(adx) == len(df)

    def test_adx_bounded_0_to_100(self):
        df  = make_ohlcv(300)
        adx = compute_adx(df).dropna()
        assert (adx >= 0).all() and (adx <= 100).all()

    def test_strong_trend_produces_high_adx(self):
        # Monotonically rising prices should produce high ADX after warmup
        close = pd.Series(range(1, 201), dtype=float)
        high  = close + 1.0
        low   = close - 1.0
        df    = pd.DataFrame({'high': high, 'low': low, 'close': close})
        adx   = compute_adx(df, period=14)
        # After 50 bars, ADX should clearly reflect a strong trend (> 25)
        assert adx.iloc[60] > 25, f"Expected ADX > 25 in steady trend, got {adx.iloc[60]:.2f}"

    def test_flat_prices_produce_low_or_nan_adx(self):
        # Flat prices → no directional movement → DI sum = 0 → ADX is NaN or near 0
        n     = 100
        df    = pd.DataFrame({'high': [50.5] * n, 'low': [49.5] * n, 'close': [50.0] * n})
        adx   = compute_adx(df, period=14)
        val   = adx.iloc[-1]
        assert pd.isna(val) or val < 5, f"Expected near-zero or NaN ADX on flat prices, got {val}"


# ── apply_all integration ─────────────────────────────────────────────────────
class TestApplyAll:
    def test_columns_added(self):
        df  = make_ohlcv()
        out = apply_all(df)
        expected = [
            'MA50', 'MA200', 'ATR', 'ATR_CHAND',
            'RSI', 'SMA200_SLOPE', 'ADX', 'HIGH200',
        ]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

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
        non_negative_cols = ['MA50', 'MA200', 'ATR', 'ATR_CHAND',
                             'RSI', 'ADX', 'HIGH200']
        for col in non_negative_cols + ['SMA200_SLOPE']:
            assert tail[col].notna().all(), f"{col} has NaN in tail"
        for col in non_negative_cols:
            assert (tail[col] >= 0).all(), f"{col} has negative values in tail"
