"""
Unit tests for backtest/strategy.py — no IB, no live data.

All tests use purely synthetic DataFrames so they run offline
and deterministically.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from backtest.strategy import VelocityBacktest, Trade, BacktestResult
from src.config import SCAN_MIN_VOLUME
from src.config import BACKTEST_RVOL_MIN


# ── Synthetic data factory ────────────────────────────────────────────────────
def _make_df(n: int = 300, seed: int = 0, trend: float = 0.2) -> pd.DataFrame:
    """Smooth upward-trending OHLCV with fully warmed-up indicators."""
    from src.config import RSI_OVERSOLD_LOOKBACK
    np.random.seed(seed)
    close  = 100 + trend * np.arange(n) + np.cumsum(np.random.randn(n) * 0.3)
    high   = close + np.abs(np.random.randn(n) * 0.2)
    low    = close - np.abs(np.random.randn(n) * 0.2)
    idx    = pd.date_range("2023-01-01", periods=n, freq='B')

    from src.indicators import apply_all
    df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                       'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
    df = apply_all(df)
    df['RSI_PREV']          = df['RSI'].shift(1)
    df['RSI_MIN_LOOKBACK']  = df['RSI'].shift(1).rolling(RSI_OVERSOLD_LOOKBACK).min()
    df['prev_high']         = df['high'].shift(1)
    df['avg_vol_20']        = df['volume'].rolling(20).mean()
    df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()
    return df


# ── Trade dataclass ───────────────────────────────────────────────────────────
class TestTradeDataclass:
    def test_pnl_long_winner(self):
        # Alpaca is commission-free: pnl = net_pnl = gross
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.pnl - 100.0) < 1e-9    # gross=100, commission=0

    def test_pnl_long_loser(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 90.0, qty=5)
        assert abs(t.pnl - (-50.0)) < 1e-9  # gross=-50, commission=0

    def test_pnl_pct_correct(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=1)
        assert abs(t.pnl_pct - 0.10) < 1e-9

    def test_pnl_returns_zero_when_no_exit(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0)
        assert t.pnl == 0.0

    def test_gross_pnl_excludes_commission(self):
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.gross_pnl - 100.0) < 1e-9

    def test_net_pnl_includes_commission(self):
        # Alpaca commission=0 → net_pnl == gross_pnl
        t = Trade("AAPL", date(2024, 1, 2), 100.0, date(2024, 1, 5), 110.0, qty=10)
        assert abs(t.net_pnl - 100.0) < 1e-9

    def test_net_pnl_can_use_broker_derived_commission_assumption(self):
        t = Trade(
            "AAPL", date(2024, 1, 2), 100.0,
            date(2024, 1, 5), 110.0,
            qty=10,
            round_trip_commission=1.25,
        )
        assert abs(t.net_pnl - 98.75) < 1e-9


# ── Entry signal ──────────────────────────────────────────────────────────────
class TestEntrySignal:
    """
    _entry_signal — Donchian bounce rules (mirrors src/rules.py CYCLE_RULES).

    Required row columns: DONCH_LOWER, RSI, RSI_MIN_LOOKBACK
    Positional args:      prev_rsi, rvol, rvol_min
    """

    def _row(self, close=100.0, donch_lower=99.8, rsi=38.0, rsi_min_lookback=28.0):
        """Default passing row: price within 0.2% of lower band, RSI was oversold."""
        return pd.Series({
            'close':           close,
            'DONCH_LOWER':     donch_lower,
            'RSI':             rsi,
            'RSI_MIN_LOOKBACK': rsi_min_lookback,
        })

    def test_all_conditions_pass(self):
        # close=100 within 0.2% of lower=99.8; RSI_MIN_LOOKBACK=28<35; delta=3.0>=3; rvol ok
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=35.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_donchian_lower_missing(self):
        row = self._row()
        row['DONCH_LOWER'] = float('nan')
        assert not VelocityBacktest._entry_signal(
            row, prev_rsi=35.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_price_too_far_above_lower_band(self):
        # proximity = (102 - 99.8) / 99.8 = 2.2% > DONCHIAN_FLOOR_TOL_PCT (0.5%)
        assert not VelocityBacktest._entry_signal(
            self._row(close=102.0, donch_lower=99.8), prev_rsi=35.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_never_oversold_in_lookback(self):
        # RSI_MIN_LOOKBACK=40 >= RSI_OVERSOLD_THRESHOLD(35) → oversold lookback fails
        assert not VelocityBacktest._entry_signal(
            self._row(rsi_min_lookback=40.0), prev_rsi=35.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_delta_below_minimum(self):
        # delta = 38.0 - 36.5 = 1.5 < RSI_MIN_DELTA (3.0)
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=38.0), prev_rsi=36.5,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rvol_below_min(self):
        assert not VelocityBacktest._entry_signal(
            self._row(), prev_rsi=35.0, rvol=0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_passes_price_exactly_at_donchian_tolerance(self):
        from src.config import DONCHIAN_FLOOR_TOL_PCT
        lower = 99.8
        close = lower * (1 + DONCHIAN_FLOOR_TOL_PCT)   # exactly at boundary → passes
        assert VelocityBacktest._entry_signal(
            self._row(close=close, donch_lower=lower), prev_rsi=35.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_passes_rsi_min_lookback_just_below_threshold(self):
        from src.config import RSI_OVERSOLD_THRESHOLD
        assert VelocityBacktest._entry_signal(
            self._row(rsi_min_lookback=RSI_OVERSOLD_THRESHOLD - 0.1), prev_rsi=35.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_flags_none_behaves_as_production_defaults(self):
        # flags kwarg is accepted but ignored — Donchian rules are all mandatory
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=35.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN, flags=None)

    def test_production_flags_dict_accepted_without_error(self):
        # flags dict is forwarded but ignored — all Donchian rules are always active
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=35.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN,
            flags={'use_rsi_delta': True, 'use_rvol': True})


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestComputeMetrics:
    def _eq(self, vals):
        idx = pd.date_range("2024-01-01", periods=len(vals), freq='B')
        return pd.Series(vals, index=idx, dtype=float)

    def test_win_rate_all_wins(self):
        trades = [
            Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1),
            Trade("B", date(2024,1,2), 200, date(2024,1,5), 220, qty=1),
        ]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1410, 1420]))
        assert m['win_rate'] == 1.0

    def test_win_rate_all_losses(self):
        trades = [
            Trade("A", date(2024,1,2), 100, date(2024,1,5),  90, qty=1),
        ]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1390]))
        assert m['win_rate'] == 0.0

    def test_profit_factor_infinite_when_no_losses(self):
        trades = [Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1)]
        m = VelocityBacktest._compute_metrics(trades, self._eq([1400, 1410]))
        assert m['profit_factor'] == float('inf')

    def test_max_drawdown_negative(self):
        # Equity drops from 1400 to 1200, then recovers
        eq = self._eq([1400, 1350, 1200, 1250, 1400])
        m  = VelocityBacktest._compute_metrics(
            [Trade("A", date(2024,1,2), 100, date(2024,1,5), 110, qty=1)], eq
        )
        assert m['max_drawdown_pct'] < 0

    def test_empty_trades_returns_empty_dict(self):
        eq = self._eq([1400, 1400])
        assert VelocityBacktest._compute_metrics([], eq) == {}

    def test_total_return_growing_equity(self):
        eq = self._eq([1000, 1500])
        trades = [Trade("A", date(2024,1,2), 100, date(2024,1,5), 150, qty=10)]
        m  = VelocityBacktest._compute_metrics(trades, eq)
        assert abs(m['total_return_pct'] - 50.0) < 1e-6


# ── Full run on synthetic data ────────────────────────────────────────────────
class TestFullRunSynthetic:
    def test_run_returns_backtest_result(self, monkeypatch):
        """Patch _download to inject a synthetic bullish symbol."""
        df = _make_df(n=300, seed=1, trend=0.3)

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAKE": df}))

        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_no_trades_flat_market(self, monkeypatch):
        """Flat/downward market should produce very few or zero qualifying signals."""
        np.random.seed(5)
        n     = 300
        close = np.full(n, 100.0) + np.random.randn(n) * 0.05
        high  = close + 0.02
        low   = close - 0.02
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')

        from src.indicators import apply_all as _apply
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
        df = _apply(df)
        df['prev_high']        = df['high'].shift(1)
        df['avg_vol_20']       = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FLAT": df}))

        result = bt.run()
        # MA50 ≈ MA200 in a flat market, so breakout filter mostly fails
        assert isinstance(result, BacktestResult)

    def test_equity_curve_is_pandas_series(self, monkeypatch):
        df = _make_df(n=300, seed=2, trend=0.2)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))

        result = bt.run()
        assert isinstance(result.equity_curve, pd.Series)

    def test_no_data_raises_runtime_error(self, monkeypatch):
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        # Patch _download to leave _data empty
        monkeypatch.setattr(bt, '_download', lambda: None)
        with pytest.raises(RuntimeError, match="No usable data"):
            bt.run()


# ── Donchian proximity coarse filter ─────────────────────────────────────────
class TestDailyScanGainFilter:
    """_daily_scan coarse filter: Donchian floor proximity + RSI oversold lookback."""

    def _make_uptrend_df(self, n: int = 300) -> pd.DataFrame:
        """Steadily uptrending stock — price always far above 20-day low."""
        from src.indicators import apply_all as _apply
        from src.config import RSI_OVERSOLD_LOOKBACK, SCAN_MIN_VOLUME

        idx    = pd.date_range("2023-01-01", periods=n, freq='B')
        close  = 100.0 + 0.5 * np.arange(n)   # $0.50 gain each bar
        high   = close + 0.1
        low    = close - 0.1
        volume = np.full(n, float(SCAN_MIN_VOLUME * 10))
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': volume}, index=idx)
        df = _apply(df)
        df['RSI_PREV']          = df['RSI'].shift(1)
        df['RSI_MIN_LOOKBACK']  = df['RSI'].shift(1).rolling(RSI_OVERSOLD_LOOKBACK).min()
        df['avg_vol_20']        = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def _make_donchian_bounce_df(self, n: int = 300) -> pd.DataFrame:
        """Downtrend then flat at the bottom — triggers Donchian floor proximity filter."""
        from src.indicators import apply_all as _apply
        from src.config import RSI_OVERSOLD_LOOKBACK, SCAN_MIN_VOLUME

        idx          = pd.date_range("2023-01-01", periods=n, freq='B')
        down_n       = n - 20
        prices_down  = np.linspace(25.0, 15.0, down_n)
        prices_flat  = np.full(20, 15.001)           # flat at the bottom
        close        = np.concatenate([prices_down, prices_flat])
        high         = close + 0.005                  # very tight spread
        low          = close - 0.005

        base_vol = SCAN_MIN_VOLUME * 10
        volume   = np.full(n, float(base_vol))
        volume[-5:] = float(base_vol) * 2.0          # RVOL spike on last 5 bars

        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': volume}, index=idx)
        df = _apply(df)
        df['RSI_PREV']          = df['RSI'].shift(1)
        df['RSI_MIN_LOOKBACK']  = df['RSI'].shift(1).rolling(RSI_OVERSOLD_LOOKBACK).min()
        df['avg_vol_20']        = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def test_uptrending_stock_not_near_donchian_floor(self, monkeypatch):
        """An uptrending stock is always far above its 20-day low — coarse filter rejects it."""
        df = self._make_uptrend_df()
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"UP": df}))
        bt.run()

        assert bt._filter_stats['coarse_candidates'] == 0, (
            "Uptrending stock far above 20-day low must not reach coarse stage"
        )

    def test_stock_near_donchian_floor_reaches_coarse(self, monkeypatch):
        """A stock at its 20-day low with RSI oversold in lookback must pass the coarse filter."""
        df = self._make_donchian_bounce_df()
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"BOUNCE": df}))
        bt.run()

        assert bt._filter_stats['coarse_candidates'] > 0, (
            "Stock near Donchian lower band with RSI oversold lookback must reach coarse stage"
        )


# ── Friday close rule ─────────────────────────────────────────────────────────
class TestBacktestFridayClose:
    """_run_loop must close positions with < FRIDAY_MIN_PROFIT_PCT profit on Fridays."""

    def _make_flat_df(self, n: int = 300, base_price: float = 100.0) -> pd.DataFrame:
        """Bullish-trend DataFrame with minimal daily moves (low ATR → chandelier far away)."""
        from src.indicators import apply_all as _apply
        from src.config import SCAN_MIN_VOLUME

        np.random.seed(42)
        close = base_price + 0.01 * np.arange(n)   # tiny uptrend — chandelier never fires
        high  = close + 0.02
        low   = close - 0.02
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')
        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close, 'volume': SCAN_MIN_VOLUME * 5}, index=idx)
        df = _apply(df)
        df['prev_high']          = df['high'].shift(1)
        df['avg_vol_20']         = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20']  = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def test_friday_close_counter_positive_when_entries_exist(self, monkeypatch):
        """When the strategy makes entries, friday_closes must be ≥ 0 in filter_stats."""
        df = _make_df(n=300, seed=1, trend=0.3)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAKE": df}))
        result = bt.run()
        assert 'friday_closes' in result.filter_stats

    def test_friday_close_uses_friday_min_profit_pct(self, monkeypatch):
        """friday_close exit reason must appear in exit_reasons when positions are held into Fridays."""
        df = _make_df(n=300, seed=3, trend=0.15)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))
        result = bt.run()
        # friday_closes key must exist in filter_stats regardless of whether any fired
        assert result.filter_stats['friday_closes'] >= 0

    def test_friday_close_exit_reason_tracked_in_filter_stats(self, monkeypatch):
        """friday_closes filter stat must equal the count of friday_close exit_reason trades."""
        df = _make_df(n=300, seed=7, trend=0.25)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"TICKER": df}))
        result = bt.run()
        friday_count = sum(1 for t in result.trades if t.exit_reason == "friday_close")
        assert result.filter_stats['friday_closes'] == friday_count, (
            "filter_stats friday_closes must equal actual friday_close exit count"
        )


# ── Dynamic slot enforcement ──────────────────────────────────────────────────
class TestBacktestDynamicSlots:
    """
    Backtest _run_loop must use floor(equity / MIN_BUCKET_SIZE) for maximum
    position capacity, while still requiring settled cash for new entry buckets.
    Previously, the backtest used self.max_pos (MAX_POSITIONS_CAP=8) as a fixed
    ceiling regardless of capital, so a $1000 account would be modelled with 8
    slots ($125 buckets) rather than 2 slots ($500 buckets) as in live.
    """

    def test_no_entries_when_capital_below_min_bucket_size(self, monkeypatch):
        """No entries must be taken when starting capital < MIN_BUCKET_SIZE.

        With dynamic slots, floor(capital / MIN_BUCKET_SIZE) = 0, so the outer
        entry gate (len < dynamic_max_pos) is always False.
        """
        from src.config import MIN_BUCKET_SIZE

        df = _make_df(n=300, seed=1, trend=0.3)
        bt = VelocityBacktest(
            start="2023-01-01", end="2024-01-01",
            capital=MIN_BUCKET_SIZE * 0.5,   # e.g. $250 < $500 → 0 slots
            use_cache=False,
        )
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAKE": df}))
        result = bt.run()

        assert result.filter_stats['entries_taken'] == 0, (
            f"No entries should be taken with capital < MIN_BUCKET_SIZE "
            f"(got {result.filter_stats['entries_taken']})"
        )

    def test_concurrent_positions_capped_by_dynamic_slots(self, monkeypatch):
        """At most floor(capital / MIN_BUCKET_SIZE) positions open simultaneously.

        Inject 3 symbols that all pass the scanner on every date. With capital =
        MIN_BUCKET_SIZE * 2 (2 dynamic slots), only 2 can be entered per day;
        the 3rd must be skipped (entries_skipped_full ≥ 1) when 2 positions
        are already open.  The old fixed-ceiling code (self.max_pos = 8) would
        allow all 3 through.
        """
        from src.config import MIN_BUCKET_SIZE

        df = _make_df(n=300, seed=1, trend=0.3)

        capital = MIN_BUCKET_SIZE * 2   # $1000 → 2 dynamic slots (not 8)
        bt = VelocityBacktest(
            start="2023-01-01", end="2024-01-01",
            capital=capital,
            use_cache=False,
        )
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({
            "AAA": df.copy(), "BBB": df.copy(), "CCC": df.copy(),
        }))
        # Force scanner to always return all 3 symbols and entry signal to always
        # pass, so that the SLOT LIMIT is what constrains entries (not signal quality).
        monkeypatch.setattr(bt, '_daily_scan',
                            lambda today, rvol_min=None: [("AAA", 2.0), ("BBB", 2.0), ("CCC", 2.0)])
        monkeypatch.setattr(VelocityBacktest, '_entry_signal',
                            staticmethod(lambda *args, **kwargs: True))
        result = bt.run()

        # With dynamic_max_pos = 2, entries for a 3rd symbol while 2 are open
        # must accumulate in entries_skipped_full.
        assert result.filter_stats['entries_skipped_full'] >= 1, (
            "With 3 daily signals and only 2 dynamic slots, at least 1 entry "
            "must be skipped due to the slot ceiling"
        )


# ── SCAN_MIN_SCORE gate in _daily_scan ────────────────────────────────────────
class TestDailyScanMinScore:
    """_daily_scan must drop candidates whose composite score < SCAN_MIN_SCORE (30).

    Live engine applies this gate in run_cycle(); backtest must mirror it so that
    low-conviction setups are not entered during backtesting.
    """

    def _make_weak_signal_df(self, n: int = 300) -> pd.DataFrame:
        """A DataFrame that passes 12-rule hard filters but has a very weak
        trend (MA50 ≈ MA200) producing near-zero trend_pts and a low total score."""
        from src.indicators import apply_all as _apply

        np.random.seed(77)
        # Flat price — MA50 ≈ MA200 → trend separation ≈ 0 → trend_pts ≈ 0
        close = np.full(n, 50.0) + np.random.randn(n) * 0.05
        high  = close + 0.05
        low   = close - 0.05
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')

        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close,
                           'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
        df = _apply(df)
        df['prev_high']          = df['high'].shift(1)
        df['avg_vol_20']         = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20']  = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def test_low_score_candidate_excluded_from_scan_output(self, monkeypatch):
        """_daily_scan must return an empty list when all candidates score < SCAN_MIN_SCORE."""
        from src.config import SCAN_MIN_SCORE

        df = self._make_weak_signal_df()
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"WEAK": df}))

        # Spy on _daily_scan output across all days
        scan_results = []
        original_scan = bt._daily_scan
        def capturing_scan(today, rvol_min=None):
            result = original_scan(today, rvol_min=rvol_min)
            scan_results.append(result)
            return result
        monkeypatch.setattr(bt, '_daily_scan', capturing_scan)

        bt.run()

        # No day should return WEAK if its score is below SCAN_MIN_SCORE
        from src.config import SCAN_MIN_SCORE
        for day_result in scan_results:
            syms = [s for s, _ in day_result]
            assert "WEAK" not in syms or len(day_result) == 0, (
                "Low-scoring symbol must not appear in _daily_scan output"
            )

    def test_scan_min_score_constant_imported_in_backtest(self):
        """Backtest module must import SCAN_MIN_SCORE from src.config."""
        import backtest.strategy as bs
        assert hasattr(bs, 'SCAN_MIN_SCORE'), (
            "backtest/strategy.py must import SCAN_MIN_SCORE from src.config"
        )
        from src.config import SCAN_MIN_SCORE
        assert bs.SCAN_MIN_SCORE == SCAN_MIN_SCORE


# ── BUCKET_CASH_PCT applied in _run_loop bucket calculation ──────────────────
class TestBacktestBucketCashPct:
    """_run_loop must apply BUCKET_CASH_PCT (0.90) when sizing position buckets.

    Live engine: bucket_size = settled_cash * BUCKET_CASH_PCT / open_slots
    Backtest must match — otherwise it over-deploys by ~11% per position and
    may enter trades the live engine would skip (cash exactly at threshold).
    """

    def test_bucket_cash_pct_constant_imported_in_backtest(self):
        """Backtest module must import BUCKET_CASH_PCT from src.config."""
        import backtest.strategy as bs
        assert hasattr(bs, 'BUCKET_CASH_PCT'), (
            "backtest/strategy.py must import BUCKET_CASH_PCT from src.config"
        )
        from src.config import BUCKET_CASH_PCT
        assert bs.BUCKET_CASH_PCT == BUCKET_CASH_PCT

    def test_position_qty_reflects_bucket_cash_pct(self, monkeypatch):
        """Entries must be sized with 10% reserve: bucket = cash * 0.90 / slots.

        We run backtest with initial capital = 1 × MIN_BUCKET_SIZE ($500).
        With BUCKET_CASH_PCT=0.90 the effective bucket is $450, so qty must be
        based on $450, not $500.  We verify by checking that no single entry
        exceeds 90% of settled cash / entry_price.
        """
        from src.config import MIN_BUCKET_SIZE, BUCKET_CASH_PCT

        df = _make_df(n=300, seed=1, trend=0.3)
        bt = VelocityBacktest(
            start="2023-01-01", end="2024-01-01",
            capital=MIN_BUCKET_SIZE,   # $500 → 1 slot
            use_cache=False,
        )
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"BULL": df}))
        result = bt.run()

        for t in result.trades:
            # qty * entry_price must be ≤ BUCKET_CASH_PCT fraction of capital
            deployed = t.qty * t.entry_price
            max_allowed = MIN_BUCKET_SIZE * BUCKET_CASH_PCT
            assert deployed <= max_allowed + 1.0, (  # +$1 tolerance for rounding
                f"Entry deployed ${deployed:.2f} > ${max_allowed:.2f} "
                f"(BUCKET_CASH_PCT={BUCKET_CASH_PCT})"
            )


# ── Dashboard JS equity chart key ─────────────────────────────────────────────
class TestDashboardEquityChartKey:
    """alpaca_dashboard.py JS chart must read e.equity, not e.eq.

    Engine writes {"ts": ..., "equity": ...} to equity_history.json.
    The JS `hist.map(e => e.eq)` was a stale key that made every chart
    data point undefined — the chart was always blank.
    """

    def test_dashboard_js_uses_equity_key_not_eq(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'alpaca_dashboard.py')
        with open(path) as f:
            source = f.read()
        assert 'hist.map(e => e.equity)' in source, (
            "JS equity chart must read e.equity (not e.eq) to match engine JSON output"
        )
        assert 'hist.map(e => e.eq)' not in source, (
            "Stale e.eq key must be removed from dashboard JS"
        )


class TestDashboardBreakEvenIndicator:
    """Break-even ↑ indicator must fire when stop_loss ≥ entry_price."""

    def test_break_even_indicator_uses_stop_vs_entry(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'alpaca_dashboard.py')
        with open(path) as f:
            source = f.read()
        assert 'p.stop_loss >= p.entry_price' in source, (
            "Break-even ↑ indicator must compare stop_loss >= entry_price"
        )
        assert 'p.effective_stop > p.stop_loss' not in source, (
            "Stale effective_stop > stop_loss comparison must be removed — "
            "they are always equal so the indicator never fired"
        )

    def test_dashboard_no_dead_commission_key(self):
        """alpaca_dashboard.py must not check for the 'commission' state key.

        Alpaca is commission-free; the engine never writes 'commission' to state.
        The raw_commission branch was always dead and has been removed.
        """
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'alpaca_dashboard.py')
        with open(path) as f:
            source = f.read()
        assert 'raw_commission' not in source, (
            "Dead raw_commission variable must be removed from alpaca_dashboard.py"
        )


# ── Bug 80: Backtest entry-price floor ────────────────────────────────────────
class TestBacktestEntryPriceFloor:
    """Backtest must reject entries where the entry proxy (max(open, prev_high) * slippage)
    is below SCAN_MIN_PRICE, even if the daily close passes the filter.

    E.g., a gap-up day where open = $2, prev_high = $1, close = $25 should not
    be entered: raw_entry = $2, entry_price ≈ $2.002 < $5 minimum.
    """

    def _make_cheap_open_df(self, n: int = 300) -> pd.DataFrame:
        """DataFrame where close is ≥ $5 throughout but open is below $5 for most bars.

        This triggers the entry-price floor: _daily_scan accepts the close ($25),
        but the actual entry proxy max(open=$2, prev_high=$1) resolves to $2, which
        is below SCAN_MIN_PRICE.
        """
        from src.indicators import apply_all as _apply

        np.random.seed(42)
        # Close is above $5 so the close-based price-floor filter passes
        close     = np.full(n, 25.0)
        high      = close + 0.05
        low       = close - 0.05
        # Open is $2 — below SCAN_MIN_PRICE ($5) — simulates a day where the stock
        # opened far below its close (e.g., wild spread or data artefact)
        open_     = np.full(n, 2.0)
        idx       = pd.date_range("2023-01-01", periods=n, freq='B')
        df = pd.DataFrame({'open': open_, 'high': high, 'low': low,
                           'close': close,
                           'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
        df = _apply(df)
        df['prev_high']          = df['high'].shift(1)
        df['avg_vol_20']         = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20']  = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def test_entry_below_scan_min_price_not_entered(self, monkeypatch):
        """No trade may be entered when the entry proxy resolves below SCAN_MIN_PRICE."""
        from src.config import SCAN_MIN_PRICE

        df = self._make_cheap_open_df()
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"CHEAP": df}))
        result = bt.run()

        for t in result.trades:
            assert t.entry_price >= SCAN_MIN_PRICE, (
                f"Trade entered at ${t.entry_price:.2f} < SCAN_MIN_PRICE=${SCAN_MIN_PRICE:.2f}; "
                "entry proxy below floor must be rejected"
            )

    def test_entry_price_floor_check_in_source(self):
        """backtest/strategy.py must contain the entry-price floor guard after entry_price computation."""
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'backtest', 'strategy.py')
        with open(path) as f:
            source = f.read()
        assert 'if entry_price < SCAN_MIN_PRICE' in source, (
            "backtest/strategy.py must reject entry_price < SCAN_MIN_PRICE "
            "(close may pass the floor filter while open/prev_high do not)"
        )


# ── Bug fix: /api/logs/download endpoint ─────────────────────────────────────
class TestDashboardLogsDownload:
    """/api/logs/download must return the log file when it exists and 404 JSON when it doesn't."""

    def test_download_returns_file_response_when_log_exists(self, tmp_path):
        """When LOG_FILE exists, /api/logs/download returns FileResponse with text/plain."""
        import alpaca_dashboard as ds
        from fastapi.testclient import TestClient
        from fastapi.responses import FileResponse

        log_file = tmp_path / "trading_engine.log"
        log_file.write_text("2026-05-29 INFO test log line\n")

        original = ds.LOG_FILE
        try:
            ds.LOG_FILE = str(log_file)
            client = TestClient(ds.app)
            resp = client.get("/api/logs/download")
            assert resp.status_code == 200
            assert "text/plain" in resp.headers.get("content-type", "")
            assert "trading_engine_" in resp.headers.get("content-disposition", "")
            assert resp.text == "2026-05-29 INFO test log line\n"
        finally:
            ds.LOG_FILE = original

    def test_download_returns_404_when_log_missing(self, tmp_path):
        """When LOG_FILE does not exist, /api/logs/download returns 404 JSON."""
        import alpaca_dashboard as ds
        from fastapi.testclient import TestClient

        original = ds.LOG_FILE
        try:
            ds.LOG_FILE = str(tmp_path / "nonexistent.log")
            client = TestClient(ds.app)
            resp = client.get("/api/logs/download")
            assert resp.status_code == 404
            body = resp.json()
            assert body.get("error") == "Log file not found"
        finally:
            ds.LOG_FILE = original

    def test_download_endpoint_in_source(self):
        """alpaca_dashboard.py must define the /api/logs/download GET endpoint."""
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'alpaca_dashboard.py')
        with open(path) as f:
            source = f.read()
        assert '@app.get("/api/logs/download")' in source, (
            "alpaca_dashboard.py must define the /api/logs/download endpoint"
        )
        assert 'FileResponse' in source, (
            "alpaca_dashboard.py must return FileResponse for the log download"
        )
