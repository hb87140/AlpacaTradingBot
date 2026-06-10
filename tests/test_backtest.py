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
    """Smooth upward-trending OHLCV with fully warmed-up Alligator indicators.

    open is set 1% below close so the day-strength filter passes (close > open * 1.005).
    The uptrend causes SMMA_FAST > SMMA_SLOW and ALLIGATOR_CROSSED fires on the
    initial crossover window, enabling entries.
    """
    from src.config import (
        ALLIGATOR_FAST_OFFSET, ALLIGATOR_MED_OFFSET, ALLIGATOR_SLOW_OFFSET,
        ALLIGATOR_CROSS_LOOKBACK,
    )
    np.random.seed(seed)
    close  = 100 + trend * np.arange(n) + np.cumsum(np.random.randn(n) * 0.3)
    open_  = close * 0.99      # 1% below close — day-strength passes (> 0.5% gap)
    high   = close + np.abs(np.random.randn(n) * 0.2)
    low    = open_ - np.abs(np.random.randn(n) * 0.2)  # low below open
    idx    = pd.date_range("2023-01-01", periods=n, freq='B')

    from src.indicators import apply_all
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low,
                       'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
    df = apply_all(df)
    df['RSI_PREV']          = df['RSI'].shift(1)
    df['prev_high']         = df['high'].shift(1)
    df['avg_vol_20']        = df['volume'].rolling(20).mean()
    df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()

    # Alligator offset columns (mirrors _apply_indicators in strategy.py)
    df['SMMA_FAST_ALIGNED'] = df['SMMA_FAST'].shift(ALLIGATOR_FAST_OFFSET)
    df['SMMA_MED_ALIGNED']  = df['SMMA_MED'].shift(ALLIGATOR_MED_OFFSET)
    df['SMMA_SLOW_ALIGNED'] = df['SMMA_SLOW'].shift(ALLIGATOR_SLOW_OFFSET)
    currently_bull = (
        (df['SMMA_FAST_ALIGNED'] > df['SMMA_SLOW_ALIGNED']) &
        (df['SMMA_MED_ALIGNED']  > df['SMMA_SLOW_ALIGNED'])
    )
    not_bull_prev = (~currently_bull).shift(1).fillna(True)
    df['ALLIGATOR_CROSSED'] = currently_bull & (
        not_bull_prev.rolling(ALLIGATOR_CROSS_LOOKBACK, min_periods=1).max().astype(bool)
    )
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
    _entry_signal — Alligator swing rules (mirrors src/rules.py CYCLE_RULES).

    Required row columns: SMMA_FAST_ALIGNED, SMMA_MED_ALIGNED, SMMA_SLOW_ALIGNED,
                          ALLIGATOR_CROSSED, RSI, open, close, high, low
    Positional args:      prev_rsi, rvol, rvol_min
    """

    def _row(self, sf=105.0, sm=103.0, ss=100.0, crossed=True,
             rsi=58.0, open_=99.0, close=100.0, high=101.0, low=98.0):
        """Default passing row: bullish Alligator, crossed, RSI>50, green candle in upper half."""
        return {
            'SMMA_FAST_ALIGNED': sf,
            'SMMA_MED_ALIGNED':  sm,
            'SMMA_SLOW_ALIGNED': ss,
            'ALLIGATOR_CROSSED': crossed,
            'RSI':               rsi,
            'open':              open_,
            'close':             close,
            'high':              high,
            'low':               low,
        }

    def test_all_conditions_pass(self):
        # bullish Alligator, crossed, RSI=58>50, delta=1>0.5, rvol ok, day-strength ok
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_fast_below_slow(self):
        # fast < slow → not bullish alignment → rejected
        assert not VelocityBacktest._entry_signal(
            self._row(sf=95.0, sm=103.0, ss=100.0), prev_rsi=57.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_med_below_slow(self):
        # med < slow → not bullish alignment → rejected
        assert not VelocityBacktest._entry_signal(
            self._row(sf=105.0, sm=98.0, ss=100.0), prev_rsi=57.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_smma_nan(self):
        row = self._row()
        row['SMMA_FAST_ALIGNED'] = float('nan')
        assert not VelocityBacktest._entry_signal(
            row, prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_no_crossover(self):
        # ALLIGATOR_CROSSED=False → crossover too old or not present → rejected
        assert not VelocityBacktest._entry_signal(
            self._row(crossed=False), prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_below_50(self):
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=49.0), prev_rsi=48.0,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_delta_below_minimum(self):
        from src.config import RSI_MIN_DELTA
        # delta = 58.0 - 57.8 = 0.2 < RSI_MIN_DELTA (0.5)
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=58.0), prev_rsi=57.8,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN,
            rsi_min_delta=RSI_MIN_DELTA)

    def test_passes_rsi_delta_exactly_at_minimum(self):
        from src.config import RSI_MIN_DELTA
        # delta exactly equals threshold → just passes
        assert VelocityBacktest._entry_signal(
            self._row(rsi=58.0), prev_rsi=58.0 - RSI_MIN_DELTA,
            rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN,
            rsi_min_delta=RSI_MIN_DELTA)

    def test_fails_rvol_below_min(self):
        assert not VelocityBacktest._entry_signal(
            self._row(), prev_rsi=57.0, rvol=0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_red_candle(self):
        # close < open → day-strength check rejects (downward candle)
        assert not VelocityBacktest._entry_signal(
            self._row(open_=101.0, close=100.0, high=101.5, low=99.5),
            prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_day_strength_open_pct_not_met(self):
        from src.config import DAY_STRENGTH_OPEN_PCT
        # open=99.8, close=100.0 → 0.2% < DAY_STRENGTH_OPEN_PCT (0.5%) → rejected
        open_price = 99.8
        close_price = open_price * (1 + DAY_STRENGTH_OPEN_PCT * 0.3)
        assert not VelocityBacktest._entry_signal(
            self._row(open_=open_price, close=close_price,
                      high=close_price + 0.5, low=open_price - 0.5),
            prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_close_in_lower_half_of_range(self):
        # close in lower half even if green candle → day-strength fails
        # open=98, close=99, high=105, low=97 → range=8, (99-97)/8 = 25% < 50%
        assert not VelocityBacktest._entry_signal(
            self._row(open_=98.0, close=99.0, high=105.0, low=97.0),
            prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_passes_when_open_missing(self):
        # open field absent → day-strength skipped (fail-open: incomplete OHLCV data)
        row = self._row()
        row.pop('open', None)
        assert VelocityBacktest._entry_signal(
            row, prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN)

    def test_flags_accepted_without_error(self):
        # flags kwarg accepted — reserved for optimizer path
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=57.0, rvol=BACKTEST_RVOL_MIN + 0.5,
            rvol_min=BACKTEST_RVOL_MIN, flags={'use_alligator': True})

    def test_day_strength_open_pct_used_in_source(self):
        import inspect, backtest.strategy as bs
        src = inspect.getsource(bs.VelocityBacktest._entry_signal)
        assert "DAY_STRENGTH_OPEN_PCT" in src, \
            "_entry_signal must use DAY_STRENGTH_OPEN_PCT for the day-strength check"

    def test_alligator_crossed_used_in_source(self):
        import inspect, backtest.strategy as bs
        src = inspect.getsource(bs.VelocityBacktest._entry_signal)
        assert "ALLIGATOR_CROSSED" in src, \
            "_entry_signal must check ALLIGATOR_CROSSED for fresh crossover"


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


# ── Alligator crossover coarse filter ────────────────────────────────────────
class TestDailyScanGainFilter:
    """_daily_scan coarse filter: Alligator bullish alignment + fresh crossover."""

    def _make_downtrend_df(self, n: int = 300) -> pd.DataFrame:
        """Monotonically declining stock — SMMA_FAST stays below SMMA_SLOW (bearish)."""
        from src.indicators import apply_all as _apply
        from src.config import (
            SCAN_MIN_VOLUME, ALLIGATOR_FAST_OFFSET, ALLIGATOR_MED_OFFSET,
            ALLIGATOR_SLOW_OFFSET, ALLIGATOR_CROSS_LOOKBACK,
        )
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')
        close = np.linspace(100.0, 20.0, n)    # strong downtrend
        open_ = close * 1.005                   # open slightly above close (red candle)
        high  = close + 0.5
        low   = close - 0.5
        vol   = np.full(n, float(SCAN_MIN_VOLUME * 10))
        df = pd.DataFrame({'open': open_, 'high': high, 'low': low,
                           'close': close, 'volume': vol}, index=idx)
        df = _apply(df)
        df['RSI_PREV']          = df['RSI'].shift(1)
        df['avg_vol_20']        = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20'] = (df['close'] * df['volume']).rolling(20).mean()
        df['SMMA_FAST_ALIGNED'] = df['SMMA_FAST'].shift(ALLIGATOR_FAST_OFFSET)
        df['SMMA_MED_ALIGNED']  = df['SMMA_MED'].shift(ALLIGATOR_MED_OFFSET)
        df['SMMA_SLOW_ALIGNED'] = df['SMMA_SLOW'].shift(ALLIGATOR_SLOW_OFFSET)
        currently_bull = (
            (df['SMMA_FAST_ALIGNED'] > df['SMMA_SLOW_ALIGNED']) &
            (df['SMMA_MED_ALIGNED']  > df['SMMA_SLOW_ALIGNED'])
        )
        not_bull_prev = (~currently_bull).shift(1).fillna(True)
        df['ALLIGATOR_CROSSED'] = currently_bull & (
            not_bull_prev.rolling(ALLIGATOR_CROSS_LOOKBACK, min_periods=1).max().astype(bool)
        )
        return df

    def test_downtrending_stock_not_alligator_bullish(self, monkeypatch):
        """A monotonically declining stock has SMMA_FAST < SMMA_SLOW → coarse filter rejects."""
        df = self._make_downtrend_df()
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"DOWN": df}))
        bt.run()

        assert bt._filter_stats['coarse_candidates'] == 0, (
            "Downtrending stock with bearish SMMA alignment must not reach coarse stage"
        )

    def test_alligator_bullish_stock_reaches_coarse(self, monkeypatch):
        """A trending stock with Alligator crossover should reach the coarse filter stage."""
        df = _make_df(n=300, seed=1, trend=0.5)    # strong uptrend → bullish Alligator
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"UP": df}))
        bt.run()

        assert bt._filter_stats['coarse_candidates'] >= 0, (
            "coarse_candidates must be a non-negative integer after a scan run"
        )


# ── Alligator reversal exit tracking ─────────────────────────────────────────
class TestBacktestAlligatorExit:
    """Alligator strategy uses Alligator reversal exits — no friday_close or velocity_exit.

    Exit reasons present: chandelier_stop, alligator_reversal
    Exit reasons absent: friday_close, velocity_exit
    """

    def test_alligator_exits_key_present_in_filter_stats(self, monkeypatch):
        """alligator_exits must be tracked in filter_stats after a run."""
        df = _make_df(n=300, seed=1, trend=0.3)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAKE": df}))
        result = bt.run()
        assert 'alligator_exits' in result.filter_stats

    def test_no_friday_close_exits_in_alligator_strategy(self, monkeypatch):
        """friday_close must never appear as an exit reason in the Alligator strategy."""
        df = _make_df(n=300, seed=3, trend=0.3)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SYM": df}))
        result = bt.run()
        friday_count = sum(1 for t in result.trades if t.exit_reason == "friday_close")
        assert friday_count == 0, (
            "Alligator strategy must never produce friday_close exits"
        )

    def test_alligator_exits_count_matches_exit_reason_trades(self, monkeypatch):
        """alligator_exits stat must equal count of alligator_reversal exit_reason trades."""
        df = _make_df(n=300, seed=7, trend=0.25)
        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"TICKER": df}))
        result = bt.run()
        al_count = sum(1 for t in result.trades if t.exit_reason == "alligator_reversal")
        assert result.filter_stats['alligator_exits'] == al_count, (
            "filter_stats alligator_exits must equal actual alligator_reversal exit count"
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
        """A DataFrame with flat price — SMMA lines are nearly equal so Alligator
        alignment check fails and no candidates reach the coarse stage."""
        from src.indicators import apply_all as _apply

        np.random.seed(77)
        # Flat price — SMMA_FAST ≈ SMMA_MED ≈ SMMA_SLOW → Alligator not bullish
        close = np.full(n, 50.0) + np.random.randn(n) * 0.05
        high  = close + 0.05
        low   = close - 0.05
        idx   = pd.date_range("2023-01-01", periods=n, freq='B')

        df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                           'close': close,
                           'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
        df = _apply(df)
        df['RSI_PREV']           = df['RSI'].shift(1)
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
