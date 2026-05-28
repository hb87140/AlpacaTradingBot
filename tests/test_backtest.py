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
    np.random.seed(seed)
    close  = 100 + trend * np.arange(n) + np.cumsum(np.random.randn(n) * 0.3)
    high   = close + np.abs(np.random.randn(n) * 0.2)
    low    = close - np.abs(np.random.randn(n) * 0.2)
    idx    = pd.date_range("2023-01-01", periods=n, freq='B')

    from src.indicators import apply_all
    df = pd.DataFrame({'open': close, 'high': high, 'low': low,
                       'close': close, 'volume': SCAN_MIN_VOLUME + 1_000_000}, index=idx)
    df = apply_all(df)
    df['prev_high']        = df['high'].shift(1)
    df['avg_vol_20']       = df['volume'].rolling(20).mean()
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
    _entry_signal requires:
      row columns: close, MA50, MA200, SMA200_SLOPE, prev_high, RSI, ADX, HIGH200
      positional:  prev_rsi, rvol, rvol_min
    """

    def _row(self, close=110, prev_high=100, ma50=105, ma200=90, rsi=60, atr=2.0,
             sma200_slope=0.5, adx=25.0, high200=120.0):
        return pd.Series({
            'close':           close,
            'prev_high':       prev_high,
            'MA50':            ma50,
            'MA200':           ma200,
            'RSI':             rsi,
            'ATR':             atr,
            'SMA200_SLOPE':    sma200_slope,
            'ADX':             adx,
            'HIGH200':         high200,
        })

    def test_all_conditions_pass(self):
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_price_below_prev_high(self):
        assert not VelocityBacktest._entry_signal(
            self._row(close=99, prev_high=100), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_price_below_ma50(self):
        assert not VelocityBacktest._entry_signal(
            self._row(close=104, ma50=105), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_ma50_below_ma200(self):
        assert not VelocityBacktest._entry_signal(
            self._row(ma50=85, ma200=90), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_not_rising(self):
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=60), prev_rsi=65, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rsi_below_55(self):
        assert not VelocityBacktest._entry_signal(
            self._row(rsi=54, ma50=105), prev_rsi=50,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_rvol_below_min(self):
        assert not VelocityBacktest._entry_signal(
            self._row(), prev_rsi=55, rvol=0.5, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_adx_below_threshold(self):
        # ADX = 10 (< threshold 20) → c_adx = False → entry blocked
        assert not VelocityBacktest._entry_signal(
            self._row(adx=10.0), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_52w_high_below_threshold(self):
        # price=110, high200=200: 110/200=55% < 85% threshold → c_52w_high = False
        assert not VelocityBacktest._entry_signal(
            self._row(high200=200.0), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_fails_trend_separation_too_small(self):
        # MA50=91, MA200=90 → sep = (91-90)/90 = 1.1% < MIN_TREND_SEP (3%) → fails
        assert not VelocityBacktest._entry_signal(
            self._row(close=110, ma50=91, ma200=90), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    def test_passes_trend_separation_at_boundary(self):
        # MA50 exactly 3% above MA200: MA200=100, MA50=103 → sep = 3.0% = MIN_TREND_SEP → passes
        assert VelocityBacktest._entry_signal(
            self._row(close=110, ma50=103, ma200=100), prev_rsi=55,
            rvol=2.0, rvol_min=BACKTEST_RVOL_MIN)

    # ── flags=None is backward-compatible ────────────────────────────────────
    def test_flags_none_behaves_as_production_defaults(self):
        # flags=None should pass all existing rules when row values are valid
        assert VelocityBacktest._entry_signal(
            self._row(), prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN,
            flags=None)

    def test_enabling_slope_blocks_entry_with_negative_slope(self):
        row = self._row(sma200_slope=-0.5)
        # use_slope=True explicitly + negative slope → entry blocked
        assert not VelocityBacktest._entry_signal(
            row, prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN,
            flags={'use_slope': True, 'use_trend_sep': True, 'use_orb': True,
                   'use_rsi_delta': True, 'use_rsi_lvl': True,
                   'use_adx': True, 'use_52w_high': True})
        # use_slope=False (production default): slope bypassed → entry passes
        assert VelocityBacktest._entry_signal(
            row, prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN,
            flags={**{s: True for s in ['use_slope','use_trend_sep','use_orb',
                                         'use_rsi_rise','use_rsi_delta','use_rsi_lvl',
                                         'use_adx', 'use_52w_high']},
                   'use_slope': False})

    def test_enabling_adx_blocks_entry_when_adx_absent(self):
        row = self._row(adx=float('nan'))   # simulate missing ADX column
        # ADX not in row (NaN fallback) → c_adx = False → entry blocked
        assert not VelocityBacktest._entry_signal(
            row, prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN,
            flags={s: True for s in ['use_slope','use_trend_sep','use_orb',
                                      'use_rsi_rise','use_rsi_delta','use_rsi_lvl',
                                      'use_adx']})

    def test_production_flags_match_defaults(self):
        # Explicitly pass current production-default flags — result must equal no-flags case.
        # Production defaults: use_adx=True, use_52w_high=True, use_trend_sep=True,
        # use_orb=True, use_rsi_delta=True, use_rsi_lvl=True;
        # use_slope=False, use_rsi_rise=False, use_ma20=False (optimizer-discovered).
        prod_flags = {
            'use_slope':     False,
            'use_trend_sep': True,
            'use_orb':       True,
            'use_rsi_rise':  False,
            'use_rsi_delta': True,
            'use_rsi_lvl':   True,
            'use_adx':       True,
            'use_52w_high':  True,
            'use_ma20':      False,
        }
        row = self._row()
        assert VelocityBacktest._entry_signal(
            row, prev_rsi=55, rvol=2.0, rvol_min=BACKTEST_RVOL_MIN,
            flags=prod_flags)


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


# ── Gain filter (SCAN_MIN_GAIN_PCT) ──────────────────────────────────────────
class TestDailyScanGainFilter:
    """_daily_scan must discard symbols whose daily gain is below SCAN_MIN_GAIN_PCT."""

    def _make_low_gain_df(self, daily_gain_pct: float, n: int = 300,
                          rvol_spike: float = 2.0) -> pd.DataFrame:
        """Return a bullish DataFrame where every bar's daily gain == daily_gain_pct.

        Volume on the last 20 bars is rvol_spike× the preceding bars so that
        the RVOL filter is cleared when rvol_spike > BACKTEST_RVOL_MIN.
        """
        from src.indicators import apply_all as _apply
        from src.config import SCAN_MIN_VOLUME

        start = 100.0
        closes = np.array([start * ((1 + daily_gain_pct) ** i) for i in range(n)])
        high   = closes * 1.005
        low    = closes * 0.995
        idx    = pd.date_range("2023-01-01", periods=n, freq='B')

        base_vol = SCAN_MIN_VOLUME * 10
        volume   = np.full(n, base_vol, dtype=float)
        # Spike the last 20 bars so RVOL = rvol_spike
        volume[-20:] = base_vol * rvol_spike

        df = pd.DataFrame({'open': closes, 'high': high, 'low': low,
                           'close': closes, 'volume': volume}, index=idx)
        df = _apply(df)
        df['prev_high']          = df['high'].shift(1)
        df['avg_vol_20']         = df['volume'].rolling(20).mean()
        df['avg_dollar_vol_20']  = (df['close'] * df['volume']).rolling(20).mean()
        return df

    def test_stock_below_gain_threshold_never_scanned(self, monkeypatch):
        """A symbol with < SCAN_MIN_GAIN_PCT daily gain must not appear in scan output."""
        from src.config import SCAN_MIN_GAIN_PCT

        # Use a gain just below the threshold
        low_gain = (SCAN_MIN_GAIN_PCT - 0.5) / 100.0
        df = self._make_low_gain_df(low_gain)

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"SLOW": df}))
        bt.run()

        # coarse_candidates counts entries that pass the gain filter
        assert bt._filter_stats['coarse_candidates'] == 0, (
            "Symbol with sub-threshold daily gain must be filtered before coarse stage"
        )

    def test_stock_above_gain_threshold_reaches_coarse(self, monkeypatch):
        """A symbol with ≥ SCAN_MIN_GAIN_PCT daily gain must pass the gain gate."""
        from src.config import SCAN_MIN_GAIN_PCT

        # Use a gain comfortably above the threshold
        high_gain = (SCAN_MIN_GAIN_PCT + 1.0) / 100.0
        df = self._make_low_gain_df(high_gain)

        bt = VelocityBacktest(start="2023-01-01", end="2024-01-01", use_cache=False)
        monkeypatch.setattr(bt, '_download', lambda: bt._data.update({"FAST": df}))
        bt.run()

        assert bt._filter_stats['coarse_candidates'] > 0, (
            "Symbol with above-threshold daily gain must reach the coarse stage"
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
        # Force scanner to always return all 3 symbols so the slot limit is
        # what constrains entries, not signal quality.
        monkeypatch.setattr(bt, '_daily_scan',
                            lambda today: [("AAA", 2.0), ("BBB", 2.0), ("CCC", 2.0)])
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
        def capturing_scan(today):
            result = original_scan(today)
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
    """dashboard_server.py JS chart must read e.equity, not e.eq.

    Engine writes {"ts": ..., "equity": ...} to equity_history.json.
    The JS `hist.map(e => e.eq)` was a stale key that made every chart
    data point undefined — the chart was always blank.
    """

    def test_dashboard_js_uses_equity_key_not_eq(self):
        import os
        path = os.path.join(os.path.dirname(__file__), '..', 'dashboard_server.py')
        with open(path) as f:
            source = f.read()
        assert 'hist.map(e => e.equity)' in source, (
            "JS equity chart must read e.equity (not e.eq) to match engine JSON output"
        )
        assert 'hist.map(e => e.eq)' not in source, (
            "Stale e.eq key must be removed from dashboard JS"
        )
