"""
Unit tests for VelocityEngine business logic (Alpaca API version).

Alpaca clients are fully mocked — no live connection required.
Tests exercise entry signals, velocity exits, position limits,
trailing stop order construction, and portfolio management.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import pytest
import pytz
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


_TZ_NY = pytz.timezone('US/Eastern')

# Shared SPY regime dict — used wherever _fetch_spy_trend is patched
_SPY_BULL_REGIME = {
    'is_bull': True, 'spy_close': 450.0, 'ema50': 440.0,
    'size_factor': 1.0, 'rvol_mult': 1.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_account(equity=2500.0, settled=2500.0):
    acc = MagicMock()
    acc.portfolio_value = str(equity)
    acc.cash            = str(settled)
    acc.id              = 'test-account'
    acc.status          = 'ACTIVE'
    return acc


def _mock_trading_client(equity=2500.0, settled=2500.0):
    tc = MagicMock()
    tc.get_account.return_value      = _mock_account(equity, settled)
    tc.get_all_positions.return_value = []
    tc.get_open_position.side_effect  = Exception("no position")
    tc.get_orders.return_value        = []
    tc.submit_order.return_value      = MagicMock(id='order-id')
    tc.cancel_order_by_id.return_value = None
    tc.get_order_by_id.return_value   = MagicMock(
        status='filled',
        filled_avg_price='100.0',
        filled_qty='1.0',
    )
    return tc


def _make_engine(equity=2500.0, settled=2500.0, trading_client=None, data_client=None):
    """Return a VelocityEngine with Alpaca clients replaced and __init__ bypassed."""
    from src.engine import VelocityEngine
    engine = VelocityEngine.__new__(VelocityEngine)
    engine.trading_client             = trading_client or _mock_trading_client(equity, settled)
    engine.data_client                = data_client or MagicMock()
    engine.screener_client            = MagicMock()
    engine.state                      = {}
    engine._last_equity               = 0.0
    engine._last_settled_cash         = 0.0
    engine._equity_initialized        = False
    engine._last_vix                  = None
    engine._vix_cache_date            = None
    engine._last_scan_ts              = None
    engine._next_scan_dt              = None
    engine._day_start_equity          = None
    engine._day_start_date            = None
    engine._bar_cache                 = {}
    engine._spy_cache                 = {}
    engine._sector_cache              = {}
    engine._analyst_cache             = {}
    engine._daily_scan_skip           = {}
    engine._insufficient_history_skip = set()
    engine._last_audit_date           = None
    engine._missing_position_counts   = {}
    engine._pnl_cache                 = None
    engine._pnl_cache_ts              = None
    return engine


def _make_snapshot(price=None, bid=0.0, ask=0.0, intraday_vol=5_000_000):
    if price is None:
        return None
    return {'live_price': price, 'bid': bid, 'ask': ask, 'intraday_vol': intraday_vol}


def _run_cycle_patched(engine, fake_now, ctx=None, symbols=None,
                       equity=1400.0, settled=1400.0):
    """Run one run_cycle() with all external calls mocked."""
    symbols = symbols or []
    with patch.object(engine, '_ensure_connected', return_value=True), \
         patch.object(engine, '_sync_positions'), \
         patch.object(engine, '_get_account_values', return_value=(equity, settled)), \
         patch.object(engine, '_fetch_vix', return_value=20.0), \
         patch.object(engine, 'check_velocity_exits', return_value={}), \
         patch.object(engine, '_audit_stop_orders'), \
         patch.object(engine, '_update_position_prices'), \
         patch.object(engine, '_write_dashboard_data'), \
         patch.object(engine, 'save_state'), \
         patch.object(engine, 'get_institutional_scan', return_value=symbols), \
         patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
         patch('src.engine.datetime') as mock_dt, \
         patch('time.sleep'):
        mock_dt.now.return_value = fake_now
        mock_dt.fromisoformat    = datetime.fromisoformat
        engine._last_equity       = equity
        engine._equity_initialized = True
        engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
        engine._day_start_equity  = equity
        if ctx is not None:
            with patch.object(engine, 'get_technical_context', return_value=ctx):
                engine.run_cycle()
        else:
            engine.run_cycle()


# ── Expert filter (entry conditions) ─────────────────────────────────────────
class TestExpertFilter:
    """
    Alligator Swing entry rules exercised via run_cycle(). Each test mutates
    exactly one field of the baseline ctx to verify the engine blocks that
    specific failing condition.

    Base ctx: price=100, Alligator bullish (5% fast/slow spread, fresh crossover),
              rsi=52/prev=45 (Δ=7.0), price 1% above open, in upper 86% of intraday
              range, rvol=3.0, spread=0.2%, dol_vol=300M.
    """

    def _base_ctx(self):
        return {
            # Alligator fields: 5% fast/slow separation → 30 pts; fresh crossover
            'smma_fast':       105.0,
            'smma_med':        102.0,
            'smma_slow':       100.0,
            'alligator_crossed': True,
            # RSI trend: delta=7.0 >= RSI_MIN_DELTA; rsi=52 >= 50
            'rsi':              52.0,   'rsi_prev':        45.0,
            'rsi_history':     [28.0, 30.0, 32.0, 35.0, 42.0],
            # Day strength: 1% above open, in upper 86% of range
            'intraday_open':   99.0,
            'intraday_high':  100.5,
            'intraday_low':    97.0,
            # Standard fields
            'live_price':     100.0,
            'atr':              2.0,   'atr_chandelier':   2.0,
            'rvol':             3.0,   'spread_pct':     0.002,
            'dollar_vol_20d': 300_000_000,
            'avg_20d_vol':  5_000_000,
            'volume':       5_000_000,
            'close':           99.5,
            'orb_high':        95.0,
            'ma50':           105.0,   'ma200':          90.0,
            'sma200_slope':     0.1,
            'price_fetched_at': _TZ_NY.localize(datetime(2024, 6, 5, 10, 30)),
        }

    def _engine_passes(self, ctx):
        """Wire ctx through run_cycle(); return True iff a BUY limit order is submitted."""
        tc      = _mock_trading_client(1400.0, 1400.0)
        engine  = _make_engine(1400.0, 1400.0, trading_client=tc)
        engine.state = {}

        from alpaca.trading.requests import LimitOrderRequest

        submitted = []

        def _submit(req):
            submitted.append(req)
            mo = MagicMock(id='buy-1')
            return mo

        tc.submit_order.side_effect    = _submit
        filled = MagicMock()
        filled.status                  = 'filled'
        filled.filled_avg_price        = '100.0'
        filled.filled_qty              = '1.0'
        tc.get_order_by_id.return_value = filled

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, 'get_technical_context',  return_value=ctx), \
             patch.object(engine, '_fetch_spy_trend',       return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_get_sector',            return_value='Technology'), \
             patch.object(engine, '_score_candidate',       return_value=75.0), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 1400.0
            engine.run_cycle()

        return any(isinstance(r, LimitOrderRequest) for r in submitted)

    def test_all_conditions_met(self):
        assert self._engine_passes(self._base_ctx()) is True

    def test_fails_when_alligator_not_crossed(self):
        """alligator_crossed=False → no fresh crossover signal → entry blocked."""
        ctx = self._base_ctx(); ctx['alligator_crossed'] = False
        assert self._engine_passes(ctx) is False

    def test_fails_when_smma_not_aligned(self):
        """fast SMMA below slow SMMA → downtrend, not bullish → entry blocked."""
        ctx = self._base_ctx()
        ctx['smma_fast'] = 90.0   # fast < slow → fails check_alligator_bullish
        ctx['smma_slow'] = 100.0
        assert self._engine_passes(ctx) is False

    def test_fails_when_rsi_not_rising(self):
        """RSI falling (rsi < rsi_prev) fails rsi_momentum delta check."""
        ctx = self._base_ctx(); ctx['rsi'] = 35.0; ctx['rsi_prev'] = 38.0
        assert self._engine_passes(ctx) is False

    def test_fails_when_rsi_delta_below_minimum(self):
        """rsi=37.5 < 50 fails check_rsi_trend (RSI not yet in bullish territory)."""
        ctx = self._base_ctx(); ctx['rsi'] = 37.5; ctx['rsi_prev'] = 35.0
        assert self._engine_passes(ctx) is False

    def test_fails_when_rsi_below_50(self):
        """RSI < 50 fails check_rsi_trend — stock not in bullish territory."""
        ctx = self._base_ctx()
        ctx['rsi']     = 48.0
        ctx['rsi_prev'] = 45.0   # delta=3 ≥ 1, but rsi=48 < 50 → fails
        assert self._engine_passes(ctx) is False

    def test_fails_when_price_in_lower_half_of_intraday_range(self):
        """Price in lower 40% of intraday range fails day_strength."""
        ctx = self._base_ctx()
        ctx['intraday_open'] = 98.0
        ctx['intraday_high'] = 103.0
        ctx['intraday_low']  = 98.0
        ctx['live_price']    = 100.0   # range_pos=(100-98)/(103-98)=0.4 < 0.5
        assert self._engine_passes(ctx) is False

    def test_fails_when_rvol_below_minimum(self):
        """RVOL below RVOL_MIN fails rvol rule."""
        ctx = self._base_ctx(); ctx['rvol'] = 0.0
        assert self._engine_passes(ctx) is False

    def test_fails_when_spread_too_wide(self):
        """Spread 1% > SPREAD_MAX_PCT=0.5% fails spread rule."""
        ctx = self._base_ctx(); ctx['spread_pct'] = 0.01
        assert self._engine_passes(ctx) is False

    def test_fails_when_dollar_volume_below_threshold(self):
        """Dollar volume 4M < SCAN_MIN_DOLLAR_VOL=5M fails dollar_vol rule."""
        ctx = self._base_ctx(); ctx['dollar_vol_20d'] = 4_000_000
        assert self._engine_passes(ctx) is False

    def test_passes_when_dollar_volume_at_threshold(self):
        """Dollar volume exactly at SCAN_MIN_DOLLAR_VOL passes dollar_vol rule."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        ctx = self._base_ctx(); ctx['dollar_vol_20d'] = SCAN_MIN_DOLLAR_VOL
        assert self._engine_passes(ctx) is True


# ── Velocity exit logic ───────────────────────────────────────────────────────
class TestVelocityExit:

    def _old_time(self, days_ago=14):
        return (datetime.now(_TZ_NY) - timedelta(days=days_ago)).isoformat()

    def test_stagnant_position_older_than_hold_bars_not_exited(self):
        """Alligator swing strategy holds positions until Alligator reversal — no time-based exit."""
        engine = _make_engine()
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        old_time  = (_safe_now - timedelta(days=14)).isoformat()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(101.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_profitable_position_not_exited_early(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'price': 100.0, 'time': self._old_time(14)}}

        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(106.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_fresh_position_not_exited(self):
        engine = _make_engine()
        fresh_time = datetime.now(_TZ_NY).isoformat()
        engine.state = {'AAPL': {'price': 100.0, 'time': fresh_time}}

        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(99.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_falls_back_to_current_price_when_snapshot_unavailable(self):
        """Stored current_price is used when live snapshot unavailable — position not exited if healthy."""
        engine = _make_engine()
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        old_time  = (_safe_now - timedelta(days=14)).isoformat()
        engine.state = {'AAPL': {
            'price': 100.0,
            'time': old_time,
            'current_price': 100.5,  # stored from previous cycle — 0.5% up, no hard stop
        }}

        with patch.object(engine, '_fetch_snapshot', return_value=None), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_missing_time_field_skips_velocity_check_without_crash(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'price': 100.0}}

        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(99.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            prices = engine.check_velocity_exits()

        mock_liq.assert_not_called()
        assert 'AAPL' in prices

    def test_malformed_time_field_skips_velocity_check_without_crash(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'price': 100.0, 'time': 'not-a-date'}}

        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(98.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            prices = engine.check_velocity_exits()

        mock_liq.assert_not_called()
        assert 'AAPL' in prices

    def test_pending_position_skipped(self):
        """Positions with pending=True must be skipped entirely."""
        engine = _make_engine()
        engine.state = {'AAPL': {
            'price': 100.0,
            'time': self._old_time(14),
            'pending': True,
        }}

        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(101.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_pending_exit_position_skipped(self):
        """Positions with pending_exit=True must be skipped (sell already submitted)."""
        engine = _make_engine()
        engine.state = {'AAPL': {
            'price': 100.0,
            'time': self._old_time(14),
            'pending_exit': True,
        }}

        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(101.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()

        mock_liq.assert_not_called()


    def test_after_market_close_returns_empty_no_sells(self):
        """Software exits must be suppressed after 16:00 ET — trailing stop handles it."""
        engine = _make_engine()
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        old_time = (_safe_now - timedelta(days=14)).isoformat()
        engine.state = {'AAPL': {'price': 100.0, 'time': old_time}}

        _after_close = _TZ_NY.localize(datetime(2024, 6, 5, 17, 28))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(94.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _after_close
            mock_dt.fromisoformat    = datetime.fromisoformat
            result = engine.check_velocity_exits()

        mock_liq.assert_not_called()
        assert result == {}


# ── Break-even floor enforcement ──────────────────────────────────────────────
class TestBreakEvenExitEnforcement:
    """check_velocity_exits() must exit programmatically when price retraces
    below entry after break-even was previously triggered (peak ≥ entry×1.04)."""

    def _make_state(self, entry, peak, cur_price):
        engine = _make_engine()
        fresh  = datetime.now(_TZ_NY).isoformat()
        engine.state = {
            'AAPL': {
                'price':      entry,
                'peak_price': peak,
                'time':       fresh,
                'qty':        10,
            }
        }
        return engine

    def test_exit_triggered_when_price_below_entry_after_break_even(self):
        from src.config import BREAK_EVEN_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT + 0.01)
        cur   = entry - 0.50

        engine = self._make_state(entry, peak, cur)
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(cur)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_called_once_with('AAPL')

    def test_no_exit_when_price_above_entry_after_break_even(self):
        from src.config import BREAK_EVEN_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT + 0.01)
        cur   = entry + 0.01

        engine = self._make_state(entry, peak, cur)
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(cur)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_no_exit_when_break_even_not_yet_triggered(self):
        from src.config import BREAK_EVEN_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT - 0.01)
        cur   = entry - 2.0

        engine = self._make_state(entry, peak, cur)
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(cur)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_not_called()

    def test_hard_stop_takes_priority(self):
        from src.config import BREAK_EVEN_PCT, HARD_STOP_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT + 0.10)
        cur   = entry * (1 - HARD_STOP_PCT - 0.01)

        engine = self._make_state(entry, peak, cur)
        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(cur)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine.check_velocity_exits()

        mock_liq.assert_called_once_with('AAPL')


# ── Position limit ────────────────────────────────────────────────────────────
class TestPositionLimit:
    def test_max_positions_cap(self):
        """run_cycle must not scan when all dynamic slots are filled."""
        from src.config import MIN_BUCKET_SIZE, MAX_POSITIONS_CAP

        n_slots    = min(int(2500 / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        state_full = {f'SYM{i}': {'price': 100, 'time': datetime.now().isoformat()}
                      for i in range(n_slots)}

        engine = _make_engine(2500.0, 2500.0)
        engine.state = state_full

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_check_portfolio_concentration', return_value=(0.5, False, False)), \
             patch.object(engine, 'get_institutional_scan', return_value=['NEW']), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 2500.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 2500.0
            engine.run_cycle()

        mock_ctx.assert_not_called()


# ── State persistence ─────────────────────────────────────────────────────────
class TestStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        engine = _make_engine()
        engine.state = {'AAPL': {'price': 123.45, 'time': '2024-01-01T10:00:00'}}

        with open(state_path, 'w') as f:
            json.dump(engine.state, f)

        with open(state_path) as f:
            loaded = json.load(f)

        assert loaded == engine.state


# ── Friday filter ─────────────────────────────────────────────────────────────
class TestFridayFilter:
    def _passes(self, dollar_vol_20d, is_friday):
        from src.config import SCAN_MIN_DOLLAR_VOL, VOL_MULT_FRIDAY
        threshold = SCAN_MIN_DOLLAR_VOL * (VOL_MULT_FRIDAY if is_friday else 1.0)
        return dollar_vol_20d >= threshold

    def test_weekday_uses_normal_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(SCAN_MIN_DOLLAR_VOL, is_friday=False) is True

    def test_friday_rejects_at_normal_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(SCAN_MIN_DOLLAR_VOL, is_friday=True) is False

    def test_friday_passes_at_double_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL, VOL_MULT_FRIDAY
        assert self._passes(SCAN_MIN_DOLLAR_VOL * VOL_MULT_FRIDAY, is_friday=True) is True

    def test_friday_rejects_below_double_threshold(self):
        from src.config import SCAN_MIN_DOLLAR_VOL
        assert self._passes(int(SCAN_MIN_DOLLAR_VOL * 1.5), is_friday=True) is False


# ── VIX risk filter ───────────────────────────────────────────────────────────
class TestVixHighBranch:
    """When VIX > threshold, velocity exits and price updates must still run."""

    def test_vix_high_still_calls_velocity_exits(self):
        engine = _make_engine()
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, '_fetch_vix', return_value=40.0), \
             patch.object(engine, 'check_velocity_exits') as mock_vel, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 2500.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 2500.0
            engine.run_cycle()

        mock_vel.assert_called_once()
        mock_upd.assert_called_once()

    def test_vix_unavailable_skips_entries(self):
        """When VIX is None, new entries are skipped but exits still run."""
        engine = _make_engine()
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(2500.0, 2500.0)), \
             patch.object(engine, '_fetch_vix', return_value=None), \
             patch.object(engine, 'check_velocity_exits') as mock_vel, \
             patch.object(engine, '_update_position_prices') as mock_upd, \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 2500.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 2500.0
            engine.run_cycle()

        mock_vel.assert_called_once()
        mock_upd.assert_called_once()


# ── Logger handler guard ──────────────────────────────────────────────────────
class TestLoggerHandlerGuard:
    def test_no_duplicate_handlers_on_reimport(self):
        import importlib
        import src.engine as eng_mod

        handler_count_before = len(logging.getLogger('VelocityEngine').handlers)
        importlib.reload(eng_mod)
        handler_count_after  = len(logging.getLogger('VelocityEngine').handlers)

        assert handler_count_after == handler_count_before, (
            f"Re-import added handlers: {handler_count_before} → {handler_count_after}"
        )


# ── Eastern formatter immunity to datetime mock ───────────────────────────────
class TestEasternFormatterTimestamp:
    def test_format_time_is_real_string_under_datetime_patch(self):
        from src.engine import _EasternFormatter

        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='hello', args=(), exc_info=None,
        )
        formatter = _EasternFormatter('%(asctime)s | %(message)s')

        with patch('src.engine.datetime', MagicMock()):
            ts = formatter.formatTime(record)

        assert 'MagicMock' not in ts
        assert len(ts) >= 19


# ── BUY order uses DAY tif ────────────────────────────────────────────────────
class TestBuyOrderTif:
    """BUY limit order must use TimeInForce.DAY."""

    def test_buy_order_uses_day_tif(self):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums   import TimeInForce

        tc = _mock_trading_client(1400.0, 1400.0)
        engine = _make_engine(1400.0, 1400.0, trading_client=tc)
        engine.state = {}

        submitted = []

        def _submit(req):
            submitted.append(req)
            return MagicMock(id='order-1')

        tc.submit_order.side_effect = _submit
        filled = MagicMock()
        filled.status           = 'filled'
        filled.filled_avg_price = '110.0'
        filled.filled_qty       = '1.0'
        tc.get_order_by_id.return_value = filled

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        ctx = {
            'live_price': 100.0, 'close': 99.5,
            'smma_fast': 105.0, 'smma_med': 102.0, 'smma_slow': 100.0,
            'alligator_crossed': True,
            'rsi': 52.0, 'rsi_prev': 45.0,
            'rsi_history': [28.0, 30.0, 32.0, 35.0, 42.0],
            'intraday_open': 99.0, 'intraday_high': 100.5, 'intraday_low': 97.0,
            'atr': 2.0, 'atr_chandelier': 2.0,
            'rvol': 3.0, 'spread_pct': 0.002,
            'dollar_vol_20d': 300_000_000, 'avg_20d_vol': 5_000_000, 'volume': 5_000_000,
            'price_fetched_at': fake_now,
        }

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, 'get_technical_context', return_value=ctx), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, '_get_sector', return_value='Technology'), \
             patch.object(engine, '_score_candidate', return_value=75.0), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 1400.0
            engine.run_cycle()

        buy_orders = [r for r in submitted if isinstance(r, LimitOrderRequest)]
        assert buy_orders, "No BUY limit order was placed"
        assert all(r.time_in_force == TimeInForce.DAY for r in buy_orders)


# ── Scanner gain config sentinel ──────────────────────────────────────────────
class TestScannerGainConfig:
    def test_scanner_gain_config_value(self):
        from src.config import SCAN_MIN_GAIN_PCT
        assert SCAN_MIN_GAIN_PCT == 2.0


# ── _daily_scan_skip — day-permanent condition caching ────────────────────────
class TestDailyScanSkip:
    """Symbols that fail day-permanent conditions are cached and bypassed next cycle."""

    def _run_cycle_at(self, engine, fake_now, ctx=None, symbols=None):
        symbols = symbols or []
        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=symbols), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            # Only set _day_start_date when not already configured by the test
            if engine._day_start_date is None:
                engine._day_start_date = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 1400.0
            if ctx is not None:
                with patch.object(engine, 'get_technical_context', return_value=ctx):
                    engine.run_cycle()
            else:
                engine.run_cycle()

    def test_permanent_fail_cached_after_first_scan(self):
        """Symbol with dollar_vol below threshold is cached as permanent-day fail."""
        from src.config import SCAN_MIN_DOLLAR_VOL
        engine = _make_engine()
        engine.state = {}

        # dollar_vol_20d=4M < SCAN_MIN_DOLLAR_VOL=5M → PERMANENT_DAY_RULES fail
        ctx = {
            'live_price': 100.0,
            'rsi': 38.0, 'rsi_prev': 35.0,
            'rsi_history': [28.0, 30.0, 32.0, 35.0, 38.0],
            'intraday_open': 99.0, 'intraday_high': 100.5, 'intraday_low': 97.0,
            'atr': 2.0, 'atr_chandelier': 2.0,
            'rvol': 3.0, 'spread_pct': 0.002,
            'dollar_vol_20d': 4_000_000,   # below 5M threshold → permanent fail
            'avg_20d_vol': 5_000_000, 'volume': 5_000_000,
            'close': 99.5,
            'price_fetched_at': _TZ_NY.localize(datetime(2024, 6, 5, 10, 30)),
        }
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        self._run_cycle_at(engine, fake_now, ctx=ctx, symbols=['MSFT'])

        assert 'MSFT' in engine._daily_scan_skip
        assert 'DolVol20d' in engine._daily_scan_skip['MSFT']

    def test_cached_symbol_skips_technical_context_fetch(self):
        engine = _make_engine()
        engine.state = {}
        engine._daily_scan_skip = {'MSFT': 'dollar_vol: DolVol20d $5.0M < $20.0M'}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        with patch.object(engine, 'get_technical_context') as mock_ctx:
            self._run_cycle_at(engine, fake_now, symbols=['MSFT'])

        mock_ctx.assert_not_called()

    def test_daily_scan_skip_cleared_on_new_day(self):
        engine = _make_engine()
        engine._daily_scan_skip  = {
            'MSFT': 'dollar_vol: DolVol20d $5.0M < $20.0M',
            'AAPL': 'dollar_vol: DolVol20d $10.0M < $20.0M',
        }
        engine._day_start_date   = '2024-06-04'
        engine._day_start_equity = 1400.0
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        self._run_cycle_at(engine, fake_now)

        assert engine._daily_scan_skip == {}

    def test_dynamic_only_fail_not_cached(self):
        """Symbol that only fails intraday-dynamic (CYCLE) conditions must NOT be day-cached."""
        engine = _make_engine()
        engine.state = {}

        # dollar_vol passes (permanent check) but rvol=0 fails (cycle check)
        ctx = {
            'live_price': 100.0,
            'rsi': 38.0, 'rsi_prev': 35.0,
            'rsi_history': [28.0, 30.0, 32.0, 35.0, 38.0],
            'intraday_open': 99.0, 'intraday_high': 100.5, 'intraday_low': 97.0,
            'atr': 2.0, 'atr_chandelier': 2.0,
            'rvol': 0.0,   # below RVOL_MIN=1.2 → cycle fail only
            'spread_pct': 0.002,
            'dollar_vol_20d': 500_000_000,   # passes permanent dollar_vol check
            'avg_20d_vol': 5_000_000, 'volume': 5_000_000,
            'close': 99.5,
            'price_fetched_at': _TZ_NY.localize(datetime(2024, 6, 5, 10, 30)),
        }
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))

        self._run_cycle_at(engine, fake_now, ctx=ctx, symbols=['NVDA'])

        assert 'NVDA' not in engine._daily_scan_skip


# ── liquidate() guards ────────────────────────────────────────────────────────
class TestLiquidateGuards:
    """liquidate() must not double-sell, must preserve state with pending_exit=True."""

    def test_skip_sell_when_position_already_zero(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {'price': 100.0}}

        tc.get_open_position.side_effect = Exception("no position")

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('AAPL')

        tc.submit_order.assert_not_called()

    def test_state_marked_pending_exit_after_liquidate(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {'price': 100.0, 'time': '2024-06-03T10:30:00-04:00'}}

        pos = MagicMock()
        pos.qty = '5.0'
        tc.get_open_position.side_effect = None
        tc.get_open_position.return_value = pos

        trade = MagicMock(id='sell-1')
        tc.submit_order.return_value = trade

        filled = MagicMock()
        filled.status = 'filled'
        tc.get_order_by_id.return_value = filled

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('AAPL')

        assert 'AAPL' in engine.state
        assert engine.state['AAPL'].get('pending_exit') is True
        assert engine.state['AAPL']['time'] == '2024-06-03T10:30:00-04:00'

    def test_state_deleted_immediately_when_position_zero(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {'price': 100.0}}

        tc.get_open_position.side_effect = Exception("no position")

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('AAPL')

        assert 'AAPL' not in engine.state

    def test_state_preserved_without_pending_exit_when_submit_raises(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {'price': 100.0}}

        pos = MagicMock()
        pos.qty = '10.0'
        tc.get_open_position.side_effect = None
        tc.get_open_position.return_value = pos

        tc.submit_order.side_effect = RuntimeError("connection dropped")

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('AAPL')

        assert 'AAPL' in engine.state
        assert not engine.state['AAPL'].get('pending_exit')

    def test_all_open_orders_cancelled_before_sell(self):
        """All open orders (including trailing stop) are cancelled before the market sell.

        Alpaca holds shares 'for orders' — the GTC TRAIL reserves the full position qty,
        so a market SELL is rejected with available=0 while the TRAIL is open.
        """
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'MSFT': {'price': 50.0}}

        pos = MagicMock()
        pos.qty = '20.0'
        tc.get_open_position.return_value = pos

        # A trailing stop and a stale LMT sell
        trail = MagicMock()
        trail.id         = 'trail-1'
        trail.symbol     = 'MSFT'
        trail.order_type = 'trailing_stop'

        stale_lmt = MagicMock()
        stale_lmt.id         = 'lmt-1'
        stale_lmt.symbol     = 'MSFT'
        stale_lmt.order_type = 'limit'

        tc.get_orders.return_value = [trail, stale_lmt]

        trade = MagicMock(id='sell-1')
        tc.submit_order.return_value = trade

        filled = MagicMock()
        filled.status = 'filled'
        tc.get_order_by_id.return_value = filled

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('MSFT')

        # Both the TRAIL and the stale LMT must be cancelled before the market sell
        cancelled_ids = [c.args[0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'lmt-1' in cancelled_ids
        assert 'trail-1' in cancelled_ids


# ── Break-even floor math ─────────────────────────────────────────────────────
class TestBreakEvenFloor:
    def test_break_even_floor_applied_when_peak_above_threshold(self):
        from src.config import BREAK_EVEN_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT + 0.01)
        chandelier_dist = (peak - entry) + 2.0

        raw_stop = round(max(entry, peak) - chandelier_dist, 2)
        if entry > 0 and peak >= entry * (1 + BREAK_EVEN_PCT):
            raw_stop = max(raw_stop, entry)

        assert raw_stop >= entry

    def test_no_break_even_floor_below_threshold(self):
        from src.config import BREAK_EVEN_PCT
        entry = 100.0
        peak  = entry * (1 + BREAK_EVEN_PCT - 0.01)
        chandelier_dist = peak - entry + 5.0

        raw_stop = round(max(entry, peak) - chandelier_dist, 2)
        if entry > 0 and peak >= entry * (1 + BREAK_EVEN_PCT):
            raw_stop = max(raw_stop, entry)

        assert raw_stop < entry

    def test_break_even_threshold_uses_config_constant(self):
        from src.config import BREAK_EVEN_PCT
        assert BREAK_EVEN_PCT == 0.06


# ── Equity history downsampling ───────────────────────────────────────────────
class TestEquityHistoryDownsampling:
    def test_snapshot_skipped_within_interval(self, tmp_path):
        import time, json
        from src.config import EQUITY_HIST_INTERVAL

        hist_path = str(tmp_path / "equity_history.json")
        dash_path = str(tmp_path / "dashboard_data.json")
        # Pre-seed: last snapshot was 1 second ago — well within the interval
        recent_ts = (datetime.now(_TZ_NY) - timedelta(seconds=1)).isoformat()
        with open(hist_path, 'w') as f:
            json.dump([{"ts": recent_ts, "equity": 1400.0}], f)

        engine = _make_engine()
        engine._last_equity       = 1450.0
        engine._equity_initialized = True

        import src.engine as eng_mod
        orig_eq   = eng_mod.EQUITY_HIST_FILE
        orig_dash = eng_mod.DASHBOARD_FILE
        eng_mod.EQUITY_HIST_FILE = hist_path
        eng_mod.DASHBOARD_FILE   = dash_path
        try:
            engine._write_dashboard_data(connected=True)
        finally:
            eng_mod.EQUITY_HIST_FILE = orig_eq
            eng_mod.DASHBOARD_FILE   = orig_dash

        with open(hist_path) as f:
            history = json.load(f)

        assert len(history) == 1

    def test_snapshot_written_after_interval(self, tmp_path):
        import time, json
        from src.config import EQUITY_HIST_INTERVAL

        hist_path = str(tmp_path / "equity_history.json")
        dash_path = str(tmp_path / "dashboard_data.json")
        old_ts = (datetime.now(_TZ_NY) - timedelta(seconds=EQUITY_HIST_INTERVAL + 60)).isoformat()
        with open(hist_path, 'w') as f:
            json.dump([{"ts": old_ts, "equity": 1400.0}], f)

        engine = _make_engine()
        engine._last_equity       = 1450.0
        engine._equity_initialized = True

        import src.engine as eng_mod
        orig_eq   = eng_mod.EQUITY_HIST_FILE
        orig_dash = eng_mod.DASHBOARD_FILE
        eng_mod.EQUITY_HIST_FILE = hist_path
        eng_mod.DASHBOARD_FILE   = dash_path
        try:
            engine._write_dashboard_data(connected=True)
        finally:
            eng_mod.EQUITY_HIST_FILE = orig_eq
            eng_mod.DASHBOARD_FILE   = orig_dash

        with open(hist_path) as f:
            history = json.load(f)

        assert len(history) == 2


# ── liquidate() position-not-found guard ─────────────────────────────────────
class TestLiquidatePositionNotFound:
    def test_liquidate_returns_early_when_position_not_found(self):
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {'price': 100.0}}

        tc.get_open_position.side_effect = Exception("position not found")

        with patch.object(engine, 'save_state') as mock_save, \
             patch('time.sleep'):
            engine.liquidate('AAPL')

        # No sell order placed; state deleted (qty=0 path)
        tc.submit_order.assert_not_called()

    def test_trailing_stop_cancelled_in_liquidate(self):
        """TRAIL SELL must be cancelled before the market sell.

        Alpaca's 'held_for_orders' mechanism reserves shares for open orders.
        Without cancelling the TRAIL first, the market sell fails with available=0.
        """
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'MSFT': {'price': 50.0}}

        pos = MagicMock()
        pos.qty = '20.0'
        tc.get_open_position.return_value = pos

        trail = MagicMock()
        trail.id         = 'trail-99'
        trail.symbol     = 'MSFT'
        trail.order_type = 'trailing_stop'
        tc.get_orders.return_value = [trail]

        trade = MagicMock(id='sell-1')
        tc.submit_order.return_value = trade
        filled = MagicMock(); filled.status = 'filled'
        tc.get_order_by_id.return_value = filled

        with patch.object(engine, 'save_state'), \
             patch('time.sleep'):
            engine.liquidate('MSFT')

        cancelled_ids = [c.args[0] for c in tc.cancel_order_by_id.call_args_list]
        assert 'trail-99' in cancelled_ids


# ── _sync_positions pending_exit handling ─────────────────────────────────────
class TestSyncPendingExit:
    """_sync_positions must correctly handle pending_exit flags."""

    def test_pending_exit_cleared_when_position_still_visible(self):
        """If position is still at Alpaca after pending_exit, clear the flag (sell rejected)."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        original_time = '2024-06-03T10:30:00-04:00'
        engine.state = {'AAPL': {
            'price': 100.0,
            'time':  original_time,
            'qty':   10.0,
            'pending_exit': True,
        }}

        pos = MagicMock()
        pos.symbol          = 'AAPL'
        pos.qty             = '10.0'
        pos.avg_entry_price = '100.0'
        tc.get_all_positions.return_value = [pos]

        with patch.object(engine, 'save_state'):
            engine._sync_positions()

        assert 'AAPL' in engine.state
        assert not engine.state['AAPL'].get('pending_exit')
        assert engine.state['AAPL']['time'] == original_time

    def test_pending_exit_state_deleted_when_position_gone(self):
        """When Alpaca confirms position is gone (2 consecutive misses), state is removed."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {'AAPL': {
            'price': 100.0,
            'time':  '2024-06-03T10:30:00-04:00',
            'qty':   10.0,
            'pending_exit': True,
        }}

        # Position gone from Alpaca
        tc.get_all_positions.return_value = []
        # No orphaned sell orders to cancel
        tc.get_orders.return_value = []

        with patch.object(engine, 'save_state'):
            engine._sync_positions()  # first miss — defers
            engine._sync_positions()  # second miss — removes

        assert 'AAPL' not in engine.state

    def test_unknown_position_readded_from_alpaca(self):
        """An untracked Alpaca position (e.g. after crash) is re-added to state."""
        tc = _mock_trading_client()
        engine = _make_engine(trading_client=tc)
        engine.state = {}

        pos = MagicMock()
        pos.symbol          = 'AAPL'
        pos.qty             = '10.0'
        pos.avg_entry_price = '100.0'
        tc.get_all_positions.return_value = [pos]

        with patch.object(engine, 'save_state'):
            engine._sync_positions()

        assert 'AAPL' in engine.state


# ── Portfolio concentration ───────────────────────────────────────────────────
class TestPortfolioConcentration:
    def test_concentration_empty_portfolio(self):
        engine = _make_engine()
        engine.state = {}
        conc, halt_entries, halt_all = engine._check_portfolio_concentration(10000.0)
        assert conc == 0.0
        assert halt_entries is False
        assert halt_all is False

    def test_concentration_single_position(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': 50.0, 'price': 150.0, 'current_price': 150.0}}
        conc, _, _ = engine._check_portfolio_concentration(10000.0)
        assert conc == 0.75

    def test_concentration_halt_entries_at_warn_threshold(self):
        from src.config import CONCENTRATION_WARN_PCT, CONCENTRATION_HALT_PCT
        # Place just above the warn threshold (but below halt threshold)
        price, equity = 150.0, 10000.0
        warn_qty = (CONCENTRATION_WARN_PCT * equity / price) + 1
        assert warn_qty * price / equity < CONCENTRATION_HALT_PCT, "qty must be below halt"
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': warn_qty, 'price': price, 'current_price': price}}
        _, halt_entries, halt_all = engine._check_portfolio_concentration(equity)
        assert halt_entries is True
        assert halt_all is False

    def test_concentration_halt_all_at_halt_threshold(self):
        from src.config import CONCENTRATION_HALT_PCT
        price, equity = 150.0, 10000.0
        halt_qty = (CONCENTRATION_HALT_PCT * equity / price) + 1
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': halt_qty, 'price': price, 'current_price': price}}
        _, halt_entries, halt_all = engine._check_portfolio_concentration(equity)
        assert halt_all is True
        assert halt_entries is True

    def test_concentration_below_thresholds(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': 50.0, 'price': 150.0, 'current_price': 150.0}}
        _, halt_entries, halt_all = engine._check_portfolio_concentration(10000.0)
        assert halt_entries is False
        assert halt_all is False

    def test_concentration_multiple_positions(self):
        engine = _make_engine()
        engine.state = {
            'AAPL': {'qty': 30.0, 'price': 150.0, 'current_price': 150.0},
            'MSFT': {'qty': 10.0, 'price': 300.0, 'current_price': 300.0},
        }
        conc, _, _ = engine._check_portfolio_concentration(20000.0)
        expected = ((30.0 * 150.0) + (10.0 * 300.0)) / 20000.0
        assert abs(conc - expected) < 0.01

    def test_concentration_ignores_invalid_price(self):
        engine = _make_engine()
        engine.state = {
            'AAPL': {'qty': 50.0, 'price': 150.0, 'current_price': 150.0},
            'MSFT': {'qty': 50.0, 'price': 0.0, 'current_price': 0.0},
        }
        conc, _, _ = engine._check_portfolio_concentration(10000.0)
        expected = (50.0 * 150.0) / 10000.0
        assert abs(conc - expected) < 0.01

    def test_concentration_zero_equity_returns_zero(self):
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': 100.0, 'price': 150.0}}
        conc, halt_entries, halt_all = engine._check_portfolio_concentration(0.0)
        assert conc == 0.0
        assert halt_entries is False
        assert halt_all is False

    def test_concentration_uses_current_price_not_entry_price(self):
        """Winning position: current_price > entry price — deployed must use current."""
        engine = _make_engine()
        # Entry at $100, now trading at $200 (doubled) — 10 shares
        engine.state = {'AAPL': {'qty': 10.0, 'price': 100.0, 'current_price': 200.0}}
        conc, _, _ = engine._check_portfolio_concentration(10000.0)
        # Deployed = 200 * 10 = 2000; pct = 0.20 (not 0.10 from entry price)
        assert abs(conc - 0.20) < 0.001

    def test_concentration_falls_back_to_price_when_no_current_price(self):
        """When current_price is absent (pending fill), entry price is the fallback."""
        engine = _make_engine()
        engine.state = {'AAPL': {'qty': 10.0, 'price': 150.0}}
        conc, _, _ = engine._check_portfolio_concentration(10000.0)
        assert abs(conc - 0.15) < 0.001

    def test_concentration_halt_triggered_by_appreciation(self):
        """A position that doubled should trigger halt even if cost basis was below 95%."""
        engine = _make_engine()
        # Cost basis: 60 * 100 = $6000 = 60% (no halt at equity $10000)
        # Current: 60 * 170 = $10200 = 102% → halt_all fires
        engine.state = {'AAPL': {'qty': 60.0, 'price': 100.0, 'current_price': 170.0}}
        _, halt_entries, halt_all = engine._check_portfolio_concentration(10000.0)
        assert halt_entries is True
        assert halt_all is True


# ── Dashboard max_positions consistency ──────────────────────────────────────
class TestDashboardMaxPositions:
    def test_returns_zero_when_settled_below_min_bucket(self):
        from src.config import MIN_BUCKET_SIZE, MAX_POSITIONS_CAP
        equity = MIN_BUCKET_SIZE - 0.01
        dyn_max_pos = (
            min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
            if equity >= MIN_BUCKET_SIZE else 0
        )
        assert dyn_max_pos == 0

    def test_matches_engine_calc_max_positions(self):
        from src.config import MIN_BUCKET_SIZE, MAX_POSITIONS_CAP
        engine = _make_engine()
        for equity in [0.0, 100.0, 499.99, 500.0, 1000.0, 2500.0, 4000.0, 10000.0]:
            engine_result = engine._calc_max_positions(equity)
            dashboard_result = (
                min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
                if equity >= MIN_BUCKET_SIZE else 0
            )
            assert engine_result == dashboard_result

    def test_engine_returns_zero_not_one_when_insufficient(self):
        from src.config import MIN_BUCKET_SIZE
        engine = _make_engine()
        assert engine._calc_max_positions(MIN_BUCKET_SIZE - 1) == 0

    def test_bucket_size_is_zero_when_all_slots_filled(self):
        from src.config import MIN_BUCKET_SIZE, MAX_POSITIONS_CAP, BUCKET_CASH_PCT
        settled_cash = MIN_BUCKET_SIZE * 3
        max_pos = min(int(settled_cash / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)
        n_positions = max_pos
        _raw_open_slots = max_pos - n_positions
        bucket_size = (
            round((settled_cash * BUCKET_CASH_PCT) / _raw_open_slots, 2)
            if _raw_open_slots > 0 else 0.0
        )
        assert bucket_size == 0.0


# ── Friday entry window close ─────────────────────────────────────────────────
class TestFridayEntryWindowClosed:
    def test_no_scan_on_friday_after_close_hour(self):
        from src.config import FRIDAY_CLOSE_HOUR
        engine = _make_engine()
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 7, FRIDAY_CLOSE_HOUR, 15))
        assert fake_now.weekday() == 4

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, 'get_technical_context') as mock_ctx, \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 1400.0
            engine.run_cycle()

        mock_ctx.assert_not_called()

    def test_entries_allowed_on_friday_before_close_hour(self):
        engine = _make_engine()
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 7, 11, 0))
        assert fake_now.weekday() == 4

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL']), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, 'get_technical_context', return_value=None) as mock_ctx, \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 1400.0
            engine.run_cycle()

        mock_ctx.assert_called_once()


# ── _tod_frac intraday volume profile ─────────────────────────────────────────
class TestTodFrac:
    def test_at_close_returns_one(self):
        from src.engine import _tod_frac
        assert abs(_tod_frac(390.0) - 1.0) < 1e-9

    def test_first_30min_returns_22pct(self):
        from src.engine import _tod_frac
        assert abs(_tod_frac(30.0) - 0.22) < 1e-9

    def test_monotonically_increasing(self):
        from src.engine import _tod_frac
        fracs = [_tod_frac(float(m)) for m in range(1, 391, 10)]
        assert all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))

    def test_always_positive(self):
        from src.engine import _tod_frac
        for m in [1, 5, 15, 30, 60, 120, 200, 300, 390]:
            assert _tod_frac(float(m)) > 0

    def test_early_session_much_higher_than_linear(self):
        from src.engine import _tod_frac
        linear_30 = 30 / 390
        assert _tod_frac(30.0) >= 2 * linear_30

    def test_interpolation_between_anchors(self):
        from src.engine import _tod_frac
        frac_30 = _tod_frac(30.0)
        frac_60 = _tod_frac(60.0)
        frac_45 = _tod_frac(45.0)
        assert frac_30 < frac_45 < frac_60

    def test_clamped_at_zero(self):
        from src.engine import _tod_frac
        assert _tod_frac(0.5) >= 0.01
        assert _tod_frac(1.0) >= 0.01


# ── pending_exit blocks duplicate sell ───────────────────────────────────────
class TestPendingExitBlocksDuplicateSell:
    """check_velocity_exits must skip pending_exit positions to prevent duplicate sells."""

    def test_pending_exit_blocks_duplicate_sell_on_next_cycle(self):
        engine = _make_engine()
        old_time = (datetime.now(_TZ_NY) - timedelta(days=14)).isoformat()
        engine.state = {'AAPL': {
            'price': 100.0,
            'time': old_time,
            'pending_exit': True,
        }}

        with patch.object(engine, '_fetch_snapshot', return_value=_make_snapshot(101.0)), \
             patch.object(engine, 'liquidate') as mock_liq, \
             patch.object(engine, 'save_state'):
            engine.check_velocity_exits()

        mock_liq.assert_not_called()


# ── Session 19 fixes ──────────────────────────────────────────────────────────

class TestBarCacheClearedOnDayRollover:
    """_bar_cache must be cleared when the trading date rolls over."""

    def _run_cycle_at(self, engine, fake_now):
        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(1400.0, 1400.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=[]), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity       = 1400.0
            engine._equity_initialized = True
            engine._day_start_equity  = 1400.0
            engine.run_cycle()

    def test_bar_cache_cleared_on_new_day(self):
        engine = _make_engine()
        engine._bar_cache      = {'AAPL': {'bars': [1, 2, 3]}, 'MSFT': {'bars': [4, 5]}}
        engine._day_start_date = '2024-06-04'
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        self._run_cycle_at(engine, fake_now)

        assert engine._bar_cache == {}

    def test_bar_cache_not_cleared_same_day(self):
        engine = _make_engine()
        engine._bar_cache      = {'AAPL': {'bars': [1, 2, 3]}}
        engine._day_start_date = '2024-06-05'
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        self._run_cycle_at(engine, fake_now)

        assert 'AAPL' in engine._bar_cache


class TestWriteDashboardDataNoDeadCommKey:
    """_write_dashboard_data must not compute or use a 'commission' variable."""

    def test_commission_key_absent_from_output(self, tmp_path, monkeypatch):
        import json as _json
        monkeypatch.setattr('src.engine.DASHBOARD_FILE', str(tmp_path / 'dashboard.json'))
        monkeypatch.setattr('src.engine.EQUITY_HIST_FILE', str(tmp_path / 'equity.json'))

        engine = _make_engine()
        engine._last_equity        = 1000.0
        engine._last_settled_cash  = 900.0
        engine._last_vix           = 18.0
        engine._last_scan_ts       = None
        engine._next_scan_dt       = None
        engine._equity_initialized = False
        engine.state = {
            'AAPL': {
                'fill_price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                'stop_dist': 10.0, 'peak_price': 155.0,
                'time': '2024-06-05T10:00:00', 'score': 75.0,
            }
        }

        with patch('src.engine.ALPACA_PAPER', True):
            engine._write_dashboard_data(connected=True)

        data = _json.loads((tmp_path / 'dashboard.json').read_text())
        pos = data['positions'][0]
        assert 'commission' not in pos
        assert pos['entry_price'] == 150.0
        assert pos['unit_price'] == 150.0

    def test_writes_last_updated_not_ts(self, tmp_path, monkeypatch):
        """dashboard_data.json must use 'last_updated', not 'ts', to match get_state() read."""
        import json as _json
        monkeypatch.setattr('src.engine.DASHBOARD_FILE', str(tmp_path / 'dashboard.json'))
        monkeypatch.setattr('src.engine.EQUITY_HIST_FILE', str(tmp_path / 'equity.json'))

        engine = _make_engine()
        engine._last_equity        = 1000.0
        engine._last_settled_cash  = 900.0
        engine._last_vix           = 18.0
        engine._last_scan_ts       = None
        engine._next_scan_dt       = None
        engine._equity_initialized = False
        engine.state = {}

        with patch('src.engine.ALPACA_PAPER', True):
            engine._write_dashboard_data(connected=True)

        data = _json.loads((tmp_path / 'dashboard.json').read_text())
        assert 'last_updated' in data, "dashboard_data.json must have 'last_updated' key"
        assert 'ts' not in data, "dashboard_data.json must not have stale 'ts' key"


class TestLogStartupSummaryUsesFillPrice:
    """_log_startup_summary must prefer fill_price over price for entry cost logging."""

    def test_uses_fill_price_when_present(self, caplog):
        engine = _make_engine()
        engine.state = {
            'AAPL': {
                'fill_price': 155.0,
                'price':      150.0,  # stale pending price — should be ignored
                'qty':        10.0,
                'stop_loss':  140.0,
                'current_price': 157.0,
            }
        }
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(equity=5000.0)

        # cost line should be based on fill_price (155), not pending price (150)
        assert '155.00' in caplog.text

    def test_falls_back_to_price_when_no_fill_price(self, caplog):
        engine = _make_engine()
        engine.state = {
            'MSFT': {
                'price':      200.0,
                'qty':        5.0,
                'stop_loss':  185.0,
                'current_price': 202.0,
            }
        }
        with caplog.at_level(logging.INFO, logger='VelocityEngine'):
            engine._log_startup_summary(equity=5000.0)

        assert '200.00' in caplog.text


class TestExitRecheckBetweenEntries:
    """check_velocity_exits must be called again before each second+ entry attempt."""

    def test_check_velocity_exits_called_between_signals(self):
        """With 2 signals and the first filling, exits are re-checked before the second."""
        engine = _make_engine()
        engine.state = {}

        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        signals_ctx = {
            'live_price': 100.0, 'close': 99.5, 'ask': 100.1,
            'smma_fast': 105.0, 'smma_med': 102.0, 'smma_slow': 100.0,
            'alligator_crossed': True,
            'rsi': 52.0, 'rsi_prev': 45.0,
            'rsi_history': [28.0, 30.0, 32.0, 35.0, 42.0],
            'intraday_open': 99.0, 'intraday_high': 100.5, 'intraday_low': 97.0,
            'atr': 2.0, 'atr_chandelier': 2.0,
            'rvol': 3.0, 'spread_pct': 0.002,
            'dollar_vol_20d': 300_000_000, 'avg_20d_vol': 5_000_000, 'volume': 5_000_000,
            'price_fetched_at': fake_now,
        }

        mock_filled = MagicMock(
            status='filled',
            filled_avg_price='110.0',
            filled_qty='5.0',
            id='order-123',
        )
        mock_order = MagicMock(id='order-123')

        exit_call_count = []

        def fake_check_exits():
            exit_call_count.append(1)
            return {}

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(5000.0, 5000.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', side_effect=fake_check_exits), \
             patch.object(engine, '_audit_stop_orders'), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=['AAPL', 'MSFT']), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch.object(engine, 'get_technical_context', return_value=signals_ctx), \
             patch.object(engine.trading_client, 'submit_order', return_value=mock_order), \
             patch.object(engine.trading_client, 'get_order_by_id', return_value=mock_filled), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine._last_equity       = 5000.0
            engine._equity_initialized = True
            engine._day_start_date    = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity  = 5000.0
            engine.run_cycle()

        # First call: step 5 of run_cycle before the entry loop.
        # Second call: inside the loop after placed=1, before the second signal.
        assert len(exit_call_count) >= 2, (
            f"check_velocity_exits called {len(exit_call_count)} time(s); "
            "expected ≥2 (pre-loop + between entries)"
        )


# ── Bug 81: Live reprice minimum price check ──────────────────────────────────

# ── Bug 82: Immediate stop audit for unprotected positions ─────────────────────

class TestStopAuditUnprotectedPositions:
    """_audit_stop_orders must fire immediately when a confirmed position has
    stop_dist <= 0, not only on the daily date-change trigger.

    Scenario: engine already ran today's audit (same date). A new position is
    added mid-day with stop_dist=0 (e.g., stop placement failed silently).
    On the next run_cycle(), the audit must fire again to protect it.
    """

    def _run_cycle_with_state(self, engine, fake_now, state):
        engine.state = state
        audit_calls = []

        def capturing_audit():
            audit_calls.append(1)

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(5000.0, 5000.0)), \
             patch.object(engine, '_fetch_vix', return_value=20.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders', side_effect=capturing_audit), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=[]), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity        = 5000.0
            engine._equity_initialized = True
            engine._day_start_date     = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity   = 5000.0
            engine.run_cycle()

        return audit_calls

    def test_audit_fires_when_position_has_zero_stop_dist(self):
        """Audit must run mid-day when a confirmed position has stop_dist=0."""
        engine = _make_engine()
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        # Mark today's audit as already done
        engine._last_audit_date = fake_now.strftime('%Y-%m-%d')

        # Confirmed position (not pending) with stop_dist=0 — no trail protection
        state = {
            'AAPL': {
                'fill_price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                'stop_dist': 0.0,  # unprotected — audit must re-run
                'peak_price': 152.0, 'time': '2024-06-05T10:00:00', 'score': 75.0,
            }
        }
        audit_calls = self._run_cycle_with_state(engine, fake_now, state)

        assert len(audit_calls) == 1, (
            "audit_stop_orders must be called when any confirmed position has stop_dist <= 0, "
            "even if today's date-based audit already ran"
        )

    def test_audit_does_not_fire_twice_when_protected(self):
        """Audit must NOT fire a second time on same day when all positions are protected."""
        engine = _make_engine()
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        # Mark today's audit as already done
        engine._last_audit_date = fake_now.strftime('%Y-%m-%d')

        # Confirmed position with a valid stop distance and confirmed stop order ID
        state = {
            'AAPL': {
                'fill_price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                'stop_dist': 10.0, 'stop_order_id': 'abc-123',  # fully protected
                'peak_price': 152.0, 'time': '2024-06-05T10:00:00', 'score': 75.0,
            }
        }
        audit_calls = self._run_cycle_with_state(engine, fake_now, state)

        assert len(audit_calls) == 0, (
            "audit_stop_orders must NOT be called mid-day when all positions already have "
            "a valid stop_dist (audit already ran today)"
        )

    def test_audit_skips_pending_positions_in_unprotected_check(self):
        """Pending-buy positions with stop_dist=0 must NOT trigger the unprotected audit.

        A pending position has no fill yet and therefore no stop order — this is
        expected and must not cause spurious audit calls.
        """
        engine = _make_engine()
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        engine._last_audit_date = fake_now.strftime('%Y-%m-%d')

        state = {
            'NVDA': {
                'price': 800.0, 'qty': 1, 'stop_dist': 0.0,
                'pending': True,  # not yet filled — stop_dist=0 is normal
                'time': '2024-06-05T10:55:00', 'score': 80.0,
            }
        }
        audit_calls = self._run_cycle_with_state(engine, fake_now, state)

        assert len(audit_calls) == 0, (
            "Pending-buy positions with stop_dist=0 must NOT trigger an unprotected-position "
            "audit — they have no fill yet so no stop order is expected"
        )

    def test_audit_fires_during_vix_high_early_return(self):
        """Stop audit must fire even when VIX is elevated and run_cycle() returns early.

        High-VIX early returns skip new entries — but positions already open still
        need their protective stops audited.  The audit runs at step 1.5 (before
        all account/VIX gates) so it cannot be bypassed by early exits.
        """
        engine = _make_engine()
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        # Today's audit has NOT yet run
        engine._last_audit_date = '2024-06-04'

        state = {
            'AAPL': {
                'fill_price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                'stop_dist': 0.0,  # unprotected
                'peak_price': 152.0, 'time': '2024-06-05T10:00:00', 'score': 75.0,
            }
        }
        engine.state = state
        audit_calls = []

        def capturing_audit():
            audit_calls.append(1)

        # VIX is 40 — above VIX_THRESHOLD — will cause early return before entry
        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', return_value=(5000.0, 5000.0)), \
             patch.object(engine, '_fetch_vix', return_value=40.0), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders', side_effect=capturing_audit), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch.object(engine, 'get_institutional_scan', return_value=[]), \
             patch.object(engine, '_fetch_spy_trend', return_value=_SPY_BULL_REGIME), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity        = 5000.0
            engine._equity_initialized = True
            engine._day_start_date     = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity   = 5000.0
            engine.run_cycle()

        assert len(audit_calls) == 1, (
            "Stop audit must fire even when VIX is high and run_cycle() returns early — "
            "unprotected positions need stops placed regardless of entry gates"
        )

    def test_audit_fires_when_account_values_fail(self):
        """Stop audit must fire even when _get_account_values() raises.

        When the Alpaca API is temporarily down, run_cycle() returns early after the
        account-values exception.  The stop audit runs at step 1.5 (before that
        exception path) so it still protects open positions.
        """
        engine = _make_engine()
        fake_now = _TZ_NY.localize(datetime(2024, 6, 5, 11, 0))

        # Today's audit has NOT yet run
        engine._last_audit_date = '2024-06-04'

        state = {
            'AAPL': {
                'fill_price': 150.0, 'qty': 10, 'stop_loss': 140.0,
                'stop_dist': 0.0,  # unprotected
                'peak_price': 152.0, 'time': '2024-06-05T10:00:00', 'score': 75.0,
            }
        }
        engine.state = state
        audit_calls = []

        def capturing_audit():
            audit_calls.append(1)

        with patch.object(engine, '_ensure_connected', return_value=True), \
             patch.object(engine, '_sync_positions'), \
             patch.object(engine, '_get_account_values', side_effect=RuntimeError("API down")), \
             patch.object(engine, 'check_velocity_exits', return_value={}), \
             patch.object(engine, '_audit_stop_orders', side_effect=capturing_audit), \
             patch.object(engine, '_update_position_prices'), \
             patch.object(engine, '_write_dashboard_data'), \
             patch.object(engine, 'save_state'), \
             patch('src.engine.datetime') as mock_dt, \
             patch('time.sleep'):
            mock_dt.now.return_value  = fake_now
            mock_dt.fromisoformat     = datetime.fromisoformat
            engine._last_equity        = 5000.0
            engine._equity_initialized = True
            engine._day_start_date     = fake_now.strftime('%Y-%m-%d')
            engine._day_start_equity   = 5000.0
            engine.run_cycle()

        assert len(audit_calls) == 1, (
            "Stop audit must fire even when _get_account_values() raises — "
            "Alpaca API outages must not leave open positions without stop protection"
        )


# ── Bug 84: Audit must restore stop_dist when confirming an existing TRAIL ────
class TestAuditRestoresStopDist:
    """_audit_stop_orders must update stop_dist in state when it finds a valid
    trailing stop for a re-synced position that has stop_dist=0.

    Without this fix, after a crash restart _sync_positions adds the position
    without stop_dist, the audit finds the TRAIL but ignores it, and:
    - _has_unprotected fires every cycle (audit API calls burn rate-limit quota)
    - _update_position_prices never updates stop_loss (skips when stop_dist=0)
    - Dashboard shows stop_loss=0.0 permanently
    - Break-even floor logic never activates
    """

    def _make_engine_with_trail_order(self, trail_dist=5.0, fill_price=100.0):
        """Return engine + a position that was re-synced (no stop_dist) + a mock TRAIL order."""
        engine = _make_engine()

        # Position as it arrives from _sync_positions after a crash restart
        engine.state['AAPL'] = {
            'fill_price': fill_price,
            'price':      fill_price,
            'peak_price': fill_price,
            'qty':        10.0,
            'stop_loss':  0.0,
            'stop_dist':  0.0,   # missing after re-sync
            'time':       '2024-06-05T10:30:00',
        }

        # Mock a TRAIL SELL order already at Alpaca
        trail_order = MagicMock()
        trail_order.symbol      = 'AAPL'
        trail_order.side        = 'sell'
        trail_order.order_type  = 'trailing_stop'
        trail_order.id          = 'trail-order-id'
        trail_order.trail_price = str(trail_dist)

        engine.trading_client.get_orders.return_value = [trail_order]
        return engine

    def test_audit_restores_stop_dist_from_existing_trail(self):
        """When the audit finds an existing TRAIL for a re-synced position with
        stop_dist=0, it must restore stop_dist and compute stop_loss correctly."""
        engine = self._make_engine_with_trail_order(trail_dist=5.0, fill_price=100.0)

        _safe_now = _TZ_NY.localize(datetime(2024, 6, 5, 10, 30))
        with patch('src.engine.datetime') as mock_dt:
            mock_dt.now.return_value = _safe_now
            mock_dt.fromisoformat    = datetime.fromisoformat
            engine._audit_stop_orders()

        assert engine.state['AAPL']['stop_dist'] == 5.0, (
            "stop_dist must be restored from the confirmed TRAIL's trail_price"
        )
        assert engine.state['AAPL']['stop_loss'] == 95.0, (
            "stop_loss must equal fill_price - trail_dist = 100.0 - 5.0 = 95.0"
        )

    def test_audit_does_not_overwrite_valid_stop_dist(self):
        """When stop_dist is already set (normal entry path), the audit must not
        overwrite it — only re-synced positions with stop_dist=0 need restoration."""
        engine = _make_engine()
        engine.state['AAPL'] = {
            'fill_price': 100.0,
            'price':      100.0,
            'qty':        10.0,
            'stop_loss':  92.0,
            'stop_dist':  8.0,   # already set from original entry
            'time':       '2024-06-05T10:30:00',
        }

        trail_order = MagicMock()
        trail_order.symbol      = 'AAPL'
        trail_order.side        = 'sell'
        trail_order.order_type  = 'trailing_stop'
        trail_order.id          = 'trail-order-id'
        trail_order.trail_price = '6.0'   # different value — must NOT be applied

        engine.trading_client.get_orders.return_value = [trail_order]
        engine._audit_stop_orders()

        assert engine.state['AAPL']['stop_dist'] == 8.0, (
            "stop_dist must remain at the original entry value when already > 0"
        )
        assert engine.state['AAPL']['stop_loss'] == 92.0, (
            "stop_loss must remain unchanged when stop_dist was already set"
        )

    def test_audit_restore_in_source(self):
        """Regression guard: the restore logic must be present in engine source."""
        import inspect
        from src.engine import VelocityEngine
        src = inspect.getsource(VelocityEngine._audit_stop_orders)
        assert "stop_dist', 0)) <= 0" in src, (
            "Audit must check 'stop_dist <= 0' before restoring from confirmed TRAIL"
        )
        assert "fill_price') or pos_data.get('price'" in src, (
            "Audit must use fill_price (not just price) for stop_loss calculation"
        )
