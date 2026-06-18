import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import (
    BASE_DIR, STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, LOG_DIR, LOG_FILE,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, ALPACA_DATA_FEED, FMP_API_KEY,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, BUCKET_CASH_PCT,
    VIX_THRESHOLD,
    ENTRY_START, ENTRY_END, EXIT_START, EXIT_END, VOL_MULT_FRIDAY, PRE_ENTRY_SYNC_TIME,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
    DAILY_HISTORY_DAYS,
    SCAN_MIN_DOLLAR_VOL,
    SCAN_INTERVAL, ERROR_WAIT,
    LOG_BACKUP_COUNT,
    EQUITY_RETRY_INTERVAL,
    EQUITY_HIST_INTERVAL,
    TICKER_BLOCKLIST,
    MAX_DAILY_LOSS_PCT,
    HARD_STOP_PCT,
    BREAK_EVEN_PCT,
    FRIDAY_CLOSE_HOUR,
    MIN_CANDLES, RVOL_MIN, SPREAD_MAX_PCT,
    CORR_MAX, CORR_LOOKBACK, MAX_SECTOR_COUNT, SMA200_SLOPE_LOOKBACK,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    LIMIT_BUF_MIN_PCT, LIMIT_BUF_MAX_PCT,
    RISK_PER_TRADE_PCT,
    SCAN_MIN_SCORE,
    CONCENTRATION_WARN_PCT, CONCENTRATION_HALT_PCT,
    REPRICE_DRIFT_MAX_PCT,
    ALLIGATOR_FAST, ALLIGATOR_MED, ALLIGATOR_SLOW,
    ALLIGATOR_FAST_OFFSET, ALLIGATOR_MED_OFFSET, ALLIGATOR_SLOW_OFFSET,
    ALLIGATOR_CROSS_LOOKBACK,
    SPY_EMA_PERIOD, SPY_REGIME_SIZE_CUT, SPY_REGIME_RVOL_MULT, SPY_FILTER_ENABLED,
    TIER_EXIT_R_MULTIPLES, TIER_EXIT_PCT,
)
from src.indicators import apply_all, compute_ma
from src.rules import (
    PERMANENT_DAY_RULES, CYCLE_RULES, check_rules, score_candidate,
)
from src.scanner import get_candidates, get_alligator_crossover_scan

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

_TZ_NY = pytz.timezone('US/Eastern')
_REAL_DT = datetime  # captured before any test can patch `datetime` in this module


# ── Logging ───────────────────────────────────────────────────────────────────
class _EasternFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = _REAL_DT.fromtimestamp(record.created, _TZ_NY)
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')


def _log_namer(default_name: str) -> str:
    if '.log.' in default_name:
        base, date_suffix = default_name.rsplit('.log.', 1)
        return f"{base}_{date_suffix}.log"
    return default_name


logger = logging.getLogger('BounceAlpha')
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=LOG_BACKUP_COUNT)
    _handler.namer = _log_namer
    _handler.setFormatter(_EasternFormatter('%(asctime)s | %(levelname)s | %(message)s'))
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(_EasternFormatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(_handler)
    logger.addHandler(_console)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tod_frac(elapsed_min: float) -> float:
    """Cumulative fraction of daily volume expected by elapsed_min.

    First 30 min = exactly 22% of daily volume; by session close (390 min) = 100%.
    Piecewise-linear approximation fitted from NYSE/NASDAQ composite profiles (2018-2024).
    """
    if elapsed_min <= 30.0:
        return max(0.01, elapsed_min / 30.0 * 0.22)
    return 0.22 + (elapsed_min - 30.0) / 360.0 * 0.78


def _count_trading_days(entry_dt: datetime, now: datetime) -> int:
    """Count complete Mon-Fri trading sessions elapsed between entry_dt and now."""
    entry_date = entry_dt.date()
    now_date   = now.date()
    count      = 0
    cursor     = entry_date
    while cursor < now_date:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


# ── Engine ────────────────────────────────────────────────────────────────────
class VelocityEngine:
    def __init__(self):
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
        # ScreenerClient provides top-gainers (market movers) endpoint
        self.screener_client = ScreenerClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
        self.state = self.load_state()

        self._last_equity:       float           = 0.0
        self._last_settled_cash: float           = 0.0
        self._equity_initialized: bool           = False
        self._last_vix:          Optional[float] = None
        self._vix_cache_date:    Optional[str]   = None  # date VIX was last fetched
        self._last_scan_ts:      Optional[str]   = None
        self._next_scan_dt:      Optional[str]   = None

        # Daily loss circuit breaker
        self._day_start_equity: Optional[float] = None
        self._day_start_date:   Optional[str]   = None

        # Caches — all keyed by symbol
        self._bar_cache:    Dict[str, dict] = {}   # daily bars, date-scoped
        self._spy_cache:    dict            = {}   # SPY trend, date-scoped
        self._sector_cache: Dict[str, str]  = {}   # stable, never invalidated
        self._analyst_cache: Dict[str, dict] = {}  # analyst ratings, session-scoped

        # Per-day skip lists to avoid re-running expensive lookups each cycle
        self._daily_scan_skip:          Dict[str, str] = {}
        self._insufficient_history_skip: set           = set()

        # Tracks consecutive Alpaca snapshot misses before removing state
        self._missing_position_counts: Dict[str, int] = {}

        # Date of last stop-order audit so we only run it once per day
        self._last_audit_date: Optional[str] = None

        # Portfolio P&L cache (refreshed every 30 min via Alpaca API)
        self._pnl_cache:    Optional[dict] = None
        self._pnl_cache_ts: Optional[datetime] = None

    # ── Connectivity ──────────────────────────────────────────────────────────
    def connect(self):
        """Validate Alpaca credentials and log account mode."""
        try:
            account = self.trading_client.get_account()
            logger.info(
                f"ENGINE CONNECTED: Alpaca paper={ALPACA_PAPER} | "
                f"account={account.id} status={account.status} | "
                f"portfolio=${float(account.portfolio_value):.2f}"
            )
            self._write_dashboard_data(connected=True)
        except Exception as e:
            logger.critical(f"CONNECTION FAILED: {e}")
            sys.exit(1)

    def _ensure_connected(self) -> bool:
        """Alpaca REST is stateless — just ping the account endpoint."""
        try:
            self.trading_client.get_account()
            return True
        except Exception as e:
            logger.warning(f"ENGINE: connectivity check failed: {e}")
            return False

    # ── State persistence ─────────────────────────────────────────────────────
    def load_state(self) -> dict:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"STATE: load failed ({e}), starting empty")
            return {}

    def _restore_blocked_today(self) -> None:
        """Reload permanent per-day filter failures from persisted state."""
        today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        for sym, data in self.state.items():
            skip_reason = data.get('_daily_skip_reason', '')
            skip_date   = data.get('_daily_skip_date', '')
            if skip_reason and skip_date == today_str:
                self._daily_scan_skip[sym] = skip_reason

    def save_state(self):
        try:
            tmp = STATE_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"STATE: save failed: {e}")

    # ── Portfolio P&L ─────────────────────────────────────────────────────────
    def _fetch_portfolio_pnl(self) -> dict:
        """Fetch P&L from Alpaca portfolio history API. Cached 30 min."""
        now = datetime.now(_TZ_NY)
        if (self._pnl_cache is not None and self._pnl_cache_ts is not None
                and (now - self._pnl_cache_ts).total_seconds() < 1800):
            return self._pnl_cache

        equity = self._last_equity
        result = {}
        for label, period, tf in [
            ('daily',   '1D', '1H'),
            ('weekly',  '1W', '1D'),
            ('monthly', '1M', '1D'),
            ('overall', '1A', '1D'),
        ]:
            try:
                h = self.trading_client.get_portfolio_history(
                    GetPortfolioHistoryRequest(
                        period=period, timeframe=tf,
                        extended_hours=(tf == '1H'),
                    )
                )
                base = float(h.base_value) if h.base_value else 0.0
                if base > 0 and equity > 0:
                    amount = round(equity - base, 2)
                    pct    = round(amount / base * 100, 2)
                    result[label] = {'amount': amount, 'pct': pct}
                else:
                    result[label] = {'amount': None, 'pct': None}
            except Exception as e:
                logger.debug(f"PNL: {label} fetch failed: {e}")
                result[label] = {'amount': None, 'pct': None}

        self._pnl_cache    = result
        self._pnl_cache_ts = now
        return result

    # ── Dashboard ─────────────────────────────────────────────────────────────
    def _write_dashboard_data(self, connected: bool = True):
        now_ny = datetime.now(_TZ_NY)
        positions = []
        for sym, d in self.state.items():
            ep      = float(d.get('fill_price') or d.get('price', 0))
            qty     = float(d.get('qty', 0))
            cur     = float(d.get('current_price', ep))
            stop    = float(d.get('stop_loss', 0))
            sd      = float(d.get('stop_dist', 0))
            peak    = float(d.get('peak_price', ep))
            eff_stop = max(stop, ep) if peak >= ep * (1 + BREAK_EVEN_PCT) else stop
            unit_price = ep if ep else None
            positions.append({
                'symbol':        sym,
                'entry_price':   ep,
                'unit_price':    unit_price,
                'current_price': cur,
                'qty':           qty,
                'stop_loss':     eff_stop,
                'stop_dist':     sd,
                'unrealized_pnl':     d.get('unrealized_pnl', 0),
                'unrealized_pnl_pct': d.get('unrealized_pnl_pct', 0),
                'score':         d.get('score'),
                'entry_time':    d.get('time', ''),
                'pending':       d.get('pending', False),
                'pending_exit':  d.get('pending_exit', False),
            })
        pnl = self._fetch_portfolio_pnl() if self._equity_initialized else {
            'daily': {'amount': None, 'pct': None},
            'weekly': {'amount': None, 'pct': None},
            'monthly': {'amount': None, 'pct': None},
            'overall': {'amount': None, 'pct': None},
        }
        data = {
            'connected':    connected,
            'equity':       self._last_equity,
            'settled_cash': self._last_settled_cash,
            'vix':          self._last_vix,
            'positions':    positions,
            'pnl':          pnl,
            'last_scan':    self._last_scan_ts,
            'next_scan':    self._next_scan_dt,
            'alpaca_paper': ALPACA_PAPER,
            'last_updated': now_ny.isoformat(),
        }
        try:
            tmp = DASHBOARD_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, DASHBOARD_FILE)
        except Exception as e:
            logger.warning(f"DASHBOARD: write failed: {e}")

        # Equity history — sampled at EQUITY_HIST_INTERVAL to keep file small
        if self._equity_initialized and self._last_equity > 0:
            try:
                hist = []
                if os.path.exists(EQUITY_HIST_FILE):
                    with open(EQUITY_HIST_FILE, 'r') as f:
                        hist = json.load(f)
                entry = {'ts': now_ny.isoformat(), 'equity': round(self._last_equity, 2)}
                if not hist or (
                    now_ny - datetime.fromisoformat(hist[-1]['ts'])
                ).total_seconds() >= EQUITY_HIST_INTERVAL:
                    hist.append(entry)
                    tmp = EQUITY_HIST_FILE + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump(hist, f)
                    os.replace(tmp, EQUITY_HIST_FILE)
            except Exception:
                pass

    # ── Account values ────────────────────────────────────────────────────────
    def _get_account_values(self) -> Tuple[float, float]:
        """Return (portfolio_value, settled_cash) with retries and last-known fallback."""
        _MAX_ATTEMPTS = 3
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                account = self.trading_client.get_account()
                equity  = float(account.portfolio_value)
                # Alpaca cash account: 'cash' is settled cash available for new purchases.
                # non_marginable_buying_power is safer for cash accounts but 'cash'
                # is the correct T+1 settled value.
                cash = float(account.cash)
                if equity > 0:
                    if cash <= 0:
                        logger.warning(
                            f"ACCOUNT: cash=${cash:.2f} ≤ 0 — "
                            "no settled cash for new entries (T+1 settlement)."
                        )
                    return equity, max(cash, 0.0)
                logger.warning(
                    f"ACCOUNT: attempt {attempt}/{_MAX_ATTEMPTS} — "
                    f"portfolio_value={equity:.2f} ≤ 0"
                )
            except Exception as e:
                logger.warning(
                    f"ACCOUNT: attempt {attempt}/{_MAX_ATTEMPTS} failed: {e}"
                )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2)

        if self._last_equity > 0:
            logger.warning(
                f"ACCOUNT: all attempts failed — using last known "
                f"equity=${self._last_equity:.2f}, settled=${self._last_settled_cash:.2f}"
            )
            return self._last_equity, self._last_settled_cash

        logger.critical("ACCOUNT: all attempts failed at startup with no fallback. Exiting.")
        self.shutdown()
        sys.exit(1)

    def _fetch_equity_with_retry(self) -> float:
        """Poll until Alpaca returns a positive portfolio value. Never gives up."""
        attempt = 0
        while True:
            attempt += 1
            try:
                account = self.trading_client.get_account()
                val = float(account.portfolio_value)
                if val > 0:
                    logger.info(f"INIT: portfolio_value=${val:.2f} (attempt {attempt})")
                    return val
                logger.warning(
                    f"INIT: attempt {attempt}: portfolio_value={val:.2f} ≤ 0, "
                    f"retrying in {EQUITY_RETRY_INTERVAL}s..."
                )
            except Exception as e:
                logger.warning(
                    f"INIT: attempt {attempt} failed: {e}, "
                    f"retrying in {EQUITY_RETRY_INTERVAL}s..."
                )
            time.sleep(EQUITY_RETRY_INTERVAL)

    # ── Startup ───────────────────────────────────────────────────────────────
    def _log_startup_summary(self, equity: float):
        if not self.state:
            logger.info("INIT: No open positions. Full capital available.")
            logger.info(
                f"INIT READY | Equity=${equity:.2f} | "
                f"Positions=0/{self._calc_max_positions(equity)}"
            )
            return
        total_cost = 0.0
        logger.info("INIT: ── Open Positions ──────────────────────────────────")
        for sym, d in self.state.items():
            ep   = float(d.get('fill_price') or d.get('price', 0))
            qty  = float(d.get('qty', 0))
            cur  = float(d.get('current_price', ep))
            sl   = float(d.get('stop_loss', 0))
            unr  = float(d.get('unrealized_pnl', (cur - ep) * qty))
            unr_pct = float(d.get('unrealized_pnl_pct', (cur - ep) / ep * 100 if ep else 0))
            cost = ep * qty
            total_cost += cost
            logger.info(
                f"INIT:  {sym:6s} | entry=${ep:.2f} cur=${cur:.2f} qty={qty:.4g} "
                f"cost=${cost:.2f} | unreal={unr:+.2f} ({unr_pct:+.1f}%) | SL=${sl:.2f}"
            )
        logger.info(
            f"INIT: Equity=${equity:.2f} | Deployed≈${total_cost:.2f} | "
            f"Positions={len(self.state)}/{self._calc_max_positions(equity)}"
        )

    def _initialize(self):
        """
        Phase 1 (immediate): fetch equity, sync positions, cancel orphaned buy orders.
        Phase 2 (timed):     sleep until PRE_ENTRY_SYNC_TIME, then re-sync + audit stops.
        """
        mode = "PAPER" if ALPACA_PAPER else "LIVE"
        logger.info(
            f"MARKET DATA: Alpaca {mode} trading. "
            f"Data feed={ALPACA_DATA_FEED.upper()}. "
            "Real-time bid/ask spread filtering is active."
        )
        logger.info("INIT: Fetching account equity from Alpaca...")
        equity = self._fetch_equity_with_retry()
        self._last_equity       = equity
        self._last_settled_cash = equity
        self._equity_initialized = True

        logger.info("INIT: Phase 1 — syncing positions...")
        self._sync_positions()
        self._restore_blocked_today()

        # Cancel orphaned open BUY orders from the previous engine session
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            orphaned_buys = [
                o for o in open_orders
                if str(o.side) in ('OrderSide.BUY', 'buy')
                and o.symbol not in self.state
            ]
            if orphaned_buys:
                logger.info(
                    f"INIT: Cancelling {len(orphaned_buys)} orphaned BUY orders "
                    f"from previous session."
                )
                for o in orphaned_buys:
                    try:
                        self.trading_client.cancel_order_by_id(o.id)
                    except Exception:
                        pass
                time.sleep(1)
        except Exception as e:
            logger.warning(f"INIT: orphan cleanup failed: {e}")

        if self.state:
            self._update_position_prices()
        self._write_dashboard_data(connected=True)

        # ── Alligator universe crossover scan ─────────────────────────────────
        # Run eagerly on every restart so the cache is warm before the first
        # entry cycle.  Takes ~2 min for the full universe; runs in the
        # foreground here while the engine is not yet cycling.
        logger.info("INIT: Pre-loading Alligator universe crossover scan...")
        get_alligator_crossover_scan(self.data_client, self.trading_client)

        # ── Phase 2 ──────────────────────────────────────────────────────────
        self._wait_for_pre_entry_sync()

        logger.info("INIT: Phase 2 — pre-entry re-sync and stop-order audit...")
        self._sync_positions()
        if self.state:
            logger.info("INIT: Auditing stop orders for open positions...")
            self._audit_stop_orders()
            self._last_audit_date = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
            logger.info("INIT: Fetching live prices for open positions...")
            self._update_position_prices()

        self._log_startup_summary(equity)
        self._write_dashboard_data(connected=True)

    def _wait_for_pre_entry_sync(self):
        """Sleep until PRE_ENTRY_SYNC_TIME (09:58 ET) with 5-min heartbeat steps."""
        h, m   = PRE_ENTRY_SYNC_TIME
        now_ny = datetime.now(_TZ_NY)
        target = now_ny.replace(hour=h, minute=m, second=0, microsecond=0)

        if now_ny >= target:
            logger.info(
                f"INIT: Already at or past {h:02d}:{m:02d} ET — "
                "running pre-entry sync immediately."
            )
            return

        wait_s = (target - now_ny).total_seconds()
        logger.info(
            f"INIT: Waiting {wait_s/60:.1f} min until {h:02d}:{m:02d} ET "
            f"for pre-entry sync (entry window opens at "
            f"{ENTRY_START[0]:02d}:{ENTRY_START[1]:02d} ET)."
        )
        _HEARTBEAT = 300
        elapsed = 0.0
        while elapsed < wait_s:
            step = min(_HEARTBEAT, wait_s - elapsed)
            time.sleep(step)
            elapsed += step
            if elapsed < wait_s:
                self._sync_positions()
                if self.state:
                    self._update_position_prices()
                self._write_dashboard_data(connected=True)
                logger.info(
                    f"INIT: {(wait_s - elapsed)/60:.1f} min remaining until "
                    f"{h:02d}:{m:02d} ET sync."
                )

    # ── Stop-order audit ──────────────────────────────────────────────────────
    def _audit_stop_orders(self):
        """Ensure every open position has exactly one chandelier trailing-stop SELL."""
        if not self.state:
            return

        now_ny    = datetime.now(_TZ_NY)
        if (now_ny.hour, now_ny.minute) < EXIT_START:
            logger.debug(
                f"AUDIT: before {EXIT_START[0]:02d}:{EXIT_START[1]:02d} ET — "
                "trailing stop placement deferred until opening volatility clears."
            )
            return
        in_market = (
            now_ny.weekday() < 5
            and (9, 30) <= (now_ny.hour, now_ny.minute) < (16, 0)
        )
        if not in_market:
            logger.info(
                "AUDIT: market is currently closed — trailing stops will be GTC "
                "and activate at the next RTH open."
            )

        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception as e:
            logger.warning(f"AUDIT: failed to fetch open orders: {e}")
            return

        # Index SELL orders by symbol
        sell_by_sym: Dict[str, list] = {}
        for o in open_orders:
            if str(o.side) in ('OrderSide.SELL', 'sell'):
                sell_by_sym.setdefault(o.symbol, []).append(o)

        for sym, pos_data in list(self.state.items()):
            if pos_data.get('pending') or pos_data.get('pending_exit'):
                continue

            qty = float(pos_data.get('qty', 0))
            if qty <= 0:
                logger.warning(f"AUDIT: {sym} — qty={qty}, cannot place stop")
                continue

            sell_orders  = sell_by_sym.get(sym, [])
            trail_orders = [
                o for o in sell_orders
                if str(o.order_type) in ('OrderType.TRAILING_STOP', 'trailing_stop')
            ]
            non_trail = [
                o for o in sell_orders
                if str(o.order_type) not in ('OrderType.TRAILING_STOP', 'trailing_stop')
            ]

            for o in non_trail:
                logger.info(
                    f"AUDIT: {sym} — cancelling non-trailing-stop SELL "
                    f"(type={o.order_type} id={o.id})"
                )
                try:
                    self.trading_client.cancel_order_by_id(o.id)
                except Exception as e:
                    logger.warning(f"AUDIT: {sym} — cancel failed: {e}")
            if non_trail:
                time.sleep(1)

            if len(trail_orders) > 1:
                trail_orders.sort(key=lambda o: o.created_at)
                for dup in trail_orders[:-1]:
                    logger.warning(
                        f"AUDIT: {sym} — duplicate trailing stop (id={dup.id}); cancelling"
                    )
                    try:
                        self.trading_client.cancel_order_by_id(dup.id)
                    except Exception as e:
                        logger.warning(f"AUDIT: {sym} — dup cancel failed: {e}")
                trail_orders = [trail_orders[-1]]
                time.sleep(1)

            if trail_orders:
                kept = trail_orders[0]
                # Restore stop_dist from the live order if state is missing it
                # (happens after a crash restart where _sync_positions re-adds the
                # position without stop_dist).  Without this, _has_unprotected fires
                # every cycle and _update_position_prices skips the stop_loss update.
                trail_dist = float(kept.trail_price or 0)
                state_changed = False
                if trail_dist > 0 and float(pos_data.get('stop_dist', 0)) <= 0:
                    entry_px = float(pos_data.get('fill_price') or pos_data.get('price', 0))
                    self.state[sym]['stop_dist'] = trail_dist
                    self.state[sym]['stop_loss'] = round(entry_px - trail_dist, 2)
                    state_changed = True
                    logger.info(
                        f"AUDIT: {sym} — trailing stop confirmed; restored stop_dist "
                        f"(id={kept.id} trail_price=${trail_dist:.2f})"
                    )
                else:
                    logger.info(
                        f"AUDIT: {sym} — trailing stop confirmed "
                        f"(id={kept.id} trail_price=${kept.trail_price})"
                    )
                # Persist the confirmed order ID so has_unprotected doesn't re-fire.
                if self.state[sym].get('stop_order_id') != str(kept.id):
                    self.state[sym]['stop_order_id'] = str(kept.id)
                    state_changed = True
                if state_changed:
                    self.save_state()
                continue

            # No trailing stop found — fetch bars and place one
            logger.info(f"AUDIT: {sym} — no trailing stop found; placing chandelier stop...")
            try:
                df = self._fetch_daily_bars(sym)
                if df is None or len(df) < CHANDELIER_PERIOD:
                    logger.error(
                        f"AUDIT: {sym} — insufficient history; position UNPROTECTED."
                    )
                    continue

                df_ind = apply_all(df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
                                   SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD)
                atr_chandelier = float(df_ind['ATR_CHAND'].iloc[-1])
                if np.isnan(atr_chandelier) or atr_chandelier <= 0:
                    logger.error(
                        f"AUDIT: {sym} — ATR_CHAND invalid; position UNPROTECTED."
                    )
                    continue

                chandelier_dist = round(atr_chandelier * CHANDELIER_MULT, 2)
                # Alpaca rejects trail_price > 25% of stock price; use 24% buffer.
                entry_px_for_cap = float(pos_data.get('fill_price') or pos_data.get('price', 0))
                max_trail = round(entry_px_for_cap * 0.24, 2) if entry_px_for_cap > 0 else chandelier_dist
                trail_dist = min(chandelier_dist, max_trail)
                if trail_dist < chandelier_dist:
                    logger.info(
                        f"AUDIT: {sym} — chandelier dist ${chandelier_dist:.2f} capped "
                        f"to ${trail_dist:.2f} (Alpaca 25% limit on ${entry_px_for_cap:.2f})"
                    )
                stop_req = TrailingStopOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_price=trail_dist,
                )
                stop_order = self.trading_client.submit_order(stop_req)
                time.sleep(2)

                # Refresh to confirm it was accepted
                try:
                    confirmed = self.trading_client.get_order_by_id(stop_order.id)
                    status = str(confirmed.status)
                    if status in ('OrderStatus.CANCELED', 'canceled',
                                  'OrderStatus.REJECTED', 'rejected',
                                  'OrderStatus.EXPIRED', 'expired'):
                        logger.error(
                            f"AUDIT: {sym} — trailing stop rejected "
                            f"(status={status}); position UNPROTECTED."
                        )
                        self.state[sym]['stop_dist'] = 0  # triggers retry next cycle
                        self.save_state()
                        continue
                except Exception:
                    pass

                entry_px = float(pos_data.get('fill_price') or pos_data.get('price', 0))
                self.state[sym]['stop_dist'] = trail_dist
                self.state[sym]['stop_loss'] = round(entry_px - trail_dist, 2)
                self.state[sym]['stop_order_id'] = str(stop_order.id)
                self.save_state()
                logger.info(
                    f"AUDIT: {sym} — chandelier stop placed "
                    f"(trail_price=${trail_dist:.2f} id={stop_order.id})"
                )
            except Exception as e:
                logger.error(
                    f"AUDIT: {sym} — stop placement failed: {e}; position UNPROTECTED."
                )
                self.state[sym]['stop_dist'] = 0  # triggers has_unprotected on next cycle
                self.save_state()

    # ── Market data helpers ───────────────────────────────────────────────────
    def _fetch_daily_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch ~400 calendar days of daily OHLCV bars from Alpaca and return as DataFrame."""
        start = datetime.now(_TZ_NY) - timedelta(days=DAILY_HISTORY_DAYS)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                feed=ALPACA_DATA_FEED,
            )
            bars = self.data_client.get_stock_bars(req)
            # bars[symbol] returns List[Bar], not a BarSet — build DataFrame from the list
            df   = pd.DataFrame([b.model_dump() for b in bars[symbol]]).reset_index(drop=True)
            # Normalise column names to lowercase so apply_all() works
            df.columns = [c.lower() for c in df.columns]
            return df[['open', 'high', 'low', 'close', 'volume']]
        except KeyError:
            # Symbol has no bars in Alpaca (warrant, delisted, non-stock) — skip for the day
            self._insufficient_history_skip.add(symbol)
            return None
        except Exception as e:
            logger.warning(f"DATA: daily bars fetch failed for {symbol}: {e}")
            return None

    def _batch_fetch_snapshots(self, symbols: list) -> dict:
        """Fetch snapshots for all symbols in a single API call.

        Returns a dict of symbol → parsed snap dict (same shape as _fetch_snapshot).
        Symbols with no IEX coverage or no data are absent from the result.
        """
        if not symbols:
            return {}
        try:
            req   = StockSnapshotRequest(symbol_or_symbols=symbols, feed=ALPACA_DATA_FEED)
            snaps = self.data_client.get_stock_snapshot(req)
        except Exception as e:
            logger.warning(f"DATA: batch snapshot fetch failed: {e}")
            return {}
        result = {}
        for sym, snap in snaps.items():
            parsed = self._parse_snapshot(snap)
            if parsed is not None:
                result[sym] = parsed
        return result

    def _parse_snapshot(self, snap) -> Optional[dict]:
        """Convert a raw Alpaca Snapshot object into the standard snap dict."""
        if snap is None:
            return None
        live_price: float = 0.0
        if snap.latest_trade and snap.latest_trade.price:
            live_price = float(snap.latest_trade.price)
        elif snap.minute_bar and snap.minute_bar.close:
            live_price = float(snap.minute_bar.close)
        bid = ask = 0.0
        if snap.latest_quote:
            bid = float(snap.latest_quote.bid_price or 0)
            ask = float(snap.latest_quote.ask_price or 0)
        intraday_vol = intraday_open = intraday_high = intraday_low = 0.0
        if snap.daily_bar:
            intraday_vol  = float(snap.daily_bar.volume or 0)
            intraday_open = float(snap.daily_bar.open   or 0)
            intraday_high = float(snap.daily_bar.high   or 0)
            intraday_low  = float(snap.daily_bar.low    or 0)
        return {
            'live_price':    live_price,
            'bid':           bid,
            'ask':           ask,
            'intraday_vol':  intraday_vol,
            'intraday_open': intraday_open,
            'intraday_high': intraday_high,
            'intraday_low':  intraday_low,
        }

    def _fetch_snapshot(self, symbol: str) -> Optional[dict]:
        """Fetch latest quote + daily bar snapshot for live price, spread, and RVOL."""
        try:
            req  = StockSnapshotRequest(
                symbol_or_symbols=[symbol],
                feed=ALPACA_DATA_FEED,
            )
            snaps = self.data_client.get_stock_snapshot(req)
            return self._parse_snapshot(snaps.get(symbol))
        except Exception as e:
            logger.debug(f"DATA: snapshot fetch failed for {symbol}: {e}")
            return None

    def _fetch_vix(self) -> Optional[float]:
        """Return the latest VIX close via yfinance, cached for the current trading day.

        Caching strategy:
        - Re-fetches only once per calendar day (yfinance returns end-of-day close).
        - On network failure, returns the last successfully fetched value so a transient
          yfinance outage does not lock out new entries for the entire session.
        - Returns None only when no value has ever been successfully fetched (first-run
          failure at startup).  Callers must treat None as "filter unavailable, warn but
          do not halt entries."
        """
        today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        if self._last_vix is not None and self._vix_cache_date == today_str:
            return self._last_vix

        try:
            hist = yf.Ticker("^VIX").history(period="5d")
            if not hist.empty:
                val = float(hist['Close'].iloc[-1])
                self._last_vix       = val
                self._vix_cache_date = today_str
                return val
        except Exception as e:
            logger.debug(f"VIX: fetch failed: {e}")

        # Return stale cached value rather than blocking entries on network hiccup
        if self._last_vix is not None:
            logger.warning(
                f"VIX: fetch failed — using last known value {self._last_vix:.1f} "
                f"(cached {self._vix_cache_date})"
            )
            return self._last_vix

        return None

    # ── SPY regime ────────────────────────────────────────────────────────────
    def _fetch_spy_trend(self) -> dict:
        """Return SPY regime dict based on 50-day EMA.

        Returns:
            is_bull      bool   True when SPY close > EMA50
            spy_close    float  latest SPY closing price
            ema50        float  SPY 50-day EMA value
            size_factor  float  1.0 in bull; (1 - SPY_REGIME_SIZE_CUT) in bear
            rvol_mult    float  1.0 in bull; SPY_REGIME_RVOL_MULT in bear

        Bearish regime does NOT block entries — it reduces position size by
        SPY_REGIME_SIZE_CUT and tightens the RVOL threshold by SPY_REGIME_RVOL_MULT.
        This allows participation in mean-reversion bounces even during weak markets,
        but with smaller, more selective positions.
        """
        today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        cached = self._spy_cache
        if cached.get('date') == today_str and 'is_bull' in cached:
            return cached

        df = self._fetch_daily_bars('SPY')
        if df is None or len(df) < SPY_EMA_PERIOD:
            logger.warning("SPY: insufficient history for regime check; assuming bull")
            result = {
                'date': today_str, 'is_bull': True,
                'spy_close': 0.0, 'ema50': 0.0,
                'size_factor': 1.0, 'rvol_mult': 1.0,
            }
            self._spy_cache = result
            return result

        ema50_series = df['close'].ewm(span=SPY_EMA_PERIOD, adjust=False).mean()
        spy_close    = float(df['close'].iloc[-1])
        ema50_val    = float(ema50_series.iloc[-1])
        is_bull      = spy_close > ema50_val
        result = {
            'date':        today_str,
            'is_bull':     is_bull,
            'spy_close':   round(spy_close, 2),
            'ema50':       round(ema50_val, 2),
            'size_factor': 1.0 if is_bull else (1.0 - SPY_REGIME_SIZE_CUT),
            'rvol_mult':   1.0 if is_bull else SPY_REGIME_RVOL_MULT,
        }
        self._spy_cache = result
        return result

    # ── Sector lookup ─────────────────────────────────────────────────────────
    def _get_sector(self, symbol: str) -> str:
        """Return GICS sector for a symbol via yfinance, cached permanently.

        Alpaca's asset API only returns asset_class ('us_equity' for all equities),
        which gives zero diversification signal.  yfinance info['sector'] returns
        the actual GICS sector (Technology, Healthcare, Financials, …), making
        MAX_SECTOR_COUNT enforcement meaningful.
        """
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        try:
            info   = yf.Ticker(symbol).info
            sector = info.get('sector') or 'Unknown'
        except Exception:
            sector = 'Unknown'
        self._sector_cache[symbol] = sector
        return sector

    def _get_analyst_ratings(self, symbol: str) -> dict:
        """Return analyst buy/hold/sell counts via Financial Modeling Prep, session-cached.

        Uses /api/v3/analyst-stock-recommendations (most recent month).
        strongBuy+buy → analyst_buy; hold → analyst_hold; sell+strongSell → analyst_sell.
        Returns zeros when FMP_API_KEY is unset or on any network/parse error so
        scoring degrades gracefully (0 bonus, no penalty).
        """
        if symbol in self._analyst_cache:
            return self._analyst_cache[symbol]
        result = {'analyst_buy': 0, 'analyst_hold': 0, 'analyst_sell': 0}
        if not FMP_API_KEY:
            self._analyst_cache[symbol] = result
            return result
        try:
            url  = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{symbol}"
            resp = httpx.get(url, params={"limit": 1, "apikey": FMP_API_KEY}, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                row  = data[0]
                buy  = int(row.get('analystRatingsStrongBuy', 0)) + int(row.get('analystRatingsbuy', 0))
                hold = int(row.get('analystRatingsHold', 0))
                sell = int(row.get('analystRatingsSell', 0)) + int(row.get('analystRatingsStrongSell', 0))
                result = {'analyst_buy': buy, 'analyst_hold': hold, 'analyst_sell': sell}
        except Exception as exc:
            logger.warning(f"ANALYST {symbol}: FMP fetch failed — {type(exc).__name__}: {exc}")
        self._analyst_cache[symbol] = result
        return result

    # ── Correlation check ─────────────────────────────────────────────────────
    def _compute_book_correlation(self, sym: str, df_daily: pd.DataFrame) -> float:
        """Maximum pairwise return correlation between sym and any open position."""
        if not self.state:
            return 0.0
        lookback = CORR_LOOKBACK
        ret_sym  = df_daily['close'].pct_change().tail(lookback)
        max_corr = 0.0
        for book_sym in self.state:
            if book_sym == sym:
                continue
            cached = self._bar_cache.get(book_sym, {})
            df_b   = cached.get('bars_daily')
            if df_b is None:
                continue
            ret_b = df_b['close'].pct_change().tail(lookback)
            try:
                n = min(len(ret_sym), len(ret_b))
                if n < 20:
                    continue
                corr = float(np.corrcoef(ret_sym.iloc[-n:], ret_b.iloc[-n:])[0, 1])
                if np.isfinite(corr):
                    max_corr = max(max_corr, abs(corr))
            except Exception:
                pass
        return max_corr

    # ── Position sizing helpers ───────────────────────────────────────────────
    def _calc_max_positions(self, equity: float) -> int:
        if equity < MIN_BUCKET_SIZE:
            return 0
        return min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP)

    @staticmethod
    def _calc_cash_entry_slots(settled: float) -> int:
        return max(0, int(settled / MIN_BUCKET_SIZE))

    def _check_portfolio_concentration(
        self, equity: float
    ) -> Tuple[float, bool, bool]:
        """Return (deployed_pct, warn, halt).  Warn ≥85%, halt ≥95%."""
        if equity <= 0 or not self.state:
            return 0.0, False, False
        deployed = sum(
            float(d.get('current_price', d.get('price', 0))) * float(d.get('qty', 0))
            for d in self.state.values()
        )
        pct = deployed / equity
        return pct, pct >= CONCENTRATION_WARN_PCT, pct >= CONCENTRATION_HALT_PCT

    # ── Position sync ─────────────────────────────────────────────────────────
    def _get_fill_time(self, symbol: str) -> str:
        """Return the filled_at timestamp of the most recent filled BUY order for symbol.

        Falls back to now() if Alpaca order history is unavailable or empty.
        """
        try:
            orders = self.trading_client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    symbols=[symbol],
                    side=OrderSide.BUY,
                    limit=10,
                )
            )
            filled = [
                o for o in orders
                if str(o.status) in ('OrderStatus.FILLED', 'filled') and o.filled_at
            ]
            if filled:
                # Most recent fill first (Alpaca returns newest-first)
                fill_time = filled[0].filled_at
                if fill_time.tzinfo is None:
                    fill_time = _TZ_NY.localize(fill_time)
                return fill_time.isoformat()
        except Exception as e:
            logger.debug(f"SYNC: could not fetch fill time for {symbol}: {e}")
        return datetime.now(_TZ_NY).isoformat()

    def _sync_positions(self):
        """Reconcile self.state against actual Alpaca positions every cycle."""
        try:
            alpaca_pos = {p.symbol: p for p in self.trading_client.get_all_positions()}
        except Exception as e:
            logger.warning(f"SYNC: position fetch failed: {e}")
            return

        missing_counts = self._missing_position_counts
        changed = False

        # Add / update positions found at Alpaca
        for sym, pos in alpaca_pos.items():
            qty      = float(pos.qty)
            avg_cost = float(pos.avg_entry_price or 0)
            if qty <= 0:
                continue

            if sym not in self.state:
                fill_time     = self._get_fill_time(sym)
                recomputed_score = self._try_rescore(sym)
                self.state[sym] = {
                    'price':        round(avg_cost, 2),
                    'fill_price':   round(avg_cost, 2),
                    'peak_price':   round(avg_cost, 2),
                    'time':         fill_time,
                    'qty':          round(qty, 4),
                    'original_qty': round(qty, 4),
                    'tier_sold':    0,
                    'stop_loss':    0.0,
                    'volume':       0,
                    'score':        recomputed_score,
                }
                score_note = f"recomputed score={recomputed_score:.1f}" if recomputed_score is not None else "score=None (no bars)"
                logger.info(
                    f"SYNC: Added {sym} from Alpaca "
                    f"(qty={qty} avg_entry=${avg_cost:.2f} filled_at={fill_time}); {score_note}"
                )
                changed = True
            else:
                missing_counts.pop(sym, None)

                if self.state[sym].pop('pending_exit', None):
                    logger.warning(
                        f"SYNC: {sym} still present at Alpaca after pending exit; "
                        "clearing pending_exit and continuing risk management."
                    )
                    changed = True

                state_qty = float(self.state[sym].get('qty', 0))
                if abs(state_qty - qty) > 1e-6:
                    logger.info(
                        f"SYNC: {sym} qty updated state={state_qty:g} → Alpaca={qty:g}"
                    )
                    self.state[sym]['qty'] = round(qty, 4)
                    changed = True

                if avg_cost > 0 and float(self.state[sym].get('fill_price', 0)) <= 0:
                    self.state[sym]['fill_price'] = round(avg_cost, 2)
                    self.state[sym]['price']      = round(avg_cost, 2)
                    changed = True
                if avg_cost > 0 and float(self.state[sym].get('peak_price', 0)) <= 0:
                    self.state[sym]['peak_price'] = round(avg_cost, 2)
                    changed = True

                # Confirm pending BUY that Alpaca now shows as a filled position
                if self.state[sym].get('pending'):
                    del self.state[sym]['pending']
                    self.state[sym]['fill_price'] = round(avg_cost, 2)
                    self.state[sym]['price']      = round(avg_cost, 2)
                    changed = True
                    logger.info(f"SYNC: {sym} pending BUY confirmed filled at ${avg_cost:.2f}")

        # Remove state entries whose Alpaca position is gone
        for sym in list(self.state.keys()):
            if sym in alpaca_pos:
                continue
            if self.state[sym].get('pending'):
                # Limit BUY not yet filled — keep watching
                continue
            if self.state[sym].get('pending_exit'):
                # MKT SELL in flight — wait for Alpaca to confirm flat
                miss_count = missing_counts.get(sym, 0) + 1
                missing_counts[sym] = miss_count
                if miss_count < 2:
                    continue
                # Two consecutive snapshots without the position — assume filled
                self._cancel_orphaned_sell_orders(sym)
                del self.state[sym]
                missing_counts.pop(sym, None)
                changed = True
                logger.info(f"SYNC: {sym} confirmed flat after pending_exit")
                continue

            miss_count = missing_counts.get(sym, 0) + 1
            missing_counts[sym] = miss_count
            if miss_count < 2:
                logger.warning(
                    f"SYNC: {sym} missing from Alpaca positions (count={miss_count}); "
                    "deferring removal until second confirming snapshot."
                )
                continue

            logger.info(f"SYNC: {sym} removed from state — no Alpaca position found")
            self._cancel_orphaned_sell_orders(sym)
            del self.state[sym]
            missing_counts.pop(sym, None)
            changed = True

        self._missing_position_counts = missing_counts
        if changed:
            self.save_state()

    def _cancel_orphaned_sell_orders(self, symbol: str) -> int:
        """Cancel leftover SELL orders after Alpaca confirms the position is flat."""
        cancelled = 0
        try:
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception as e:
            logger.warning(f"SYNC: {symbol} — orphan SELL query failed: {e}")
            return 0

        for o in orders:
            if o.symbol != symbol:
                continue
            if str(o.side) not in ('OrderSide.SELL', 'sell'):
                continue
            try:
                self.trading_client.cancel_order_by_id(o.id)
                cancelled += 1
            except Exception as e:
                logger.warning(f"SYNC: {symbol} — orphan cancel failed: {e}")

        if cancelled:
            logger.info(f"SYNC: {symbol} — cancelled {cancelled} orphaned SELL orders")
            time.sleep(1)
        return cancelled

    # ── Price updates ─────────────────────────────────────────────────────────
    def _update_position_prices(
        self, prefetched: Optional[Dict[str, float]] = None
    ):
        """Fetch live price for every open position and persist unrealised P&L."""
        if not self.state:
            return
        changed = False
        for sym in list(self.state.keys()):
            cur: Optional[float] = None
            if prefetched and sym in prefetched:
                cur = prefetched[sym]
            else:
                snap = self._fetch_snapshot(sym)
                if snap:
                    cur = snap.get('live_price')

            if cur and cur > 0:
                self.state[sym]['current_price'] = round(cur, 2)
                self.state[sym]['price_checked_at'] = datetime.now(_TZ_NY).isoformat()

                ep  = float(self.state[sym].get('price', 0))
                qty = float(self.state[sym].get('qty', 0))
                if ep > 0 and qty > 0:
                    self.state[sym]['unrealized_pnl']     = round((cur - ep) * qty, 2)
                    self.state[sym]['unrealized_pnl_pct'] = round((cur - ep) / ep * 100, 2)

                # Update peak and dashboard effective stop
                sd   = float(self.state[sym].get('stop_dist', 0))
                peak = max(float(self.state[sym].get('peak_price', cur) or cur), cur)
                self.state[sym]['peak_price'] = round(peak, 2)
                if sd > 0:
                    effective_stop = round(peak - sd, 2)
                    if ep > 0 and peak >= ep * (1 + BREAK_EVEN_PCT):
                        effective_stop = max(effective_stop, ep)
                    self.state[sym]['stop_loss'] = effective_stop

                # Lazy-load analyst ratings for positions re-synced after restart
                if 'analyst_buy' not in self.state[sym]:
                    self.state[sym].update(self._get_analyst_ratings(sym))

                changed = True

        if changed:
            self.save_state()

    # ── Technical context ─────────────────────────────────────────────────────
    def get_technical_context(self, symbol: str, snap: Optional[dict] = None) -> Optional[dict]:
        """Fetch all indicator data needed for entry screening.

        snap: pre-fetched snapshot dict from _batch_fetch_snapshots; if None,
        falls back to an individual _fetch_snapshot API call.
        Returns None if data is unavailable or insufficient.
        """
        now_ny    = datetime.now(_TZ_NY)
        today_str = now_ny.strftime('%Y-%m-%d')

        # Bar cache — valid for one trading day
        cached = self._bar_cache.get(symbol)
        if cached and cached.get('date') == today_str:
            bars_daily = cached['bars_daily']
        else:
            bars_daily = self._fetch_daily_bars(symbol)
            if bars_daily is None:
                return None
            self._bar_cache[symbol] = {
                'date':       today_str,
                'bars_daily': bars_daily,
            }

        if len(bars_daily) < MIN_CANDLES:
            logger.debug(f"SCAN {symbol}: insufficient daily bars ({len(bars_daily)} < {MIN_CANDLES})")
            self._insufficient_history_skip.add(symbol)
            return None

        df = apply_all(
            bars_daily, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
            SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD,
            alligator_fast=ALLIGATOR_FAST,
            alligator_med=ALLIGATOR_MED,
            alligator_slow=ALLIGATOR_SLOW,
        )
        if np.isnan(float(df['MA200'].iloc[-1])):
            logger.debug(f"SCAN {symbol}: MA200 is NaN, skipping")
            return None

        avg_20d_vol    = float(df['volume'].tail(20).mean())
        dollar_vol_20d = float((df['close'] * df['volume']).tail(20).mean())

        # ── Alligator SMMA values with offsets applied ────────────────────────
        # Each SMMA line is "displaced" forward by its offset bars for chart alignment.
        # In live trading this means comparing the value `offset` bars ago with the
        # current price — the offset projects the historical line to the present bar.
        def _smma_at(col: str, offset: int) -> float:
            idx = -1 - offset
            if len(df) > abs(idx):
                val = float(df[col].iloc[idx])
                return val if not np.isnan(val) else float('nan')
            return float('nan')

        smma_fast_val = _smma_at('SMMA_FAST', ALLIGATOR_FAST_OFFSET)
        smma_med_val  = _smma_at('SMMA_MED',  ALLIGATOR_MED_OFFSET)
        smma_slow_val = _smma_at('SMMA_SLOW',  ALLIGATOR_SLOW_OFFSET)

        # Crossover detection: fast+med are currently above slow AND within
        # ALLIGATOR_CROSS_LOOKBACK bars they were NOT (fresh bullish crossover).
        alligator_crossed = False
        if (not any(np.isnan(v) for v in (smma_fast_val, smma_med_val, smma_slow_val))
                and smma_fast_val > smma_slow_val and smma_med_val > smma_slow_val):
            for k in range(1, ALLIGATOR_CROSS_LOOKBACK + 1):
                pf = _smma_at('SMMA_FAST', ALLIGATOR_FAST_OFFSET + k)
                pm = _smma_at('SMMA_MED',  ALLIGATOR_MED_OFFSET  + k)
                ps = _smma_at('SMMA_SLOW',  ALLIGATOR_SLOW_OFFSET + k)
                if not any(np.isnan(v) for v in (pf, pm, ps)):
                    if not (pf > ps and pm > ps):
                        alligator_crossed = True
                        break

        # Live snapshot — price, bid/ask, intraday volume, and intraday OHLC
        if snap is None:
            snap = self._fetch_snapshot(symbol)
        if snap is None:
            logger.debug(f"SCAN {symbol}: snapshot unavailable, skipping")
            return None

        live_price = snap.get('live_price', 0.0)
        if not live_price or live_price <= 0:
            logger.debug(f"SCAN {symbol}: live price unavailable, skipping")
            return None

        bid = snap.get('bid', 0.0)
        ask = snap.get('ask', 0.0)
        if bid > 0 and ask > bid:
            spread_pct = (ask - bid) / ((bid + ask) / 2)
        else:
            spread_pct = 0.0

        intraday_vol = snap.get('intraday_vol', 0.0)
        market_open  = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed_min  = max(1.0, (now_ny - market_open).total_seconds() / 60)
        tod_frac     = _tod_frac(elapsed_min)
        rvol         = (intraday_vol / avg_20d_vol / tod_frac) if avg_20d_vol > 0 else 0.0

        return {
            # Alligator SMMA values (primary entry signal)
            'smma_fast':         smma_fast_val,
            'smma_med':          smma_med_val,
            'smma_slow':         smma_slow_val,
            'alligator_crossed': alligator_crossed,
            # RSI for momentum filter and scoring
            'rsi':               float(df['RSI'].iloc[-1]),
            'rsi_prev':          float(df['RSI'].iloc[-2]),
            # Retained for stop sizing (not entry rules)
            'ma50':              float(df['MA50'].iloc[-1]),
            'ma200':             float(df['MA200'].iloc[-1]),
            'atr':               float(df['ATR'].iloc[-1]),
            'atr_chandelier':    float(df['ATR_CHAND'].iloc[-1]),
            'sma200_slope':      float(df['SMA200_SLOPE'].iloc[-1]),
            # Price and liquidity
            'close':             float(df['close'].iloc[-1]),
            'live_price':        live_price,
            'spread_pct':        spread_pct,
            'rvol':              rvol,
            'volume':            int(df['volume'].iloc[-1]),
            'dollar_vol_20d':    dollar_vol_20d,
            'avg_20d_vol':       avg_20d_vol,
            'bid':               bid,
            'ask':               ask,
            # Intraday OHLC for day-strength check
            'intraday_open':     snap.get('intraday_open', 0.0),
            'intraday_high':     snap.get('intraday_high', live_price),
            'intraday_low':      snap.get('intraday_low',  live_price),
            'price_fetched_at':  now_ny,
            'df_daily':          df,
        }

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _score_candidate(self, ctx: dict) -> float:
        """Delegate to src.rules.score_candidate (Alligator swing formula).

        Scores 0-100:  Alligator Alignment (30) · RVOL (25) · RSI Momentum (25) · Liquidity (20).
        All weights and thresholds are in src/config.py.
        """
        return score_candidate(ctx)

    def _try_rescore(self, sym: str) -> Optional[float]:
        """Attempt to re-compute the entry score from current technical data.

        Called when a position is found in Alpaca but not in local state (e.g.
        after an ephemeral-filesystem restart on Render).  Returns None if bars
        are unavailable or if any step raises an exception.
        """
        try:
            ctx = self.get_technical_context(sym)
            if ctx is None:
                return None
            return round(self._score_candidate(ctx), 2)
        except Exception as e:
            logger.debug(f"RESCORE {sym}: failed ({e})")
            return None

    def _execute_tier_sell(self, symbol: str, tier_qty: int, tier_num: int) -> bool:
        """Partial market sell for a tiered profit exit.

        Flow:
        1. Cancel ALL open orders for the symbol — Alpaca blocks sells while the
           GTC trailing stop holds the full qty.
        2. Submit a market SELL for tier_qty shares.
        3. Clear stop_order_id so _has_unprotected fires and _audit_stop_orders
           re-places the TRAIL for the reduced remaining qty on the next cycle.

        Returns True if the sell was submitted without error.
        """
        # Step 1: cancel open orders (including the trailing stop)
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            for o in open_orders:
                if o.symbol == symbol:
                    try:
                        self.trading_client.cancel_order_by_id(o.id)
                    except Exception as ce:
                        logger.warning(f"TIER SELL {symbol}: cancel {o.id} failed: {ce}")
            if any(o.symbol == symbol for o in open_orders):
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"TIER SELL {symbol}: failed to fetch open orders: {e}")
            return False

        # Step 2: submit partial market sell
        sell_req = MarketOrderRequest(
            symbol=symbol,
            qty=tier_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        try:
            self.trading_client.submit_order(sell_req)
        except Exception as e:
            logger.error(f"TIER SELL {symbol}: market sell {tier_qty} shares failed: {e}")
            return False

        logger.info(
            f"TIER EXIT {symbol}: sold {tier_qty} shares at tier {tier_num}/2 "
            f"({TIER_EXIT_R_MULTIPLES[tier_num - 1]:.2f}R ATR threshold)"
        )

        # Step 3: update state — qty is reconciled by _sync_positions next cycle
        if symbol in self.state:
            self.state[symbol]['tier_sold'] = tier_num
            self.state[symbol].pop('stop_order_id', None)  # triggers audit to re-place TRAIL
            self.save_state()
        return True

    # ── Exit management ───────────────────────────────────────────────────────
    def check_velocity_exits(self) -> Dict[str, float]:
        """Manage all software-enforced exits: hard stop, break-even, Alligator reversal.

        Friday forced-close and velocity (time-based) exits are intentionally absent:
        Alligator swing trades need days-to-weeks to mature; the Alligator reversal
        signal (both fast+med SMMA crossing below slow) is the correct time-to-exit
        signal, and the chandelier GTC trailing stop covers overnight/weekend gaps.

        Returns {symbol: current_price} for positions surviving this cycle.
        """
        now_et          = datetime.now(_TZ_NY)
        if (now_et.hour, now_et.minute) < EXIT_START:
            logger.debug(
                f"EXIT: before {EXIT_START[0]:02d}:{EXIT_START[1]:02d} ET — "
                f"software exits suppressed during opening print volatility."
            )
            return {}
        if (now_et.hour, now_et.minute) >= EXIT_END:
            logger.debug(
                f"EXIT: after {EXIT_END[0]:02d}:{EXIT_END[1]:02d} ET — "
                f"software exits suppressed after market close; GTC trailing stop is active."
            )
            return {}
        prefetched: Dict[str, float] = {}

        for sym in list(self.state.keys()):
            data = self.state[sym]
            if data.get('pending'):
                continue
            if data.get('pending_exit'):
                continue

            entry_price = float(data.get('price', 0))
            if entry_price <= 0:
                logger.warning(f"EXIT: {sym} invalid entry price, skipping.")
                continue

            snap = self._fetch_snapshot(sym)
            if snap:
                cur = snap.get('live_price') or 0.0
                if cur > 0:
                    self.state[sym]['current_price'] = round(cur, 2)
                    self.state[sym]['price_checked_at'] = now_et.isoformat()
                    prefetched[sym] = cur
            else:
                cur = float(data.get('current_price', 0))
                if cur <= 0:
                    logger.warning(
                        f"EXIT: {sym} — no live price; skipping exit checks this cycle."
                    )
                    continue

            # 1. Intraday hard stop
            drawdown = (cur - entry_price) / entry_price
            if drawdown <= -HARD_STOP_PCT:
                logger.warning(
                    f"HARD STOP: {sym} down {drawdown*100:.1f}% from entry "
                    f"(${cur:.2f} vs ${entry_price:.2f}). Forcing exit."
                )
                self.liquidate(sym)
                continue

            # 1b. Software stop_loss enforcement — safety net for when the Alpaca GTC
            # trailing stop is absent (e.g., cancelled during a tier sell and not yet
            # re-placed) or was rejected. stop_loss is the break-even-floored chandelier
            # level written by _update_position_prices each cycle.
            sl = float(data.get('stop_loss', 0))
            if sl > 0 and cur <= sl:
                logger.warning(
                    f"STOP LOSS HIT: {sym} price ${cur:.2f} ≤ stop_loss ${sl:.2f} "
                    f"(entry=${entry_price:.2f}). Alpaca GTC may be absent — forcing exit."
                )
                self.liquidate(sym)
                continue

            # 2. Keep peak_price current so _update_position_prices (which runs after
            # this method) uses the latest price when recomputing the chandelier stop.
            peak_price = max(
                float(data.get('peak_price', entry_price) or entry_price), cur
            )
            if peak_price != float(data.get('peak_price', 0) or 0):
                self.state[sym]['peak_price'] = round(peak_price, 2)

            # NOTE: break-even retrace exit removed — exit 1b (software stop_loss
            # floor) already covers it. stop_loss is floored at entry_price once
            # peak >= BREAK_EVEN_PCT, so `cur <= stop_loss` fires at the same
            # price level. Keeping both was redundant dead code.

            # 3. Tiered profit exits — sell 25% at 0.75R and 1.25R ATR multiples.
            # R = stop_dist (chandelier ATR distance set at entry), so thresholds scale
            # with volatility rather than using fixed percentage targets.
            # Remaining 50% rides the chandelier trailing stop.
            tier_sold = int(data.get('tier_sold', 0))
            if tier_sold < len(TIER_EXIT_R_MULTIPLES):
                tier_threshold = TIER_EXIT_R_MULTIPLES[tier_sold]
                stop_dist = float(data.get('stop_dist', 0))
                profit_pts = cur - entry_price
                if stop_dist > 0 and profit_pts >= tier_threshold * stop_dist:
                    original_qty = float(data.get('original_qty', 0) or 0)
                    if original_qty <= 0:
                        original_qty = float(data.get('qty', 0))
                    tier_qty = int(original_qty * TIER_EXIT_PCT)  # floor = nearest round number
                    current_qty = int(float(data.get('qty', 0)))
                    if tier_qty >= 1 and tier_qty <= current_qty:
                        logger.info(
                            f"TIER EXIT {sym}: profit=${profit_pts:.2f} ≥ {tier_threshold:.2f}R "
                            f"(R=${stop_dist:.2f}) — selling {tier_qty}/{current_qty} "
                            f"shares (tier {tier_sold+1}/2)"
                        )
                        self._execute_tier_sell(sym, tier_qty, tier_sold + 1)
                        prefetched[sym] = cur
                        continue  # remaining shares handled next cycle

            # 4. Alligator reversal exit — re-fetch daily bars to get current SMMA state.
            # Full confirmed reversal: both fast and medium SMMAs cross below slow.
            # Only fires when SMMA values are available (requires fetched daily bars).
            try:
                cached = self._bar_cache.get(sym, {})
                df_exit = cached.get('bars_daily')
                if df_exit is not None and len(df_exit) >= ALLIGATOR_SLOW + ALLIGATOR_SLOW_OFFSET + 2:
                    from src.indicators import compute_smma as _smma

                    def _val_at(series, offset):
                        idx = -1 - offset
                        v = float(series.iloc[idx]) if len(series) > abs(idx) else float('nan')
                        return v if not np.isnan(v) else float('nan')

                    sf = _val_at(_smma(df_exit['close'], ALLIGATOR_FAST), ALLIGATOR_FAST_OFFSET)
                    sm = _val_at(_smma(df_exit['close'], ALLIGATOR_MED),  ALLIGATOR_MED_OFFSET)
                    ss = _val_at(_smma(df_exit['close'], ALLIGATOR_SLOW), ALLIGATOR_SLOW_OFFSET)
                    if (not any(np.isnan(v) for v in (sf, sm, ss))
                            and sf < ss and sm < ss):
                        logger.info(
                            f"ALLIGATOR EXIT: {sym} — fast ({sf:.2f}) and med ({sm:.2f}) "
                            f"SMMAs crossed below slow ({ss:.2f}). Bearish reversal confirmed."
                        )
                        self.liquidate(sym)
                        continue
            except Exception:
                pass  # Non-fatal: trailing stop remains the primary protection

            prefetched[sym] = cur

        self.save_state()
        return prefetched

    # ── Liquidation ───────────────────────────────────────────────────────────
    def liquidate(self, symbol: str):
        """Submit a market SELL for the full position.

        - Cancels non-trailing-stop SELL orders first (preserving the trailing stop
          as last-resort protection if the market sell is rejected).
        - Sets pending_exit=True; state removal is deferred until _sync_positions()
          confirms the position is flat (prevents hold-time clock reset on retry).
        """
        # Cancel ALL open orders for this symbol (including the trailing stop).
        # Alpaca holds shares "for orders" — the GTC TRAIL reserves the full position
        # qty, so a market SELL is rejected with available=0 while the TRAIL is open.
        # The audit's has_unprotected check re-places the TRAIL if the sell fails.
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception as e:
            logger.warning(f"LIQUIDATE {symbol}: failed to fetch open orders: {e}")
            open_orders = []

        cancellable = [o for o in open_orders if o.symbol == symbol]
        for o in cancellable:
            try:
                self.trading_client.cancel_order_by_id(o.id)
            except Exception as e:
                logger.warning(f"LIQUIDATE {symbol}: cancel {o.id} failed: {e}")
        if cancellable:
            time.sleep(1)

        # Find the position
        qty = 0.0
        try:
            pos = self.trading_client.get_open_position(symbol)
            qty = float(pos.qty)
        except Exception:
            pass

        if qty <= 0:
            logger.info(
                f"LIQUIDATE {symbol}: no Alpaca position found; "
                "cancelling orphaned exits and removing stale state."
            )
            self._cancel_orphaned_sell_orders(symbol)
            if symbol in self.state:
                del self.state[symbol]
                self.save_state()
            return

        if symbol in self.state:
            self.state[symbol]['pending_exit'] = True
            self.save_state()

        sell_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        try:
            trade = self.trading_client.submit_order(sell_req)
        except Exception as e:
            if symbol in self.state:
                self.state[symbol].pop('pending_exit', None)
                self.save_state()
            logger.error(
                f"LIQUIDATE {symbol}: market SELL placement failed: {e}; "
                "state retained for retry."
            )
            return

        # Poll for fill confirmation (up to 30 s)
        deadline = time.time() + 30
        status   = ''
        while time.time() < deadline:
            try:
                o = self.trading_client.get_order_by_id(trade.id)
                status = str(o.status)
                if status in ('OrderStatus.FILLED', 'filled',
                              'OrderStatus.CANCELED', 'canceled',
                              'OrderStatus.REJECTED', 'rejected',
                              'OrderStatus.EXPIRED', 'expired'):
                    break
            except Exception:
                pass
            time.sleep(1)

        if status in ('OrderStatus.CANCELED', 'canceled',
                      'OrderStatus.REJECTED', 'rejected',
                      'OrderStatus.EXPIRED', 'expired'):
            if symbol in self.state:
                self.state[symbol].pop('pending_exit', None)
                self.save_state()
            logger.error(
                f"LIQUIDATE {symbol}: market SELL {status}; "
                "state retained for retry."
            )
            return

        logger.info(
            f"LIQUIDATE {symbol}: market SELL {status} "
            f"(qty={qty:g}); state pending until Alpaca confirms flat."
        )

    # ── Scanner + entry cycle ─────────────────────────────────────────────────
    def get_institutional_scan(self) -> List[str]:
        """Fetch candidates from all four sources: gainers, actives-vol, actives-trades, universe crossover."""
        return get_candidates(self.data_client, self.screener_client, self.trading_client)

    def run_cycle(self):
        # 0. Connectivity check
        if not self._ensure_connected():
            logger.error("ENGINE: Alpaca connectivity failed — skipping cycle")
            self._write_dashboard_data(connected=False)
            return

        # 1. Position sync
        self._sync_positions()

        # 1.5. Stop-order audit — runs before all account/VIX/entry gates so that
        # positions are protected even when account summary or regime data is unavailable
        _audit_today = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        _has_unprotected = any(
            float(d.get('stop_dist', 0)) <= 0 or 'stop_order_id' not in d
            for d in self.state.values()
            if not d.get('pending') and not d.get('pending_exit')
        )
        if (self._last_audit_date != _audit_today or _has_unprotected) and self.state:
            self._audit_stop_orders()
            self._last_audit_date = _audit_today

        # 2. Account values
        try:
            equity, settled = self._get_account_values()
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"ACCOUNT: values unavailable ({e}); managing positions only.")
            self.check_velocity_exits()
            self._update_position_prices()
            self._write_dashboard_data(connected=True)
            return

        max_pos        = self._calc_max_positions(equity)
        capacity_slots = max(0, max_pos - len(self.state))
        cash_slots     = self._calc_cash_entry_slots(settled)
        open_slots     = min(capacity_slots, cash_slots)
        bucket_size    = (settled * BUCKET_CASH_PCT) / open_slots if open_slots > 0 else 0.0
        self._last_equity       = equity
        self._last_settled_cash = settled
        self._equity_initialized = True

        bucket_text = f"${bucket_size:.2f}" if open_slots > 0 else "N/A"
        logger.info(
            f"HEARTBEAT: Equity ${equity:.2f} | Settled ${settled:.2f} | "
            f"EntrySlots {open_slots}/{max_pos} | CashSlots {cash_slots} | "
            f"Bucket {bucket_text} | Positions: {list(self.state.keys()) or 'none'}"
        )

        # Daily loss circuit breaker
        today_str = datetime.now(_TZ_NY).strftime('%Y-%m-%d')
        if self._day_start_date != today_str:
            self._day_start_date   = today_str
            self._day_start_equity = equity
            # Clear day-scoped caches on date rollover
            self._daily_scan_skip.clear()
            self._insufficient_history_skip.clear()
            self._bar_cache.clear()
            # Refresh the Alligator crossover universe for the new trading day.
            # Runs on the first cycle after midnight ET (markets closed), so the
            # ~2-min scan completes hours before the 09:35 entry window opens.
            # Skip weekends — Saturday/Sunday have no new bar data and no entries.
            if datetime.now(_TZ_NY).weekday() < 5:
                logger.info("SCANNER: Daily crossover scan refresh (new trading day)...")
                get_alligator_crossover_scan(self.data_client, self.trading_client)
        elif (
            self._day_start_equity is not None
            and equity < self._day_start_equity * (1 - MAX_DAILY_LOSS_PCT)
        ):
            logger.warning(
                f"CIRCUIT BREAKER: daily loss limit hit "
                f"(equity ${equity:.2f} vs open ${self._day_start_equity:.2f} "
                f"= {(1-equity/self._day_start_equity)*100:.1f}% loss). "
                "No new entries for the rest of today."
            )
            self.check_velocity_exits()
            self._update_position_prices()
            self._write_dashboard_data(connected=True)
            return

        # 3. Portfolio concentration guard
        concentration, halt_entries, halt_all = self._check_portfolio_concentration(equity)
        if halt_all:
            logger.error(
                f"CONCENTRATION: {concentration*100:.0f}% deployed "
                f"≥ {CONCENTRATION_HALT_PCT*100:.0f}% — halting all new orders."
            )
            self.check_velocity_exits()
            self._update_position_prices()
            self._write_dashboard_data(connected=True)
            return
        if halt_entries:
            logger.warning(
                f"CONCENTRATION: {concentration*100:.0f}% deployed "
                f"≥ {CONCENTRATION_WARN_PCT*100:.0f}% — no new entries."
            )

        # 4. VIX risk filter
        vix_price = self._fetch_vix()
        if vix_price is None:
            # Never fetched successfully — warn but allow entries; the 12-rule screener
            # will gate quality independently.  Blocking all entries on a DNS hiccup
            # at startup is worse than trading without the VIX macro filter.
            logger.warning("VIX: unavailable at startup — proceeding without VIX filter.")
        else:
            self._last_vix = vix_price
            if vix_price > VIX_THRESHOLD:
                logger.warning(f"VIX HIGH ({vix_price:.2f}). Risk Off — no new entries.")
                self.check_velocity_exits()
                self._update_position_prices()
                self._write_dashboard_data(connected=True)
                return

        # 5. Manage existing positions
        prefetched = self.check_velocity_exits()

        # 7. Entry window check
        now_ny = datetime.now(_TZ_NY)
        if not (now_ny.weekday() < 5 and ENTRY_START <= (now_ny.hour, now_ny.minute) <= ENTRY_END):
            self._update_position_prices(prefetched)
            self._write_dashboard_data(connected=True)
            return

        # Block new entries on Friday after FRIDAY_CLOSE_HOUR (weekend gap risk)
        if now_ny.weekday() == 4 and now_ny.hour >= FRIDAY_CLOSE_HOUR:
            self._update_position_prices(prefetched)
            self._write_dashboard_data(connected=True)
            return

        if halt_entries or open_slots <= 0:
            if open_slots <= 0:
                logger.info(
                    f"SCAN: no entry slots "
                    f"(capacity={capacity_slots}, cash_slots={cash_slots})"
                )
            self._update_position_prices(prefetched)
            self._write_dashboard_data(connected=True)
            return

        # 8. SPY regime (soft — bearish cuts size + tightens RVOL, does not block)
        # SPY_FILTER_ENABLED=False by default: Alligator swing works across regime types.
        if SPY_FILTER_ENABLED:
            regime           = self._fetch_spy_trend()
            is_bull          = regime['is_bull']
            effective_rvol   = RVOL_MIN * regime['rvol_mult']
            if not is_bull:
                logger.warning(
                    f"REGIME: SPY bearish (close={regime['spy_close']:.2f} < "
                    f"EMA{SPY_EMA_PERIOD}={regime['ema50']:.2f}) — "
                    f"size cut {SPY_REGIME_SIZE_CUT*100:.0f}%, "
                    f"RVOL threshold raised to {effective_rvol:.2f}x"
                )
                bucket_size = round(bucket_size * regime['size_factor'], 2)
        else:
            is_bull        = True
            effective_rvol = RVOL_MIN

        is_friday = (now_ny.weekday() == 4)

        # 9. Scan and score candidates
        watchlist = self.get_institutional_scan()
        logger.info(
            f"SCAN: {len(watchlist)} candidates → {watchlist}"
            + (" [FRIDAY: 2× volume threshold]" if is_friday else "")
            + ("" if is_bull else f" [BEARISH REGIME: size=${bucket_size:.0f}]")
        )

        book_sectors: Dict[str, int] = {}
        for book_sym in self.state:
            s = self._get_sector(book_sym)
            book_sectors[s] = book_sectors.get(s, 0) + 1

        # Batch-fetch snapshots for all candidates in one API call (vs one call per symbol)
        scan_candidates = [
            s for s in watchlist
            if s not in self.state
            and s not in TICKER_BLOCKLIST
            and s not in self._daily_scan_skip
            and s not in self._insufficient_history_skip
        ]
        batch_snaps = self._batch_fetch_snapshots(scan_candidates)

        signals      = []
        n_portfolio  = 0
        n_blocked    = 0
        n_history    = 0
        n_no_ctx     = 0
        n_day        = 0
        n_cycle      = 0
        _rule_fails: dict = {}

        for sym in watchlist:
            if sym in self.state:
                n_portfolio += 1
                logger.info(f"SCAN {sym}: SKIP — already in portfolio")
                continue
            if sym in TICKER_BLOCKLIST:
                n_blocked += 1
                continue
            if sym in self._daily_scan_skip:
                n_day += 1
                logger.debug(
                    f"SCAN {sym}: SKIP — day-filtered ({self._daily_scan_skip[sym]})"
                )
                continue
            if sym in self._insufficient_history_skip:
                n_history += 1
                continue

            ctx = self.get_technical_context(sym, snap=batch_snaps.get(sym))
            if not ctx:
                n_no_ctx += 1
                continue

            # Inject runtime overrides into ctx so rule functions can read them
            ctx['_is_friday']          = is_friday
            ctx['_effective_rvol_min'] = effective_rvol

            # ── Permanent per-day filter checks (dollar volume) ───────────────
            perm_passed, perm_fails = check_rules(ctx, PERMANENT_DAY_RULES)
            if not perm_passed:
                reason = ', '.join(r for _, r in perm_fails)
                self._daily_scan_skip[sym] = reason
                n_day += 1
                if sym in self.state:
                    self.state[sym]['_daily_skip_reason'] = reason
                    self.state[sym]['_daily_skip_date']   = today_str
                logger.debug(f"SCAN {sym}: SKIP (day) — {reason}")
                continue

            # ── Cycle-by-cycle filter checks (all 7 Alligator swing rules) ────
            cycle_passed, cycle_fails = check_rules(ctx, CYCLE_RULES)

            price       = ctx['live_price']
            smma_fast   = ctx.get('smma_fast', 0.0)
            smma_slow   = ctx.get('smma_slow', 0.0)
            rsi         = ctx['rsi']
            rsi_p       = ctx['rsi_prev']
            rvol        = ctx.get('rvol', 0.0)
            spread_pct  = ctx.get('spread_pct', 0.0)
            dol_vol_20d = ctx['dollar_vol_20d']
            crossed     = ctx.get('alligator_crossed', False)

            scan_detail = (
                f"SCAN {sym}: price=${price:.2f} "
                f"SMMA(fast={smma_fast:.2f} slow={smma_slow:.2f} crossed={crossed}) "
                f"RSI={rsi:.1f}(Δ{rsi-rsi_p:+.1f}) RVOL={rvol:.2f}x(≥{effective_rvol:.2f}) "
                f"spread={spread_pct*100:.2f}% DolVol20d=${dol_vol_20d/1e6:.0f}M"
            )

            if not cycle_passed:
                n_cycle += 1
                for name, reason in cycle_fails:
                    _rule_fails[name] = _rule_fails.get(name, 0) + 1
                logger.debug(f"{scan_detail}")
                logger.debug(
                    f"SCAN {sym}: NO SIGNAL — failed: "
                    f"{[name for name, _ in cycle_fails]}"
                )
                continue

            # ── Correlation + sector clustering (expensive — only for passing) ─
            df_daily = ctx.get('df_daily')
            if df_daily is not None and self.state:
                max_corr = self._compute_book_correlation(sym, df_daily)
                if max_corr > CORR_MAX:
                    logger.debug(
                        f"SCAN {sym}: SKIP — corr={max_corr:.2f} > {CORR_MAX}"
                    )
                    continue

            sector = self._get_sector(sym)
            if book_sectors.get(sector, 0) >= MAX_SECTOR_COUNT:
                logger.debug(
                    f"SCAN {sym}: SKIP — sector '{sector}' already at "
                    f"{book_sectors[sector]}/{MAX_SECTOR_COUNT} positions"
                )
                continue

            # Analyst ratings — fetched only for symbols passing all rules (session-cached).
            # Injecting here (not in get_technical_context) avoids a yfinance call for every
            # scan candidate; only the handful that reach scoring actually hit the API.
            ctx.update(self._get_analyst_ratings(sym))

            score = self._score_candidate(ctx)
            if score < SCAN_MIN_SCORE:
                logger.debug(
                    f"SCAN {sym}: SKIP — score={score:.1f} < min {SCAN_MIN_SCORE:.0f}"
                )
                continue
            signals.append((score, sym, ctx))
            logger.info(scan_detail)
            logger.info(
                f"SIGNAL {sym}: score={score:.1f}/100 | "
                f"SMMA fast={smma_fast:.2f} slow={smma_slow:.2f} "
                f"RVOL={rvol:.2f}x RSIδ={rsi-rsi_p:.1f} spread={spread_pct*100:.2f}%"
            )

        if not signals:
            cycle_detail = ' '.join(
                f'{k}:{v}' for k, v in sorted(_rule_fails.items(), key=lambda x: -x[1])
            )
            logger.info(
                f"SCAN: No signals — "
                f"{n_blocked} ETF/blocked, {n_history} no-history, "
                f"{n_no_ctx} no-snapshot, {n_day} day-filtered(DolVol), "
                f"{n_cycle} cycle-filtered"
                + (f" [{cycle_detail}]" if cycle_detail else "")
            )
            self._update_position_prices(prefetched)
            self._write_dashboard_data(connected=True)
            return

        signals.sort(key=lambda x: x[0], reverse=True)
        ranked = [(s, sym) for s, sym, _ in signals]
        logger.info(
            f"RANKED: {ranked} — attempting up to {min(open_slots, len(signals))}"
        )

        placed = 0
        for score, sym, ctx in signals:
            if placed >= open_slots:
                break

            # Each fill poll blocks up to 30 s — re-run exits before the next entry
            # so hard-stop / break-even protection stays current across multi-signal cycles
            if placed > 0:
                self.check_velocity_exits()

            # Re-fetch live price if the scan snapshot is stale (>60 s)
            fetched_at = ctx.get('price_fetched_at', now_ny)
            age_s = (datetime.now(_TZ_NY) - fetched_at).total_seconds()
            if age_s > 60:
                snap2 = self._fetch_snapshot(sym)
                if snap2 and snap2.get('live_price', 0) > 0:
                    new_price = snap2['live_price']
                    drift = abs(new_price - ctx['live_price']) / ctx['live_price']
                    if drift > REPRICE_DRIFT_MAX_PCT:
                        logger.warning(
                            f"SKIP {sym}: price drifted {drift*100:.2f}% since scan"
                        )
                        continue
                    price = new_price
                    ask_now = snap2.get('ask', 0.0)
                else:
                    logger.warning(f"SKIP {sym}: stale price and reprice failed")
                    continue
            else:
                price   = ctx['live_price']
                ask_now = ctx.get('ask', 0.0)

            atr          = ctx['atr']
            atr_chand    = ctx['atr_chandelier']
            rvol_now     = ctx.get('rvol', RVOL_MIN)

            if np.isnan(atr) or atr <= 0:
                logger.warning(f"SKIP {sym}: invalid ATR ({atr:.4f})")
                continue
            if np.isnan(atr_chand) or atr_chand <= 0:
                logger.warning(f"SKIP {sym}: invalid ATR_CHAND ({atr_chand:.4f}) — zero-width trail stop")
                continue

            # Adaptive limit price — ATR/RVOL scaled slippage buffer above ask
            ref       = ask_now if ask_now > 0 else price
            atr_15min = atr / (26 ** 0.5)
            rvol_scale = (rvol_now / RVOL_MIN) ** 0.5
            drift_buf  = max(
                ref * LIMIT_BUF_MIN_PCT,
                min(atr_15min * rvol_scale, ref * LIMIT_BUF_MAX_PCT),
            )
            limit_price = round(ref + drift_buf, 2)
            logger.debug(
                f"LIMIT {sym}: ref=${ref:.2f} drift=${drift_buf:.3f} "
                f"→ limit=${limit_price:.2f}"
            )

            chandelier_dist = round(atr_chand * CHANDELIER_MULT, 2)
            hard_stop_dist  = round(limit_price * HARD_STOP_PCT, 2)
            risk_stop_dist  = max(min(chandelier_dist, hard_stop_dist), 0.01)

            risk_per_trade = equity * RISK_PER_TRADE_PCT
            qty_by_risk    = int(risk_per_trade / risk_stop_dist) if risk_stop_dist > 0 else 0
            qty_by_bucket  = int(bucket_size / limit_price)
            qty            = min(qty_by_risk, qty_by_bucket)
            if qty < 1:
                logger.warning(
                    f"SKIP {sym}: qty=0 after sizing "
                    f"(risk={qty_by_risk} bucket={qty_by_bucket})"
                )
                continue

            order_cost = round(qty * limit_price, 2)
            if settled < order_cost:
                logger.warning(
                    f"SKIP {sym}: insufficient settled cash "
                    f"(need ${order_cost:.2f}, have ${settled:.2f})"
                )
                continue

            # ── Submit limit buy ──────────────────────────────────────────────
            buy_req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            try:
                buy_order = self.trading_client.submit_order(buy_req)
            except Exception as e:
                logger.warning(f"SKIP {sym}: limit BUY submission failed: {e}")
                continue

            # Mark as pending so the slot is visible in the dashboard
            self.state[sym] = {
                'price':    limit_price,
                'time':     datetime.now(_TZ_NY).isoformat(),
                'qty':      qty,
                'original_qty': qty,
                'tier_sold': 0,
                'stop_loss': round(limit_price - chandelier_dist, 2),
                'stop_dist': chandelier_dist,
                'peak_price': limit_price,
                'volume':   ctx.get('volume', 0),
                'score':    score,
                'analyst_buy':  ctx.get('analyst_buy',  0),
                'analyst_hold': ctx.get('analyst_hold', 0),
                'analyst_sell': ctx.get('analyst_sell', 0),
                'pending':  True,
                'entry_order_id': str(buy_order.id),
            }
            self.save_state()

            logger.info(
                f"BUY SUBMITTED: {sym} | score={score:.1f} qty={qty} "
                f"limit=${limit_price:.2f} | "
                f"Chandelier dist=${chandelier_dist:.2f} | "
                f"order_id={buy_order.id}"
            )

            # ── Poll for fill (up to 30 s) ────────────────────────────────────
            fill_price = None
            filled_qty = 0.0
            deadline   = time.time() + 30
            while time.time() < deadline:
                try:
                    o = self.trading_client.get_order_by_id(buy_order.id)
                    status = str(o.status)
                    if status in ('OrderStatus.FILLED', 'filled'):
                        fill_price = float(o.filled_avg_price or limit_price)
                        filled_qty = float(o.filled_qty or qty)
                        break
                    if status in (
                        'OrderStatus.CANCELED', 'canceled',
                        'OrderStatus.REJECTED', 'rejected',
                        'OrderStatus.EXPIRED', 'expired',
                    ):
                        logger.warning(
                            f"ENTRY {sym}: BUY {status}; removing from state."
                        )
                        del self.state[sym]
                        self.save_state()
                        fill_price = None
                        break
                except Exception:
                    pass
                time.sleep(1)

            if fill_price is None or fill_price <= 0:
                # Not filled within timeout — cancel and move on
                try:
                    self.trading_client.cancel_order_by_id(buy_order.id)
                except Exception:
                    pass
                if sym in self.state and self.state[sym].get('pending'):
                    del self.state[sym]
                    self.save_state()
                logger.warning(
                    f"ENTRY {sym}: BUY not filled within timeout; cancelled."
                )
                continue

            # ── Submit chandelier trailing stop ───────────────────────────────
            # Deferred until EXIT_START to avoid opening-print volatility fills.
            now_for_stop = datetime.now(_TZ_NY)
            if (now_for_stop.hour, now_for_stop.minute) < EXIT_START:
                logger.info(
                    f"ENTRY {sym}: trailing stop deferred until "
                    f"{EXIT_START[0]:02d}:{EXIT_START[1]:02d} ET — audit will place it."
                )
                stop_order = None
            else:
                # Alpaca rejects trail_price > 25% of stock price; use 24% to leave a
                # buffer for intraday price movement between fill and stop submission.
                max_trail_entry = round(fill_price * 0.24, 2)
                trail_dist_entry = min(chandelier_dist, max_trail_entry)
                if trail_dist_entry < chandelier_dist:
                    logger.info(
                        f"ENTRY {sym}: chandelier dist ${chandelier_dist:.2f} capped "
                        f"to ${trail_dist_entry:.2f} (Alpaca 25% limit on ${fill_price:.2f})"
                    )
                stop_req = TrailingStopOrderRequest(
                    symbol=sym,
                    qty=filled_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_price=trail_dist_entry,
                )
                try:
                    stop_order = self.trading_client.submit_order(stop_req)
                    chandelier_dist = trail_dist_entry  # use capped value in state write
                except Exception as stop_err:
                    logger.error(
                        f"ENTRY {sym}: trailing stop placement failed ({stop_err}). "
                        "Position open WITHOUT stop — audit will retry next cycle."
                    )
                    stop_order = None
                    chandelier_dist = 0  # triggers has_unprotected → audit retries

            # Update state with confirmed fill details
            self.state[sym] = {
                'fill_price':     round(fill_price, 2),
                'price':          round(fill_price, 2),
                'entry_order_id': str(buy_order.id),
                'time':           datetime.now(_TZ_NY).isoformat(),
                'qty':            filled_qty,
                'original_qty':   filled_qty,
                'tier_sold':      0,
                'stop_loss':      round(fill_price - chandelier_dist, 2),
                'stop_dist':      chandelier_dist,
                'peak_price':     round(fill_price, 2),
                'volume':         ctx.get('volume', 0),
                'score':          score,
                'analyst_buy':    ctx.get('analyst_buy',  0),
                'analyst_hold':   ctx.get('analyst_hold', 0),
                'analyst_sell':   ctx.get('analyst_sell', 0),
            }
            if stop_order:
                self.state[sym]['stop_order_id'] = str(stop_order.id)
            self.save_state()

            actual_cost = round(filled_qty * fill_price, 2)
            settled    -= actual_cost
            placed     += 1

            capacity_slots = max(0, max_pos - len(self.state))
            cash_slots     = self._calc_cash_entry_slots(settled)
            open_slots     = min(capacity_slots, cash_slots)
            bucket_size    = (settled * BUCKET_CASH_PCT) / open_slots if open_slots > 0 else 0.0

            logger.info(
                f"ORDER CONFIRMED: {sym} | score={score:.1f} qty={filled_qty:g} "
                f"limit=${limit_price:.2f} fill=${fill_price:.2f} "
                f"chandelier=${round(fill_price-chandelier_dist,2):.2f} "
                f"(dist=${chandelier_dist:.2f}) | settled remaining=${settled:.2f}"
            )

        self._update_position_prices(prefetched)
        self._write_dashboard_data(connected=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        logger.info("=" * 40)
        mode = "PAPER" if ALPACA_PAPER else "LIVE"
        logger.info(f"ENGINE DEPLOYED — Alpaca {mode} Trading")
        logger.info("=" * 40)
        self._initialize()
        logger.info("ENGINE READY: Starting main loop...")
        while True:
            try:
                self.run_cycle()
                now_ny = datetime.now(_TZ_NY)
                self._last_scan_ts = now_ny.strftime("%H:%M:%S %Z")
                self._next_scan_dt = (now_ny + timedelta(seconds=SCAN_INTERVAL)).isoformat()
                self._write_dashboard_data(connected=True)
                time.sleep(SCAN_INTERVAL)
            except Exception:
                logger.exception("RUNTIME ERROR")
                self._next_scan_dt = (
                    datetime.now(_TZ_NY) + timedelta(seconds=ERROR_WAIT)
                ).isoformat()
                self._write_dashboard_data(connected=self._ensure_connected())
                time.sleep(ERROR_WAIT)

    def shutdown(self):
        """Cancel all pending BUY orders and disconnect cleanly."""
        logger.info("SHUTDOWN: cancelling pending BUY orders...")
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            for o in open_orders:
                if str(o.side) in ('OrderSide.BUY', 'buy'):
                    try:
                        self.trading_client.cancel_order_by_id(o.id)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"SHUTDOWN: order cancel failed: {e}")
        logger.info("SHUTDOWN: complete.")
