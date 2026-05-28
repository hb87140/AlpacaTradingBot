"""
Velocity Strategy Backtester — Production-Grade Edition
────────────────────────────────────────────────────────
Key design decisions:
  1. RVOL uses BACKTEST_RVOL_MIN (1.2×) not RVOL_MIN (2.5×).
     End-of-day volume does not spike the same way intraday RVOL does;
     2.5× was eliminating ~95% of valid daily setups.
  2. bars_held counts actual trading bars open (not calendar days).
     Previously, Friday entry → Saturday + Sunday counted as 2 bars,
     firing velocity_exit after only 1 real trading session.
  3. Break-even floor: once profit ≥ BREAK_EVEN_PCT (4%), the effective
     stop cannot fall below entry price — locks in break-even.
  4. ATR-based position sizing: risk RISK_PER_TRADE_PCT (2%) of equity
     per trade, using the tighter of chandelier or 7% hard-stop as the
     risk distance.  Capped by the per-bucket dollar limit.
  5. 0.1% entry slippage added to entry_price for realism.
  6. Commission configurable via BACKTEST_COMMISSION_PER_ORDER (default $0.00;
     Alpaca is commission-free). Set VELOCITY_BACKTEST_COMMISSION_PER_ORDER env
     var to simulate a non-zero commission broker.
  7. Composite scanner score = Trend(30pts) + RVOL(25pts) + Momentum(25pts);
     Liquidity(20pts) omitted (no bid/ask spread in daily OHLCV data).
     Mirrors live _score_candidate() — ranks high-conviction setups ahead of thin movers.
  8. Data caching: downloaded + indicator-enriched DataFrames are
     pickled to backtest/.cache/ so re-runs skip the 5-min download.
  9. Filter funnel stats printed at end: shows exactly where signals
     are lost across each filter stage.

Universe discovery (mirrors the live Alpaca top-gainers + most-actives screener):
  - Candidate pool  : NASDAQ Global Select/Market + NYSE equities
  - Daily scan      : each bar, rank by composite score and keep top
                      all scanner-passed stocks unless --scan-count is set

ORB approximation: previous day's high acts as the opening-range breakout
level.  A close above it on the signal day mirrors the live "price > orb_high"
check.

Entry rules (production signal combination — optimizer rank #1):
  1. Data sufficiency    : ≥ MIN_CANDLES (210) bars of history
  2. Trend               : close > MA50 > MA200
  3. Trend separation    : (MA50 - MA200) / MA200 ≥ MIN_TREND_SEP (3%) — confirmed uptrend, not fresh crossover
  4. ADX                 : ADX(14) ≥ ADX_THRESHOLD (20) — trend has real momentum
  5. 52-week high        : close ≥ HIGH200_MIN_PCT (85%) of 200-day rolling high — momentum leadership
  6. RVOL                : volume / 20d avg ≥ BACKTEST_RVOL_MIN (1.2×)
  7. Spread              : not available in daily data — skipped
  8. SPY regime          : SPY close > SMA50 > SMA200 (optional)
  9. Correlation         : not practical in daily batch — skipped
  10. Sector clustering  : not practical in daily batch — skipped
  11. RSI delta          : RSI rose vs previous bar by ≥ RSI_MIN_DELTA (1.0 pt) — momentum accelerating
  12. RSI level          : RSI > RSI_THRESHOLD (55) — confirms bullish momentum
      ORB proxy          : close > prev_high
      Gap cap            : open ≤ prev_high × (1 + GAP_MAX_PCT)

  Removed from prior 12-rule set (optimizer rank #1 result):
    - SMA200 slope (use_slope=False by default)
    - RSI rising flag (use_rsi_rise=False; replaced by rsi_delta check)

Exit rules (production):
  • Chandelier trailing stop : peak_high - ATR_CHAND × CHANDELIER_MULT
  • Hard stop                : entry × (1 - HARD_STOP_PCT) = 7% from entry
  • Break-even floor         : if profit > BREAK_EVEN_PCT, stop ≥ entry
  • Velocity time exit       : held ≥ hold_bars (2 trading days default) and profit < PROFIT_MIN_THRESHOLD (5%)
  • Friday close              : on Fridays, close positions with profit < FRIDAY_MIN_PROFIT_PCT (3%) to avoid weekend gap risk
  • (No take-profit bracket — removed from production)
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    PROFIT_MIN_THRESHOLD,
    RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW, RSI_THRESHOLD,
    BACKTEST_INITIAL_CAPITAL, MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, BACKTEST_SCAN_COUNT,
    SCAN_MIN_PRICE, SCAN_MIN_VOLUME, SCAN_MIN_DOLLAR_VOL, SCAN_MIN_GAIN_PCT,
    VIX_THRESHOLD,
    MIN_CANDLES, BACKTEST_RVOL_MIN, GAP_MAX_PCT,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    RSI_MIN_DELTA, HARD_STOP_PCT,
    SMA200_SLOPE_LOOKBACK,
    RISK_PER_TRADE_PCT, BREAK_EVEN_PCT,
    BACKTEST_COMMISSION_PER_ORDER,
    MIN_TREND_SEP,
    BACKTEST_HOLD_BARS, BACKTEST_SLIPPAGE, BACKTEST_EXIT_SLIPPAGE,
    VOL_MULT_FRIDAY, FRIDAY_MIN_PROFIT_PCT,
    ADX_THRESHOLD, HIGH200_MIN_PCT,
    SCAN_MIN_SCORE, BUCKET_CASH_PCT,
)
from src.indicators import apply_all

warnings.filterwarnings("ignore", category=FutureWarning)

_NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
_OTHER_LISTED_URL  = 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
_CACHE_DIR         = os.path.join(os.path.dirname(__file__), ".cache")
_DEFAULT_ROUND_TRIP_COST = BACKTEST_COMMISSION_PER_ORDER * 2

# Columns required to be non-NaN before calling _entry_signal
_REQUIRED_ENTRY_COLS = [
    'MA50', 'MA200', 'RSI', 'ATR', 'ATR_CHAND',
    'SMA200_SLOPE', 'prev_high',
]
# Columns snapshotted into the pre-computed candidate dicts
_PRECOMPUTE_COLS = (
    'close', 'open', 'MA50', 'MA200', 'RSI', 'ATR', 'ATR_CHAND',
    'SMA200_SLOPE', 'prev_high', 'ADX', 'HIGH200', 'EMA20',
)


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    symbol:      str
    entry_date:  date
    entry_price: float
    exit_date:   Optional[date]  = None
    exit_price:  Optional[float] = None
    exit_reason: str             = ""
    qty:         float           = 0.0
    round_trip_commission: float = _DEFAULT_ROUND_TRIP_COST

    @property
    def gross_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def net_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.gross_pnl - self.round_trip_commission

    # Alias kept for backward compat with print_report / metrics
    @property
    def pnl(self) -> float:
        return self.net_pnl

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    trades:       List[Trade]
    equity_curve: pd.Series
    metrics:      Dict
    filter_stats: Dict


# ── Core backtester ───────────────────────────────────────────────────────────
class VelocityBacktest:
    """
    Replay the full VelocityEngine production signal logic on historical
    daily OHLCV data downloaded via yfinance.

    Parameters
    ----------
    start           : backtest start date  (YYYY-MM-DD)
    end             : backtest end date    (YYYY-MM-DD)
    capital         : starting capital in USD
    max_pos         : max simultaneous positions
    hold_bars       : trading bars before velocity time-exit check (default 2)
    scan_count      : top-N from daily scanner considered for entry each bar; <=0 means all
    min_price       : minimum close price filter
    min_volume      : minimum daily share volume filter
    min_dollar_vol  : minimum 20-day avg dollar volume
    use_spy_filter  : if True, skip entries when SPY < SMA50 or SMA50 < SMA200
    use_vix_filter  : if True, skip new entries when VIX > VIX_THRESHOLD
    rvol_min             : daily RVOL threshold (1.2× optimal — much lower than live 2.5× intraday)
    break_even_pct       : once profit exceeds this, floor the stop at entry (0.04 optimal)
    profit_min_threshold : velocity exit fires if profit < this after hold_bars (0.05 optimal with cm=2.0)
    chandelier_mult      : ATR multiplier for trailing stop (2.0 optimal — harvests gains faster)
    use_cache            : load/save downloaded data from backtest/.cache/
    """

    def __init__(
        self,
        start:          str   = "2025-01-01",
        end:            str   = "2026-05-01",
        capital:        float = BACKTEST_INITIAL_CAPITAL,
        max_pos:        int   = MAX_POSITIONS_CAP,
        hold_bars:      int   = BACKTEST_HOLD_BARS,
        scan_count:     int   = BACKTEST_SCAN_COUNT,
        min_price:      float = SCAN_MIN_PRICE,
        min_volume:     float = SCAN_MIN_VOLUME,
        min_dollar_vol: float = SCAN_MIN_DOLLAR_VOL,
        use_spy_filter: bool  = True,
        use_vix_filter: bool  = False,
        rvol_min:             float = BACKTEST_RVOL_MIN,
        break_even_pct:       float = BREAK_EVEN_PCT,
        profit_min_threshold: float = PROFIT_MIN_THRESHOLD,
        chandelier_mult:      float = CHANDELIER_MULT,
        commission_per_order: float = BACKTEST_COMMISSION_PER_ORDER,
        use_cache:            bool  = True,
    ):
        self.start                 = start
        self.end                   = end
        self.capital               = capital
        self.max_pos               = max_pos
        self.hold_bars             = hold_bars
        self._scan_count           = scan_count
        self._min_price            = min_price
        self._min_volume           = min_volume
        self._min_dollar_vol       = min_dollar_vol
        self._use_spy_filter       = use_spy_filter
        self._use_vix_filter       = use_vix_filter
        self._rvol_min             = rvol_min
        self._break_even_pct       = break_even_pct
        self._profit_min_threshold = profit_min_threshold
        self._chandelier_mult      = chandelier_mult
        self._round_trip_cost      = max(0.0, float(commission_per_order)) * 2.0
        self._use_cache            = use_cache

        self._data:        Dict[str, pd.DataFrame] = {}
        self._vix_series:  Optional[pd.Series]     = None
        self._spy_bull:    Optional[pd.Series]      = None

        # Download starts early enough to warm up MA200 + chandelier ATR
        _trade_start     = date.fromisoformat(start)
        self._data_start = (_trade_start - timedelta(days=400)).isoformat()

        # Filter funnel accumulators (populated during run)
        self._filter_stats: Dict = {
            'scan_days':            0,
            'coarse_candidates':    0,   # pass price/vol/dollar-vol/trend/rvol
            'fine_signals':         0,   # pass full _entry_signal (12-rule)
            'entries_taken':        0,   # actually opened a position
            'entries_skipped_full': 0,   # signal fired but max_pos already full
            'spy_blocked_days':     0,   # trading days blocked by SPY filter
            'vix_blocked_days':     0,   # trading days blocked by VIX filter
            'friday_closes':        0,   # positions closed by Friday profit gate
            'total_commissions':    0.0,
        }

        # Cached across _run_loop calls (same data, different flags)
        self._all_dates:   Optional[list] = None
        self._date_to_idx: Optional[dict] = None

    # ── Universe discovery ────────────────────────────────────────────────────
    @staticmethod
    def _fetch_universe() -> List[str]:
        """Fetch NASDAQ Global Select/Market + NYSE ordinary equities."""
        import io
        import urllib.request

        ua = {'User-Agent': 'Mozilla/5.0 (compatible; VelocityBacktest/1.0)'}

        def _get(url: str) -> bytes:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()

        tickers: set = set()

        try:
            text  = _get(_NASDAQ_LISTED_URL).decode('utf-8')
            df_nq = pd.read_csv(io.StringIO(text), sep='|')
            df_nq = df_nq[
                (df_nq['ETF']               == 'N') &
                (df_nq['Test Issue']        == 'N') &
                (df_nq['Market Category'].isin(['Q', 'G'])) &
                (df_nq['Symbol'].str.len()  <= 5) &
                (~df_nq['Symbol'].str.contains(r'[\^+$\.]', regex=True, na=False))
            ]
            tickers.update(df_nq['Symbol'].dropna().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch NASDAQ listing: {e}")

        try:
            text   = _get(_OTHER_LISTED_URL).decode('utf-8')
            df_oth = pd.read_csv(io.StringIO(text), sep='|')
            df_oth = df_oth[
                (df_oth['ETF']             == 'N') &
                (df_oth['Test Issue']      == 'N') &
                (df_oth['Exchange']        == 'N') &
                (df_oth['ACT Symbol'].str.len()  <= 5) &
                (~df_oth['ACT Symbol'].str.contains(r'[\^+$\.]', regex=True, na=False))
            ]
            tickers.update(df_oth['ACT Symbol'].dropna().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch other-exchange listing: {e}")

        return sorted(tickers)

    # ── Cache helpers ─────────────────────────────────────────────────────────
    def _cache_path(self) -> str:
        key = (
            f"{self._data_start}_{self.end}"
            f"_dv{int(self._min_dollar_vol/1e6)}"
            f"_rv{self._rvol_min}"
            f"_ch{self._chandelier_mult}"
        )
        h   = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:10]
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(_CACHE_DIR, f"bt_{h}.pkl")

    def _try_load_cache(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        try:
            print(f"  Loading cached data from {path} …")
            with open(path, 'rb') as f:
                cached = pickle.load(f)
            self._data = cached.get('data', {})
            print(f"  Loaded {len(self._data):,} symbols from cache.")
            return True
        except Exception as e:
            print(f"  Cache load failed ({e}), re-downloading …")
            return False

    def _save_cache(self) -> None:
        path = self._cache_path()
        try:
            with open(path, 'wb') as f:
                pickle.dump({'data': self._data}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Data cached → {path}")
        except Exception as e:
            print(f"  Cache save failed: {e}")

    # ── Data download ─────────────────────────────────────────────────────────
    def _download(self) -> None:
        """
        Download and indicator-enrich daily OHLCV for all institutional-grade
        US-listed equities.
        """
        tickers = self._fetch_universe()
        seed = int(hashlib.md5(self.start.encode(), usedforsecurity=False).hexdigest(), 16) & 0xFFFFFFFF
        rng  = random.Random(seed)
        rng.shuffle(tickers)
        print(
            f"  Universe : {len(tickers):,} US equities "
            f"(NASDAQ Global Select + Global Market + NYSE)\n"
            f"  Filters  : price>${self._min_price:.0f}  |  "
            f"vol>{self._min_volume/1e6:.0f}M shares  |  "
            f"20d avg dollar-vol>${self._min_dollar_vol/1e6:.0f}M  |  "
            f"{'all scanner-passed stocks' if self._scan_count <= 0 else f'top-{self._scan_count} by RVOL-weighted momentum'}"
        )
        print(f"  Downloading {len(tickers):,} tickers (from {self._data_start}) …")

        try:
            raw = yf.download(
                tickers,
                start=self._data_start,
                end=self.end,
                auto_adjust=True,
                progress=True,
                group_by='ticker',
                threads=True,
            )
        except Exception as e:
            raise RuntimeError(f"Data download failed: {e}")

        loaded = 0
        single = len(tickers) == 1
        for sym in tickers:
            try:
                df = raw.copy() if single else raw[sym].copy()
                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                if len(df) < MIN_CANDLES + 5:
                    continue
                df = apply_all(
                    df, RSI_PERIOD, ATR_PERIOD, MA_FAST, MA_SLOW,
                    SMA200_SLOPE_LOOKBACK, CHANDELIER_PERIOD
                )
                df['prev_high']         = df['high'].shift(1)
                df['avg_vol_20']        = df['volume'].rolling(20).mean()
                df['avg_dollar_vol_20'] = (
                    (df['close'] * df['volume']).rolling(20).mean()
                )
                self._data[sym] = df
                loaded += 1
            except Exception:
                continue

        print(f"  Loaded   : {loaded:,} symbols with ≥{MIN_CANDLES + 5} bars.")
        self._download_regime_data()

    def _download_regime_data(self) -> None:
        """
        Download SPY and VIX regime data fresh on every run (not cached).
        SPY regime requires close > SMA50 AND SMA50 > SMA200 (golden cross).
        This blocks both bear markets and early recoveries where the golden cross
        has not yet re-established, reducing false-breakout entries.
        """
        try:
            vix_raw = yf.download('^VIX', start=self.start, end=self.end,
                                  auto_adjust=True, progress=False)
            if not vix_raw.empty:
                vix_raw.columns = [
                    c[0].lower() if isinstance(c, tuple) else c.lower()
                    for c in vix_raw.columns
                ]
                self._vix_series = vix_raw['close']
            else:
                print("  WARNING: VIX data download returned empty DataFrame — VIX filter disabled")
        except Exception as e:
            print(f"  WARNING: VIX data download failed ({e}) — VIX filter disabled")

        if self._use_spy_filter:
            try:
                spy_raw = yf.download('SPY', start=self._data_start, end=self.end,
                                      auto_adjust=True, progress=False)
                if not spy_raw.empty:
                    spy_raw.columns = [
                        c[0].lower() if isinstance(c, tuple) else c.lower()
                        for c in spy_raw.columns
                    ]
                    sc    = spy_raw['close']
                    ma50  = sc.rolling(50).mean()
                    ma200 = sc.rolling(200).mean()
                    # Require all three conditions matching live engine _fetch_spy_trend():
                    # price > MA50 AND MA50 > MA200 AND SMA200 slope > 0.
                    # The slope check blocks recovery rallies where price has crossed
                    # above a still-falling SMA200 — the highest-false-breakout window.
                    sma200_slope = ma200.diff(SMA200_SLOPE_LOOKBACK)
                    self._spy_bull = (sc > ma50) & (ma50 > ma200) & (sma200_slope > 0)
                else:
                    print("  WARNING: SPY data download returned empty DataFrame — SPY regime filter disabled")
            except Exception as e:
                print(f"  WARNING: SPY data download failed ({e}) — SPY regime filter disabled")

    # ── Daily scanner simulation ──────────────────────────────────────────────
    def _daily_scan(self, today) -> List[Tuple[str, float]]:
        """
        Simulate the Alpaca top-gainers + most-actives screener with production pre-filters.
        Returns list of (symbol, rvol) tuples, sorted by composite score
        (% daily gain × RVOL) descending.  scan_count <= 0 means all scanner-passed
        symbols are returned.
        Fine signal rules are applied in _entry_signal.
        """
        scored: List[tuple] = []

        for sym, df in self._data.items():
            if today not in df.index:
                continue
            idx = df.index.get_loc(today)
            if idx < 1:
                continue

            row      = df.loc[today]
            prev_row = df.iloc[idx - 1]

            # Price and volume floor (mirrors Alpaca screener parameters)
            if row['close'] < self._min_price:
                continue
            if row['volume'] < self._min_volume:
                continue

            # Dollar-volume gate (20-day average).
            # Apply Friday multiplier matching live engine: Friday liquidity thins
            # after 12 PM ET and weekend gaps increase risk, so we require 2×
            # the normal threshold before taking a Friday position.
            avg_dvol = row.get('avg_dollar_vol_20', row['close'] * row['volume'])
            friday_mult = VOL_MULT_FRIDAY if pd.Timestamp(today).dayofweek == 4 else 1.0
            if pd.isna(avg_dvol) or avg_dvol < self._min_dollar_vol * friday_mult:
                continue

            # Trend filter (coarse pass)
            if pd.isna(row['MA50']) or pd.isna(row['MA200']):
                continue
            if not (row['close'] > row['MA50'] > row['MA200']):
                continue

            # RVOL filter (backtest threshold — daily close RVOL proxy)
            avg_vol = row.get('avg_vol_20', 0)
            if pd.isna(avg_vol) or avg_vol <= 0:
                continue
            rvol = row['volume'] / avg_vol
            if rvol < self._rvol_min:
                continue

            # Minimum daily gain filter — mirrors Alpaca screener's percent_change threshold
            pct = (row['close'] - prev_row['close']) / prev_row['close']
            if pct < SCAN_MIN_GAIN_PCT / 100.0:
                continue

            self._filter_stats['coarse_candidates'] += 1

            # Composite score mirroring live engine's _score_candidate() formula:
            # Trend(30) + RVOL(25) + Momentum(25); Liquidity(20) omitted (no spread in OHLCV).
            ma50  = row.get('MA50',  0)
            ma200 = row.get('MA200', 1) or 1
            rsi   = row.get('RSI',   50)
            prev_rsi = prev_row.get('RSI', rsi)

            sep           = (ma50 - ma200) / ma200 * 100
            ma_sep_pts    = max(0.0, min(sep / 6.0 * 22.0, 22.0))
            adx_val_scan  = row.get('ADX', float('nan'))
            adx_qual_pts  = (
                min(8.0, max(0.0, (float(adx_val_scan) - 25.0) / 25.0 * 8.0))
                if not pd.isna(adx_val_scan) and float(adx_val_scan) >= 25.0
                else 0.0
            )
            trend_pts   = ma_sep_pts + adx_qual_pts
            rvol_excess = max(0.0, rvol - self._rvol_min)
            rvol_pts    = min(25.0, rvol_excess / (5.0 - self._rvol_min) * 25.0)
            rsi_delta   = (rsi - prev_rsi) if not (pd.isna(rsi) or pd.isna(prev_rsi)) else 0.0
            accel       = min(max(rsi_delta / 5.0 * 15.0, 0.0), 15.0)
            rsi_lvl_pts = min(10.0, max(0.0, (rsi - RSI_THRESHOLD) / 20.0 * 10.0))
            momentum_pts = accel + rsi_lvl_pts

            score = trend_pts + rvol_pts + momentum_pts
            if score < SCAN_MIN_SCORE:
                continue
            scored.append((sym, score, rvol))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored if self._scan_count <= 0 else scored[:self._scan_count]
        return [(sym, rvol) for sym, _, rvol in selected]

    # ── Signal check ─────────────────────────────────────────────────────────
    @staticmethod
    def _entry_signal(row: pd.Series, prev_rsi: float, rvol: float,
                      rvol_min: float,
                      min_trend_sep: float = MIN_TREND_SEP,
                      flags: dict = None) -> bool:
        """Full 12-rule production filter (daily-bar approximation).

        flags controls which optional rules are active.  None → all production
        defaults.
        """
        g = flags if flags is not None else {}

        # ── Mandatory (never toggled off) ──────────────────────────────────
        c_trend = row['close'] > row['MA50'] > row['MA200']
        c_rvol  = rvol >= rvol_min

        # ── Togglable rules (optimizer-discoverable) ───────────────────────
        c_slope = (
            not pd.isna(row['SMA200_SLOPE']) and row['SMA200_SLOPE'] > 0
        ) if g.get('use_slope', False) else True

        c_trend_sep = (
            row['MA200'] > 0
            and (row['MA50'] - row['MA200']) / row['MA200'] >= min_trend_sep
        ) if g.get('use_trend_sep', True) else True

        c_orb = (
            not pd.isna(row['prev_high']) and row['close'] > row['prev_high']
        ) if g.get('use_orb', True) else True

        rsi_delta   = row['RSI'] - prev_rsi
        c_rsi_rise  = (row['RSI'] > prev_rsi)           if g.get('use_rsi_rise',  False) else True
        c_rsi_delta = (rsi_delta >= RSI_MIN_DELTA)       if g.get('use_rsi_delta', True) else True
        c_rsi_lvl   = (row['RSI'] > RSI_THRESHOLD)       if g.get('use_rsi_lvl',   True) else True

        adx_val   = row.get('ADX',    float('nan'))
        h200_val  = row.get('HIGH200', float('nan'))
        ema20_val = row.get('EMA20',  float('nan'))

        c_adx = (
            not pd.isna(adx_val) and adx_val > ADX_THRESHOLD
        ) if g.get('use_adx', True) else True

        c_52w_high = (
            not pd.isna(h200_val) and h200_val > 0
            and row['close'] >= h200_val * HIGH200_MIN_PCT
        ) if g.get('use_52w_high', True) else True

        c_ma20 = (
            not pd.isna(ema20_val) and row['close'] > ema20_val
        ) if g.get('use_ma20', False) else True

        return (
            c_trend and c_rvol
            and c_slope and c_trend_sep and c_orb
            and c_rsi_rise and c_rsi_delta and c_rsi_lvl
            and c_adx and c_52w_high and c_ma20
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> BacktestResult:
        if self._use_cache and self._try_load_cache():
            # Cache only stores stock data — always refresh regime signals live
            self._download_regime_data()
        else:
            self._download()   # _download() calls _download_regime_data() at the end
            if self._use_cache:
                self._save_cache()
        if not self._data:
            raise RuntimeError("No usable data downloaded. Check tickers / dates.")
        return self._run_loop()

    # ── Optimizer helpers ─────────────────────────────────────────────────────
    def _prepare_optimizer_signals(self) -> None:
        """Lazily add EMA20 column to cached DataFrames for the optimizer.

        ADX and HIGH200 are now computed by apply_all() and already present.
        EMA20 is an optimizer-only candidate signal (not in production apply_all).
        Subsequent calls are no-ops (column already present).
        """
        for df in self._data.values():
            if 'EMA20' not in df.columns:
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

    def _precompute_scans_enriched(self) -> dict:
        """Pre-compute _daily_scan + row data for the signal optimizer.

        Returns dict[date → list[(sym, rvol, row_dict, prev_rsi)]] where every
        entry has already passed the NaN and gap-cap pre-checks.  Call once
        after run() and _prepare_optimizer_signals(); pass the result to
        run_with_flags(precomputed_scans=…) to skip the O(n_symbols × n_days)
        scan overhead on every combination (50-100× speedup).
        """
        if self._all_dates is None:
            self._all_dates   = sorted(set(d for df in self._data.values() for d in df.index))
            self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}

        result: dict = {}
        for today in self._all_dates:
            enriched = []
            for sym, rvol in self._daily_scan(today):
                df = self._data.get(sym)
                if df is None or today not in df.index:
                    continue
                idx = df.index.get_loc(today)
                if idx < 1:
                    continue
                row = df.loc[today]
                if pd.isna(row[_REQUIRED_ENTRY_COLS]).any():
                    continue
                if float(row['open']) > float(row['prev_high']) * (1 + GAP_MAX_PCT):
                    continue
                prev_rsi = float(df.iloc[idx - 1]['RSI'])
                row_data = {c: row.get(c, float('nan')) for c in _PRECOMPUTE_COLS}
                enriched.append((sym, rvol, row_data, prev_rsi))
            result[today] = enriched
        return result

    def run_with_flags(self, flags: dict, precomputed_scans: dict = None) -> BacktestResult:
        """Run _run_loop with custom signal flags; data must already be loaded.

        Intended for the optimizer: call run() once to load data, then call
        run_with_flags() repeatedly without re-downloading.  Pass precomputed_scans
        (from _precompute_scans_enriched) to skip the O(n_symbols × n_days) scan
        overhead and achieve 50-100× speedup.
        """
        if not self._data:
            raise RuntimeError("No data loaded. Call run() first.")
        # Reset accumulators
        for k in list(self._filter_stats):
            self._filter_stats[k] = 0.0 if k == 'total_commissions' else 0
        return self._run_loop(flags=flags, precomputed_scans=precomputed_scans)

    def _run_loop(self, flags: dict = None, precomputed_scans: dict = None) -> BacktestResult:
        """
        Strategy loop:
        - Chandelier trailing stop + hard stop + break-even floor exit
        - ATR-based position sizing with entry slippage (BACKTEST_SLIPPAGE)
        - Commission per order configurable via BACKTEST_COMMISSION_PER_ORDER (default $0.00)
        - Trading-bar hold count (not calendar days)
        - T+1 settlement: sale proceeds are not available for re-entry until the
          next trading day, matching the live cash-account constraint.
        """
        trades: List[Trade]             = []
        open_positions: Dict            = {}
        settled_cash                    = self.capital   # cash not tied up in open positions
        # T+1 settlement queue: list of (settle_date, amount) — sale proceeds
        # become available on the next trading day, not on the exit day.
        pending_settlements: list       = []
        equity_curve: Dict[date, float] = {}

        # Cache across repeated _run_loop calls (optimizer runs same data, different flags)
        if self._all_dates is None:
            self._all_dates   = sorted(set(d for df in self._data.values() for d in df.index))
            self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}
        all_dates    = self._all_dates
        _date_to_idx = self._date_to_idx
        trade_start = pd.Timestamp(self.start)

        for today in all_dates:
            # ── T+1: credit any settlements due today or earlier ──────────
            remaining_settlements = []
            for settle_date, amount in pending_settlements:
                if today >= settle_date:
                    settled_cash += amount
                else:
                    remaining_settlements.append((settle_date, amount))
            pending_settlements = remaining_settlements

            # ── Exit checks ───────────────────────────────────────────────
            for sym in list(open_positions.keys()):
                t  = open_positions[sym]
                df = self._data.get(sym)
                if df is None or today not in df.index:
                    continue

                row = df.loc[today]

                # Increment trading-bar count (real trading days, not calendar)
                t.__dict__['_bars_held'] = t.__dict__.get('_bars_held', 0) + 1
                bars_held = t.__dict__['_bars_held']

                atr_chand  = t.__dict__.get('_atr_chand', float(row['ATR']))
                chand_dist = t.__dict__.get('_chand_dist', atr_chand * self._chandelier_mult)

                # Track peak high since entry (Chandelier ratchet)
                peak_high = max(t.__dict__.get('_peak_high', t.entry_price),
                                float(row['high']))
                t.__dict__['_peak_high'] = peak_high

                # Chandelier stop: fixed dollar distance from peak — matches Alpaca
                # TrailingStopOrderRequest(trail_price=chand_dist) semantics exactly.
                # The stop rises as the stock rises but always stays chand_dist below
                # the peak, never widening or narrowing in percentage terms.
                chand_stop = peak_high - chand_dist

                # Hard stop: flat 7% below entry
                hard_stop = t.entry_price * (1 - HARD_STOP_PCT)

                # Break-even floor: once up BREAK_EVEN_PCT, stop ≥ entry
                if peak_high >= t.entry_price * (1 + self._break_even_pct):
                    be_stop = t.entry_price
                else:
                    be_stop = 0.0   # inactive until profit threshold reached

                effective_stop = max(chand_stop, hard_stop, be_stop)

                profit_pct = (float(row['close']) - t.entry_price) / t.entry_price

                exit_reason = None
                exit_price  = float(row['close'])

                if float(row['low']) <= effective_stop:
                    exit_reason = "chandelier_stop"
                    # If the stock gapped below the stop at open, fill at open price —
                    # TRAIL orders execute at (or near) the open on gap-down sessions,
                    # not at the stop level. Using min() prevents the optimistic
                    # assumption that we always filled exactly at the stop price.
                    exit_price  = round(min(float(row['open']), effective_stop), 4)
                elif pd.Timestamp(today).dayofweek == 4 and profit_pct < FRIDAY_MIN_PROFIT_PCT:
                    # Friday afternoon close: market order — apply exit slippage.
                    exit_reason = "friday_close"
                    exit_price  = round(float(row['close']) * (1 - BACKTEST_EXIT_SLIPPAGE), 4)
                elif bars_held >= self.hold_bars and profit_pct < self._profit_min_threshold:
                    # Velocity exit: market order — apply exit slippage.
                    exit_reason = "velocity_exit"
                    exit_price  = round(float(row['close']) * (1 - BACKTEST_EXIT_SLIPPAGE), 4)

                if exit_reason:
                    t.exit_date   = today.date() if hasattr(today, 'date') else today
                    t.exit_price  = exit_price
                    t.exit_reason = exit_reason
                    # T+1 settlement: queue proceeds for the next trading day so the
                    # backtest matches the live cash-account constraint — proceeds from
                    # a sale cannot fund a new entry on the same day.
                    _proceeds = t.entry_price * t.qty + t.pnl
                    _today_idx = _date_to_idx.get(today, -1)
                    if _today_idx + 1 < len(all_dates):
                        pending_settlements.append((all_dates[_today_idx + 1], _proceeds))
                    else:
                        settled_cash += _proceeds   # last trading day — credit immediately
                    self._filter_stats['total_commissions'] += t.round_trip_commission
                    if exit_reason == "friday_close":
                        self._filter_stats['friday_closes'] += 1
                    trades.append(t)
                    del open_positions[sym]

            # ── Regime gates ──────────────────────────────────────────────
            skip_entries  = False
            past_start    = pd.Timestamp(today) >= trade_start

            if self._use_vix_filter and self._vix_series is not None and past_start:
                try:
                    vix_val = self._vix_series.get(today)
                    if vix_val is not None and float(vix_val) > VIX_THRESHOLD:
                        skip_entries = True
                        self._filter_stats['vix_blocked_days'] += 1
                except Exception:
                    pass

            if (not skip_entries and self._use_spy_filter
                    and self._spy_bull is not None and past_start):
                try:
                    bull = self._spy_bull.get(today)
                    if bull is not None and not bool(bull):
                        skip_entries = True
                        self._filter_stats['spy_blocked_days'] += 1
                except Exception:
                    pass

            # MTM equity for ATR risk sizing and dynamic max-position capacity —
            # mirrors the live engine which uses
            # NetLiquidation (settled cash + current market value of open positions).
            # Using realized-only equity would undersize after winning trades and
            # oversize after losing trades while positions are still held open.
            _pre_entry_open_mtm = sum(
                float(self._data[s].loc[today]['close']) * open_positions[s].qty
                for s in open_positions
                if s in self._data and today in self._data[s].index
            )
            equity_mtm = settled_cash + _pre_entry_open_mtm

            # Mirror live engine:
            # - maximum simultaneous positions compounds with total equity;
            # - new entry slots are additionally constrained by settled cash.
            dynamic_max_pos = (
                min(int(equity_mtm / MIN_BUCKET_SIZE), self.max_pos)
                if equity_mtm >= MIN_BUCKET_SIZE else 0
            )

            if past_start:
                self._filter_stats['scan_days'] += 1

            if (
                not skip_entries
                and len(open_positions) < dynamic_max_pos
                and past_start
            ):
                # Build today's candidate list.
                # Fast path: use pre-computed scan+enrichment (optimizer).
                # Slow path: compute on-the-fly (normal backtest).
                if precomputed_scans is not None:
                    _today_cands = precomputed_scans.get(today, [])
                else:
                    _today_cands = []
                    for sym, rvol in self._daily_scan(today):
                        df = self._data.get(sym)
                        if df is None or today not in df.index:
                            continue
                        idx = df.index.get_loc(today)
                        if idx < 1:
                            continue
                        row      = df.loc[today]
                        prev_rsi = float(df.iloc[idx - 1]['RSI'])
                        if pd.isna(row[_REQUIRED_ENTRY_COLS]).any():
                            continue
                        # Gap cap: skip if already gapped beyond GAP_MAX_PCT
                        if float(row['open']) > float(row['prev_high']) * (1 + GAP_MAX_PCT):
                            continue
                        _today_cands.append((sym, rvol, row, prev_rsi))

                for sym, rvol, row, prev_rsi in _today_cands:
                    capacity_slots = max(0, dynamic_max_pos - len(open_positions))
                    cash_slots = int(settled_cash / MIN_BUCKET_SIZE) if settled_cash >= MIN_BUCKET_SIZE else 0
                    entry_slots = min(capacity_slots, cash_slots)
                    if sym in open_positions or entry_slots <= 0:
                        if sym not in open_positions:
                            self._filter_stats['entries_skipped_full'] += 1
                        continue

                    if self._entry_signal(row, prev_rsi, rvol, self._rvol_min,
                                          flags=flags):
                        self._filter_stats['fine_signals'] += 1

                        # Entry at open or prev_high, plus BACKTEST_SLIPPAGE to simulate impact
                        raw_entry   = max(float(row['open']), float(row['prev_high']))
                        entry_price = round(raw_entry * (1 + BACKTEST_SLIPPAGE), 4)

                        # ATR-based position sizing: risk 2% of equity per trade
                        atr_chand_val   = float(row['ATR_CHAND'])
                        chand_dist      = atr_chand_val * self._chandelier_mult
                        hard_stop_dist  = entry_price * HARD_STOP_PCT
                        # The tighter stop is the one that fires first → defines risk
                        risk_stop_dist  = min(chand_dist, hard_stop_dist)
                        risk_stop_dist  = max(risk_stop_dist, 0.01)  # floor at 1¢

                        # Bucket = settled_cash / cash-qualified entry slots.  Max
                        # position count compounds with equity, but spending remains
                        # constrained by settled cash.
                        bucket        = settled_cash * BUCKET_CASH_PCT / entry_slots

                        risk_dollars  = equity_mtm * RISK_PER_TRADE_PCT
                        qty_risk      = risk_dollars / risk_stop_dist
                        qty_bucket    = bucket / entry_price
                        qty           = int(min(qty_risk, qty_bucket))   # whole shares only

                        if qty < 1:   # no cash left or qty rounded to zero
                            continue

                        # Deduct actual cost from settled cash immediately
                        settled_cash -= entry_price * qty

                        t = Trade(
                            symbol      = sym,
                            entry_date  = today.date() if hasattr(today, 'date') else today,
                            entry_price = entry_price,
                            qty         = qty,
                            round_trip_commission = self._round_trip_cost,
                        )
                        # Store dollar distance — matches Alpaca trail_price semantics.
                        # Stop always stays exactly chand_dist below the peak,
                        # rising with the price but never changing in dollar terms.
                        t.__dict__['_chand_dist'] = chand_dist
                        t.__dict__['_atr_chand']  = atr_chand_val
                        t.__dict__['_peak_high']      = entry_price
                        t.__dict__['_bars_held']      = 0
                        open_positions[sym]        = t
                        self._filter_stats['entries_taken'] += 1

            # Mark-to-market equity: settled cash + current market value of open positions
            # + any sale proceeds in transit (pending T+1 settlement).
            # Including pending amounts prevents artificial equity dips on exit days —
            # the cash is still ours, just not yet in the "settled" bucket.
            open_mtm = sum(
                float(self._data[s].loc[today]['close']) * open_positions[s].qty
                for s in open_positions
                if s in self._data and today in self._data[s].index
            )
            _pending_amt = sum(amt for _, amt in pending_settlements)
            equity_curve[today] = settled_cash + open_mtm + _pending_amt

        # Close any positions still open at end of period
        for sym, t in open_positions.items():
            df = self._data.get(sym)
            if df is not None and not df.empty:
                t.exit_price  = float(df['close'].iloc[-1])
                t.exit_date   = df.index[-1].date()
                t.exit_reason = "end_of_period"
                settled_cash += t.entry_price * t.qty + t.pnl
                self._filter_stats['total_commissions'] += t.round_trip_commission
                trades.append(t)

        eq_series = pd.Series(equity_curve).sort_index()
        metrics   = self._compute_metrics(trades, eq_series)
        return BacktestResult(
            trades=trades,
            equity_curve=eq_series,
            metrics=metrics,
            filter_stats=dict(self._filter_stats),
        )

    # ── Performance metrics ───────────────────────────────────────────────────
    @staticmethod
    def _compute_metrics(trades: List[Trade], equity: pd.Series) -> Dict:
        if not trades:
            return {}

        completed = [t for t in trades if t.exit_price is not None]
        pnls      = [t.pnl     for t in completed]   # net pnl (post-commission)
        pnl_pcts  = [t.pnl_pct for t in completed]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p < 0]

        eq_vals  = equity.values.astype(float)
        peak     = np.maximum.accumulate(eq_vals)
        drawdown = (eq_vals - peak) / np.where(peak == 0, 1, peak)
        max_dd   = drawdown.min()

        daily_ret = equity.pct_change().dropna()
        sharpe    = (
            (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            if daily_ret.std() > 0 else 0.0
        )

        total_return = (
            (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
            if len(equity) > 1 else 0.0
        )

        avg_hold_bars = (
            np.mean([t.__dict__.get('_bars_held', 0) for t in completed])
            if completed else 0.0
        )

        return {
            "total_trades":     len(pnls),
            "win_rate":         len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl":        sum(pnls),
            "total_return_pct": total_return * 100,
            "avg_win":          np.mean(wins)    if wins    else 0.0,
            "avg_loss":         np.mean(losses)  if losses  else 0.0,
            "avg_win_pct":      np.mean([p for p in pnl_pcts if p > 0]) * 100 if wins else 0.0,
            "avg_loss_pct":     np.mean([p for p in pnl_pcts if p < 0]) * 100 if losses else 0.0,
            "avg_hold_bars":    avg_hold_bars,
            "profit_factor":    (
                sum(wins) / abs(sum(losses))
                if losses and sum(losses) != 0 else float('inf')
            ),
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio":     sharpe,
            "exit_reasons":     pd.Series(
                [t.exit_reason for t in completed]
            ).value_counts().to_dict(),
        }

    # ── Console report ────────────────────────────────────────────────────────
    @staticmethod
    def print_report(result: BacktestResult, capital: float = BACKTEST_INITIAL_CAPITAL) -> None:
        m  = result.metrics
        fs = result.filter_stats
        if not m:
            print("No trades executed.")
            VelocityBacktest._print_filter_stats(fs)
            return

        final_equity = capital + m['total_pnl']

        print("\n" + "=" * 65)
        print("  VELOCITY STRATEGY — FORWARD BACKTEST REPORT")
        print("  12-filter entry | Chandelier stop | Break-even floor")
        print("=" * 65)
        print("  *** SURVIVORSHIP BIAS WARNING ***")
        print("  Universe = current NASDAQ/NYSE listing.  Bankrupt and")
        print("  delisted tickers from the backtest window are absent.")
        print("  Reported returns are likely overstated by 5-15 ppts.")
        print("-" * 65)
        print(f"  Starting Capital  : ${capital:,.2f}")
        print(f"  Final Equity      : ${final_equity:,.2f}")
        print(f"  Total Return      : {m['total_return_pct']:.2f}%  (net of commissions)")
        print(f"  Total P&L (net)   : ${m['total_pnl']:,.2f}  "
              f"(comm: ${fs.get('total_commissions', 0):,.2f})")
        print("-" * 65)
        print(f"  Total Trades      : {m['total_trades']}")
        print(f"  Win Rate          : {m['win_rate']:.1%}")
        print(f"  Avg Win (net)     : ${m['avg_win']:,.2f}  ({m['avg_win_pct']:.2f}%)")
        print(f"  Avg Loss (net)    : ${m['avg_loss']:,.2f}  ({m['avg_loss_pct']:.2f}%)")
        print(f"  Profit Factor     : {m['profit_factor']:.2f}")
        print(f"  Avg Hold          : {m['avg_hold_bars']:.1f} trading bars")
        print("-" * 65)
        print(f"  Max Drawdown      : {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio      : {m['sharpe_ratio']:.2f}")
        print("  Exit Breakdown    :")
        for reason, count in sorted(m['exit_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason:<22}: {count}")
        print("=" * 65)

        VelocityBacktest._print_filter_stats(fs)
        VelocityBacktest._print_yearly_breakdown(result, capital)
        VelocityBacktest._print_monthly_breakdown(result, capital)

    @staticmethod
    def _print_yearly_breakdown(result: BacktestResult, capital: float) -> None:
        """Print per-calendar-year performance: return, trades, win rate, profit factor, max DD."""
        eq = result.equity_curve
        if eq.empty:
            return
        trades = [t for t in result.trades if t.exit_price is not None]
        if not trades:
            return

        years = sorted(set(eq.index.year) if hasattr(eq.index, 'year')
                       else set(d.year for d in eq.index))

        print("\n  YEAR-BY-YEAR BREAKDOWN")
        print("  " + "-" * 83)
        print(f"  {'Year':<6} {'Start Eq':>10} {'End Eq':>10} {'Return':>8} "
              f"{'Trades':>7} {'WinRate':>8} {'PF':>6} {'MaxDD':>7}")
        print("  " + "-" * 83)

        running_equity = capital
        for yr in years:
            # Equity slice for this calendar year
            if hasattr(eq.index, 'year'):
                yr_eq = eq[eq.index.year == yr]
            else:
                yr_eq = eq[[d.year == yr for d in eq.index]]

            if yr_eq.empty:
                continue

            start_eq = running_equity
            end_eq   = float(yr_eq.iloc[-1])

            # Trades closed in this year (needed for the zero-activity filter below)
            yr_trades = [t for t in trades
                         if (t.exit_date.year if hasattr(t.exit_date, 'year')
                             else t.exit_date) == yr]

            # Suppress years with no trading activity — these are pre-backtest-start
            # warm-up periods where the equity curve exists but no trades were placed.
            if len(yr_trades) == 0 and abs(end_eq - start_eq) < 0.01:
                running_equity = end_eq
                continue

            ret_pct  = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0.0

            # Max drawdown within year
            vals = yr_eq.values.astype(float)
            peak = np.maximum.accumulate(vals)
            dd   = (vals - peak) / np.where(peak == 0, 1, peak)
            max_dd = dd.min() * 100

            n_trades = len(yr_trades)
            wins     = [t for t in yr_trades if t.pnl > 0]
            losses   = [t for t in yr_trades if t.pnl < 0]
            win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
            pf_num   = sum(t.pnl for t in wins)
            pf_den   = abs(sum(t.pnl for t in losses))
            pf       = pf_num / pf_den if pf_den > 0 else float('inf')

            arrow = "▲" if ret_pct >= 0 else "▼"
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            print(f"  {yr:<6} ${start_eq:>9,.0f} ${end_eq:>9,.0f} "
                  f"{arrow}{abs(ret_pct):>6.1f}%  "
                  f"{n_trades:>6}  {win_rate:>7.1%}  {pf_str:>6}  {max_dd:>6.2f}%")

            running_equity = end_eq

        print("  " + "-" * 83)
        total_ret = (running_equity - capital) / capital * 100 if capital > 0 else 0.0
        arrow = "▲" if total_ret >= 0 else "▼"
        print(f"  {'TOTAL':<6} ${capital:>9,.0f} ${running_equity:>9,.0f} "
              f"{arrow}{abs(total_ret):>6.1f}%  {len(trades):>6}")
        print()

    @staticmethod
    def _print_monthly_breakdown(result: BacktestResult, capital: float) -> None:
        """Print month-by-month equity with running cumulative return."""
        eq = result.equity_curve
        if eq.empty:
            return

        # Build a proper DatetimeIndex for resampling
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq = eq.copy()
            eq.index = pd.to_datetime([str(d) for d in eq.index])

        # Month-end equity (last observation per calendar month)
        monthly = eq.resample('ME').last().dropna()
        if monthly.empty:
            return

        print("  MONTHLY EQUITY PROGRESSION  (cumulative from start)")
        print("  " + "-" * 55)
        print(f"  {'Month':<10} {'Equity':>10} {'Month Ret':>10} {'Cumul Ret':>10}")
        print("  " + "-" * 55)

        prev_eq  = capital
        start_eq = capital
        for dt, eq_val in monthly.items():
            month_ret = (eq_val - prev_eq) / prev_eq * 100 if prev_eq > 0 else 0.0
            cumul_ret = (eq_val - start_eq) / start_eq * 100 if start_eq > 0 else 0.0
            m_arrow   = "▲" if month_ret >= 0 else "▼"
            c_arrow   = "▲" if cumul_ret >= 0 else "▼"
            print(f"  {dt.strftime('%Y-%m'):<10} ${eq_val:>9,.0f} "
                  f"{m_arrow}{abs(month_ret):>8.1f}%  "
                  f"{c_arrow}{abs(cumul_ret):>8.1f}%")
            prev_eq = eq_val

        print("  " + "-" * 55)
        print()

    @staticmethod
    def _print_filter_stats(fs: Dict) -> None:
        if not fs:
            return
        print("\n  FILTER FUNNEL")
        print("  " + "-" * 40)
        print(f"  Scan days           : {fs.get('scan_days', 0):,}")
        spy_d = fs.get('spy_blocked_days', 0)
        vix_d = fs.get('vix_blocked_days', 0)
        if spy_d:
            print(f"  SPY-blocked days    : {spy_d:,}")
        if vix_d:
            print(f"  VIX-blocked days    : {vix_d:,}")
        print(f"  Coarse candidates   : {fs.get('coarse_candidates', 0):,}  "
              f"(price/vol/trend/rvol pass)")
        print(f"  Fine signals        : {fs.get('fine_signals', 0):,}  "
              f"(full 12-rule pass)")
        print(f"  Entries taken       : {fs.get('entries_taken', 0):,}")
        skipped = fs.get('entries_skipped_full', 0)
        if skipped:
            print(f"  Skipped (pos full)  : {skipped:,}")
        fri = fs.get('friday_closes', 0)
        if fri:
            print(f"  Friday closes       : {fri:,}  (profit < {FRIDAY_MIN_PROFIT_PCT*100:.0f}% at week-end)")
        print()

    # ── Per-trade log ─────────────────────────────────────────────────────────
    @staticmethod
    def print_trades(result: BacktestResult, top_n: int = 20) -> None:
        """Print the top_n trades by absolute gross P&L."""
        trades = sorted(
            [t for t in result.trades if t.exit_price is not None],
            key=lambda t: abs(t.gross_pnl), reverse=True
        )[:top_n]
        if not trades:
            print("No completed trades.")
            return
        print(f"\n{'SYM':<6}  {'ENTRY':>10}  {'EXIT':>10}  "
              f"{'ENTRY $':>8}  {'EXIT $':>8}  {'NET PNL':>8}  {'PNL%':>7}  "
              f"{'BARS':>4}  {'REASON'}")
        print("-" * 90)
        for t in trades:
            bars = t.__dict__.get('_bars_held', '?')
            print(
                f"{t.symbol:<6}  {str(t.entry_date):>10}  {str(t.exit_date):>10}  "
                f"${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                f"${t.net_pnl:>7.2f}  {t.pnl_pct*100:>6.1f}%  "
                f"{str(bars):>4}  {t.exit_reason}"
            )
