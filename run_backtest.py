#!/usr/bin/env python
"""
CLI entry-point for the Velocity Strategy forward backtester.

Defaults to 2025-01-01 → today (out-of-sample relative to the
2023-2024 development period).

Key design decisions:
  • RVOL threshold 1.2× (daily close; not intraday 2.5×)
  • ATR-based position sizing (2% equity risk per trade)
  • Break-even stop floor at 4% profit
  • Trading-bar hold count (not calendar days)
  • 0.1% entry slippage; commission configurable via --commission-per-order
    (default $0.00 — Alpaca is commission-free)
  • Data caching to backtest/.cache/ (use --no-cache to force re-download)
  • Filter funnel stats printed after each run

Usage:
    venv/bin/python run_backtest.py
    venv/bin/python run_backtest.py --start 2025-01-01 --end 2026-05-01
    venv/bin/python run_backtest.py --capital 1400
    venv/bin/python run_backtest.py --no-spy-filter
    venv/bin/python run_backtest.py --rvol 1.2 --hold-bars 7
    venv/bin/python run_backtest.py --trades
    venv/bin/python run_backtest.py --no-cache        # force fresh download
"""

import argparse
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from backtest.strategy import VelocityBacktest
from src.config import (
    BACKTEST_SCAN_COUNT, BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION_PER_ORDER,
    BACKTEST_RVOL_MIN, BREAK_EVEN_PCT, SCAN_MIN_SCORE,
    CHANDELIER_MULT, CHANDELIER_PERIOD, PROFIT_MIN_THRESHOLD,
    BACKTEST_HOLD_BARS, BACKTEST_SLIPPAGE, BACKTEST_EXIT_SLIPPAGE,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, FRIDAY_MIN_PROFIT_PCT, DONCHIAN_PERIOD,
    BACKTEST_DONCHIAN_TOL_PCT, RSI_OVERSOLD_THRESHOLD, RSI_BOUNCE_MAX,
    RSI_MIN_DELTA, RSI_OVERSOLD_LOOKBACK,
    RISK_PER_TRADE_PCT, HARD_STOP_PCT,
    SCAN_MIN_DOLLAR_VOL, BACKTEST_MIN_BODY_PCT,
    SCAN_MIN_PRICE, SCAN_MIN_VOLUME,
)


def parse_args():
    p = argparse.ArgumentParser(description="Velocity Strategy Forward Backtester")
    break_even_default = f"{BREAK_EVEN_PCT:.0%}".replace("%", "%%")
    p.add_argument("--start",           default="2025-01-01",        help="Start date YYYY-MM-DD")
    p.add_argument("--end",             default=date.today().isoformat(), help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--capital",         default=BACKTEST_INITIAL_CAPITAL, type=float,
                   help=f"Starting capital USD (default: ${BACKTEST_INITIAL_CAPITAL:,.0f})")
    p.add_argument("--scan-count",      default=BACKTEST_SCAN_COUNT,      type=int,
                   help="Top-N daily scanner picks; 0 means all scanner-passed stocks (default: all)")
    p.add_argument("--commission-per-order", default=BACKTEST_COMMISSION_PER_ORDER, type=float,
                   help=f"Backtest commission assumption per order (default: ${BACKTEST_COMMISSION_PER_ORDER:.2f}; Alpaca is commission-free)")
    p.add_argument("--hold-bars",       default=BACKTEST_HOLD_BARS,   type=int,
                   help=f"Trading bars before velocity-exit check (default: {BACKTEST_HOLD_BARS} = matches live HOLD_TRADING_BARS)")
    p.add_argument("--rvol",            default=BACKTEST_RVOL_MIN,    type=float,
                   help=f"Daily RVOL threshold (default: {BACKTEST_RVOL_MIN}×)")
    p.add_argument("--min-score",       default=SCAN_MIN_SCORE,       type=float,
                   help=f"Minimum composite score gate 0-100 (default: {SCAN_MIN_SCORE:.0f})")
    p.add_argument("--break-even-pct",  default=BREAK_EVEN_PCT,       type=float,
                   help=f"Break-even stop activation threshold (default: {break_even_default})")
    p.add_argument("--profit-min",        default=PROFIT_MIN_THRESHOLD, type=float,
                   help=f"Velocity exit: min profit threshold (default: {PROFIT_MIN_THRESHOLD*100:.0f}%%)")
    p.add_argument("--friday-min-profit", default=FRIDAY_MIN_PROFIT_PCT, type=float,
                   help=f"Friday close: exit if profit < this pct (default: {FRIDAY_MIN_PROFIT_PCT*100:.0f}%%); set 0 to disable")
    p.add_argument("--chandelier-mult",    default=CHANDELIER_MULT,    type=float,
                   help=f"Chandelier ATR multiplier (default: {CHANDELIER_MULT})")
    p.add_argument("--chandelier-period", default=CHANDELIER_PERIOD,  type=int,
                   help=f"Chandelier ATR lookback period (default: {CHANDELIER_PERIOD})")
    p.add_argument("--donchian-period",  default=DONCHIAN_PERIOD,          type=int,
                   help=f"Donchian channel lookback period (default: {DONCHIAN_PERIOD})")
    p.add_argument("--donchian-tol",     default=BACKTEST_DONCHIAN_TOL_PCT, type=float,
                   help=f"Donchian floor tolerance (default: {BACKTEST_DONCHIAN_TOL_PCT*100:.0f}%%)")
    p.add_argument("--rsi-oversold",     default=RSI_OVERSOLD_THRESHOLD,   type=float,
                   help=f"RSI oversold threshold (default: {RSI_OVERSOLD_THRESHOLD})")
    p.add_argument("--rsi-bounce-max",   default=RSI_BOUNCE_MAX,           type=float,
                   help=f"RSI bounce max at entry (default: {RSI_BOUNCE_MAX})")
    p.add_argument("--rsi-min-delta",    default=RSI_MIN_DELTA,            type=float,
                   help=f"Minimum RSI point rise for momentum confirmation (default: {RSI_MIN_DELTA})")
    p.add_argument("--rsi-lookback",     default=RSI_OVERSOLD_LOOKBACK,    type=int,
                   help=f"Days to look back for RSI oversold condition (default: {RSI_OVERSOLD_LOOKBACK})")
    p.add_argument("--risk-per-trade",   default=RISK_PER_TRADE_PCT,       type=float,
                   help=f"Equity fraction risked per trade (default: {RISK_PER_TRADE_PCT:.0%})")
    p.add_argument("--hard-stop-pct",    default=HARD_STOP_PCT,            type=float,
                   help=f"Hard stop loss from entry (default: {HARD_STOP_PCT:.0%})")
    p.add_argument("--spy-filter",       action="store_true",
                   help="Enable SPY regime filter (default: OFF — Donchian bounce improves without it)")
    p.add_argument("--min-dollar-vol",  default=SCAN_MIN_DOLLAR_VOL, type=float,
                   help=f"Minimum 20-day avg dollar volume (default: ${SCAN_MIN_DOLLAR_VOL/1e6:.0f}M)")
    p.add_argument("--min-body-pct",    default=BACKTEST_MIN_BODY_PCT, type=float,
                   help=f"Day-strength: min candle body pct close/open-1 (default: {BACKTEST_MIN_BODY_PCT:.1%}; 0=disable)")
    p.add_argument("--min-price",       default=SCAN_MIN_PRICE, type=float,
                   help=f"Minimum stock price filter (default: ${SCAN_MIN_PRICE:.0f})")
    p.add_argument("--min-volume",      default=SCAN_MIN_VOLUME, type=float,
                   help=f"Minimum 20-day avg share volume (default: {SCAN_MIN_VOLUME/1e6:.0f}M)")
    p.add_argument("--vix-filter",      action="store_true",
                   help="Enable VIX > 35 regime gate")
    p.add_argument("--trades",          action="store_true",
                   help="Print top-20 trade log after summary")
    p.add_argument("--no-cache",        action="store_true",
                   help="Force fresh data download (ignore backtest/.cache/)")
    return p.parse_args()


def main():
    args = parse_args()
    print("\nVELOCITY STRATEGY — FORWARD BACKTEST")
    print(f"{'─' * 50}")
    print(f"  Period        : {args.start} → {args.end}")
    print(f"  Capital       : ${args.capital:,.2f}")
    _init_slots      = min(int(args.capital / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) if args.capital >= MIN_BUCKET_SIZE else 0
    _init_bucket_str = f"${args.capital / _init_slots:,.2f}" if _init_slots > 0 else "N/A"
    print(f"  Max pos       : {MAX_POSITIONS_CAP} cap  |  Dynamic max = floor(equity / ${MIN_BUCKET_SIZE:.0f}/slot)  |  Initial slots={_init_slots}, bucket≈{_init_bucket_str}")
    print("  Entry rules   : 12-filter production screener")
    print(f"  RVOL min      : {args.rvol:.1f}× (daily close proxy)")
    print(f"  Min score     : {args.min_score:.0f}/100 composite gate")
    print(f"  Exit          : Chandelier (ATR{CHANDELIER_PERIOD}×{CHANDELIER_MULT}) + 7% hard stop + {args.break_even_pct:.0%} break-even")
    print(f"  Velocity exit : profit_min {args.profit_min:.0%} after {args.hold_bars} bars")
    print(f"  Hold bars     : {args.hold_bars} trading days before velocity check")
    print("  Position size : ATR-based (2% equity risk) capped by bucket")
    print(f"  Slippage      : {BACKTEST_SLIPPAGE:.1%} entry, {BACKTEST_EXIT_SLIPPAGE:.1%} exit (mkt orders)  |  Commission: ${args.commission_per_order*2:.2f}/round-trip")
    print(f"  SPY filter    : {'ON (EMA50 soft regime)' if args.spy_filter else 'OFF (mean-reversion default)'}")
    print(f"  VIX filter    : {'ON (VIX > 35 blocks entries)' if args.vix_filter else 'OFF'}")
    print(f"  Cache         : {'OFF (forced re-download)' if args.no_cache else 'ON (backtest/.cache/)'}")
    print()

    bt = VelocityBacktest(
        start              = args.start,
        end                = args.end,
        capital            = args.capital,
        scan_count         = args.scan_count,
        commission_per_order = args.commission_per_order,
        hold_bars          = args.hold_bars,
        rvol_min           = args.rvol,
        min_score          = args.min_score,
        break_even_pct     = args.break_even_pct,
        profit_min_threshold = args.profit_min,
        friday_min_profit  = args.friday_min_profit,
        chandelier_mult    = args.chandelier_mult,
        chandelier_period  = args.chandelier_period,
        donchian_period    = args.donchian_period,
        donchian_tol_pct   = args.donchian_tol,
        rsi_oversold_threshold = args.rsi_oversold,
        rsi_bounce_max     = args.rsi_bounce_max,
        rsi_min_delta      = args.rsi_min_delta,
        rsi_oversold_lookback = args.rsi_lookback,
        risk_per_trade_pct = args.risk_per_trade,
        hard_stop_pct      = args.hard_stop_pct,
        min_dollar_vol     = args.min_dollar_vol,
        min_body_pct       = args.min_body_pct,
        min_price          = args.min_price,
        min_volume         = args.min_volume,
        use_spy_filter     = args.spy_filter,
        use_vix_filter     = args.vix_filter,
        use_cache          = not args.no_cache,
    )
    result = bt.run()
    VelocityBacktest.print_report(result, capital=args.capital)

    if args.trades:
        VelocityBacktest.print_trades(result)


if __name__ == "__main__":
    main()
