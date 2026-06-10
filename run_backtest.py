#!/usr/bin/env python
"""
CLI entry-point for the Alligator Swing Strategy forward backtester.

Defaults to 2021-01-01 → today (multi-year out-of-sample run).

Key design decisions:
  • Entry: Bill Williams Alligator SMMA crossover (periods 5/8/13, offsets 3/5/8)
  • RVOL threshold 1.2× (daily close proxy; live uses intraday 2.5×)
  • ATR-based position sizing (2% equity risk per trade)
  • Break-even stop floor at 6% profit
  • Exit: Chandelier trail + hard stop + break-even + Alligator reversal
    (no velocity time-exit, no forced Friday close)
  • 0.1% entry slippage; commission configurable via --commission-per-order
    (default $0.00 — Alpaca is commission-free)
  • Data caching to backtest/.cache/ (use --no-cache to force re-download)
  • Filter funnel stats printed after each run

Usage:
    venv/bin/python run_backtest.py
    venv/bin/python run_backtest.py --start 2021-01-01 --end 2026-06-01
    venv/bin/python run_backtest.py --capital 1400
    venv/bin/python run_backtest.py --spy-filter
    venv/bin/python run_backtest.py --rvol 1.2 --chandelier-mult 2.5
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
    CHANDELIER_MULT, CHANDELIER_PERIOD,
    BACKTEST_SLIPPAGE, BACKTEST_EXIT_SLIPPAGE,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE,
    RSI_MIN_DELTA,
    RISK_PER_TRADE_PCT, HARD_STOP_PCT,
    SCAN_MIN_DOLLAR_VOL,
    SCAN_MIN_PRICE, SCAN_MIN_VOLUME,
)


def parse_args():
    p = argparse.ArgumentParser(description="Alligator Swing Strategy Forward Backtester")
    break_even_default = f"{BREAK_EVEN_PCT:.0%}".replace("%", "%%")
    p.add_argument("--start",           default="2021-01-01",        help="Start date YYYY-MM-DD")
    p.add_argument("--end",             default=date.today().isoformat(), help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--capital",         default=BACKTEST_INITIAL_CAPITAL, type=float,
                   help=f"Starting capital USD (default: ${BACKTEST_INITIAL_CAPITAL:,.0f})")
    p.add_argument("--max-pos",         default=MAX_POSITIONS_CAP,        type=int,
                   help=f"Max simultaneous positions cap (default: {MAX_POSITIONS_CAP})")
    p.add_argument("--bucket-size",     default=MIN_BUCKET_SIZE,          type=float,
                   help=f"Min equity per position slot in $ (default: ${MIN_BUCKET_SIZE:.0f})")
    p.add_argument("--scan-count",      default=BACKTEST_SCAN_COUNT,      type=int,
                   help="Top-N daily scanner picks; 0 means all scanner-passed stocks (default: all)")
    p.add_argument("--commission-per-order", default=BACKTEST_COMMISSION_PER_ORDER, type=float,
                   help=f"Backtest commission assumption per order (default: ${BACKTEST_COMMISSION_PER_ORDER:.2f}; Alpaca is commission-free)")
    p.add_argument("--rvol",            default=BACKTEST_RVOL_MIN,    type=float,
                   help=f"Daily RVOL threshold (default: {BACKTEST_RVOL_MIN}×)")
    p.add_argument("--min-score",       default=SCAN_MIN_SCORE,       type=float,
                   help=f"Minimum composite score gate 0-100 (default: {SCAN_MIN_SCORE:.0f})")
    p.add_argument("--break-even-pct",  default=BREAK_EVEN_PCT,       type=float,
                   help=f"Break-even stop activation threshold (default: {break_even_default})")
    p.add_argument("--chandelier-mult",    default=CHANDELIER_MULT,    type=float,
                   help=f"Chandelier ATR multiplier (default: {CHANDELIER_MULT})")
    p.add_argument("--chandelier-period", default=CHANDELIER_PERIOD,  type=int,
                   help=f"Chandelier ATR lookback period (default: {CHANDELIER_PERIOD})")
    p.add_argument("--rsi-min-delta",    default=RSI_MIN_DELTA,            type=float,
                   help=f"Minimum RSI point rise for momentum confirmation (default: {RSI_MIN_DELTA})")
    p.add_argument("--risk-per-trade",   default=RISK_PER_TRADE_PCT,       type=float,
                   help=f"Equity fraction risked per trade (default: {RISK_PER_TRADE_PCT:.0%})")
    p.add_argument("--hard-stop-pct",    default=HARD_STOP_PCT,            type=float,
                   help=f"Hard stop loss from entry (default: {HARD_STOP_PCT:.0%})")
    p.add_argument("--spy-filter",       action="store_true",
                   help="Enable SPY regime filter (default: OFF)")
    p.add_argument("--min-dollar-vol",  default=SCAN_MIN_DOLLAR_VOL, type=float,
                   help=f"Minimum 20-day avg dollar volume (default: ${SCAN_MIN_DOLLAR_VOL/1e6:.0f}M)")
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
    p.add_argument("--sweep",           default="",
                   metavar="PARAM=V1,V2,...",
                   help="In-process sweep: load data once, simulate for each value. "
                        "PARAM is the CLI flag name without '--' (e.g. chandelier-mult=2.0,2.5,3.0). "
                        "Parameters that change indicator columns (chandelier-period) "
                        "require full recomputation and cannot be swept this way.")
    return p.parse_args()


# Maps CLI flag name → (instance_attr, type, needs_indicator_recompute)
_SWEEP_PARAM_MAP = {
    "rsi-min-delta":    ("_rsi_min_delta",            float, False),
    "chandelier-mult":  ("_chandelier_mult",          float, False),
    "hard-stop-pct":    ("_hard_stop_pct",            float, False),
    "break-even-pct":   ("_break_even_pct",           float, False),
    "min-dollar-vol":   ("_min_dollar_vol",           float, False),
    "rvol":             ("_rvol_min",                 float, False),
    "min-score":        ("_min_score",                float, False),
    "min-price":        ("_min_price",                float, False),
    # Changes indicator columns — expensive.
    "chandelier-period":("_chandelier_period",        int,   True),
}

# Parameters that only affect exit logic — daily scan result is identical
# across all values, so precomputed scans can be reused for a speedup.
_EXIT_ONLY_PARAMS = frozenset([
    "hard-stop-pct", "break-even-pct", "chandelier-mult",
])


def _build_backtest(args) -> "VelocityBacktest":
    return VelocityBacktest(
        start              = args.start,
        end                = args.end,
        capital            = args.capital,
        max_pos            = args.max_pos,
        min_bucket_size    = args.bucket_size,
        scan_count         = args.scan_count,
        commission_per_order = args.commission_per_order,
        rvol_min           = args.rvol,
        min_score          = args.min_score,
        break_even_pct     = args.break_even_pct,
        chandelier_mult    = args.chandelier_mult,
        chandelier_period  = args.chandelier_period,
        rsi_min_delta      = args.rsi_min_delta,
        risk_per_trade_pct = args.risk_per_trade,
        hard_stop_pct      = args.hard_stop_pct,
        min_dollar_vol     = args.min_dollar_vol,
        min_price          = args.min_price,
        min_volume         = args.min_volume,
        use_spy_filter     = args.spy_filter,
        use_vix_filter     = args.vix_filter,
        use_cache          = not args.no_cache,
    )


def main():
    args = parse_args()

    if args.sweep:
        _run_sweep(args)
        return

    print("\nALLIGATOR SWING STRATEGY — FORWARD BACKTEST")
    print(f"{'─' * 50}")
    print(f"  Period        : {args.start} → {args.end}")
    print(f"  Capital       : ${args.capital:,.2f}")
    _init_slots      = min(int(args.capital / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) if args.capital >= MIN_BUCKET_SIZE else 0
    _init_bucket_str = f"${args.capital / _init_slots:,.2f}" if _init_slots > 0 else "N/A"
    print(f"  Max pos       : {MAX_POSITIONS_CAP} cap  |  Dynamic max = floor(equity / ${MIN_BUCKET_SIZE:.0f}/slot)  |  Initial slots={_init_slots}, bucket≈{_init_bucket_str}")
    print("  Entry rules   : Alligator crossover + RSI trend + RVOL + day-strength")
    print(f"  RVOL min      : {args.rvol:.1f}× (daily close proxy)")
    print(f"  Min score     : {args.min_score:.0f}/100 composite gate")
    print(f"  Exit          : Chandelier (ATR{CHANDELIER_PERIOD}×{CHANDELIER_MULT}) + {args.hard_stop_pct:.0%} hard stop + {args.break_even_pct:.0%} break-even + Alligator reversal")
    print("  Position size : ATR-based (2% equity risk) capped by bucket")
    print(f"  Slippage      : {BACKTEST_SLIPPAGE:.1%} entry, {BACKTEST_EXIT_SLIPPAGE:.1%} exit (mkt orders)  |  Commission: ${args.commission_per_order*2:.2f}/round-trip")
    print(f"  SPY filter    : {'ON (EMA50 soft regime)' if args.spy_filter else 'OFF'}")
    print(f"  VIX filter    : {'ON (VIX > 35 blocks entries)' if args.vix_filter else 'OFF'}")
    print(f"  Cache         : {'OFF (forced re-download)' if args.no_cache else 'ON (backtest/.cache/)'}")
    print()

    bt = _build_backtest(args)
    result = bt.run()
    VelocityBacktest.print_report(result, capital=args.capital)

    if args.trades:
        VelocityBacktest.print_trades(result)


def _run_sweep(args) -> None:
    """In-process parameter sweep: load indicator cache once, simulate for each value."""
    try:
        param_str, values_str = args.sweep.split("=", 1)
    except ValueError:
        print(f"ERROR: --sweep must be PARAM=V1,V2,... (got: {args.sweep!r})", file=sys.stderr)
        sys.exit(1)

    param_str = param_str.strip()
    if param_str not in _SWEEP_PARAM_MAP:
        print(f"ERROR: unknown sweep param {param_str!r}. Known: {', '.join(_SWEEP_PARAM_MAP)}", file=sys.stderr)
        sys.exit(1)

    attr, cast, needs_recompute = _SWEEP_PARAM_MAP[param_str]
    values = [cast(v.strip()) for v in values_str.split(",")]

    if needs_recompute:
        print(f"WARNING: {param_str} changes indicator columns — each value triggers full indicator recomputation.", file=sys.stderr)

    bt = _build_backtest(args)
    print(f"Loading data for sweep: {param_str} ∈ {values} …", flush=True)
    bt.load_data()
    print(f"  Data loaded: {len(bt._data):,} symbols. Starting sweep …\n", flush=True)

    precomputed = None
    if param_str in _EXIT_ONLY_PARAMS:
        print("  Precomputing scan candidates (exit-only param) …", flush=True)
        precomputed = bt._precompute_scans_enriched()
        print(f"  Precomputed {len(precomputed):,} trading days.\n", flush=True)

    for val in values:
        setattr(bt, attr, val)
        if needs_recompute:
            bt._apply_indicators({s: df[bt._RAW_COLS] for s, df in bt._data.items()})
            bt._save_ind_cache()
        result = bt.run_with_flags({}, precomputed_scans=precomputed)
        sharpe = result.metrics.get("sharpe_ratio", 0.0)
        print(f"  {param_str}={val}:   Sharpe={sharpe:.2f}", flush=True)


if __name__ == "__main__":
    main()
