#!/usr/bin/env python
"""
Signal combination optimizer for the Velocity Strategy.

Exhaustively tests all 2^11 = 2048 combinations of 8 existing toggleable
rules and 3 new candidate signals. Loads the cached price data once, then
calls _run_loop() for each combination — no redundant downloads.

Usage:
    venv/bin/python scripts/run_optimizer.py
    venv/bin/python scripts/run_optimizer.py --start 2021-01-01 --end 2025-12-31

Output:
    backtest/optimizer_results.csv  — full ranked results
    Console                         — top-30 combinations + analysis

WARNING: Exhaustive search on historical data carries a significant overfitting
risk.  The "best" combination on any single backtest window may not generalise.
Look for PATTERNS across the top combinations (which signals appear consistently)
rather than blindly implementing the rank-1 combo.
"""

import argparse
import csv
import itertools
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategy import VelocityBacktest

# ── Search space ──────────────────────────────────────────────────────────────
# Existing rules (default ON in production — toggling OFF removes them).
EXISTING_SIGNALS = [
    'use_slope',        # SMA200 slope > 0
    'use_trend_sep',    # MA50 ≥ 3% above MA200
    'use_orb',          # close > previous day's high (ORB proxy)
    'use_rsi_rise',     # RSI rising day-over-day
    'use_rsi_delta',    # RSI delta ≥ RSI_MIN_DELTA
    'use_rsi_lvl',      # RSI > RSI_THRESHOLD (55)
]
# New candidate signals (default OFF — toggling ON adds them).
NEW_SIGNALS = [
    'use_adx',          # ADX(14) > 20: trend strength filter
    'use_52w_high',     # close within 15% of 200-day high: momentum leadership
    'use_ma20',         # close > EMA(20): near-term uptrend confirmation
]

ALL_SIGNALS      = EXISTING_SIGNALS + NEW_SIGNALS
N_SIGNALS        = len(ALL_SIGNALS)          # 9
N_COMBINATIONS   = 2 ** N_SIGNALS            # 512

# ── Scoring constraints ────────────────────────────────────────────────────────
MIN_TRADES    = 100    # below this, statistics are unreliable
MIN_WIN_RATE  = 0.55   # basic quality floor

# ── Baseline flags = production default (optimised combination) ───────────────
# ON:  trend_sep, orb, rsi_delta, rsi_lvl, adx, 52w_high
# OFF: slope, rsi_rise, ma20
BASELINE_FLAGS = {
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def score(metrics: dict) -> float:
    """Composite score = Sharpe × Profit Factor (higher is better)."""
    if not metrics:
        return -999.0
    if metrics.get('total_trades', 0) < MIN_TRADES:
        return -999.0
    if metrics.get('win_rate', 0.0) < MIN_WIN_RATE:
        return -999.0
    sharpe = metrics.get('sharpe_ratio', 0.0)
    pf     = metrics.get('profit_factor', 0.0)
    if pf == float('inf'):
        pf = 10.0   # cap inf (0 losses) to avoid misleading scores
    return sharpe * pf


def flags_label(flags: dict) -> str:
    """Human-readable label: active signals separated by +."""
    parts = []
    for s in ALL_SIGNALS:
        if flags[s]:
            label = s.replace('use_', '')
            parts.append(label)
    return '+'.join(parts) if parts else '(none)'


def iter_combinations() -> List[dict]:
    """Generate all 2^11 flag dicts."""
    combos = []
    for bits in itertools.product([False, True], repeat=N_SIGNALS):
        combos.append(dict(zip(ALL_SIGNALS, bits)))
    return combos


def format_row(rank: int, flags: dict, m: dict, s: float) -> str:
    wr      = m.get('win_rate', 0)
    pf      = m.get('profit_factor', 0)
    pf_str  = f"{pf:.2f}" if pf != float('inf') else "inf"
    ret     = m.get('total_return_pct', 0)
    dd      = m.get('max_drawdown_pct', 0)
    sharpe  = m.get('sharpe_ratio', 0)
    trades  = m.get('total_trades', 0)
    label   = flags_label(flags)
    return (
        f"{rank:<4} {s:<8.3f} {sharpe:<7.3f} {pf_str:<6} "
        f"{wr:<6.1%} {trades:<7} {ret:<9.1f}% {dd:<8.1f}% {label}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Velocity Strategy — Signal Optimizer")
    p.add_argument("--start",    default="2021-01-01")
    p.add_argument("--end",      default="2025-12-31")
    p.add_argument("--capital",  default=2000.0, type=float)
    p.add_argument("--top",      default=30, type=int,
                   help="Number of top combinations to print (default: 30)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "backtest")

    print("\nVELOCITY STRATEGY — SIGNAL OPTIMIZER")
    print(f"{'─' * 60}")
    print(f"  Period      : {args.start} → {args.end}")
    print(f"  Capital     : ${args.capital:,.2f}")
    print(f"  Signals     : {N_SIGNALS} toggleable  ({len(EXISTING_SIGNALS)} existing,"
          f" {len(NEW_SIGNALS)} new candidates)")
    print(f"  Combinations: {N_COMBINATIONS:,}")
    print(f"  Score metric: Sharpe × Profit Factor  (min {MIN_TRADES} trades,"
          f" {MIN_WIN_RATE:.0%} win rate)")
    print()

    # ── Step 1: load data once ────────────────────────────────────────────────
    print("Loading backtest data (cached)…")
    bt = VelocityBacktest(
        start=args.start, end=args.end,
        capital=args.capital, use_cache=True,
    )
    baseline_result = bt.run()
    bm = baseline_result.metrics
    baseline_score  = score(bm)
    print(f"  Baseline (production defaults): "
          f"score={baseline_score:.3f}  sharpe={bm.get('sharpe_ratio',0):.3f}  "
          f"PF={bm.get('profit_factor',0):.2f}  "
          f"WR={bm.get('win_rate',0):.1%}  "
          f"trades={bm.get('total_trades',0)}  "
          f"return={bm.get('total_return_pct',0):.1f}%")

    # ── Step 2: prepare new signal columns (one-time, fast) ───────────────────
    print("\nPreparing new signal columns (ADX, 200-day high, EMA20)…")
    t0 = time.time()
    bt._prepare_optimizer_signals()
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 2b: pre-compute daily scan candidates once (flags-independent) ───
    print("\nPre-computing daily scan candidates (fast-path for optimizer)…")
    t0 = time.time()
    precomputed_scans = bt._precompute_scans_enriched()
    n_cands  = sum(len(v) for v in precomputed_scans.values())
    n_active = sum(1 for v in precomputed_scans.values() if v)
    print(f"  Done in {time.time() - t0:.1f}s  —  "
          f"{n_cands:,} candidates across {n_active:,} scan dates", flush=True)

    # ── Step 3: enumerate and score all 2^11 combinations ─────────────────────
    combos = iter_combinations()
    print(f"\nScoring {N_COMBINATIONS:,} combinations…", flush=True)
    t_start = time.time()

    records: List[dict] = []
    for i, flags in enumerate(combos):
        result  = bt.run_with_flags(flags, precomputed_scans=precomputed_scans)
        m       = result.metrics
        s       = score(m)

        records.append({
            'rank':          0,
            'score':         round(s, 4),
            'sharpe':        round(m.get('sharpe_ratio', 0), 4),
            'profit_factor': round(m.get('profit_factor', 0) if m.get('profit_factor', 0) != float('inf') else 10.0, 4),
            'win_rate':      round(m.get('win_rate', 0), 4),
            'trades':        m.get('total_trades', 0),
            'total_return':  round(m.get('total_return_pct', 0), 2),
            'max_drawdown':  round(m.get('max_drawdown_pct', 0), 2),
            'avg_hold':      round(m.get('avg_hold_bars', 0), 2),
            **{k: int(v) for k, v in flags.items()},
        })

        if (i + 1) % 256 == 0:
            elapsed   = time.time() - t_start
            rate      = (i + 1) / elapsed
            remaining = (N_COMBINATIONS - i - 1) / rate
            print(f"  {i+1:>4}/{N_COMBINATIONS}  —  "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining",
                  flush=True)
            # Checkpoint: partial sorted CSV in case of crash
            _partial = sorted(records, key=lambda r: r['score'], reverse=True)
            _ckpt    = os.path.join(out_dir, "optimizer_results_partial.csv")
            with open(_ckpt, 'w', newline='') as _f:
                _w = csv.DictWriter(_f, fieldnames=records[0].keys())
                _w.writeheader()
                _w.writerows(_partial)

    total_time = time.time() - t_start
    print(f"  Finished {N_COMBINATIONS} runs in {total_time:.1f}s "
          f"({total_time / N_COMBINATIONS * 1000:.0f}ms/run)")

    # ── Step 4: rank and save ──────────────────────────────────────────────────
    records.sort(key=lambda r: r['score'], reverse=True)
    for i, r in enumerate(records, 1):
        r['rank'] = i

    csv_path = os.path.join(out_dir, "optimizer_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nFull results → {csv_path}")

    # ── Step 5: print top N ───────────────────────────────────────────────────
    top_n    = args.top
    print(f"\n{'─' * 115}")
    print(f"TOP {top_n} COMBINATIONS  (sorted by Sharpe × Profit Factor)")
    print(f"{'─' * 115}")
    header = (f"{'Rank':<4} {'Score':<8} {'Sharpe':<7} {'PF':<6} "
              f"{'WR':<6} {'Trades':<7} {'Return':<10} {'MaxDD':<9} Active signals")
    print(header)
    print('─' * 115)

    # Reconstruct flags dict from record for display
    def rec_to_flags(r):
        return {s: bool(r[s]) for s in ALL_SIGNALS}

    for r in records[:top_n]:
        flags = rec_to_flags(r)
        m_fake = {
            'total_trades': r['trades'],
            'win_rate':     r['win_rate'],
            'profit_factor': r['profit_factor'],
            'total_return_pct': r['total_return'],
            'max_drawdown_pct': r['max_drawdown'],
            'sharpe_ratio':  r['sharpe'],
        }
        print(format_row(r['rank'], flags, m_fake, r['score']))

    # ── Step 6: signal frequency analysis ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("SIGNAL FREQUENCY IN TOP-50  (how often each rule is ON)")
    print(f"{'─' * 60}")
    top50 = records[:50]
    for sig in ALL_SIGNALS:
        count      = sum(1 for r in top50 if r[sig])
        tag        = '  ← NEW' if sig in NEW_SIGNALS else ''
        prod_state = 'ON ' if BASELINE_FLAGS[sig] else 'OFF'
        print(f"  {sig:<18}  {count:>3}/50 ({count*2:>3}%)  [prod={prod_state}]{tag}")

    # ── Step 7: report baseline rank ─────────────────────────────────────────
    baseline_rank = next(
        (r['rank'] for r in records
         if all(bool(r[s]) == BASELINE_FLAGS[s] for s in ALL_SIGNALS)),
        None,
    )
    print(f"\n  Production baseline ranks #{baseline_rank} of {N_COMBINATIONS}")
    print("\nDone.")


if __name__ == "__main__":
    main()
