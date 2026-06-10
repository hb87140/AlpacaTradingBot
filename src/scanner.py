"""
Alpaca candidate scanner.

Combines four candidate sources into one pool each scan cycle:

  1. get_top_gainers         — stocks with the largest intraday % gain (Alpaca max 50).
  2. get_most_actives_vol    — stocks with the highest intraday share volume (max 100).
  3. get_most_actives_trades — stocks with the most intraday transactions (max 100);
                               different ranking than volume — catches high-frequency
                               breakout activity on stocks with moderate dollar volume.
  4. get_alligator_crossover_scan — daily pre-scan of the full Alpaca tradeable universe
                               (all active NYSE/NASDAQ/BATS/ARCA/AMEX stocks).
                               Downloads daily bars in batches, computes offset-adjusted
                               SMMAs, and returns symbols where the bullish Alligator
                               crossover occurred within ALLIGATOR_CROSS_LOOKBACK bars.
                               The universe is fetched from Alpaca's assets API once per
                               process lifetime (cached); the crossover scan runs once
                               per trading day (date-keyed cache). Zero extra API cost
                               on subsequent scan cycles within the same session.

Sources 1-3 surface intraday momentum candidates. Source 4 ensures that stocks whose
Alligator signal fired on a quiet day (below the gainers/actives threshold) are never
missed, regardless of their daily rank.

Early filters applied at scan time:
  • Price floor   — gainers endpoint returns price; sub-$10 stocks discarded.
  • Blocklist     — ETFs, leveraged/inverse products removed from all sources.
  • Non-stocks    — warrants, rights, and other non-equity instruments removed.
  • Exchange      — universe limited to NYSE / NASDAQ / BATS / ARCA / AMEX.
  • Min % gain    — gainers below SCAN_MIN_GAIN_PCT discarded.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import MarketType, MostActivesBy
from alpaca.trading.client import TradingClient

from src.config import (
    ALPACA_DATA_FEED,
    ALPACA_SCANNER_TOP,
    ALPACA_SCANNER_TOP_GAINERS,
    ALLIGATOR_CROSS_LOOKBACK,
    ALLIGATOR_FAST,
    ALLIGATOR_FAST_OFFSET,
    ALLIGATOR_MED,
    ALLIGATOR_MED_OFFSET,
    ALLIGATOR_SLOW,
    ALLIGATOR_SLOW_OFFSET,
    SCAN_MIN_GAIN_PCT,
    SCAN_MIN_PRICE,
    TICKER_BLOCKLIST,
)
from src.indicators import compute_smma

logger = logging.getLogger('BounceAlpha')

# Warrants (BASE+W), rights (BASE.RT), warrant series (BASE.WS), and other
# non-standard instruments never qualify — filter them before hitting the API.
_NON_STOCK_RE = re.compile(
    r'\.'            # contains a period  (e.g. GLED.RT, GRAF.WS)
    r'|WS$|WD$|WT$'  # warrant-series / warrant-deed suffixes
    r'|RT$|RW$'      # rights suffixes
)

# Exchanges considered legitimate for the Alligator universe.
_VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'BATS', 'ARCA', 'AMEX'}


def _is_non_stock(sym: str) -> bool:
    """Return True for warrants, rights, and other non-standard instruments."""
    if _NON_STOCK_RE.search(sym):
        return True
    # 5+ char symbols ending in W are almost always warrants (4-char base + W suffix)
    if len(sym) >= 5 and sym.endswith('W'):
        return True
    return False


def _keep(sym: str) -> bool:
    """True when a symbol passes the cheap scanner-level pre-filters."""
    return sym not in TICKER_BLOCKLIST and not _is_non_stock(sym)


# ── Dynamic Alpaca universe ────────────────────────────────────────────────────
# Fetched once per process from Alpaca's assets API, then cached for the session.
# New IPOs and delistings are picked up on the next process restart (daily on Render).
_alpaca_universe_cache: Optional[List[str]] = None

# Day-level crossover cache: expensive bar download + SMMA computation runs
# once per trading day; subsequent calls return the cached list instantly.
_crossover_cache: dict = {'date': None, 'symbols': []}

_UNIVERSE_BATCH_SIZE = 50   # symbols per Alpaca bars API request
_UNIVERSE_LOOKBACK   = 60   # calendar days of bars (≈ 42 trading days — enough for SMMA warmup)


def _fetch_alpaca_universe(trading_client: TradingClient) -> List[str]:
    """Return the full list of active, tradeable US equity symbols from Alpaca.

    Fetches once per process and caches the result.  Filtered to:
      • active + tradable
      • NYSE / NASDAQ / BATS / ARCA / AMEX exchanges (no OTC / crypto)
      • symbol passes _keep() and _is_non_stock() checks
      • symbol length ≤ 5 (eliminates most special instruments not caught above)
    Result is capped at ALLIGATOR_UNIVERSE_MAX and sorted shortest-symbol-first
    (a lightweight proxy for established / liquid names).
    """
    global _alpaca_universe_cache
    if _alpaca_universe_cache is not None:
        return _alpaca_universe_cache

    try:
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        req    = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = trading_client.get_all_assets(req)

        symbols: List[str] = []
        for a in assets:
            if not a.tradable:
                continue
            if a.exchange.value not in _VALID_EXCHANGES:
                continue
            sym = a.symbol
            if len(sym) > 5:
                continue
            if not _keep(sym) or _is_non_stock(sym):
                continue
            symbols.append(sym)

        # Shorter symbols first as a lightweight proxy for established / liquid names
        symbols.sort(key=lambda s: (len(s), s))
        _alpaca_universe_cache = symbols

        logger.info(
            f"SCANNER: Alpaca universe built — {len(_alpaca_universe_cache)} symbols "
            f"(from {len(assets)} raw assets)"
        )
    except Exception as e:
        logger.warning(f"SCANNER: Alpaca asset list fetch failed ({e}); universe will be empty this session")
        _alpaca_universe_cache = []

    return _alpaca_universe_cache


def get_top_gainers(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP_GAINERS) -> List[str]:
    """Return top gaining stock symbols filtered by gain %, price floor, and blocklist."""
    try:
        req  = MarketMoversRequest(top=top, market_type=MarketType.STOCKS)
        resp = screener_client.get_market_movers(req)
        raw  = resp.gainers

        symbols = []
        n_gain = n_price = n_block = 0
        for m in raw:
            if m.percent_change < SCAN_MIN_GAIN_PCT:
                n_gain += 1
                continue
            price = getattr(m, 'price', None)
            if price is not None and float(price) < SCAN_MIN_PRICE:
                n_price += 1
                continue
            if not _keep(m.symbol):
                n_block += 1
                continue
            symbols.append(m.symbol)

        logger.debug(
            f"SCANNER: top-gainers {len(symbols)} kept from {len(raw)} "
            f"(dropped: gain<{SCAN_MIN_GAIN_PCT:.0f}%={n_gain} "
            f"price<${SCAN_MIN_PRICE:.0f}={n_price} blocked={n_block})"
        )
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: top-gainers fetch failed: {e}")
        return []


def get_most_actives_vol(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return most-active stock symbols ranked by intraday share volume."""
    try:
        response = screener_client.get_most_actives(MostActivesRequest(top=top, by=MostActivesBy.VOLUME))
        symbols  = [item.symbol for item in response.most_actives if _keep(item.symbol)]
        logger.debug(f"SCANNER: most-actives(vol) {len(symbols)} kept from {len(response.most_actives)}")
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives(vol) fetch failed: {e}")
        return []


# Backward-compatible alias
get_most_actives = get_most_actives_vol


def get_most_actives_trades(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return most-active stock symbols ranked by intraday transaction count.

    Trade-count ranking surfaces stocks with many small orders — often a sign of
    retail accumulation or early institutional positioning ahead of an Alligator
    crossover. Complements the volume-ranked list with different candidates.
    """
    try:
        response = screener_client.get_most_actives(MostActivesRequest(top=top, by=MostActivesBy.TRADES))
        symbols  = [item.symbol for item in response.most_actives if _keep(item.symbol)]
        logger.debug(f"SCANNER: most-actives(trades) {len(symbols)} kept from {len(response.most_actives)}")
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives(trades) fetch failed: {e}")
        return []


def get_alligator_crossover_scan(
    data_client:    StockHistoricalDataClient,
    trading_client: TradingClient,
) -> List[str]:
    """Daily batch scan: returns universe symbols with an active Alligator crossover.

    On the first call each trading day:
      1. Fetch / use the cached Alpaca symbol universe (up to ALLIGATOR_UNIVERSE_MAX).
      2. Download 60 calendar days of daily bars in batches of 50.
      3. Compute offset-adjusted SMMAs for each symbol.
      4. Return symbols where the bullish crossover fired within ALLIGATOR_CROSS_LOOKBACK bars.

    Subsequent calls the same day return the cached list with no API calls.
    """
    today_str = date.today().isoformat()
    if _crossover_cache['date'] == today_str:
        return _crossover_cache['symbols']

    universe = _fetch_alpaca_universe(trading_client)
    if not universe:
        _crossover_cache['date']    = today_str
        _crossover_cache['symbols'] = []
        return []

    logger.info(
        f"SCANNER: Alligator crossover scan — {len(universe)} symbols "
        f"({len(universe) // _UNIVERSE_BATCH_SIZE + 1} batches)"
    )
    start      = datetime.now(timezone.utc) - timedelta(days=_UNIVERSE_LOOKBACK)
    candidates: List[str] = []
    n_batches  = 0

    n_failed = 0
    for i in range(0, len(universe), _UNIVERSE_BATCH_SIZE):
        batch     = universe[i: i + _UNIVERSE_BATCH_SIZE]
        n_batches += 1
        if n_batches > 1:
            time.sleep(0.2)  # avoid rate-limiting across 100+ consecutive batch requests
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                feed=ALPACA_DATA_FEED,
            )
            bars = data_client.get_stock_bars(req)
        except Exception as e:
            n_failed += 1
            logger.warning(f"SCANNER: universe batch {n_batches} failed: {e}")
            continue

        for sym in batch:
            try:
                raw_bars = bars[sym]
            except (KeyError, TypeError):
                continue
            if not raw_bars or len(raw_bars) < 22:
                continue

            df = pd.DataFrame([b.model_dump() for b in raw_bars]).reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]

            smma_f = compute_smma(df['close'], ALLIGATOR_FAST)
            smma_m = compute_smma(df['close'], ALLIGATOR_MED)
            smma_s = compute_smma(df['close'], ALLIGATOR_SLOW)

            def _at(series: pd.Series, offset: int) -> float:
                idx = -1 - offset
                if len(series) > abs(idx):
                    v = float(series.iloc[idx])
                    return v if not np.isnan(v) else float('nan')
                return float('nan')

            fast_now = _at(smma_f, ALLIGATOR_FAST_OFFSET)
            med_now  = _at(smma_m, ALLIGATOR_MED_OFFSET)
            slow_now = _at(smma_s, ALLIGATOR_SLOW_OFFSET)

            if any(np.isnan(v) for v in (fast_now, med_now, slow_now)):
                continue
            if not (fast_now > slow_now and med_now > slow_now):
                continue

            # Crossover: within CROSS_LOOKBACK bars there must have been a bar
            # where fast+med were NOT both above slow (i.e. the crossover is fresh)
            crossed = False
            for k in range(1, ALLIGATOR_CROSS_LOOKBACK + 1):
                pf = _at(smma_f, ALLIGATOR_FAST_OFFSET + k)
                pm = _at(smma_m, ALLIGATOR_MED_OFFSET  + k)
                ps = _at(smma_s, ALLIGATOR_SLOW_OFFSET + k)
                if not any(np.isnan(v) for v in (pf, pm, ps)):
                    if not (pf > ps and pm > ps):
                        crossed = True
                        break

            if crossed:
                candidates.append(sym)

    logger.info(
        f"SCANNER: Alligator crossover scan complete — "
        f"{len(candidates)} crossover candidates from {len(universe)} symbols "
        f"({n_batches} batches, {n_failed} failed)"
    )
    _crossover_cache['date']    = today_str
    _crossover_cache['symbols'] = candidates
    return candidates


def get_candidates(
    data_client:    StockHistoricalDataClient,
    screener_client: ScreenerClient,
    trading_client: Optional[TradingClient] = None,
) -> List[str]:
    """Return the combined candidate pool from all four sources, deduplicated.

    Priority order (higher-conviction sources first):
      1. Top gainers             — strongest intraday % movers
      2. Most-actives by volume  — highest dollar-flow stocks
      3. Most-actives by trades  — most transactions (breakout accumulation signal)
      4. Alligator universe scan — all liquid stocks with active SMMA crossovers

    Deduplication preserves order so higher-priority sources are evaluated first.
    """
    gainers  = get_top_gainers(screener_client)
    act_vol  = get_most_actives_vol(screener_client)
    act_trds = get_most_actives_trades(screener_client)
    universe = get_alligator_crossover_scan(data_client, trading_client) if trading_client else []

    seen:   set       = set()
    result: List[str] = []
    for sym in gainers + act_vol + act_trds + universe:
        if sym not in seen and _keep(sym) and not _is_non_stock(sym):
            seen.add(sym)
            result.append(sym)

    logger.debug(
        f"SCANNER: combined pool — "
        f"{len(gainers)}g + {len(act_vol)}av + {len(act_trds)}at + {len(universe)}u "
        f"= {len(result)} unique candidates"
    )
    return result
