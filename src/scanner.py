"""
Alpaca candidate scanner.

Combines two Alpaca screener endpoints into one candidate pool:
  1. get_top_gainers  — stocks with the largest intraday % price gain.
     These are the primary momentum candidates a breakout strategy wants.
  2. get_most_actives — stocks with the highest intraday share volume.
     Supplements gainers with high-volume movers not yet showing large % moves.

Both lists are fetched each scan cycle and merged (gainers first, then actives).
Duplicates are removed while preserving order.  The combined list is fed to the
engine's screener, which performs full technical filtering.

Early filters applied at scan time (before any technical context is built):
  • Price floor   — gainers endpoint returns price; sub-$10 stocks discarded immediately.
  • Blocklist     — ETFs, leveraged/inverse products removed from both sources.
  • Non-stocks    — warrants, rights, and other non-equity instruments removed.
  • Min % gain    — gainers below SCAN_MIN_GAIN_PCT discarded.
These same checks are redundantly enforced in the engine screener as a backstop.
"""

from __future__ import annotations

import logging
import re
from typing import List

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest
from alpaca.data.enums import MarketType

from src.config import (
    ALPACA_SCANNER_TOP,
    ALPACA_SCANNER_TOP_GAINERS,
    SCAN_MIN_GAIN_PCT,
    SCAN_MIN_PRICE,
    TICKER_BLOCKLIST,
)

logger = logging.getLogger('VelocityEngine')

# Warrants (BASE+W), rights (BASE.RT), warrant series (BASE.WS), and other
# non-standard instruments never qualify — filter them before hitting the API.
_NON_STOCK_RE = re.compile(
    r'\.'            # contains a period  (e.g. GLED.RT, GRAF.WS)
    r'|WS$|WD$|WT$'  # warrant-series / warrant-deed suffixes
    r'|RT$|RW$'      # rights suffixes
)


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


def get_top_gainers(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP_GAINERS) -> List[str]:
    """Return top gaining stock symbols filtered by gain %, price floor, and blocklist.

    Filters applied here (gainers endpoint provides live price):
      • percent_change >= SCAN_MIN_GAIN_PCT   — ignore flat/tiny movers
      • price >= SCAN_MIN_PRICE               — discard sub-$10 stocks immediately
      • not in TICKER_BLOCKLIST               — drop ETFs / leveraged products
      • not a warrant / right / non-stock     — drop instrument noise
    """
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


def get_most_actives(screener_client: ScreenerClient, top: int = ALPACA_SCANNER_TOP) -> List[str]:
    """Return most-active stock symbols filtered by blocklist and non-stock check.

    Most-actives endpoint does not expose price, so the price floor is enforced
    later in the engine screener.  Blocklist and non-stock filters are cheap and
    applied here to cut obvious noise (e.g. leveraged ETFs, warrants).
    """
    try:
        request  = MostActivesRequest(top=top, by='volume')
        response = screener_client.get_most_actives(request)
        raw      = response.most_actives

        symbols  = []
        n_block  = 0
        for item in raw:
            if not _keep(item.symbol):
                n_block += 1
                continue
            symbols.append(item.symbol)

        logger.debug(
            f"SCANNER: most-actives {len(symbols)} kept from {len(raw)} "
            f"(dropped: blocked={n_block})"
        )
        return symbols
    except Exception as e:
        logger.warning(f"SCANNER: most-actives fetch failed: {e}")
        return []


def get_candidates(
    data_client: StockHistoricalDataClient,
    screener_client: ScreenerClient,
) -> List[str]:
    """Return the union of top-gainers and most-actives, deduplicated.

    Gainers appear first — they are higher-conviction momentum candidates.
    Most-actives follow to catch high-volume setups that may not yet show
    a large percentage gain (e.g., early-session breakouts).
    """
    gainers = get_top_gainers(screener_client)
    actives = get_most_actives(screener_client)

    seen:   set       = set()
    result: List[str] = []
    for sym in gainers + actives:
        if sym not in seen and not _is_non_stock(sym):
            seen.add(sym)
            result.append(sym)

    logger.debug(
        f"SCANNER: combined pool — {len(gainers)} gainers + {len(actives)} actives "
        f"= {len(result)} unique candidates"
    )
    return result
