#!/usr/bin/env python
"""
AlligatorAlpha Web Dashboard
─────────────────────────────
Standalone FastAPI server — completely independent of the trading engine.

Start:   venv/bin/python alpaca_dashboard.py
Open:    http://localhost:8080

The server only reads JSON files written by the engine:
  • engine_state.json    — open positions
  • dashboard_data.json  — equity, VIX, connection status, scan times
  • equity_history.json  — rolling 60-day equity snapshots for P&L

Closing/restarting this server never affects the running alpaca_auto_trader.
"""

import collections
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import pytz
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from src.config import (
    STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, LOG_FILE,
    MAX_POSITIONS_CAP, MIN_BUCKET_SIZE, BUCKET_CASH_PCT, VIX_THRESHOLD,
    SCAN_MIN_VOLUME, SCAN_MIN_DOLLAR_VOL,
    SPREAD_MAX_PCT, RVOL_MIN,
    CHANDELIER_PERIOD, CHANDELIER_MULT,
    HARD_STOP_PCT, BREAK_EVEN_PCT,
    MAX_DAILY_LOSS_PCT, CORR_MAX, MAX_SECTOR_COUNT,
    ENTRY_START, ENTRY_END, FRIDAY_CLOSE_HOUR,
    RSI_PERIOD, RSI_MIN_DELTA,
    ALLIGATOR_FAST, ALLIGATOR_MED, ALLIGATOR_SLOW, ALLIGATOR_CROSS_LOOKBACK,
    DAY_STRENGTH_OPEN_PCT,
    SCAN_MIN_SCORE,
    SCORE_ALLIGATOR_MAX, SCORE_RVOL_MAX, SCORE_RSI_DELTA_MAX, SCORE_LIQUIDITY_MAX, SCORE_ANALYST_MAX,
    ALPACA_PAPER,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AlligatorAlpha Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _count_trading_days(entry_dt: datetime, now: datetime) -> int:
    """Count complete Mon–Fri trading sessions elapsed between entry_dt and now.
    Mirrors engine._count_trading_days exactly so hold-time display matches
    the velocity-exit logic."""
    entry_date = entry_dt.date()
    now_date   = now.date()
    count  = 0
    cursor = entry_date
    while cursor < now_date:
        if cursor.weekday() < 5:   # Mon=0 … Fri=4; skip Sat=5, Sun=6
            count += 1
        cursor += timedelta(days=1)
    return count


def _read_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _read_history() -> list:
    try:
        if os.path.exists(EQUITY_HIST_FILE):
            with open(EQUITY_HIST_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _pnl(equity_now: float) -> dict:
    """Compute daily / weekly / monthly / overall P&L from equity history."""
    history = _read_history()
    tz_ny   = pytz.timezone('US/Eastern')
    now     = datetime.now(tz_ny)

    def _parse_ts(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else tz_ny.localize(dt)

    def _find_base(days_ago: int) -> Optional[float]:
        cutoff = now - timedelta(days=days_ago)
        past   = [e for e in history if _parse_ts(e["ts"]) <= cutoff]
        if past:
            e = past[-1]
            return float(e.get("equity") or e.get("eq") or 0) or None
        return None

    def _entry(base: Optional[float]) -> dict:
        if base is None or base == 0:
            return {"amount": None, "pct": None}
        amount = round(equity_now - base, 2)
        pct    = round(amount / base * 100, 2)
        return {"amount": amount, "pct": pct}

    # Overall: oldest snapshot in history (first Alpaca account reading)
    if history:
        h0 = history[0]
        overall_base = float(h0.get("equity") or h0.get("eq") or 0) or None
    else:
        overall_base = None

    return {
        "daily":   _entry(_find_base(1)),
        "weekly":  _entry(_find_base(7)),
        "monthly": _entry(_find_base(30)),
        "overall": _entry(overall_base),
    }


def _market_open() -> bool:
    tz_ny  = pytz.timezone("US/Eastern")
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    return (9, 30) <= (now_ny.hour, now_ny.minute) < (16, 0)


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/state")
def get_state():
    state     = _read_json(STATE_FILE)
    dash_data = _read_json(DASHBOARD_FILE)

    equity         = float(dash_data.get("equity") or 0)
    raw_settled    = dash_data.get("settled_cash")
    position_value = sum(
        float(d.get("current_price", d.get("price", 0))) * float(d.get("qty", 0))
        for d in state.values()
        if not d.get("pending")
    )
    settled_cash = (
        float(raw_settled) if raw_settled is not None
        else round(equity - position_value, 2)
    )

    tz_ny = pytz.timezone('US/Eastern')
    now   = datetime.now(tz_ny)
    positions        = []
    total_unrealized = 0.0
    for sym, d in state.items():
        if d.get("pending"):
            continue
        ep  = float(d.get("fill_price") or d.get("price", 0))
        qty = float(d.get("qty", 0))
        if qty <= 0:
            continue
        unit_price = round(ep, 4) if d.get("fill_price") else None
        cur        = float(d.get("current_price", ep))
        sl         = float(d.get("stop_loss", 0))
        # Engine writes break-even-floored chandelier stop directly into stop_loss via
        # _update_position_prices(); no separate effective_stop key in state.
        effective_sl = sl
        vol          = float(d.get("volume", 0))
        entry_ts     = d.get("time", now.isoformat())
        try:
            entry_dt = datetime.fromisoformat(entry_ts)
            if entry_dt.tzinfo is None:
                entry_dt = tz_ny.localize(entry_dt)
            hold_h            = (now - entry_dt).total_seconds() / 3600
            hold_trading_days = _count_trading_days(entry_dt, now)
        except ValueError:
            hold_h            = 0.0
            hold_trading_days = 0
        unreal     = round((cur - ep) * qty, 2)
        unreal_pct = round((cur - ep) / ep * 100, 2) if ep else 0.0
        total_unrealized += unreal
        positions.append({
            "symbol":            sym,
            "entry_price":       ep,
            "unit_price":        unit_price,
            "current_price":     cur,
            "qty":               qty,
            "total_amount":      round(ep * qty, 2),
            "unrealized":        unreal,
            "unrealized_pct":    unreal_pct,
            "stop_loss":         sl,
            "effective_stop":    effective_sl,
            "volume":            vol,
            "hold_hours":        round(hold_h, 2),
            "hold_trading_days": hold_trading_days,
            "entry_time":        entry_ts,
            "score":             d.get("score"),
            "analyst_buy":       d.get("analyst_buy",  0),
            "analyst_hold":      d.get("analyst_hold", 0),
            "analyst_sell":      d.get("analyst_sell", 0),
        })

    _dyn_max_pos    = min(int(equity / MIN_BUCKET_SIZE), MAX_POSITIONS_CAP) if equity >= MIN_BUCKET_SIZE else 0
    _capacity_slots = max(0, _dyn_max_pos - len(positions))
    _cash_slots     = int(settled_cash / MIN_BUCKET_SIZE) if settled_cash >= MIN_BUCKET_SIZE else 0
    _entry_slots    = min(_capacity_slots, _cash_slots)
    bucket_size     = round((settled_cash * BUCKET_CASH_PCT) / _entry_slots, 2) if _entry_slots > 0 else 0.0
    return JSONResponse({
        "equity":            equity,
        "mkt_value":         round(position_value, 2),
        "cash":              settled_cash,
        "allocation_pct":    round((position_value / equity * 100) if equity else 0, 1),
        "bucket_size":       bucket_size,
        "position_count":    len(positions),
        "max_positions":     _dyn_max_pos,
        "entry_slots":       _entry_slots,
        "positions":         positions,
        "total_unrealized":  round(total_unrealized, 2),
        "pnl":               dash_data.get("pnl") or _pnl(equity),
        "connected":         bool(dash_data.get("connected", False)),
        "market_open":       _market_open(),
        "paper_mode":        ALPACA_PAPER,
        "vix":               dash_data.get("vix"),
        "vix_threshold":     VIX_THRESHOLD,
        "last_scan":         dash_data.get("last_scan"),
        "next_scan":         dash_data.get("next_scan"),
        "last_updated":      dash_data.get("last_updated"),
        "blocked_today":     dash_data.get("blocked_today", []),
        "entry_start":       list(ENTRY_START),
        "entry_end":         list(ENTRY_END),
    })


@app.get("/api/equity_history")
def get_equity_history():
    return JSONResponse(_read_history())


@app.get("/api/logs")
def get_logs(n: int = 100):
    """Return the last n lines from the trading engine log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = list(collections.deque(f, maxlen=n))
        return JSONResponse({"lines": [ln.rstrip() for ln in lines]})
    except FileNotFoundError:
        return JSONResponse({"lines": [], "error": "Log file not found"})
    except OSError as e:
        return JSONResponse({"lines": [], "error": str(e)})


@app.get("/api/logs/download")
def download_logs():
    """Download the full trading engine log file as plain text."""
    if os.path.exists(LOG_FILE):
        filename = f"trading_engine_{datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d')}.log"
        return FileResponse(LOG_FILE, filename=filename, media_type="text/plain")
    return JSONResponse({"error": "Log file not found"}, status_code=404)


# ── Dashboard HTML ────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlligatorAlpha</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:      #080b12;
  --bg2:     #0c1018;
  --bg3:     #101520;
  --bg4:     #141c28;
  --bg5:     #1a2333;
  --border:  #1e2d42;
  --border2: #253850;
  --text:    #c8d6e5;
  --text2:   #7b92a8;
  --text3:   #4a5f72;
  --green:   #10b981;
  --red:     #ef4444;
  --yellow:  #f59e0b;
  --cyan:    #06b6d4;
  --blue:    #3b82f6;
  --purple:  #8b5cf6;
  --ui: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --num: 'Fira Code', 'Cascadia Code', 'Consolas', 'Courier New', monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--ui);font-size:13px;line-height:1.5;
  padding-bottom:28px;min-height:100vh;
}
.num{font-family:var(--num);}

/* ── progress bar ── */
#topbar{position:fixed;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--cyan),var(--green));
  z-index:1000;opacity:.5;}
#progress{height:100%;width:0%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.9));
  transition:width .1s ease;}

/* ── header ── */
.hdr{
  background:linear-gradient(135deg,#090e1a 0%,#0c1320 100%);
  border-bottom:1px solid var(--border2);
  padding:0 24px;display:flex;align-items:center;
  justify-content:space-between;height:54px;gap:16px;
  position:sticky;top:2px;z-index:100;
}
.hdr-left{display:flex;align-items:center;gap:12px;}
.logo{
  font-family:var(--num);font-size:15px;font-weight:700;letter-spacing:3px;
  background:linear-gradient(90deg,var(--cyan),var(--green));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.mode-badge{
  font-size:9px;font-weight:700;letter-spacing:2px;
  padding:3px 9px;border-radius:3px;
  background:rgba(245,158,11,.15);color:var(--yellow);
  border:1px solid rgba(245,158,11,.3);
}
.mode-badge.live{background:rgba(239,68,68,.15);color:var(--red);border-color:rgba(239,68,68,.3);}
.hdr-clock{
  font-family:var(--num);font-size:19px;font-weight:700;
  color:var(--text);letter-spacing:1px;flex:1;text-align:center;
}
.hdr-right{display:flex;align-items:center;gap:20px;}
.hdr-pill{
  display:flex;align-items:center;gap:6px;
  font-size:11px;font-weight:600;
}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;}
.dot-on{background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 1.8s infinite;}
.dot-off{background:var(--red);}
.dot-warn{background:var(--yellow);box-shadow:0 0 5px var(--yellow);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}
.hdr-updated{font-size:10px;color:var(--text3);}

/* ── page wrap ── */
.page{max-width:1440px;margin:0 auto;padding:16px 20px;}

/* ── metrics strip ── */
.metrics{
  display:grid;grid-template-columns:repeat(5,1fr);
  gap:12px;margin-bottom:14px;
}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr);}}
@media(max-width:680px){.metrics{grid-template-columns:1fr 1fr;}}
.metric{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;
  display:flex;flex-direction:column;gap:5px;
  transition:border-color .2s;position:relative;overflow:hidden;
}
.metric::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--mc,var(--border));opacity:.9;
}
.metric:hover{border-color:var(--border2);}
.metric-lbl{
  font-size:9px;font-weight:700;letter-spacing:2px;
  color:var(--text3);text-transform:uppercase;
}
.metric-val{
  font-family:var(--num);font-size:22px;font-weight:700;
  color:var(--text);line-height:1.1;
}
.metric-sub{font-size:11px;color:var(--text2);display:flex;align-items:center;gap:4px;}

/* ── colour helpers ── */
.g{color:var(--green);}  .r{color:var(--red);}
.y{color:var(--yellow);} .c{color:var(--cyan);}
.d{color:var(--text3);}  .p{color:var(--purple);}
.pos{color:var(--green);} .neg{color:var(--red);}
.neu{color:var(--text2);}

/* ── mid row ── */
.mid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
@media(max-width:800px){.mid{grid-template-columns:1fr;}}

/* ── panels ── */
.panel{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;padding:18px 20px;margin-bottom:14px;
}
.ptitle{
  font-size:10px;font-weight:700;letter-spacing:2.5px;
  color:var(--text3);text-transform:uppercase;
  margin-bottom:14px;padding-bottom:10px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.ptitle-left{display:flex;align-items:center;gap:8px;}
.ptitle-icon{font-size:14px;}

/* ── P&L rows ── */
.pnl-rows{display:flex;flex-direction:column;gap:8px;}
.pnl-row{
  display:grid;grid-template-columns:60px 1fr 110px 72px;
  align-items:center;gap:10px;
  padding:8px 10px;background:var(--bg3);
  border:1px solid var(--border);border-radius:7px;
}
.pnl-period{font-size:9px;color:var(--text3);font-weight:700;letter-spacing:1.5px;}
.pbar-wrap{background:var(--bg5);border-radius:3px;height:5px;overflow:hidden;}
.pbar{height:100%;border-radius:3px;transition:width .5s ease;min-width:2px;}
.pnl-amt{font-family:var(--num);font-size:13px;font-weight:700;text-align:right;}
.pnl-pct{font-family:var(--num);font-size:11px;text-align:right;opacity:.85;}

/* ── status rows ── */
.srows{display:flex;flex-direction:column;gap:7px;}
.srow{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 12px;background:var(--bg3);
  border:1px solid var(--border);border-radius:7px;
}
.slbl{font-size:10px;color:var(--text3);font-weight:600;letter-spacing:1px;}
.sval{font-weight:700;font-size:12px;font-family:var(--num);}

/* ── risk summary strip ── */
.risk-strip{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:10px;margin-bottom:14px;
}
@media(max-width:900px){.risk-strip{grid-template-columns:1fr 1fr;}}
.rcard{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:8px;padding:12px 14px;
}
.rlbl{font-size:9px;color:var(--text3);letter-spacing:2px;font-weight:700;margin-bottom:6px;}
.rval{font-family:var(--num);font-size:18px;font-weight:700;}
.rsub{font-size:10px;color:var(--text3);margin-top:3px;}

/* ── portfolio table ── */
.tbl-wrap{overflow-x:auto;border-radius:7px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead tr{background:var(--bg4);}
th{
  padding:10px 12px;text-align:right;
  font-size:9px;letter-spacing:1.5px;color:var(--text3);
  font-weight:700;border-bottom:2px solid var(--border2);
  white-space:nowrap;font-family:var(--ui);
}
th:first-child{text-align:left;}
tbody tr{border-bottom:1px solid var(--border);transition:background .12s;}
tbody tr:hover{background:var(--bg4);}
tbody tr:last-child{border-bottom:none;}
td{padding:12px 12px;text-align:right;white-space:nowrap;font-family:var(--num);}
td:first-child{text-align:left;}
td.ui-font{font-family:var(--ui);}
.sym{font-size:14px;font-weight:700;color:var(--cyan);letter-spacing:.5px;}
.sc-wrap{display:flex;align-items:center;gap:6px;justify-content:flex-end;}
.sc-bar{height:4px;border-radius:2px;flex-shrink:0;}
.prc-cell{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}
.prc-chg{font-size:10px;}
.unr-cell{display:flex;flex-direction:column;align-items:flex-end;gap:3px;}
.unr-mini{height:3px;border-radius:2px;width:48px;background:var(--bg5);}
.unr-fill{height:100%;border-radius:2px;}
.risk-cell{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}
.risk-dist{font-size:10px;opacity:.75;}
.hold-cell{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}
.vel-tag{font-size:9px;color:var(--yellow);}
.badge{
  display:inline-block;font-size:8px;font-weight:700;letter-spacing:.5px;
  padding:2px 6px;border-radius:3px;font-family:var(--ui);
}
.badge-be{background:rgba(16,185,129,.18);color:var(--green);border:1px solid rgba(16,185,129,.3);}
.badge-vel{background:rgba(245,158,11,.18);color:var(--yellow);border:1px solid rgba(245,158,11,.3);}
.badge-strong{background:rgba(6,182,212,.15);color:var(--cyan);border:1px solid rgba(6,182,212,.3);}
.empty td{
  text-align:center;color:var(--text3);
  font-style:italic;padding:36px;font-family:var(--ui);
}

/* ── equity chart ── */
.chart-wrap{position:relative;height:240px;}

/* ── logs ── */
.log-box{
  background:var(--bg);border:1px solid var(--border);
  border-radius:7px;padding:10px 12px;
  font-family:var(--num);font-size:11px;line-height:1.85;
  height:260px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:var(--border2) transparent;
}
.log-box::-webkit-scrollbar{width:4px;}
.log-box::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.ll{display:block;padding:1px 0;border-bottom:1px solid rgba(30,45,66,.5);word-break:break-all;}
.ll:last-child{border-bottom:none;}
.le{color:var(--red);}  .lw{color:var(--yellow);}
.lb{color:var(--green);} .ls{color:var(--cyan);}
.lh{color:var(--text3);} .lt{color:var(--purple);}
.ln{color:var(--text2);}

/* ── collapsible ── */
.tog-btn{
  background:none;border:none;cursor:pointer;
  font-size:10px;color:var(--text3);font-family:var(--ui);
  font-weight:700;letter-spacing:.5px;
  display:flex;align-items:center;gap:5px;
  transition:color .15s;padding:0;
}
.tog-btn:hover{color:var(--text);}
.chev{transition:transform .3s;display:inline-block;font-style:normal;}
.chev.open{transform:rotate(180deg);}
.coll-body{overflow:hidden;transition:max-height .35s ease;max-height:0;}
.coll-body.open{max-height:3000px;}

/* ── conditions ── */
.cond-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;padding-top:4px;}
@media(max-width:900px){.cond-grid{grid-template-columns:1fr;}}
.cond{display:flex;align-items:flex-start;gap:8px;padding:7px 8px;border-radius:6px;transition:background .15s;}
.cond:hover{background:var(--bg4);}
.cn{font-family:var(--num);color:var(--yellow);font-weight:700;font-size:10px;min-width:22px;margin-top:1px;opacity:.8;}
.cname{font-size:11px;font-weight:600;min-width:130px;padding-right:8px;}
.cname.en{color:var(--green);}  .cname.ex{color:var(--red);}
.cdesc{color:var(--text2);font-size:11px;font-family:var(--ui);}

/* ── footer ── */
footer{text-align:center;color:var(--text3);font-size:9px;letter-spacing:2px;padding:12px 0 4px;}
footer a{color:var(--text3);text-decoration:none;}
footer a:hover{color:var(--text2);}
</style>
</head>
<body>

<div id="topbar"><div id="progress"></div></div>

<!-- ── HEADER ─────────────────────────────────────────────────── -->
<div class="hdr">
  <div class="hdr-left">
    <span class="logo">🐊 ALLIGATORALPHA</span>
    <span class="mode-badge" id="mode-badge">PAPER</span>
  </div>
  <div class="hdr-clock num" id="clock">—</div>
  <div class="hdr-right">
    <div class="hdr-pill">
      <span class="dot dot-off" id="conn-dot"></span>
      <span id="conn-text" class="r">—</span>
    </div>
    <div class="hdr-pill">
      <span class="dot dot-warn" id="mkt-dot"></span>
      <span id="mkt-text" class="y">—</span>
    </div>
    <div class="hdr-updated">Updated <span id="lu">—</span></div>
  </div>
</div>

<div class="page">

<!-- ── METRICS STRIP ────────────────────────────────────────────── -->
<div class="metrics">
  <div class="metric" style="--mc:var(--green)">
    <div class="metric-lbl">Total Equity</div>
    <div class="metric-val num g" id="m-equity">—</div>
    <div class="metric-sub" id="m-equity-sub"><span class="neu">portfolio value</span></div>
  </div>
  <div class="metric" style="--mc:var(--cyan)">
    <div class="metric-lbl">Settled Cash</div>
    <div class="metric-val num c" id="m-cash">—</div>
    <div class="metric-sub" id="m-cash-sub"><span class="neu">available for entries</span></div>
  </div>
  <div class="metric" style="--mc:var(--purple)">
    <div class="metric-lbl">Deployed</div>
    <div class="metric-val num p" id="m-mktval">—</div>
    <div class="metric-sub" id="m-alloc-sub"><span class="neu">of equity</span></div>
  </div>
  <div class="metric" style="--mc:var(--yellow)">
    <div class="metric-lbl">Unrealized P&amp;L</div>
    <div class="metric-val num" id="m-unreal">—</div>
    <div class="metric-sub" id="m-unreal-sub"><span class="neu">open positions</span></div>
  </div>
  <div class="metric" style="--mc:var(--red)">
    <div class="metric-lbl">VIX</div>
    <div class="metric-val num" id="m-vix">—</div>
    <div class="metric-sub" id="m-vix-sub"><span class="neu">volatility index</span></div>
  </div>
</div>

<!-- ── MID ROW ──────────────────────────────────────────────────── -->
<div class="mid">

  <!-- P&L Performance -->
  <div class="panel">
    <div class="ptitle">
      <div class="ptitle-left"><span class="ptitle-icon">📈</span> P&amp;L PERFORMANCE</div>
    </div>
    <div class="pnl-rows">
      <div class="pnl-row">
        <div class="pnl-period">TODAY</div>
        <div class="pbar-wrap"><div class="pbar" id="pb-daily"></div></div>
        <div class="pnl-amt neu" id="pa-daily">—</div>
        <div class="pnl-pct neu" id="pp-daily">—</div>
      </div>
      <div class="pnl-row">
        <div class="pnl-period">WEEKLY</div>
        <div class="pbar-wrap"><div class="pbar" id="pb-weekly"></div></div>
        <div class="pnl-amt neu" id="pa-weekly">—</div>
        <div class="pnl-pct neu" id="pp-weekly">—</div>
      </div>
      <div class="pnl-row">
        <div class="pnl-period">MONTHLY</div>
        <div class="pbar-wrap"><div class="pbar" id="pb-monthly"></div></div>
        <div class="pnl-amt neu" id="pa-monthly">—</div>
        <div class="pnl-pct neu" id="pp-monthly">—</div>
      </div>
      <div class="pnl-row">
        <div class="pnl-period">OVERALL</div>
        <div class="pbar-wrap"><div class="pbar" id="pb-overall"></div></div>
        <div class="pnl-amt neu" id="pa-overall">—</div>
        <div class="pnl-pct neu" id="pp-overall">—</div>
      </div>
    </div>
  </div>

  <!-- Engine Status -->
  <div class="panel">
    <div class="ptitle">
      <div class="ptitle-left"><span class="ptitle-icon">📡</span> ENGINE STATUS</div>
    </div>
    <div class="srows">
      <div class="srow"><span class="slbl">ALPACA API</span>      <span class="sval" id="s-api">—</span></div>
      <div class="srow"><span class="slbl">MARKET</span>          <span class="sval" id="s-mkt">—</span></div>
      <div class="srow"><span class="slbl">POSITIONS USED</span>  <span class="sval" id="s-pos">—</span></div>
      <div class="srow"><span class="slbl">BUCKET SIZE</span>     <span class="sval" id="s-bucket">—</span></div>
      <div class="srow"><span class="slbl">ENTRY WINDOW</span>    <span class="sval" id="s-window">—</span></div>
      <div class="srow"><span class="slbl">LAST SCAN</span>       <span class="sval d" id="s-lscan">—</span></div>
      <div class="srow"><span class="slbl">NEXT SCAN IN</span>    <span class="sval" id="s-nscan">—</span></div>
    </div>
  </div>

</div>

<!-- ── PORTFOLIO ─────────────────────────────────────────────────── -->
<div class="panel">
  <div class="ptitle">
    <div class="ptitle-left"><span class="ptitle-icon">💼</span> OPEN PORTFOLIO</div>
    <span id="pos-summary" style="font-size:11px;color:var(--text2);font-weight:400;letter-spacing:0"></span>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th style="text-align:left">SYMBOL</th>
          <th>SCORE</th>
          <th>ANALYSTS</th>
          <th>ENTRY</th>
          <th>CURRENT</th>
          <th>QTY</th>
          <th>COST BASIS</th>
          <th>UNREALIZED P&amp;L</th>
          <th>STOP LOSS</th>
          <th>RISK TO STOP</th>
          <th>HOLD</th>
          <th>STATUS</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr class="empty"><td colspan="12">Waiting for data…</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── RISK SUMMARY ───────────────────────────────────────────────── -->
<div class="risk-strip" id="risk-strip" style="display:none">
  <div class="rcard">
    <div class="rlbl">MAX LOSS IF ALL STOPS HIT</div>
    <div class="rval r num" id="r-maxloss">—</div>
    <div class="rsub">if all trailing stops trigger simultaneously</div>
  </div>
  <div class="rcard">
    <div class="rlbl">EQUITY AT RISK</div>
    <div class="rval y num" id="r-pct">—</div>
    <div class="rsub">% of total equity at risk right now</div>
  </div>
  <div class="rcard">
    <div class="rlbl">TOTAL DEPLOYED</div>
    <div class="rval p num" id="r-deployed">—</div>
    <div class="rsub">market value of all open positions</div>
  </div>
  <div class="rcard">
    <div class="rlbl">AVG DIST TO STOP</div>
    <div class="rval d num" id="r-avgstop">—</div>
    <div class="rsub">average % price must fall to hit stop</div>
  </div>
</div>

<!-- ── EQUITY CURVE ────────────────────────────────────────────────── -->
<div class="panel">
  <div class="ptitle">
    <div class="ptitle-left"><span class="ptitle-icon">📉</span> EQUITY CURVE — 60-DAY ROLLING</div>
  </div>
  <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
</div>

<!-- ── LIVE LOGS ──────────────────────────────────────────────────── -->
<div class="panel">
  <div class="ptitle">
    <div class="ptitle-left">
      <span class="ptitle-icon">🖥</span> LIVE ENGINE LOG
      <span style="background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.2);
                   padding:1px 7px;border-radius:3px;font-size:8px;letter-spacing:1px;margin-left:4px">LIVE</span>
    </div>
    <button class="tog-btn" onclick="tog('log-body','log-chev','log-lbl')">
      <span id="log-chev" class="chev open">▼</span>
      <span id="log-lbl">HIDE</span>
    </button>
  </div>
  <div id="log-body" class="coll-body open">
    <div class="log-box" id="log-lines">Loading…</div>
  </div>
</div>

<!-- ── ENTRY CONDITIONS ───────────────────────────────────────────── -->
<div class="panel">
  <div class="ptitle" style="color:var(--green)">
    <div class="ptitle-left"><span class="ptitle-icon">✅</span> ENTRY CONDITIONS — ALL MUST BE MET</div>
    <button class="tog-btn" onclick="tog('ec-body','ec-chev','ec-lbl')">
      <span id="ec-chev" class="chev">▼</span>
      <span id="ec-lbl">SHOW</span>
    </button>
  </div>
  <div id="ec-body" class="coll-body">
    <div class="cond-grid" id="entry-conds"></div>
  </div>
</div>

<!-- ── EXIT CONDITIONS ────────────────────────────────────────────── -->
<div class="panel">
  <div class="ptitle" style="color:var(--red)">
    <div class="ptitle-left"><span class="ptitle-icon">🚪</span> EXIT CONDITIONS — ANY ONE TRIGGERS CLOSE</div>
    <button class="tog-btn" onclick="tog('xc-body','xc-chev','xc-lbl')">
      <span id="xc-chev" class="chev">▼</span>
      <span id="xc-lbl">SHOW</span>
    </button>
  </div>
  <div id="xc-body" class="coll-body">
    <div class="cond-grid" id="exit-conds"></div>
  </div>
</div>

<footer>
  ALLIGATOR ENGINE &nbsp;·&nbsp;
  <a href="/api/state"      target="_blank">API JSON</a> &nbsp;·&nbsp;
  <a href="/api/logs?n=500" target="_blank">RAW LOGS</a> &nbsp;·&nbsp;
  Auto-refresh 5 s
</footer>

</div><!-- /page -->

<script>
// ── Entry / Exit conditions ──────────────────────────────────────
const ENTRY_CONDITIONS = [
  __ENTRY_EXIT_CONDITIONS_PLACEHOLDER__
function renderConds(arr, id) {
  document.getElementById(id).innerHTML = arr.map(([n,name,cls,desc]) =>
    `<div class="cond">
      <span class="cn">${n}.</span>
      <span class="cname ${cls}">${name}</span>
      <span class="cdesc">${desc}</span>
    </div>`
  ).join('');
}
renderConds(ENTRY_CONDITIONS, 'entry-conds');
renderConds(EXIT_CONDITIONS,  'exit-conds');

// ── Collapsible ─────────────────────────────────────────────────
function tog(bodyId, chevId, lblId) {
  const body = document.getElementById(bodyId);
  const chev = document.getElementById(chevId);
  const lbl  = document.getElementById(lblId);
  const open = body.classList.toggle('open');
  chev.classList.toggle('open', open);
  lbl.textContent = open ? 'HIDE' : 'SHOW';
}

// ── Clock ────────────────────────────────────────────────────────
function tick() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-US',{
      hour:'2-digit',minute:'2-digit',second:'2-digit',
      hour12:false,timeZone:'America/New_York'
    }) + ' ET';
}
setInterval(tick,1000); tick();

// ── Next-scan countdown ─────────────────────────────────────────
let nextMs = null;
function countdown() {
  if (!nextMs) return;
  const s  = Math.max(0, Math.floor((nextMs - Date.now())/1000));
  const el = document.getElementById('s-nscan');
  el.textContent = String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
  el.className   = 'sval num ' + (s < 60 ? 'r' : s < 180 ? 'y' : 'g');
}
setInterval(countdown, 1000);

// ── Progress flash ──────────────────────────────────────────────
function flash() {
  const p = document.getElementById('progress');
  p.style.transition='none'; p.style.width='0%';
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    p.style.transition='width 4.8s linear'; p.style.width='100%';
  }));
}

// ── Formatters ──────────────────────────────────────────────────
const $f  = v => '$' + Math.abs(+v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const $fs = v => (v>=0?'+$':'-$') + Math.abs(+v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const pf  = v => (v>=0?'+':'') + (+v).toFixed(2) + '%';
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── PnL row renderer ────────────────────────────────────────────
function setPnlRow(suffix, data) {
  const bar = document.getElementById('pb-'+suffix);
  const amt = document.getElementById('pa-'+suffix);
  const pct = document.getElementById('pp-'+suffix);
  if (!data || data.amount == null) {
    bar.style.width='2px'; bar.style.background='var(--border2)';
    amt.textContent='—'; amt.className='pnl-amt neu';
    pct.textContent='—'; pct.className='pnl-pct neu';
    return;
  }
  const pos = data.amount >= 0;
  const col = pos ? 'var(--green)' : 'var(--red)';
  bar.style.width = Math.min(100, Math.abs(data.pct)/5*100).toFixed(1)+'%';
  bar.style.background = col;
  amt.textContent = $fs(data.amount); amt.className = 'pnl-amt num '+(pos?'pos':'neg');
  pct.textContent = pf(data.pct);    pct.className = 'pnl-pct num '+(pos?'pos':'neg');
}

// ── Main render ─────────────────────────────────────────────────
function render(d) {

  // header
  const badge = document.getElementById('mode-badge');
  if (d.paper_mode != null) {
    badge.textContent = d.paper_mode ? 'PAPER' : 'LIVE';
    badge.className   = 'mode-badge' + (d.paper_mode ? '' : ' live');
  }
  const connDot  = document.getElementById('conn-dot');
  const connText = document.getElementById('conn-text');
  if (d.connected) {
    connDot.className='dot dot-on'; connText.textContent='CONNECTED'; connText.className='g';
  } else {
    connDot.className='dot dot-off'; connText.textContent='DISCONNECTED'; connText.className='r';
  }
  const mktDot  = document.getElementById('mkt-dot');
  const mktText = document.getElementById('mkt-text');
  if (d.market_open) {
    mktDot.className='dot dot-on'; mktText.textContent='MARKET OPEN'; mktText.className='g';
  } else {
    mktDot.className='dot dot-warn'; mktText.textContent='MARKET CLOSED'; mktText.className='y';
  }
  if (d.last_updated) {
    document.getElementById('lu').textContent =
      new Date(d.last_updated).toLocaleTimeString('en-US',
        {timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  }

  // metrics strip
  document.getElementById('m-equity').textContent  = $f(d.equity||0);
  document.getElementById('m-cash').textContent    = $f(d.cash||0);
  document.getElementById('m-mktval').textContent  = $f(d.mkt_value||0);

  const slots = (d.entry_slots != null) ? d.entry_slots : Math.max(0,(d.max_positions||0)-d.position_count);
  document.getElementById('m-cash-sub').innerHTML =
    `<span class="${slots>0?'pos':'neu'}">${slots} entry slot${slots!==1?'s':''} available</span>`;

  document.getElementById('m-alloc-sub').innerHTML =
    `<span class="p">${(d.allocation_pct||0).toFixed(1)}%</span>&nbsp;<span class="neu">of equity</span>`;

  const tu  = d.total_unrealized ?? 0;
  const uEl = document.getElementById('m-unreal');
  uEl.textContent = (tu>=0?'+$':'-$') + Math.abs(tu).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  uEl.className   = 'metric-val num ' + (tu>0?'pos':tu<0?'neg':'neu');
  const uPct = d.equity ? (tu/d.equity*100) : 0;
  document.getElementById('m-unreal-sub').innerHTML =
    `<span class="${tu>=0?'pos':'neg'}">${tu>=0?'▲':'▼'} ${Math.abs(uPct).toFixed(2)}%</span>`+
    `&nbsp;<span class="neu">${d.position_count||0} position${d.position_count!==1?'s':''}</span>`;

  const vix = d.vix, vThr = d.vix_threshold ?? 35;
  const vEl  = document.getElementById('m-vix');
  const vSub = document.getElementById('m-vix-sub');
  if (vix != null) {
    vEl.textContent = (+vix).toFixed(2);
    if (vix > vThr) {
      vEl.className='metric-val num neg';
      vSub.innerHTML='<span class="neg">⚠ RISK-OFF — NO NEW ENTRIES</span>';
    } else if (vix > vThr*0.71) {
      vEl.className='metric-val num warn';
      vSub.innerHTML=`<span class="warn">ELEVATED</span>&nbsp;<span class="neu">threshold ${vThr}</span>`;
    } else {
      vEl.className='metric-val num pos';
      vSub.innerHTML=`<span class="pos">NORMAL</span>&nbsp;<span class="neu">threshold ${vThr}</span>`;
    }
  } else {
    vEl.textContent='—'; vEl.className='metric-val num d';
    vSub.innerHTML='<span class="neu">unavailable</span>';
  }

  // P&L rows
  if (d.pnl) {
    setPnlRow('daily',   d.pnl.daily);
    setPnlRow('weekly',  d.pnl.weekly);
    setPnlRow('monthly', d.pnl.monthly);
    setPnlRow('overall', d.pnl.overall);
  }

  // Status panel
  document.getElementById('s-api').innerHTML = d.connected
    ? '<span class="g">● CONNECTED</span>'
    : '<span class="r">● DISCONNECTED</span>';
  const smkt = document.getElementById('s-mkt');
  smkt.textContent = d.market_open ? 'OPEN' : 'CLOSED';
  smkt.className   = 'sval num '+(d.market_open?'g':'y');
  document.getElementById('s-pos').innerHTML =
    `<span class="c">${d.position_count||0}</span><span class="d"> / ${d.max_positions||'—'} slots</span>`;
  const sb = document.getElementById('s-bucket');
  sb.textContent = d.bucket_size>0 ? $f(d.bucket_size) : '—';
  sb.className   = 'sval num c';

  // Entry window — times driven by ENTRY_START / ENTRY_END from /api/state
  const ny      = new Date(new Date().toLocaleString('en-US',{timeZone:'America/New_York'}));
  const h       = ny.getHours(), mn = ny.getMinutes();
  const isWkd   = ny.getDay()===0||ny.getDay()===6;
  const [sh,sm] = d.entry_start || [10,5];
  const [eh,em] = d.entry_end   || [14,0];
  const fmt2    = n => String(n).padStart(2,'0');
  const afterStart = h > sh || (h === sh && mn >= sm);
  const beforeEnd  = h < eh || (h === eh && mn <= em);
  const inWin      = d.market_open && afterStart && beforeEnd;
  const swin  = document.getElementById('s-window');
  if (isWkd)             { swin.textContent='WEEKEND';        swin.className='sval num y'; }
  else if (!d.market_open) { swin.textContent='CLOSED';       swin.className='sval num y'; }
  else if (inWin)        { swin.textContent=`ACTIVE ${fmt2(sh)}:${fmt2(sm)}–${fmt2(eh)}:${fmt2(em)}`; swin.className='sval num g'; }
  else                   { swin.textContent='OUTSIDE WINDOW'; swin.className='sval num d'; }

  document.getElementById('s-lscan').textContent = d.last_scan||'—';
  if (d.next_scan) { nextMs=new Date(d.next_scan).getTime(); countdown(); }

  // Portfolio
  const tb   = document.getElementById('tbody');
  const hbar = d.hold_trading_bars ?? 2;

  if (!d.positions || d.positions.length===0) {
    tb.innerHTML='<tr class="empty"><td colspan="12">No open positions — scanning for signals</td></tr>';
    document.getElementById('risk-strip').style.display='none';
    document.getElementById('pos-summary').textContent='';
    return;
  }

  // risk aggregates
  let totalRisk=0, totalDeployed=0, stopDists=[];
  for (const p of d.positions) {
    totalDeployed += p.current_price*p.qty;
    if (p.stop_loss>0) {
      totalRisk += (p.current_price-p.stop_loss)*p.qty;
      stopDists.push((p.current_price-p.stop_loss)/p.current_price*100);
    }
  }
  const avgDist = stopDists.length ? stopDists.reduce((a,b)=>a+b,0)/stopDists.length : 0;
  document.getElementById('risk-strip').style.display='grid';
  document.getElementById('r-maxloss').textContent  = '-'+$f(totalRisk);
  document.getElementById('r-pct').textContent      = (totalRisk/(d.equity||1)*100).toFixed(2)+'%';
  document.getElementById('r-deployed').textContent = $f(totalDeployed);
  document.getElementById('r-avgstop').textContent  = avgDist.toFixed(2)+'%';

  document.getElementById('pos-summary').innerHTML =
    `<span class="num">${d.position_count}</span> position${d.position_count!==1?'s':''}&nbsp;·&nbsp;`+
    `P&amp;L: <span class="num ${tu>=0?'g':'r'}">${$fs(tu)}</span>`;

  tb.innerHTML = d.positions.map(p => {
    const unr  = p.unrealized ?? 0;
    const unrP = p.unrealized_pct ?? 0;
    const ucls = unr>0?'pos':unr<0?'neg':'neu';
    const chg  = p.entry_price>0 ? (p.current_price-p.entry_price)/p.entry_price*100 : 0;

    // score bar
    const sc    = p.score!=null ? p.score.toFixed(1) : '—';
    const scCol = p.score!=null ? (p.score>=70?'var(--green)':p.score>=45?'var(--yellow)':'var(--red)') : 'var(--text3)';
    const scW   = p.score!=null ? p.score.toFixed(0) : 0;

    // risk
    const riskUsd = p.stop_loss>0 ? (p.current_price-p.stop_loss)*p.qty : 0;
    const riskPct = p.stop_loss>0&&p.current_price>0 ? (p.current_price-p.stop_loss)/p.current_price*100 : 0;
    const rCls    = riskPct>5?'pos':riskPct>2?'warn':'neg';

    // hold
    const hd     = p.hold_trading_days ?? 0;
    const hh     = ((+p.hold_hours||0)%24).toFixed(0);
    const velWin = hd >= hbar;

    // status badges
    let badges='';
    if (p.stop_loss >= p.entry_price) badges += '<span class="badge badge-be">BREAK-EVEN ↑</span> ';
    if (velWin && unrP < 5)           badges += '<span class="badge badge-vel">VEL WINDOW ⚠</span> ';
    if (unrP >= 10)                    badges += `<span class="badge badge-strong">STRONG +${unrP.toFixed(1)}%</span> `;

    // mini pnl bar
    const mW = Math.min(100, Math.abs(unrP)/10*100).toFixed(0);
    const mC = unr>=0 ? 'var(--green)' : 'var(--red)';

    // analyst cell
    const ab = p.analyst_buy  ?? 0;
    const ah = p.analyst_hold ?? 0;
    const as_ = p.analyst_sell ?? 0;
    const tot = ab + ah + as_;
    const anaHtml = tot > 0
      ? `<span class="g" style="font-weight:700">${ab}B</span>`+
        `<span class="d"> / </span>`+
        `<span class="y">${ah}H</span>`+
        `<span class="d"> / </span>`+
        `<span class="r">${as_}S</span>`
      : `<span class="d" style="font-size:10px">—</span>`;

    return `<tr>
      <td><span class="sym">${p.symbol}</span></td>
      <td>
        <div class="sc-wrap">
          <div class="sc-bar" style="width:${scW}%;max-width:44px;background:${scCol}"></div>
          <span style="color:${scCol};font-weight:700">${sc}</span>
        </div>
      </td>
      <td style="text-align:center">${anaHtml}</td>
      <td class="d">${$f(p.entry_price)}</td>
      <td>
        <div class="prc-cell">
          <span style="font-weight:700">${$f(p.current_price)}</span>
          <span class="prc-chg ${chg>=0?'pos':'neg'}">${chg>=0?'▲':'▼'} ${Math.abs(chg).toFixed(2)}%</span>
        </div>
      </td>
      <td>${parseFloat((+p.qty).toFixed(4))}</td>
      <td class="d">${$f(p.total_amount)}</td>
      <td>
        <div class="unr-cell">
          <span class="num ${ucls}" style="font-weight:700">${$fs(unr)}</span>
          <span class="num ${ucls}" style="font-size:10px">${pf(unrP)}</span>
          <div class="unr-mini"><div class="unr-fill" style="width:${mW}%;background:${mC}"></div></div>
        </div>
      </td>
      <td><span class="num r" style="font-weight:700">${$f(p.stop_loss)}</span></td>
      <td>
        <div class="risk-cell">
          <span class="num ${rCls}" style="font-weight:700">-${$f(riskUsd)}</span>
          <span class="risk-dist ${rCls}">${riskPct.toFixed(2)}% to stop</span>
        </div>
      </td>
      <td>
        <div class="hold-cell">
          <span class="num ${velWin?'warn':'d'}">${hd}d ${hh}h</span>
          ${velWin ? '<span class="vel-tag">VEL WINDOW</span>' : ''}
        </div>
      </td>
      <td class="ui-font" style="text-align:right">${badges||'<span class="d" style="font-size:10px">—</span>'}</td>
    </tr>`;
  }).join('');
}

// ── Equity chart ─────────────────────────────────────────────────
let eqChart = null;
async function refreshChart() {
  try {
    const r = await fetch('/api/equity_history');
    if (!r.ok) return;
    const hist = await r.json();
    if (!hist || hist.length===0) return;
    const labels = hist.map(e => new Date(e.ts).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'America/New_York'}));
    const data   = hist.map(e => e.equity);
    const isUp   = data[data.length-1] >= data[0];
    const lc     = isUp ? '#10b981' : '#ef4444';
    const ctx    = document.getElementById('eqChart').getContext('2d');
    const grad   = ctx.createLinearGradient(0,0,0,240);
    grad.addColorStop(0, isUp ? 'rgba(16,185,129,.18)' : 'rgba(239,68,68,.18)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    if (eqChart) {
      eqChart.data.labels=labels;
      eqChart.data.datasets[0].data=data;
      eqChart.data.datasets[0].borderColor=lc;
      eqChart.data.datasets[0].backgroundColor=grad;
      eqChart.data.datasets[0].pointRadius=hist.length>30?0:2;
      eqChart.update('none');
    } else {
      eqChart = new Chart(ctx, {
        type:'line',
        data:{labels, datasets:[{
          data, borderColor:lc, backgroundColor:grad,
          borderWidth:2, pointRadius:hist.length>30?0:2,
          pointHoverRadius:4, tension:0.3, fill:true,
        }]},
        options:{
          responsive:true, maintainAspectRatio:false, animation:{duration:600},
          plugins:{
            legend:{display:false},
            tooltip:{
              backgroundColor:'#101520', borderColor:'#253850', borderWidth:1,
              titleColor:'#7b92a8', bodyColor:'#c8d6e5',
              callbacks:{label:c=>' $'+c.parsed.y.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
            }
          },
          scales:{
            x:{ticks:{color:'#4a5f72',font:{size:10},maxTicksLimit:10}, grid:{color:'#1e2d42'}, border:{color:'#1e2d42'}},
            y:{ticks:{color:'#4a5f72',font:{size:10},
                callback:v=>'$'+v.toLocaleString('en-US',{maximumFractionDigits:0})},
               grid:{color:'#1e2d42'}, border:{color:'#1e2d42'}}
          }
        }
      });
    }
  } catch(e) { /* silent */ }
}
refreshChart();
setInterval(refreshChart, 60000);

// ── Log viewer ───────────────────────────────────────────────────
async function refreshLogs() {
  try {
    const r = await fetch('/api/logs?n=60');
    if (!r.ok) return;
    const data  = await r.json();
    const box   = document.getElementById('log-lines');
    const atBot = box.scrollHeight - box.scrollTop - box.clientHeight < 50;
    box.innerHTML = (data.lines||[]).map(line => {
      let cls='ln';
      if (/ERROR|CRASH|FATAL/i.test(line))              cls='le';
      else if (/WARNING|WARN|CIRCUIT|VIX HIGH/i.test(line)) cls='lw';
      else if (/\bBUY\b|ENTRY|SIGNAL|FILLED.*BUY/i.test(line)) cls='lb';
      else if (/\bSELL\b|LIQUIDATE|EXIT|FILLED.*SELL/i.test(line)) cls='ls';
      else if (/HEARTBEAT/i.test(line))                 cls='lh';
      else if (/TRAIL/i.test(line))                     cls='lt';
      return `<span class="ll ${cls}">${esc(line)}</span>`;
    }).join('');
    if (atBot) box.scrollTop = box.scrollHeight;
  } catch(e) { /* silent */ }
}
refreshLogs();
setInterval(refreshLogs, 8000);

// ── Fetch loop ────────────────────────────────────────────────────
async function refresh() {
  flash();
  try {
    const r = await fetch('/api/state');
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  } catch(e) {
    document.getElementById('conn-text').textContent='SERVER OFFLINE';
    document.getElementById('conn-text').className='r';
    document.getElementById('conn-dot').className='dot dot-off';
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""

# Inject config-driven condition descriptions so the dashboard always reflects
# current strategy parameters from config.py.
_COND_JS = (
    f'["1",  "Alligator Align",   "en", "Fast SMMA({ALLIGATOR_FAST}) > Med SMMA({ALLIGATOR_MED}) > Slow SMMA({ALLIGATOR_SLOW}) — all three lines stacked bullish (mouth open upward)"],\n'
    f'  ["2",  "Fresh Crossover",  "en", "Bullish crossover occurred within the last {ALLIGATOR_CROSS_LOOKBACK} bars — catches the start of the move, not an exhausted tail"],\n'
    f'  ["3",  "RSI Trend",        "en", "RSI({RSI_PERIOD}) ≥ 50 AND rising ≥ {RSI_MIN_DELTA:.0f} pt — stock in bullish territory with building momentum, not yet overextended"],\n'
    f'  ["4",  "Scanner Universe", "en", "Alpaca scan: Top gainers + most actives + full Alligator crossover universe | Avg vol > {SCAN_MIN_VOLUME/1e6:.0f}M shares | Avg dollar vol > ${SCAN_MIN_DOLLAR_VOL/1e6:.0f}M/day"],\n'
    f'  ["5",  "Spread Filter",    "en", "Bid-ask spread ≤ {SPREAD_MAX_PCT*100:.1f}% — filters illiquid stocks where slippage would erode edge"],\n'
    f'  ["6",  "RVOL Gate",        "en", "Intraday relative volume ≥ {RVOL_MIN:.1f}× (time-adjusted CDF normalizer) — confirms unusual buying activity above typical pace"],\n'
    f'  ["7",  "Day Strength",     "en", "Price ≥ {DAY_STRENGTH_OPEN_PCT*100:.1f}% above today\'s open — sustained buying pressure, not a dead-cat bounce fading into close"],\n'
    f'  ["8",  "VIX Filter",       "en", "VIX ≤ {VIX_THRESHOLD} — VIX > {VIX_THRESHOLD} suspends all new entries (Risk-Off regime, Alligator trends less reliable)"],\n'
    f'  ["9",  "Session Window",   "en", "Entries {ENTRY_START[0]:02d}:{ENTRY_START[1]:02d}–{ENTRY_END[0]:02d}:{ENTRY_END[1]:02d} ET Mon–Fri — avoids opening volatility and late-day reversals"],\n'
    f'  ["10", "Position Limit",   "en", "Max {MAX_POSITIONS_CAP} positions — dynamic: floor(equity/${MIN_BUCKET_SIZE:.0f}), capped at {MAX_POSITIONS_CAP}. Max {MAX_SECTOR_COUNT} per sector. Settled cash gates T+1 entries."],\n'
    f'  ["11", "Score Gate",       "en", "Composite score ≥ {SCAN_MIN_SCORE:.0f}/100 required: Alligator {SCORE_ALLIGATOR_MAX:.0f}pts + RVOL {SCORE_RVOL_MAX:.0f}pts + RSI {SCORE_RSI_DELTA_MAX:.0f}pts + Liquidity {SCORE_LIQUIDITY_MAX:.0f}pts + Analyst Consensus {SCORE_ANALYST_MAX:.0f}pts bonus (total capped 100)"]\n'
    f'];\n'
    f'const EXIT_CONDITIONS = [\n'
    f'  ["1", "Chandelier Trail",   "ex", "TRAIL SELL at ATR({CHANDELIER_PERIOD})×{CHANDELIER_MULT:.1f} from peak price — Alpaca raises the stop automatically as price climbs; dollar distance fixed at entry"],\n'
    f'  ["2", "Alligator Reversal", "ex", "Fast SMMA({ALLIGATOR_FAST}) AND Med SMMA({ALLIGATOR_MED}) both cross below Slow SMMA({ALLIGATOR_SLOW}) — confirmed bearish reversal, trend structure broken"],\n'
    f'  ["3", "Hard Stop",          "ex", "{HARD_STOP_PCT*100:.0f}% drawdown from fill price triggers immediate Market SELL regardless of ATR stop distance"],\n'
    f'  ["4", "Break-Even Floor",   "ex", "Once profit ≥ {BREAK_EVEN_PCT*100:.0f}%, chandelier stop is floored at fill price — software-enforced, prevents winners from becoming losers"],\n'
    f'  ["5", "VIX Risk-Off",       "ex", "VIX > {VIX_THRESHOLD} blocks new entries; existing positions exit via chandelier stop or hard stop as normal"],\n'
    f'  ["6", "Daily Loss Halt",    "ex", "{MAX_DAILY_LOSS_PCT*100:.0f}% intraday equity drawdown halts all new entries for the rest of the trading day (circuit breaker)"]\n'
    f'];\n'
)
_HTML = _HTML.replace("  __ENTRY_EXIT_CONDITIONS_PLACEHOLDER__", _COND_JS)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


@app.get("/health")
def health():
    """Liveness probe for cloud platforms (Render, Railway, Fly.io)."""
    return JSONResponse({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AlligatorAlpha Web Dashboard")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", default=int(os.getenv("PORT", "8080")), type=int,
                   help="Port (default: $PORT env or 8080)")
    args = p.parse_args()

    print("\n  🐊  AlligatorAlpha Dashboard")
    print(f"  Open → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
