"""
Pytest configuration:
  1. Suppress engine logger so CRITICAL/WARNING messages from mocked failure
     paths never bleed into the live logs/ directory.
  2. Redirect all production file paths (state, dashboard, equity history, log)
     to per-test tmp directories so tests can never corrupt live data files.
  3. Redirect the engine module's file handler (installed at import time) to the
     temp log dir so test runs never write to logs/trading_engine.log.
"""

import logging
import os
import pytest
import src.config as cfg
import src.engine as eng


@pytest.fixture(autouse=True)
def silence_engine_logger():
    """Raise the engine logger's level to CRITICAL+1 for every test."""
    logger = logging.getLogger("VelocityEngine")
    original = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    yield
    logger.setLevel(original)


@pytest.fixture(autouse=True)
def isolate_production_files(tmp_path):
    """
    Redirect STATE_FILE, DASHBOARD_FILE, EQUITY_HIST_FILE, LOG_DIR and LOG_FILE
    to a per-test temp directory so tests can never touch live data or log files.
    Also swaps the engine's file handler (installed at import time with the
    production path) to a throwaway temp-dir handler.
    """
    orig_state   = cfg.STATE_FILE
    orig_dash    = cfg.DASHBOARD_FILE
    orig_equity  = cfg.EQUITY_HIST_FILE
    orig_log_dir = cfg.LOG_DIR
    orig_log     = cfg.LOG_FILE

    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)
    cfg.STATE_FILE       = str(tmp_path / "engine_state.json")
    cfg.DASHBOARD_FILE   = str(tmp_path / "dashboard_data.json")
    cfg.EQUITY_HIST_FILE = str(tmp_path / "equity_history.json")
    cfg.LOG_DIR          = log_dir
    cfg.LOG_FILE         = str(tmp_path / "logs" / "trading_engine.log")

    eng.STATE_FILE       = cfg.STATE_FILE
    eng.DASHBOARD_FILE   = cfg.DASHBOARD_FILE
    eng.EQUITY_HIST_FILE = cfg.EQUITY_HIST_FILE
    eng.LOG_DIR          = cfg.LOG_DIR
    eng.LOG_FILE         = cfg.LOG_FILE

    # The engine module installs a TimedRotatingFileHandler pointing at the
    # production log path at import time.  Swap it out so test runs don't
    # write to logs/trading_engine.log even if log-level guards are bypassed.
    engine_logger  = logging.getLogger("VelocityEngine")
    removed_handlers = [h for h in engine_logger.handlers[:]
                        if isinstance(h, logging.FileHandler)]
    for h in removed_handlers:
        engine_logger.removeHandler(h)
        h.close()
    temp_fh = logging.FileHandler(cfg.LOG_FILE)
    engine_logger.addHandler(temp_fh)

    yield

    engine_logger.removeHandler(temp_fh)
    temp_fh.close()
    for h in removed_handlers:
        engine_logger.addHandler(h)

    cfg.STATE_FILE       = orig_state
    cfg.DASHBOARD_FILE   = orig_dash
    cfg.EQUITY_HIST_FILE = orig_equity
    cfg.LOG_DIR          = orig_log_dir
    cfg.LOG_FILE         = orig_log

    eng.STATE_FILE       = orig_state
    eng.DASHBOARD_FILE   = orig_dash
    eng.EQUITY_HIST_FILE = orig_equity
    eng.LOG_DIR          = orig_log_dir
    eng.LOG_FILE         = orig_log
