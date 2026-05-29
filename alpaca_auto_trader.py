"""
VelocityEngine — entry point
────────────────────────────
All logic lives in src/engine.py.
Run:  venv/bin/python alpaca_auto_trader.py
"""

import sys
import os
import signal
import logging
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.engine import VelocityEngine

# Reuse the engine's named logger so restart/crash messages land in the same
# rotating log file (trading_engine.log) and are not silently dropped when
# running under nohup or systemd where stdout is not captured.
logger = logging.getLogger('VelocityEngine')

_engine: VelocityEngine | None = None


def _handle_shutdown(signum, frame):
    """Graceful shutdown: cancel pending orders, then exit."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} — initiating graceful shutdown.")
    if _engine is not None:
        try:
            _engine.shutdown()
        except Exception as exc:
            logger.error(f"Error during shutdown: {exc}")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers only when run as the entry point, not on import.
    # signal.signal() can only be called from the main thread; registering at
    # module level would silently overwrite the caller's handlers if
    # alpaca_auto_trader were ever imported rather than executed directly.
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT,  _handle_shutdown)

    while True:
        try:
            _engine = VelocityEngine()
            _engine.run()
            # run() returned normally — exit cleanly
            break
        except KeyboardInterrupt:
            # SIGINT already handled above, but catch here as safety net
            break
        except Exception as exc:
            logger.exception(f"Unhandled exception in VelocityEngine.run(): {exc}")
            logger.info("Restarting engine in 60 seconds …")
            time.sleep(60)
