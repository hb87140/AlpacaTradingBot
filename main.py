"""
Combined launcher for cloud deployment.

Starts the trading engine in a background thread and the FastAPI dashboard
in the main process. Both share the local filesystem for JSON state files.

Usage (local):  python main.py
Usage (cloud):  set PORT env var; cloud platforms do this automatically.
"""

import os
import sys
import threading
import logging
import time

# ── Engine thread ─────────────────────────────────────────────────────────────

def _run_engine():
    """Restart-looping engine worker — mirrors alpaca_auto_trader.py logic."""
    from src.engine import VelocityEngine
    logger = logging.getLogger('VelocityEngine')
    while True:
        try:
            engine = VelocityEngine()
            engine.run()
            break  # clean exit
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Engine crashed — restarting in 60 s")
            time.sleep(60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # Ensure log directory exists before engine starts writing
    from src.config import LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    # Start the engine in a daemon thread so it dies when the main process exits
    engine_thread = threading.Thread(target=_run_engine, daemon=True, name="TradingEngine")
    engine_thread.start()

    # Start the dashboard in the main process (uvicorn must own the main thread)
    port = int(os.getenv("PORT", "8080"))
    import uvicorn
    from alpaca_dashboard import app
    print(f"\n  VelocityEngine — trading engine + dashboard")
    print(f"  Dashboard → http://0.0.0.0:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
