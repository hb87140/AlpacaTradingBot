#!/usr/bin/env bash
# Download the day's trading log from Render after market close.
# Saves to logs/remote/trading_engine_YYYYMMDD.log
#
# Usage:
#   bash scripts/download_daily_log.sh              # downloads today's log
#   bash scripts/download_daily_log.sh 2026-06-04   # specific date (for docs only, filename)
#
# Cron (runs Mon–Fri at 16:20 ET — adjust timezone offset for your shell):
#   20 16 * * 1-5 /home/harika/MyLearning/AI/AlpacaTradingBot/scripts/download_daily_log.sh >> /home/harika/MyLearning/AI/AlpacaTradingBot/logs/download_cron.log 2>&1

set -euo pipefail

RENDER_URL="https://alpacatradingbot-pe9m.onrender.com/api/logs/download"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_LOG_DIR="$REPO_DIR/logs/remote"
DATE="${1:-$(date +%Y-%m-%d)}"
OUTFILE="$REMOTE_LOG_DIR/trading_engine_${DATE}.log"

mkdir -p "$REMOTE_LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading log for $DATE from Render..."

HTTP_CODE=$(curl -s -o "$OUTFILE" -w "%{http_code}" "$RENDER_URL")

if [ "$HTTP_CODE" = "200" ]; then
    SIZE=$(wc -l < "$OUTFILE")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saved $SIZE lines → $OUTFILE"
elif [ "$HTTP_CODE" = "404" ]; then
    rm -f "$OUTFILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: log file not found on Render (404)."
    exit 1
else
    rm -f "$OUTFILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: unexpected HTTP $HTTP_CODE from Render."
    exit 1
fi
