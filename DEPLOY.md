# Deploying VelocityEngine to a Free Cloud

Both the trading engine and the dashboard run in a **single service** on the same
container — they share the filesystem, so no database or message queue is needed.

The free URL you get looks like:
- Render:  `https://velocity-trading-bot.onrender.com`
- Railway: `https://velocity-trading-bot.up.railway.app`

---

## Option A — Render (recommended free tier)

Render gives you a **free always-on web service** with a `*.onrender.com` HTTPS URL.
Free tier sleeps after 15 min of no HTTP traffic — use [UptimeRobot](https://uptimerobot.com)
(free) to ping `/health` every 5 minutes and keep it awake.

### Steps

1. Push your code to a GitHub repository.

2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo.

3. Render auto-detects `render.yaml` and pre-fills the settings. Confirm:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python main.py`
   - **Health check path**: `/health`

4. Add environment variables under **Environment**:
   ```
   ALPACA_API_KEY      = <your key>
   ALPACA_SECRET_KEY   = <your secret>
   ALPACA_PAPER        = true
   ALPACA_DATA_FEED    = iex
   ```

5. Click **Create Web Service**. Wait ~2 min for the first deploy.

6. Open your `*.onrender.com` URL — the dashboard loads immediately.

7. **Keep-alive (optional but recommended for free tier)**:
   Sign up at [uptimerobot.com](https://uptimerobot.com) → Add monitor →
   HTTP(s) → URL: `https://your-app.onrender.com/health` → Interval: 5 min.

---

## Option B — Railway

Railway gives $5/month free credit. At ~$0.50/day the bot stays within the credit
if it runs only during market hours (bot auto-idles when markets are closed).

### Steps

1. Push your code to GitHub.

2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.

3. Railway auto-detects `railway.toml`. Add environment variables in the **Variables** tab:
   ```
   ALPACA_API_KEY      = <your key>
   ALPACA_SECRET_KEY   = <your secret>
   ALPACA_PAPER        = true
   ALPACA_DATA_FEED    = iex
   ```

4. Under **Settings → Networking**, click **Generate Domain** to get a
   `*.up.railway.app` URL.

5. Deploy. The dashboard is live at your generated URL.

---

## Option C — Docker (any VPS / self-hosted)

Use this if you have a free Oracle Cloud Always Free VM or any VPS.

```bash
# Build
docker build -t velocity-bot .

# Run
docker run -d \
  -p 8080:8080 \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  -e ALPACA_PAPER=true \
  -e ALPACA_DATA_FEED=iex \
  --name velocity-bot \
  velocity-bot

# Dashboard → http://your-server-ip:8080
```

For persistent state across container restarts, mount the data directory:
```bash
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e ALPACA_API_KEY=your_key \
  ...
```

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALPACA_API_KEY` | Yes | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes | — | Alpaca secret key |
| `ALPACA_PAPER` | No | `true` | `true` = paper, `false` = live |
| `ALPACA_DATA_FEED` | No | `iex` | `iex` (free) or `sip` (paid) |
| `PORT` | No | `8080` | Set automatically by cloud platforms |

Copy `.env.example` to `.env` for local development (never commit `.env`).

---

## Local development

```bash
# Install dependencies
python -m venv venv
venv/bin/pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env
# edit .env

# Run everything (engine + dashboard)
venv/bin/python main.py
# Dashboard → http://localhost:8080

# Run tests
venv/bin/python -m pytest tests/ -q
```

---

## Architecture notes

- `main.py` starts the trading engine in a background thread and uvicorn (dashboard) in the main process.
- Both share the local filesystem; JSON state files are written by the engine and read by the dashboard.
- State files (`engine_state.json`, `dashboard_data.json`) are ephemeral on free-tier cloud — lost on restart. The engine re-syncs open positions from Alpaca on startup automatically.
- `logs/` is created at startup if it doesn't exist.
