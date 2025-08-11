AI Crypto Trading Agent (Vite + React + TypeScript)

Overview
This is a client‑side crypto trading dashboard that unifies live market data, manual and automated trading (paper), portfolio persistence (optional), and AI‑assisted market insights.

Tech stack
- React 18 + TypeScript (Vite)
- TailwindCSS UI
- Alpaca (paper trading REST v2)
- CoinGecko markets (public or Pro key)
- Groq LLM for AI insights
- Supabase (auth + Postgres) optional

Key features
- Automated trading agent (opt‑in)
  - Momentum + indicator blend, risk gating (drawdown pause, correlation cap)
  - Cooldown control and confidence threshold
- Manual trading controls
  - Market and limit orders via Alpaca paper API
  - Quick buy/sell actions, favorite trades
- AI Market Insights
  - LLM‑generated insights and overall market sentiment
  - Blended confidence score (LLM + price breadth/volatility + naive news tone)
- Whale Alerts panel (large transfers)
- Performance Analytics (v1)
  - Realized trade pairing (FIFO long), daily return series, Sharpe, win rate, best/worst
- Draggable/resizable dashboard with saved layouts
- API Status modal with live probes
- Auth via Supabase (optional demo mode supported)

Prerequisites
- Node.js >= 18 (recommended 18.18+)
- npm >= 9

Environment variables (VITE_*)
Create a .env file in the project root:

```
# Supabase (optional)
VITE_SUPABASE_URL="https://<your-project>.supabase.co"
VITE_SUPABASE_ANON_KEY="<supabase-anon-key>"

# CoinGecko
# If you have a Pro key, also set VITE_USE_COINGECKO_PRO=true
VITE_COINGECKO_API_KEY="<optional-pro-key>"
VITE_USE_COINGECKO_PRO="false"

# Alpaca paper trading
VITE_ALPACA_API_KEY="<alpaca-key>"
VITE_ALPACA_SECRET_KEY="<alpaca-secret>"

# Groq LLM
VITE_GROQ_API_KEY="<groq-api-key>"

# (Optional) Whale Alert API key
VITE_WHALE_ALERT_API_KEY="<optional>"
```

Notes
- This app is client‑only. VITE_* values are exposed in the browser by design. Never put secrets that must remain private at runtime.
- Without some keys the UI will use curated fallbacks (keeps the demo working):
  - CoinGecko throttled → fallback market snapshot
  - Alpaca errors → empty tables + demo account
  - Groq errors → fallback insights
  - Whale Alert missing key → sample events

Install & run
```
npm ci
npm run dev
```
Open http://localhost:5173

Build & preview
```
npm run build
npm run preview
```

Lint (optional)
```
npm run lint
```

Deployment (Netlify, prebuilt dist)
1) Ensure .env is present locally and build:
```
npm run build
```
2) Deploy dist/ to Netlify:
```
npx netlify-cli deploy --prod --dir=dist
```
or drag‑and‑drop dist/ in the Netlify UI. For SPA deep links, add a file dist/_redirects with:
```
/* /index.html 200
```

How the agent works (high level)
- Signals: momentum + RSI/MAs/MACD + volume + rank/ATH distance
- Risk manager: drawdown pause, correlation cap (BTC/ETH), confidence‑scaled sizing, leverage suggestion
- Trades: agent submits market orders via Alpaca when confidence exceeds threshold and cooldown allows

Performance analytics (v1)
- Pairs BUY/SELL fills per symbol FIFO to compute realized PnL and trade returns
- Aggregates daily returns for Sharpe
- Stores/reads metrics from Supabase table performance_metrics if enabled; otherwise computes on the fly

Troubleshooting
- CoinGecko 429: The service retries once, then falls back to a curated snapshot. Provide VITE_COINGECKO_API_KEY and set VITE_USE_COINGECKO_PRO=true to reduce throttling.
- CORS: This app avoids serverless calls. News tone is computed naively from available headlines; Groq insights don’t require server proxies.
- Alpaca credentials: Use paper API keys and ensure crypto trading is enabled on your account.

Roadmap / ideas
- Binance Futures integration + unified exchange manager
- Backtesting + paper trading framework
- Multi‑model ML ensemble + regime detection
- Real trailing stops / OCOs via exchange
- News/social ingestion from multiple sources

License
MIT

# tradingbot
