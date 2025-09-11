Project Brief: AI Crypto Trading Agent

Overview
- React + TypeScript dashboard that unifies crypto market data, paper trading, portfolio persistence, and AI insights.
- Integrations: CoinGecko (markets, global), Alpaca (paper trading), Supabase (auth + DB), Groq (AI analysis).

Primary Goals
- Authenticated users view account summary, positions, orders, watchlist, global stats, and AI insights.
- Manual market/limit orders via Alpaca paper trading.
- Reliable fallbacks for external API failures to keep the demo functional.
- Enforce Supabase RLS so users only access their own data.

Non‑Goals (current scope)
- Live websockets or HFT; multi-exchange execution; on-chain wallet flows; complex risk engines.

Target Users
- Individual traders and builders evaluating a cohesive trading dashboard with common integrations.

Success Criteria
- Auth works; dashboard loads with live or fallback data.
- Orders can be placed and appear in recent orders.
- AI insights render with clear sentiment and confidence.
- API status indicator and modal communicate connectivity and issues.

High-Level Features
- Auth (Supabase), Market data (CoinGecko), Trading (Alpaca), Portfolio tables (Supabase), AI (Groq), Draggable dashboard (react-grid-layout), Tailwind UI.

Risks & Constraints
- VITE_* env vars required at build/runtime; external rate limits; RLS misconfig can block data; browser fetch subject to CORS/network.


