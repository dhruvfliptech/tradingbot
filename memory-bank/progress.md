Progress

What works
- Supabase auth flow and session handling via `useAuth`.
- Dashboard layout with draggable/resizable widgets and persistent layouts.
- Market data from CoinGecko with robust fallback; Global market header; Fear & Greed index.
- Trading agent mock signals; AI market insights via Groq with fallback.
- Manual order placement against Alpaca paper trading.
- API status indicator + detailed modal probes.

What’s left / enhancements
- Unify order/position persistence: UI uses Alpaca for placing orders; Supabase has schema and portfolio service but not wired into UI views.
- Add refresh cadence or live updates for account/positions/orders.
- Improve error surfaces around missing env vars at runtime.

Known issues / caveats
- Mixed data sources (CoinGecko vs Alpaca) can diverge in fields and pricing.
- Without required VITE_* env vars, services fall back but features may degrade.

Current status
- Functional demo with graceful degradation. Ready for iterative feature wiring and polish.


