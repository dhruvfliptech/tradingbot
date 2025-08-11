Tech Context

Stack
- React 18, TypeScript, Vite 5, TailwindCSS
- Supabase JS v2 (auth + Postgres)
- CoinGecko REST (markets, global) with optional Pro key
- Alpaca paper trading REST v2
- Groq OpenAI-compatible chat completions API
- react-grid-layout for dashboard

Dev commands
- `npm run dev` — start dev server
- `npm run build` — production build
- `npm run preview` — preview build
- `npm run lint` — lint TS/TSX

Environment variables (VITE_*)
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`
- `VITE_COINGECKO_API_KEY` (optional; enables Pro header)
- `VITE_ALPACA_API_KEY`
- `VITE_ALPACA_SECRET_KEY`
- `VITE_GROQ_API_KEY`

Data model (public schema)
- portfolios(id, user_id, balance_usd, balance_btc, total_trades, created_at, updated_at)
- positions(id, portfolio_id, symbol, name, image, quantity, avg_cost_basis, side, created_at, updated_at)
- orders(id, portfolio_id, symbol, quantity, side, order_type, status, filled_quantity, filled_avg_price, limit_price, submitted_at, filled_at)

Notable implementation details
- `coinGeckoService` adds 10s timeout, 429 retry, and detailed fallbacks.
- `groqService` parses JSON arrays/objects from model output via regex match and enforces 3s rate limit.
- `DraggableGrid` resolves item overlaps and persists per-breakpoint layouts.


