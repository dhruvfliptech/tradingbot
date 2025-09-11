System Patterns

Architecture
- UI: React + Tailwind components organized under `src/components`.
- State: Local component state; auth via `useAuth` hook listening to Supabase session.
- Services layer: `alpacaService`, `coinGeckoService`, `portfolioService`, `groqService`, `tradingAgent` encapsulate integrations and business logic.
- Types: Centralized in `src/types/trading.ts`.
- Data persistence: Supabase tables `portfolios`, `positions`, `orders` with RLS.

Key technical decisions
- Fallback-first strategy: CoinGecko and Alpaca services provide deterministic fallback data on failures/timeouts.
- Rate limiting: Groq service enforces a 3s min interval between requests.
- Layout persistence: `DraggableGrid` saves layouts per breakpoint in `localStorage` and resolves overlaps.
- Auth gating: App renders `AuthModal` and minimal UI until `user` is present.
- API health surfacing: Compact indicator in header + detailed `ApiStatusModal` probe.

Component relationships
- `App` orchestrates data fetch, API status, and renders `GlobalMarketHeader` + `DraggableGrid` with widgets (AccountSummary, PositionsTable, MarketWatchlist, OrdersTable, FearGreedIndexWidget, TradingControls, TradingSignals, MarketInsights).
- `TradingControls` places orders via Alpaca; `TradingSignals` uses `tradingAgent` and/or `cryptoData`.
- `MarketInsights` calls `groqService` using current `cryptoData`.

Error handling
- Services catch and log errors; CoinGecko/Alpaca return curated fallback payloads to keep UI responsive.
- API status checks run independently to classify overall vs partial outages.

Security & RLS
- Final migration enforces authenticated-only RLS on all tables constrained by `auth.uid()`.


