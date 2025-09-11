Active Context

Current focus
- Establish Memory Bank to persist project knowledge across sessions.
- Verify environment variables and onboarding docs clarity.

Recent changes
- Added Memory Bank core docs and `.cursorrules`.

Next steps
- Wire portfolio persistence path end-to-end (UI currently places orders via Alpaca; `portfolioService` route is present but not used in UI).
- Add periodic refresh/auto-refetch strategy for account/positions/orders (currently one-shot on load).
- Consider consolidating crypto price source to one service in UI to avoid divergence between Alpaca vs CoinGecko shapes.

Active decisions
- Keep fallback data paths to ensure demo resilience.
- Maintain authenticated-only RLS (no anonymous demo policies in production).

Open questions
- Should manual orders write to Supabase `orders` for a unified history when placed via Alpaca, or are these separate concerns?
- Do we want websocket/streaming updates later (Alpaca data, CoinGecko websockets via third-party)?


