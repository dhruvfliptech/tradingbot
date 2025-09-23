# Session-Test Variant Plan

## Goals
- Run the trading agent continuously on the backend, independent of the browser UI session.
- Keep the existing frontend repository intact while experimenting on a separate code path.
- Establish clear separation between headless trading logic and presentation concerns (React/browser APIs).

## High-Level Architecture
1. **Shared Core (`session-test/core`)**
   - Extract trading agent logic (signal generation, validators, risk management, broker calls) into a set of pure TypeScript modules with no browser dependencies.
   - Define interfaces for persistence, settings, event logging, and broker execution so both frontend and backend can supply their own adapters.

2. **Frontend Adapters (`session-test/src`)**
   - Wrap the core with adapters that continue to use localStorage, window events, and React state.
   - No major frontend changes initially; focus on making sure the adapters satisfy the new core contracts.

3. **Backend Runner (`session-test/backend/agent-runner`)**
   - Create a Node entry point that loads env vars, authenticates with Supabase using a service account/bot credentials, and instantiates the core with backend-specific adapters.
   - Provide scheduling (setInterval/cron) to execute the trading loop, with graceful shutdown hooks and structured logging.

## Key Workstreams
- [ ] **Core extraction**: carve tradingAgentV2 + dependencies into `core/agent`. Replace direct imports of browser-only utilities with injected abstractions (e.g., `PersistenceAdapter`, `BrokerAdapter`).
- [ ] **Backend adapters**: implement Supabase persistence, audit logging, trade history, and broker execution classes for Node.
- [ ] **Runner skeleton**: new command (e.g., `npm run agent:backend`) that bootstraps the backend agent. Includes configuration schema, health logging, and error handling.
- [ ] **Session handling**: backend uses service role key or stored refresh token; include helper to refresh access tokens and rotate when needed.
- [ ] **Telemetry & control**: expose simple status endpoint / log output so we can monitor agent health. Optionally integrate with existing `TradingService` scaffolding.

## Open Questions
- Preferred auth mechanism (service role vs bot user access/refresh tokens)?
- Deployment target for the backend runner (Docker, serverless, dedicated VM)?
- Do we need multi-user support in this variant, or is a single bot account sufficient?

## Next Steps
1. Audit current `tradingAgentV2` dependencies to document browser-only APIs.
2. Design TypeScript interfaces for adapters (persistence, broker, audit log, notification, state cache).
3. Begin moving reusable logic into `core/` with unit tests.
4. Build backend runner wiring once the core is compilable in Node.
