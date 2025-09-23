# Session Test Agent Runner

Execute the trading agent outside the browser using Supabase service credentials.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `SUPABASE_URL` | ✅ | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✅ | Service role key (keep secret) |
| `TRADING_USER_ID` | ✅ | Supabase user id whose data the agent manages |
| `COINGECKO_API_KEY` | ❌ | Optional CoinGecko Pro key |
| `AGENT_INTERVAL_MS` | ❌ | Cycle interval (default 60000) |

Ensure the following tables exist (`agent_state`, `audit_logs`, `trade_history`) with `user_id` columns matching RLS expectations.

## Running Locally

```bash
cd session-test/backend
npm install
npm run agent
```

The default broker adapter is a no-op; integrate real Alpaca/Binance adapters before enabling live trading.
