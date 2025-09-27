# Features Not Needed for MVP
*These can be implemented after successful launch*

## Descoped from Initial Release

### Epic 1: Exchange Integration
- **1.2 Kraken Backup Integration** - Single exchange (Binance) is sufficient for MVP
  - Failover complexity not justified initially
  - Can add once primary exchange is stable

### Epic 6: UI/UX
- **6.2 Liquidity Heat Maps** - Nice visualization but not critical
  - Current order book display sufficient
  - Can enhance UI in version 2

### Epic 7: Communications
- **7.1 Telegram Bot Integration** - Manual monitoring sufficient initially
  - Can watch dashboard directly
  - Add once automated trading is stable

- **7.2 Email Performance Reports** - Manual review sufficient
  - Dashboard provides real-time metrics
  - Can export data manually if needed

### Epic 8: Security & Infrastructure
- **8.2 Multi-Factor Authentication** - Basic auth sufficient for MVP
  - Supabase auth is secure enough initially
  - Can add MFA once user base grows

- **8.4 Disaster Recovery Documentation** - Can document after launch
  - Focus on getting system running first
  - Document procedures based on actual operations

## Already Completed Elsewhere

### Paper Trading
- **Completed with Alpaca** - No need to redo
  - Already validated strategies
  - Have performance metrics
  - Ready for live trading

## Why These Aren't Needed

1. **Complexity**: Each adds significant development time
2. **Risk**: More features = more potential bugs
3. **Focus**: MVP should prove core trading works
4. **Iteration**: Can add based on actual user needs

## When to Revisit

- **After 1 month** of stable live trading
- **If specific need arises** (e.g., Binance goes down frequently)
- **When scaling beyond** personal use
- **Based on actual** trading performance data

## Current Priority

Focus 100% on:
1. Getting backend deployed
2. Securing API keys
3. Starting live trading
4. Monitoring initial performance

Everything else can wait.