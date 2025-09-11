Product Context

Why this exists
- Provide a streamlined crypto trading cockpit that combines portfolio, execution, market intelligence, and AI analysis.

Problems solved
- Fragmented workflows across multiple tools/APIs.
- Lack of insight layering (AI) over live market context.
- Demo resilience: graceful fallbacks keep the UI useful when APIs throttle/fail.

How it should work
- After sign-in, the dashboard populates: account, positions, orders, watchlist, signals, global stats, and AI insights.
- Manual orders flow through Alpaca paper trading.
- Users can customize the dashboard layout; it persists locally.

User experience goals
- Fast, legible, mobile-friendly UI with status visibility.
- Clear success/failure states and confidence visualizations for AI/Signals.
- Minimal blocking: show fallback data rather than blank screens when possible.


