# V4 Trading Agent - AI-Powered Crypto Trading Bot

An advanced cryptocurrency trading bot with institutional-grade analytics, comprehensive risk management, and AI-driven decision making powered by Groq LLM.

## Core Features

### 🤖 AI Trading Engine
- **6-Validator System** for trading decisions:
  - Trend Validator (20% weight)
  - Volume Validator (15% weight)
  - Volatility Validator (15% weight)
  - Risk/Reward Validator (20% weight)
  - Sentiment Validator (15% weight)
  - Position Size Validator (15% weight)
- **Groq AI Integration** for market analysis
- **Virtual Portfolio** with $50K baseline capital
- **Automated trade execution** with manual override capabilities

### 📊 Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- SMA/EMA (Simple/Exponential Moving Averages)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- Support & Resistance Levels

### 📈 Portfolio Analytics
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk measurement
- **Calmar Ratio** - Return vs maximum drawdown
- **Win Rate & Profit Factor**
- **Maximum Drawdown tracking**
- **Daily performance snapshots**

### ⚡ Real-Time Features
- WebSocket integration for live price updates
- Funding rates sentiment analysis (Binance)
- Whale alerts monitoring
- Fear & Greed Index tracking
- Market watchlist with real-time updates

### 🛡️ Risk Management
- **Kelly Criterion** position sizing
- Dynamic stop-loss and take-profit
- Maximum drawdown limits (15% default)
- Per-trade risk limits (1% default)
- Volatility-based position adjustments
- Correlation analysis for portfolio risk

### 🎛️ Agent Controls
- Trading window configuration
- Risk tolerance settings
- Strategy preferences (shorting, margin, leverage)
- Agent pause/resume capabilities
- Manual trade override options
- Comprehensive audit logging

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **UI**: Tailwind CSS + Lucide Icons
- **State Management**: React Context + Hooks
- **Database**: Supabase (PostgreSQL)
- **AI**: Groq LLM API
- **Market Data**: Alpaca Markets API
- **Crypto Prices**: CoinGecko API
- **Dashboard**: React Grid Layout (draggable widgets)

## Project Structure

```
src/
├── components/
│   ├── Auth/           # Authentication components
│   ├── Dashboard/      # Dashboard widgets
│   └── Settings/       # Settings management
├── hooks/              # Custom React hooks
├── pages/
│   ├── Dashboard.tsx   # Main dashboard with 8 widgets
│   ├── AgentControlsEnhanced.tsx # Comprehensive agent controls
│   └── AuditLogs.tsx   # Audit trail viewer
├── services/
│   ├── tradingAgentV2.ts    # Core trading engine
│   ├── validatorSystem.ts   # 6-validator implementation
│   ├── technicalIndicators.ts # TA library
│   ├── portfolioAnalytics.ts # Performance metrics
│   ├── fundingRatesService.ts # Sentiment analysis
│   ├── riskManagerV2.ts     # Risk management
│   └── websocketService.ts  # Real-time updates
└── persistence/        # State persistence layer
```

## Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Supabase account
- API keys for Alpaca, Groq, and CoinGecko

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Jkinney331/V4-Trading-Agent.git
cd V4-Trading-Agent
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file with your API keys:
```bash
cp .env.example .env
```

4. Configure your API keys in `.env`:
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_ALPACA_API_KEY=your_alpaca_key
VITE_ALPACA_SECRET_KEY=your_alpaca_secret
VITE_GROQ_API_KEY=your_groq_key
VITE_COINGECKO_API_KEY=your_coingecko_key
```

5. Run database migrations in Supabase:
```sql
-- Execute the migration file at:
-- supabase/migrations/001_trading_persistence.sql
```

6. Start the development server:
```bash
npm run dev
```

## Dashboard Widgets

1. **Account Summary** - Portfolio value, P&L, available capital
2. **Trading Signals** - Real-time buy/sell signals with technical indicators
3. **Portfolio Analytics** - Sharpe ratio, win rate, performance metrics
4. **Performance Calendar** - Daily P&L heat map
5. **Positions Table** - Current open positions
6. **Market Watchlist** - Tracked cryptocurrencies
7. **Fear & Greed Index** - Market sentiment gauge
8. **Whale Alerts** - Large transaction monitoring

## Trading Configuration

### Risk Settings
- Per-trade risk: 0.5% - 5% of portfolio
- Maximum position size: 5% - 20% of portfolio
- Maximum drawdown: 10% - 25%
- Confidence threshold: 60% - 90%
- Risk/reward minimum: 2:1 to 5:1

### Strategy Options
- Enable/disable shorting
- Margin trading toggle
- Leverage limits (1x - 3x)
- Trading time windows
- Unorthodox strategies toggle

## API Endpoints

The trading agent communicates with various services:

- **Supabase**: Authentication, data persistence, audit logs
- **Alpaca**: Trade execution, account data
- **CoinGecko**: Cryptocurrency prices, market data
- **Groq**: AI market analysis
- **Binance** (optional): Funding rates for sentiment

## Security

- API keys stored securely in environment variables
- Row-level security (RLS) in Supabase
- User authentication required for all operations
- Audit logging for all trading decisions
- No hardcoded secrets in codebase

## Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint
```

## License

MIT License - See LICENSE file for details

## Support

For issues, feature requests, or questions, please open an issue on GitHub.

## Disclaimer

This is a virtual trading system for educational purposes. Always conduct thorough research and consider consulting with financial advisors before making real investment decisions.