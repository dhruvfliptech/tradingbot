# AI-Powered Crypto Trading Bot 🤖📈

An advanced cryptocurrency trading bot featuring Groq AI reasoning engine, multi-exchange support, and institutional-grade analytics. Built with React, TypeScript, and cutting-edge AI technology.

## 🌟 Key Features

### AI Intelligence
- **Groq AI Integration**: Leverages Mixtral-8x7b model for sophisticated trading decisions
- **6-Validator System**: Cross-validates signals from multiple independent indicators
- **Funding Rate Analysis**: Monitors perpetual futures funding across exchanges for overleveraged conditions
- **Sentiment Analysis**: Real-time market sentiment evaluation

### Advanced Analytics
- **Portfolio Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, Alpha/Beta calculations
- **Trade History**: Complete tracking with filtering and CSV export
- **Correlation Heatmap**: Visual risk assessment across portfolio assets
- **P&L Tracking**: Daily, weekly, monthly, and all-time performance views

### Trading Features
- **Multi-Exchange Support**: Binance, Bybit, OKX, Coinglass integration
- **Market Hours Enforcement**: 9 AM - 6 PM EST with override capability
- **Risk Management**: 2:1 minimum risk/reward ratio, position sizing limits
- **Demo Mode**: Full functionality without live trading accounts

### Technical Indicators
- Professional-grade calculations (RSI, MACD, Bollinger Bands, ATR, VWAP)
- Real-time data aggregation from multiple sources
- Comprehensive caching strategy for optimal performance

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- API keys (optional, demo mode available)

### Installation

```bash
# Clone the repository
git clone https://github.com/Jkinney331/crypto-trading-bot-ai.git
cd crypto-trading-bot-ai

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start development server
npm run dev
```

### Environment Setup

Create a `.env` file with your API keys:

```env
# Core APIs
VITE_GROQ_API_KEY=your_groq_api_key
VITE_ALPACA_API_KEY=your_alpaca_key
VITE_ALPACA_SECRET_KEY=your_alpaca_secret
VITE_COINGECKO_API_KEY=your_coingecko_key

# Optional APIs
VITE_WHALE_ALERT_API_KEY=your_whale_alert_key
VITE_ETHERSCAN_API_KEY=your_etherscan_key
VITE_BITQUERY_TOKEN=your_bitquery_token
VITE_COINGLASS_API_KEY=your_coinglass_key

# Supabase (optional, demo mode available)
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key

# Configuration
VITE_PORT=3001
```

## 📊 Architecture

### Frontend Stack
- **React 18** with TypeScript
- **Vite** for fast development
- **TailwindCSS** for styling
- **Lucide React** for icons

### Backend Services
- **Groq AI** for intelligent reasoning
- **Alpaca Markets** for paper trading
- **CoinGecko** for market data
- **Supabase** for data persistence

### Key Components

```
src/
├── components/
│   ├── Dashboard/
│   │   ├── PortfolioAnalytics.tsx    # Advanced metrics
│   │   ├── TradeHistory.tsx          # Trade tracking
│   │   ├── CorrelationHeatmap.tsx    # Risk visualization
│   │   └── AutoTradeActivity.tsx     # Live trading feed
│   └── ui/
│       └── ConfirmDialog.tsx         # Safety confirmations
├── services/
│   ├── ai/
│   │   └── groqReasoningEngine.ts    # AI decision engine
│   ├── analytics/
│   │   └── portfolioAnalytics.ts     # Performance calculations
│   ├── data/
│   │   ├── dataAggregator.ts         # Multi-source aggregation
│   │   └── fundingRatesService.ts    # Exchange funding rates
│   └── indicators/
│       └── technicalIndicators.ts    # TA calculations
```

## 🧠 AI Trading Logic

The bot uses a sophisticated multi-validator system:

1. **Technical Analysis**: RSI, MACD, Moving Averages
2. **Momentum Analysis**: 24h price change and volume
3. **Sentiment Analysis**: AI-powered market sentiment
4. **On-Chain Metrics**: Network activity indicators
5. **Market Position**: Market cap ranking analysis
6. **Funding Rates**: Perpetual futures overleveraging detection

Each validator contributes to a confidence score (0-100), with trades executed only when:
- Multiple validators agree
- Risk/reward ratio exceeds 2:1
- Market hours are active (or override enabled)
- Confirmation thresholds are met

## 🔒 Security Features

- **Confirmation Dialogs**: Critical actions require explicit confirmation
- **Large Order Protection**: Orders >$10,000 trigger warnings
- **API Key Management**: Secure storage with retry logic
- **Demo Mode**: Safe testing without real funds
- **Position Limits**: Maximum 10% portfolio allocation per trade

## 📈 Performance Metrics

The system tracks comprehensive performance indicators:

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation focus
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits vs losses
- **Alpha/Beta**: Market outperformance metrics

## 🛠️ Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript checks
```

### Testing

```bash
npm test             # Run test suite
npm run test:watch   # Watch mode
npm run test:coverage # Coverage report
```

## 📝 Configuration

### Trading Parameters

Edit `src/services/tradingAgent.ts` for:
- Confidence thresholds
- Position sizing
- Cooldown periods
- Symbol watchlists

### Market Hours

Configure in `src/services/tradingSchedule.ts`:
- Trading windows (default: 9 AM - 6 PM EST)
- Weekend/holiday handling
- Override capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for lightning-fast AI inference
- **Alpaca Markets** for paper trading capabilities
- **CoinGecko** for comprehensive market data
- **Claude** for development assistance

## ⚠️ Disclaimer

This bot is for educational and research purposes only. Cryptocurrency trading carries significant risk. Always do your own research and never invest more than you can afford to lose.

## 📞 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/Jkinney331/crypto-trading-bot-ai/issues)
- Check the [Wiki](https://github.com/Jkinney331/crypto-trading-bot-ai/wiki) for documentation

---

Built with ❤️ by the AI Crypto Trading Team