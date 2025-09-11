# 🔑 API Keys Required for Trading Bot

This document lists all API keys needed to run the AI Trading Bot. Some are essential for basic operation, while others enable additional features.

---

## 📋 Quick Setup Checklist

### Essential Keys (Required to Start)
- [ ] Alpaca API Key & Secret
- [ ] Supabase URL & Anon Key
- [ ] Groq API Key (for sentiment)

### Recommended Keys (For Full Features)
- [ ] CoinGecko API Key
- [ ] Telegram Bot Token
- [ ] WhaleAlert API Key
- [ ] Etherscan API Key

---

## 🔴 Essential API Keys (System Won't Run Without These)

### 1. Alpaca Trading API
**Purpose:** Paper trading and order execution
**Get It From:** https://alpaca.markets/
```env
VITE_ALPACA_API_KEY=your_alpaca_api_key_here
VITE_ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```
**Free Tier:** Yes (paper trading unlimited)
**Setup Time:** 5 minutes

### 2. Supabase (Database)
**Purpose:** Data persistence, user authentication
**Get It From:** https://supabase.com/
```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```
**Free Tier:** Yes (500MB database, 2GB bandwidth)
**Setup Time:** 10 minutes

### 3. Groq AI
**Purpose:** Market sentiment analysis
**Get It From:** https://console.groq.com/
```env
VITE_GROQ_API_KEY=your_groq_api_key_here
```
**Free Tier:** Yes (limited requests)
**Setup Time:** 2 minutes

---

## 🟡 Recommended API Keys (For Better Performance)

### 4. CoinGecko
**Purpose:** Comprehensive crypto market data
**Get It From:** https://www.coingecko.com/api/pricing
```env
VITE_COINGECKO_API_KEY=your_coingecko_key_here
```
**Free Tier:** Yes (10-50 calls/minute)
**Note:** System works without it but with limited data

### 5. Telegram Bot
**Purpose:** Real-time notifications and alerts
**Get It From:** Create bot via @BotFather on Telegram
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```
**Free Tier:** Yes (unlimited)
**Setup Guide:** 
1. Message @BotFather on Telegram
2. Create new bot with `/newbot`
3. Get your chat ID from @userinfobot

### 6. Redis (If Not Using Docker)
**Purpose:** Caching and performance optimization
**Get It From:** https://redis.com/try-free/
```env
REDIS_URL=redis://localhost:6379
# Or cloud Redis
REDIS_URL=redis://:password@your-redis-host:6379
```
**Free Tier:** Yes (30MB on Redis Cloud)
**Note:** Included in Docker setup

---

## 🟢 Optional API Keys (Enhanced Features)

### 7. WhaleAlert
**Purpose:** Track large crypto transactions
**Get It From:** https://whale-alert.io/
```env
VITE_WHALE_ALERT_API_KEY=your_whale_alert_key_here
```
**Free Tier:** Yes (10 requests/minute)
**Benefit:** Smart money tracking

### 8. Etherscan
**Purpose:** On-chain Ethereum data
**Get It From:** https://etherscan.io/apis
```env
ETHERSCAN_API_KEY=your_etherscan_key_here
```
**Free Tier:** Yes (5 calls/second)
**Benefit:** On-chain analytics

### 9. Bitquery
**Purpose:** Blockchain analytics via GraphQL
**Get It From:** https://bitquery.io/
```env
BITQUERY_API_KEY=your_bitquery_key_here
```
**Free Tier:** Yes (limited)
**Benefit:** Cross-chain data

### 10. Binance (Future Migration)
**Purpose:** Live trading (when ready to switch from Alpaca)
**Get It From:** https://www.binance.com/en/my/settings/api-management
```env
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET_KEY=your_binance_secret_here
```
**Note:** Only needed when migrating from paper trading

---

## 🚀 Quick Configuration Guide

### Method 1: Environment File (Recommended)
1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` with your keys:
```bash
nano .env  # or use any text editor
```

3. Add your keys:
```env
# Essential
VITE_ALPACA_API_KEY=pk_abc123...
VITE_ALPACA_SECRET_KEY=sk_xyz789...
VITE_SUPABASE_URL=https://abc.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
VITE_GROQ_API_KEY=gsk_...

# Recommended
VITE_COINGECKO_API_KEY=CG-...
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=987654321

# Optional
ETHERSCAN_API_KEY=ABC123...
VITE_WHALE_ALERT_API_KEY=xyz...
```

### Method 2: Settings UI (User-Friendly)
1. Start the application:
```bash
npm run dev
```

2. Navigate to Settings → API Keys

3. Enter keys in the secure form

4. Keys are encrypted and stored in Supabase

---

## 🔒 Security Best Practices

### DO's ✅
- Store keys in `.env` files (never commit)
- Use environment variables in production
- Rotate keys regularly
- Use read-only keys where possible
- Enable IP whitelisting when available

### DON'Ts ❌
- Never commit API keys to Git
- Don't share keys in documentation
- Don't use production keys in development
- Don't expose keys in frontend code
- Never log API keys

### Git Security
Add to `.gitignore`:
```
.env
.env.local
.env.production
*.key
*.secret
```

---

## 🧪 Testing Without All Keys

The system can run with minimal keys for testing:

### Minimal Setup (Basic Testing)
```env
# Just these 3 for basic operation
VITE_ALPACA_API_KEY=your_key
VITE_ALPACA_SECRET_KEY=your_secret
VITE_GROQ_API_KEY=your_key
```

### What Works Without Full Keys
- ✅ Paper trading via Alpaca
- ✅ Basic market data (public APIs)
- ✅ Sentiment analysis (Groq)
- ✅ Local backtesting
- ❌ Whale tracking (needs WhaleAlert)
- ❌ On-chain data (needs Etherscan)
- ❌ Telegram alerts (needs bot token)

---

## 📝 API Key Features Matrix

| API Key | Trading | Data | Alerts | Analytics | Required |
|---------|---------|------|--------|-----------|----------|
| Alpaca | ✅ | ✅ | ❌ | ❌ | Yes |
| Supabase | ❌ | ✅ | ❌ | ✅ | Yes |
| Groq | ❌ | ❌ | ❌ | ✅ | Yes |
| CoinGecko | ❌ | ✅ | ❌ | ✅ | No |
| Telegram | ❌ | ❌ | ✅ | ❌ | No |
| WhaleAlert | ❌ | ✅ | ✅ | ✅ | No |
| Etherscan | ❌ | ✅ | ❌ | ✅ | No |
| Redis | ❌ | ✅ | ❌ | ❌ | No* |

*Included in Docker setup

---

## 🆘 Troubleshooting

### Common Issues

#### "Invalid API Key" Error
- Check for extra spaces or quotes
- Verify key is active on provider's dashboard
- Ensure using correct environment (paper vs live)

#### "Rate Limit Exceeded"
- Implement caching (Redis)
- Upgrade to paid tier
- Add request throttling

#### "Connection Refused"
- Check firewall settings
- Verify API endpoint URLs
- Test with curl/Postman first

### Testing API Keys
```bash
# Test Alpaca
curl -H "APCA-API-KEY-ID: YOUR_KEY" \
     -H "APCA-API-SECRET-KEY: YOUR_SECRET" \
     https://paper-api.alpaca.markets/v2/account

# Test Supabase
curl https://your-project.supabase.co/rest/v1/ \
     -H "apikey: YOUR_ANON_KEY"

# Test CoinGecko
curl "https://api.coingecko.com/api/v3/ping?x_cg_demo_api_key=YOUR_KEY"
```

---

## 📚 Additional Resources

### API Documentation
- [Alpaca API Docs](https://docs.alpaca.markets/)
- [Supabase Docs](https://supabase.com/docs)
- [CoinGecko API](https://docs.coingecko.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Etherscan API](https://docs.etherscan.io/)

### Support
- Most APIs have free tiers sufficient for development
- Upgrade to paid tiers for production use
- Check rate limits for each service
- Use caching to minimize API calls

---

**Note:** The system is designed to gracefully handle missing optional API keys. Start with essential keys and add others as needed.

*Last Updated: August 16, 2025*