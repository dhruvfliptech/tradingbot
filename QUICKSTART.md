# 🚀 Quick Start Guide - Live Trading with Binance.US

## ⚠️ IMPORTANT: Live Trading Warning
This system is configured for **LIVE TRADING** with real money on Binance.US. Please read carefully before starting.

## Prerequisites
- Node.js 18+ installed
- Binance.US account with API keys (no IP whitelist needed)
- Funds in your Binance.US account

## 🔐 Security Setup Completed
Your API keys have been securely configured:
- ✅ Frontend `.env` - Only public variables (Supabase)
- ✅ Backend `.env` - All sensitive API keys (never exposed to browser)
- ✅ Binance.US endpoints configured (not Binance.com)

## 🚦 Starting the System

### 1. Quick Start (Recommended)
```bash
./start-local.sh
```

This will:
- Check all prerequisites
- Verify environment files
- Install dependencies if needed
- Start backend (port 3001)
- Start frontend (port 5173)
- Display safety warnings

### 2. Manual Start (Advanced)
```bash
# Terminal 1 - Backend
cd backend
npm run dev

# Terminal 2 - Frontend
npm run dev

# Terminal 3 - Monitor (optional)
./monitor-system.sh
```

## 📊 Accessing the Dashboard

Once running, open: **http://localhost:5173**

### Key Features:
1. **Binance Account Summary** - Shows balances in USD and BTC
2. **Live Trading Bot** - Auto-trades based on signals
3. **Performance Metrics** - Sharpe ratio, P&L, win rate
4. **Real-time Updates** - WebSocket connections for live data
5. **Emergency Stop** - Press `Ctrl+Shift+E` anywhere in app

## 💰 Your Binance Balances

The dashboard will display:
- **Total USD Value** - All holdings converted to USD
- **Total BTC Value** - All holdings in BTC equivalent
- **Asset Breakdown** - Individual coin balances
- **Available Cash** - USDT/USD for trading

Updates every 30 seconds automatically.

## 🤖 Starting Auto-Trading

### Safe Start Procedure:
1. **Review Settings** in `backend/.env`:
   ```
   MAX_POSITION_SIZE_USD=1000    # Max per trade
   MAX_DAILY_LOSS_USD=500        # Daily stop loss
   MAX_OPEN_POSITIONS=5          # Max concurrent
   ```

2. **Start Bot from Dashboard**:
   - Click "Start Trading" button
   - Bot will use conservative settings initially
   - Monitor via Trading Bot Status widget

3. **Monitor Closely**:
   - Watch the Account Summary for balance changes
   - Check Performance Metrics
   - Review Trading Signals
   - Use Emergency Stop if needed

## 📈 Trading Configuration

Edit `trading-config.json` for strategy settings:
- `live` profile - Production trading
- `test` profile - Minimal risk testing

Current limits (from backend/.env):
- Max Position: $1,000
- Max Daily Loss: $500
- Max Open Positions: 5

## 🛑 Safety Features

1. **Emergency Stop**: `Ctrl+Shift+E` - Stops all trading immediately
2. **Daily Loss Limit**: Auto-stops at $500 loss
3. **Position Limits**: Max 5 concurrent positions
4. **Stop Loss**: 2% default on all positions

## 📡 System Monitoring

Run monitoring in separate terminal:
```bash
./monitor-system.sh
```

Shows:
- Service health
- System resources
- Recent alerts
- Bot status
- Account summary
- Performance metrics

## 🔧 Troubleshooting

### Backend won't start:
- Check port 3001 is free: `lsof -i :3001`
- Verify `backend/.env` exists
- Check API keys are correct

### Frontend won't connect:
- Ensure backend is running first
- Check port 5173 is free
- Verify `.env` has correct backend URL

### Binance API errors:
- Confirm using Binance.US (not .com) account
- Check API key permissions (spot trading enabled)
- Ensure sufficient balance for minimum order sizes

### No balance showing:
- Wait 30 seconds for initial fetch
- Check browser console for errors
- Verify Binance API keys in `backend/.env`

## 📝 Important Notes

1. **Minimum Order Sizes**: Binance.US has minimums (usually $10)
2. **API Rate Limits**: System respects Binance rate limits
3. **Market Hours**: Crypto trades 24/7
4. **Fees**: Binance.US charges 0.1% per trade

## 🚨 Emergency Procedures

If something goes wrong:

1. **In App**: Press `Ctrl+Shift+E` for emergency stop
2. **Terminal**: Press `Ctrl+C` to stop all services
3. **Binance.US**: Login and manually close positions if needed
4. **API Keys**: Can be disabled in Binance.US account settings

## 📊 What You'll See

When everything is working:
- ✅ Green "Connected" status in header
- ✅ Binance Account Summary shows your balances
- ✅ Real-time price updates in Market Watchlist
- ✅ Bot status shows "Running" when active
- ✅ Orders appear in Orders Table when trades execute

## 🎯 Next Steps

1. Start with the TEST profile first
2. Monitor for 1 hour before increasing limits
3. Gradually increase position sizes
4. Review daily performance
5. Adjust strategy based on results

---

**Remember**: Start small, monitor closely, and use the emergency stop if needed. This is REAL MONEY trading on Binance.US!

For issues or questions, check the logs in both terminal windows or review `backend/logs/`.