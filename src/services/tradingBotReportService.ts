// Trading Bot P&L Report Service
// Generates realistic trading bot performance data for client demonstration

export interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  value: number;
  pnl?: number;
  strategy: string;
}

export interface DailyPerformance {
  date: string;
  trades: Trade[];
  totalPnL: number;
  totalVolume: number;
  winRate: number;
  bestTrade: number;
  worstTrade: number;
}

export interface TradingBotReport {
  period: string;
  totalPnL: number;
  totalVolume: number;
  totalTrades: number;
  winRate: number;
  dailyPerformance: DailyPerformance[];
  currentPositions: {
    symbol: string;
    quantity: number;
    avgPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
  }[];
}

// Realistic price data based on actual September 2025 prices
const priceData = {
  'BTC': {
    '2025-09-08': { open: 111300, high: 112500, low: 110800, close: 111608.79 },
    '2025-09-09': { open: 111608.79, high: 112200, low: 111100, close: 111850.50 },
    '2025-09-10': { open: 111850.50, high: 114500, low: 111600, close: 114293.00 }
  },
  'ETH': {
    '2025-09-08': { open: 4301.42, high: 4350, low: 4280, close: 4320.15 },
    '2025-09-09': { open: 4320.15, high: 4380, low: 4300, close: 4341.38 },
    '2025-09-10': { open: 4341.38, high: 4420, low: 4330, close: 4399.34 }
  },
  'SOL': {
    '2025-09-08': { open: 213.64, high: 216.50, low: 212.10, close: 214.80 },
    '2025-09-09': { open: 214.80, high: 218.20, low: 213.50, close: 210.16 },
    '2025-09-10': { open: 210.16, high: 225.50, low: 209.80, close: 224.25 }
  }
};

// Generate realistic trading patterns
function generateTrades(symbol: string, date: string): Trade[] {
  const prices = priceData[symbol as keyof typeof priceData][date];
  if (!prices) return [];

  const trades: Trade[] = [];
  const strategies = ['Momentum', 'Mean Reversion', 'Breakout', 'Arbitrage', 'Grid Trading'];
  
  // Generate 4-8 trades per day per symbol
  const numTrades = Math.floor(Math.random() * 5) + 4;
  
  for (let i = 0; i < numTrades; i++) {
    const hour = 9 + Math.floor((i / numTrades) * 8); // Spread trades across trading day
    const minute = Math.floor(Math.random() * 60);
    const timestamp = `${date}T${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}:00Z`;
    
    // Price varies between daily high/low
    const priceRange = prices.high - prices.low;
    const price = prices.low + (Math.random() * priceRange);
    
    // Realistic quantities based on symbol
    let quantity: number;
    if (symbol === 'BTC') {
      quantity = 0.1 + Math.random() * 0.5; // 0.1 to 0.6 BTC
    } else if (symbol === 'ETH') {
      quantity = 1 + Math.random() * 10; // 1 to 11 ETH
    } else { // SOL
      quantity = 10 + Math.random() * 90; // 10 to 100 SOL
    }
    
    const type = Math.random() > 0.5 ? 'BUY' : 'SELL';
    const strategy = strategies[Math.floor(Math.random() * strategies.length)];
    
    trades.push({
      id: `${symbol}-${date}-${i + 1}`,
      timestamp,
      symbol,
      type,
      quantity: parseFloat(quantity.toFixed(symbol === 'BTC' ? 4 : 2)),
      price: parseFloat(price.toFixed(2)),
      value: parseFloat((quantity * price).toFixed(2)),
      strategy
    });
  }
  
  // Calculate P&L for paired trades
  const buyTrades = trades.filter(t => t.type === 'BUY');
  const sellTrades = trades.filter(t => t.type === 'SELL');
  
  // Match buy/sell trades and calculate P&L
  const minPairs = Math.min(buyTrades.length, sellTrades.length);
  for (let i = 0; i < minPairs; i++) {
    const buyTrade = buyTrades[i];
    const sellTrade = sellTrades[i];
    const pnl = (sellTrade.price - buyTrade.price) * Math.min(buyTrade.quantity, sellTrade.quantity);
    sellTrade.pnl = parseFloat(pnl.toFixed(2));
  }
  
  return trades.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function calculateDailyPerformance(trades: Trade[], date: string): DailyPerformance {
  const totalPnL = trades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
  const totalVolume = trades.reduce((sum, trade) => sum + trade.value, 0);
  const tradesWithPnL = trades.filter(trade => trade.pnl !== undefined);
  const winningTrades = tradesWithPnL.filter(trade => (trade.pnl || 0) > 0);
  const winRate = tradesWithPnL.length > 0 ? (winningTrades.length / tradesWithPnL.length) * 100 : 0;
  
  const pnlValues = tradesWithPnL.map(trade => trade.pnl || 0);
  const bestTrade = pnlValues.length > 0 ? Math.max(...pnlValues) : 0;
  const worstTrade = pnlValues.length > 0 ? Math.min(...pnlValues) : 0;
  
  return {
    date,
    trades,
    totalPnL: parseFloat(totalPnL.toFixed(2)),
    totalVolume: parseFloat(totalVolume.toFixed(2)),
    winRate: parseFloat(winRate.toFixed(1)),
    bestTrade: parseFloat(bestTrade.toFixed(2)),
    worstTrade: parseFloat(worstTrade.toFixed(2))
  };
}

export function generateTradingBotReport(): TradingBotReport {
  const dates = ['2025-09-08', '2025-09-09', '2025-09-10'];
  const symbols = ['BTC', 'ETH', 'SOL'];
  
  const dailyPerformance: DailyPerformance[] = [];
  let allTrades: Trade[] = [];
  
  // Generate trades for each day
  for (const date of dates) {
    const dayTrades: Trade[] = [];
    
    for (const symbol of symbols) {
      const symbolTrades = generateTrades(symbol, date);
      dayTrades.push(...symbolTrades);
    }
    
    const performance = calculateDailyPerformance(dayTrades, date);
    dailyPerformance.push(performance);
    allTrades.push(...dayTrades);
  }
  
  // Calculate overall statistics
  const totalPnL = dailyPerformance.reduce((sum, day) => sum + day.totalPnL, 0);
  const totalVolume = dailyPerformance.reduce((sum, day) => sum + day.totalVolume, 0);
  const totalTrades = allTrades.length;
  const tradesWithPnL = allTrades.filter(trade => trade.pnl !== undefined);
  const winningTrades = tradesWithPnL.filter(trade => (trade.pnl || 0) > 0);
  const overallWinRate = tradesWithPnL.length > 0 ? (winningTrades.length / tradesWithPnL.length) * 100 : 0;
  
  // Calculate current positions (final day holdings)
  const currentPositions = symbols.map(symbol => {
    const symbolTrades = allTrades.filter(t => t.symbol === symbol);
    const buyTrades = symbolTrades.filter(t => t.type === 'BUY');
    const sellTrades = symbolTrades.filter(t => t.type === 'SELL');
    
    const totalBought = buyTrades.reduce((sum, t) => sum + t.quantity, 0);
    const totalSold = sellTrades.reduce((sum, t) => sum + t.quantity, 0);
    const netQuantity = totalBought - totalSold;
    
    const avgBuyPrice = buyTrades.length > 0 
      ? buyTrades.reduce((sum, t) => sum + (t.price * t.quantity), 0) / buyTrades.reduce((sum, t) => sum + t.quantity, 0)
      : 0;
    
    const currentPrice = priceData[symbol as keyof typeof priceData]['2025-09-10'].close;
    const unrealizedPnL = netQuantity * (currentPrice - avgBuyPrice);
    
    return {
      symbol,
      quantity: parseFloat(netQuantity.toFixed(symbol === 'BTC' ? 4 : 2)),
      avgPrice: parseFloat(avgBuyPrice.toFixed(2)),
      currentPrice,
      unrealizedPnL: parseFloat(unrealizedPnL.toFixed(2))
    };
  }).filter(pos => Math.abs(pos.quantity) > 0.001); // Only show significant positions
  
  return {
    period: 'September 8-10, 2025',
    totalPnL: parseFloat(totalPnL.toFixed(2)),
    totalVolume: parseFloat(totalVolume.toFixed(2)),
    totalTrades,
    winRate: parseFloat(overallWinRate.toFixed(1)),
    dailyPerformance,
    currentPositions
  };
}

// Export singleton instance for consistent data
let cachedReport: TradingBotReport | null = null;

export function getTradingBotReport(): TradingBotReport {
  if (!cachedReport) {
    cachedReport = generateTradingBotReport();
  }
  return cachedReport;
}

// Reset cache for new report generation
export function refreshTradingBotReport(): TradingBotReport {
  cachedReport = null;
  return getTradingBotReport();
}
