/**
 * Technical Analysis Module
 * Implements RSI, MACD, Moving Averages, and Support/Resistance calculations
 */

const axios = require('axios');

// Simple Moving Average
function calculateSMA(prices, period) {
  if (prices.length < period) return null;
  const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
  return sum / period;
}

// Exponential Moving Average
function calculateEMA(prices, period) {
  if (prices.length < period) return null;
  
  const multiplier = 2 / (period + 1);
  let ema = prices[0]; // Start with first price
  
  for (let i = 1; i < prices.length; i++) {
    ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
  }
  
  return ema;
}

// Relative Strength Index (RSI)
function calculateRSI(prices, period = 14) {
  if (prices.length < period + 1) return null;
  
  let gains = 0;
  let losses = 0;
  
  // Calculate initial average gains and losses
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses += Math.abs(change);
    }
  }
  
  let avgGain = gains / period;
  let avgLoss = losses / period;
  
  // Calculate subsequent values using Wilder's smoothing
  for (let i = period + 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    let gain = 0;
    let loss = 0;
    
    if (change > 0) {
      gain = change;
    } else {
      loss = Math.abs(change);
    }
    
    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;
  }
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  return rsi;
}

// MACD (Moving Average Convergence Divergence)
function calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (prices.length < slowPeriod) return null;
  
  const fastEMA = calculateEMA(prices, fastPeriod);
  const slowEMA = calculateEMA(prices, slowPeriod);
  
  if (!fastEMA || !slowEMA) return null;
  
  const macdLine = fastEMA - slowEMA;
  
  // For signal line, we need historical MACD values
  // For simplicity, we'll use a single MACD value
  return {
    macd: macdLine,
    signal: macdLine * 0.9, // Simplified signal line
    histogram: macdLine - (macdLine * 0.9)
  };
}

// Bollinger Bands
function calculateBollingerBands(prices, period = 20, stdDev = 2) {
  if (prices.length < period) return null;
  
  const sma = calculateSMA(prices, period);
  if (!sma) return null;
  
  // Calculate standard deviation
  const recentPrices = prices.slice(-period);
  const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
  const standardDeviation = Math.sqrt(variance);
  
  return {
    upper: sma + (standardDeviation * stdDev),
    middle: sma,
    lower: sma - (standardDeviation * stdDev)
  };
}

// Support and Resistance levels
function calculateSupportResistance(prices, lookbackPeriod = 20) {
  if (prices.length < lookbackPeriod) return null;
  
  const recentPrices = prices.slice(-lookbackPeriod);
  const highs = [];
  const lows = [];
  
  // Find local highs and lows
  for (let i = 1; i < recentPrices.length - 1; i++) {
    if (recentPrices[i] > recentPrices[i-1] && recentPrices[i] > recentPrices[i+1]) {
      highs.push(recentPrices[i]);
    }
    if (recentPrices[i] < recentPrices[i-1] && recentPrices[i] < recentPrices[i+1]) {
      lows.push(recentPrices[i]);
    }
  }
  
  // Calculate support and resistance levels
  const resistance = highs.length > 0 ? highs.reduce((a, b) => a + b, 0) / highs.length : null;
  const support = lows.length > 0 ? lows.reduce((a, b) => a + b, 0) / lows.length : null;
  
  return {
    support: support,
    resistance: resistance,
    currentPrice: prices[prices.length - 1]
  };
}

// Volume analysis
function analyzeVolume(volumes, prices, period = 14) {
  if (volumes.length < period || prices.length < period) return null;
  
  const recentVolumes = volumes.slice(-period);
  const recentPrices = prices.slice(-period);
  
  const avgVolume = recentVolumes.reduce((a, b) => a + b, 0) / period;
  const currentVolume = volumes[volumes.length - 1];
  const currentPrice = prices[prices.length - 1];
  const priceChange = ((currentPrice - prices[prices.length - 2]) / prices[prices.length - 2]) * 100;
  
  return {
    volumeRatio: currentVolume / avgVolume,
    priceChange: priceChange,
    volumeTrend: currentVolume > avgVolume ? 'high' : 'normal',
    priceVolumeConfirmation: (priceChange > 0 && currentVolume > avgVolume) || (priceChange < 0 && currentVolume > avgVolume)
  };
}

// Technical Analysis Signal Generator
function generateTechnicalSignals(symbol, currentPrice, priceHistory, volumeHistory, timeframes = []) {
  if (!priceHistory || priceHistory.length < 26) {
    return {
      symbol,
      signals: [],
      indicators: {},
      confidence: 0,
      recommendation: 'HOLD',
      reason: 'Insufficient data for technical analysis'
    };
  }
  
  // Calculate technical indicators
  const rsi = calculateRSI(priceHistory, 14);
  const macd = calculateMACD(priceHistory);
  const sma20 = calculateSMA(priceHistory, 20);
  const sma50 = calculateSMA(priceHistory, 50);
  const ema12 = calculateEMA(priceHistory, 12);
  const ema26 = calculateEMA(priceHistory, 26);
  const bollinger = calculateBollingerBands(priceHistory, 20, 2);
  const supportResistance = calculateSupportResistance(priceHistory, 20);
  const volumeAnalysis = volumeHistory ? analyzeVolume(volumeHistory, priceHistory) : null;
  
  const indicators = {
    rsi: rsi ? Math.round(rsi * 100) / 100 : null,
    macd: macd,
    sma20: sma20 ? Math.round(sma20 * 100) / 100 : null,
    sma50: sma50 ? Math.round(sma50 * 100) / 100 : null,
    ema12: ema12 ? Math.round(ema12 * 100) / 100 : null,
    ema26: ema26 ? Math.round(ema26 * 100) / 100 : null,
    bollinger: bollinger ? {
      upper: Math.round(bollinger.upper * 100) / 100,
      middle: Math.round(bollinger.middle * 100) / 100,
      lower: Math.round(bollinger.lower * 100) / 100
    } : null,
    supportResistance: supportResistance ? {
      support: Math.round(supportResistance.support * 100) / 100,
      resistance: Math.round(supportResistance.resistance * 100) / 100,
      currentPrice: Math.round(supportResistance.currentPrice * 100) / 100
    } : null,
    volume: volumeAnalysis
  };
  
  // Generate trading signals based on technical indicators
  const signals = [];
  let buySignals = 0;
  let sellSignals = 0;
  let confidence = 0;
  
  // RSI Signals
  if (rsi !== null) {
    if (rsi < 30) {
      signals.push({ indicator: 'RSI', signal: 'BUY', strength: 'strong', value: rsi, reason: 'RSI oversold' });
      buySignals += 2;
      confidence += 0.3;
    } else if (rsi > 70) {
      signals.push({ indicator: 'RSI', signal: 'SELL', strength: 'strong', value: rsi, reason: 'RSI overbought' });
      sellSignals += 2;
      confidence += 0.3;
    } else if (rsi < 40) {
      signals.push({ indicator: 'RSI', signal: 'BUY', strength: 'weak', value: rsi, reason: 'RSI approaching oversold' });
      buySignals += 1;
      confidence += 0.1;
    } else if (rsi > 60) {
      signals.push({ indicator: 'RSI', signal: 'SELL', strength: 'weak', value: rsi, reason: 'RSI approaching overbought' });
      sellSignals += 1;
      confidence += 0.1;
    }
  }
  
  // MACD Signals
  if (macd && macd.macd && macd.signal) {
    if (macd.macd > macd.signal && macd.histogram > 0) {
      signals.push({ indicator: 'MACD', signal: 'BUY', strength: 'medium', value: macd.macd, reason: 'MACD bullish crossover' });
      buySignals += 1.5;
      confidence += 0.2;
    } else if (macd.macd < macd.signal && macd.histogram < 0) {
      signals.push({ indicator: 'MACD', signal: 'SELL', strength: 'medium', value: macd.macd, reason: 'MACD bearish crossover' });
      sellSignals += 1.5;
      confidence += 0.2;
    }
  }
  
  // Moving Average Signals
  if (sma20 && sma50 && currentPrice) {
    if (currentPrice > sma20 && sma20 > sma50) {
      signals.push({ indicator: 'SMA', signal: 'BUY', strength: 'medium', value: sma20, reason: 'Price above SMA20, bullish trend' });
      buySignals += 1;
      confidence += 0.15;
    } else if (currentPrice < sma20 && sma20 < sma50) {
      signals.push({ indicator: 'SMA', signal: 'SELL', strength: 'medium', value: sma20, reason: 'Price below SMA20, bearish trend' });
      sellSignals += 1;
      confidence += 0.15;
    }
  }
  
  // EMA Signals
  if (ema12 && ema26 && currentPrice) {
    if (ema12 > ema26 && currentPrice > ema12) {
      signals.push({ indicator: 'EMA', signal: 'BUY', strength: 'medium', value: ema12, reason: 'EMA12 above EMA26, bullish momentum' });
      buySignals += 1;
      confidence += 0.15;
    } else if (ema12 < ema26 && currentPrice < ema12) {
      signals.push({ indicator: 'EMA', signal: 'SELL', strength: 'medium', value: ema12, reason: 'EMA12 below EMA26, bearish momentum' });
      sellSignals += 1;
      confidence += 0.15;
    }
  }
  
  // Bollinger Bands Signals
  if (bollinger && currentPrice) {
    if (currentPrice <= bollinger.lower) {
      signals.push({ indicator: 'BB', signal: 'BUY', strength: 'strong', value: currentPrice, reason: 'Price at lower Bollinger Band' });
      buySignals += 1.5;
      confidence += 0.2;
    } else if (currentPrice >= bollinger.upper) {
      signals.push({ indicator: 'BB', signal: 'SELL', strength: 'strong', value: currentPrice, reason: 'Price at upper Bollinger Band' });
      sellSignals += 1.5;
      confidence += 0.2;
    }
  }
  
  // Support/Resistance Signals
  if (supportResistance && currentPrice) {
    const supportDistance = Math.abs(currentPrice - supportResistance.support) / supportResistance.support * 100;
    const resistanceDistance = Math.abs(currentPrice - supportResistance.resistance) / supportResistance.resistance * 100;
    
    if (supportDistance < 2) {
      signals.push({ indicator: 'S/R', signal: 'BUY', strength: 'medium', value: supportResistance.support, reason: 'Price near support level' });
      buySignals += 1;
      confidence += 0.1;
    } else if (resistanceDistance < 2) {
      signals.push({ indicator: 'S/R', signal: 'SELL', strength: 'medium', value: supportResistance.resistance, reason: 'Price near resistance level' });
      sellSignals += 1;
      confidence += 0.1;
    }
  }
  
  // Volume Confirmation
  if (volumeAnalysis && volumeAnalysis.priceVolumeConfirmation) {
    signals.push({ indicator: 'Volume', signal: buySignals > sellSignals ? 'BUY' : 'SELL', strength: 'weak', value: volumeAnalysis.volumeRatio, reason: 'Volume confirms price movement' });
    confidence += 0.1;
  }
  
  // Determine final recommendation
  let recommendation = 'HOLD';
  let reason = 'No clear technical signal';
  
  if (buySignals > sellSignals && buySignals > 2) {
    recommendation = 'BUY';
    reason = `${buySignals.toFixed(1)} buy signals vs ${sellSignals.toFixed(1)} sell signals`;
  } else if (sellSignals > buySignals && sellSignals > 2) {
    recommendation = 'SELL';
    reason = `${sellSignals.toFixed(1)} sell signals vs ${buySignals.toFixed(1)} buy signals`;
  } else if (Math.abs(buySignals - sellSignals) <= 1) {
    recommendation = 'HOLD';
    reason = 'Mixed signals - waiting for clearer direction';
  }
  
  // Cap confidence at 0.95
  confidence = Math.min(confidence, 0.95);
  
  return {
    symbol,
    signals,
    indicators,
    confidence: Math.round(confidence * 100) / 100,
    recommendation,
    reason,
    signalStrength: {
      buy: buySignals,
      sell: sellSignals,
      net: buySignals - sellSignals
    }
  };
}

// Helper function to get historical data from Binance
async function getHistoricalData(symbol, limit = 100) {
  try {
    const baseUrl = process.env.VITE_BINANCE_BASE_URL || process.env.BINANCE_BASE_URL || 'https://api.binance.com';
    console.log(`Fetching ${limit} candles for ${symbol}USDT from ${baseUrl}`);
    
    const response = await axios.get(`${baseUrl}/api/v3/klines`, {
      params: {
        symbol: `${symbol}USDT`,
        interval: '1h', // 1-hour candles
        limit: limit
      },
      timeout: 15000,
      headers: {
        'User-Agent': 'TradingBot/1.0'
      }
    });
    
    if (!response.data || !Array.isArray(response.data)) {
      throw new Error('Invalid response format from Binance');
    }
    
    const klines = response.data;
    console.log(`Received ${klines.length} candles for ${symbol}`);
    
    const prices = klines.map(k => parseFloat(k[4])); // Close prices
    const volumes = klines.map(k => parseFloat(k[5])); // Volume
    
    // Validate data
    if (prices.length === 0 || prices.some(p => isNaN(p))) {
      throw new Error('Invalid price data received');
    }
    
    console.log(`${symbol} historical data: ${prices.length} prices, latest: $${prices[prices.length - 1]}`);
    
    return { prices, volumes };
  } catch (error) {
    console.error(`Error fetching historical data for ${symbol}:`, error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return { prices: null, volumes: null };
  }
}

module.exports = {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateSupportResistance,
  analyzeVolume,
  generateTechnicalSignals,
  getHistoricalData
};
