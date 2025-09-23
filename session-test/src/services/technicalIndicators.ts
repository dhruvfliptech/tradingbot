/**
 * Technical Indicators Library
 * Provides comprehensive technical analysis calculations for trading decisions
 */

export interface PriceData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorResult {
  value: number;
  signal?: 'buy' | 'sell' | 'neutral';
  strength?: number; // 0-100
}

export interface MACDResult {
  macd: number;
  signal: number;
  histogram: number;
  trend: 'bullish' | 'bearish' | 'neutral';
}

export interface BollingerBandsResult {
  upper: number;
  middle: number;
  lower: number;
  bandwidth: number;
  percentB: number;
}

export interface StochasticResult {
  k: number;
  d: number;
  signal: 'overbought' | 'oversold' | 'neutral';
}

export class TechnicalIndicators {
  /**
   * Calculate Simple Moving Average (SMA)
   */
  static SMA(prices: number[], period: number): number {
    if (prices.length < period) return 0;
    const slice = prices.slice(-period);
    return slice.reduce((sum, price) => sum + price, 0) / period;
  }

  /**
   * Calculate Exponential Moving Average (EMA)
   */
  static EMA(prices: number[], period: number): number {
    if (prices.length === 0) return 0;
    
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
    }
    
    return ema;
  }

  /**
   * Calculate Relative Strength Index (RSI)
   */
  static RSI(prices: number[], period: number = 14): IndicatorResult {
    if (prices.length < period + 1) {
      return { value: 50, signal: 'neutral', strength: 0 };
    }

    let gains = 0;
    let losses = 0;

    // Calculate initial average gain/loss
    for (let i = 1; i <= period; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    // Calculate subsequent values using smoothing
    for (let i = period + 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) {
        avgGain = (avgGain * (period - 1) + change) / period;
        avgLoss = (avgLoss * (period - 1)) / period;
      } else {
        avgGain = (avgGain * (period - 1)) / period;
        avgLoss = (avgLoss * (period - 1) - change) / period;
      }
    }

    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = avgLoss === 0 ? 100 : 100 - (100 / (1 + rs));

    // Determine signal
    let signal: 'buy' | 'sell' | 'neutral' = 'neutral';
    let strength = 50;

    if (rsi < 30) {
      signal = 'buy';
      strength = (30 - rsi) * 3.33; // Scale 0-30 to 0-100
    } else if (rsi > 70) {
      signal = 'sell';
      strength = (rsi - 70) * 3.33; // Scale 70-100 to 0-100
    }

    return { value: rsi, signal, strength };
  }

  /**
   * Calculate MACD (Moving Average Convergence Divergence)
   */
  static MACD(prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): MACDResult {
    if (prices.length < slowPeriod) {
      return { macd: 0, signal: 0, histogram: 0, trend: 'neutral' };
    }

    const emaFast = this.EMA(prices, fastPeriod);
    const emaSlow = this.EMA(prices, slowPeriod);
    const macdLine = emaFast - emaSlow;

    // Calculate signal line (EMA of MACD)
    const macdValues: number[] = [];
    for (let i = slowPeriod - 1; i < prices.length; i++) {
      const subPrices = prices.slice(0, i + 1);
      const fast = this.EMA(subPrices, fastPeriod);
      const slow = this.EMA(subPrices, slowPeriod);
      macdValues.push(fast - slow);
    }

    const signalLine = this.EMA(macdValues, signalPeriod);
    const histogram = macdLine - signalLine;

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (histogram > 0 && macdLine > 0) trend = 'bullish';
    else if (histogram < 0 && macdLine < 0) trend = 'bearish';

    return {
      macd: macdLine,
      signal: signalLine,
      histogram,
      trend
    };
  }

  /**
   * Calculate Bollinger Bands
   */
  static BollingerBands(prices: number[], period: number = 20, stdDev: number = 2): BollingerBandsResult {
    if (prices.length < period) {
      const current = prices[prices.length - 1] || 0;
      return {
        upper: current,
        middle: current,
        lower: current,
        bandwidth: 0,
        percentB: 0.5
      };
    }

    const sma = this.SMA(prices, period);
    
    // Calculate standard deviation
    const slice = prices.slice(-period);
    const variance = slice.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
    const standardDeviation = Math.sqrt(variance);

    const upper = sma + (standardDeviation * stdDev);
    const lower = sma - (standardDeviation * stdDev);
    const bandwidth = (upper - lower) / sma;
    const currentPrice = prices[prices.length - 1];
    const percentB = (currentPrice - lower) / (upper - lower);

    return {
      upper,
      middle: sma,
      lower,
      bandwidth,
      percentB
    };
  }

  /**
   * Calculate ATR (Average True Range)
   */
  static ATR(data: PriceData[], period: number = 14): number {
    if (data.length < period + 1) return 0;

    const trueRanges: number[] = [];
    
    for (let i = 1; i < data.length; i++) {
      const high = data[i].high;
      const low = data[i].low;
      const prevClose = data[i - 1].close;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trueRanges.push(tr);
    }

    // Calculate ATR using EMA method
    return this.EMA(trueRanges, period);
  }

  /**
   * Calculate VWAP (Volume Weighted Average Price)
   */
  static VWAP(data: PriceData[]): number {
    if (data.length === 0) return 0;

    let totalVolume = 0;
    let totalVolumePrice = 0;

    for (const candle of data) {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      totalVolumePrice += typicalPrice * candle.volume;
      totalVolume += candle.volume;
    }

    return totalVolume === 0 ? 0 : totalVolumePrice / totalVolume;
  }

  /**
   * Calculate Stochastic Oscillator
   */
  static Stochastic(data: PriceData[], kPeriod: number = 14, dPeriod: number = 3): StochasticResult {
    if (data.length < kPeriod) {
      return { k: 50, d: 50, signal: 'neutral' };
    }

    const kValues: number[] = [];
    
    for (let i = kPeriod - 1; i < data.length; i++) {
      const slice = data.slice(i - kPeriod + 1, i + 1);
      const highs = slice.map(d => d.high);
      const lows = slice.map(d => d.low);
      
      const highest = Math.max(...highs);
      const lowest = Math.min(...lows);
      const current = data[i].close;
      
      const k = highest === lowest ? 50 : ((current - lowest) / (highest - lowest)) * 100;
      kValues.push(k);
    }

    const currentK = kValues[kValues.length - 1] || 50;
    const d = this.SMA(kValues, dPeriod);

    // Determine signal
    let signal: 'overbought' | 'oversold' | 'neutral' = 'neutral';
    if (currentK > 80) signal = 'overbought';
    else if (currentK < 20) signal = 'oversold';

    return { k: currentK, d, signal };
  }

  /**
   * Calculate Support and Resistance Levels
   */
  static SupportResistance(data: PriceData[], lookback: number = 20): { support: number[]; resistance: number[] } {
    if (data.length < lookback) {
      return { support: [], resistance: [] };
    }

    const slice = data.slice(-lookback);
    const highs = slice.map(d => d.high);
    const lows = slice.map(d => d.low);
    
    // Find local maxima and minima
    const resistance: number[] = [];
    const support: number[] = [];
    
    for (let i = 1; i < highs.length - 1; i++) {
      // Resistance levels (local maxima)
      if (highs[i] > highs[i - 1] && highs[i] > highs[i + 1]) {
        resistance.push(highs[i]);
      }
      
      // Support levels (local minima)
      if (lows[i] < lows[i - 1] && lows[i] < lows[i + 1]) {
        support.push(lows[i]);
      }
    }

    // Sort and return top 3 levels
    resistance.sort((a, b) => b - a);
    support.sort((a, b) => a - b);
    
    return {
      resistance: resistance.slice(0, 3),
      support: support.slice(0, 3)
    };
  }

  /**
   * Calculate composite technical score
   */
  static getCompositeScore(data: PriceData[]): { score: number; signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell' } {
    if (data.length < 26) {
      return { score: 50, signal: 'neutral' };
    }

    const prices = data.map(d => d.close);
    let score = 0;
    let signals = 0;

    // RSI Signal (25% weight)
    const rsi = this.RSI(prices);
    if (rsi.signal === 'buy') score += 25;
    else if (rsi.signal === 'sell') score -= 25;
    
    // MACD Signal (25% weight)
    const macd = this.MACD(prices);
    if (macd.trend === 'bullish') score += 25;
    else if (macd.trend === 'bearish') score -= 25;

    // Bollinger Bands (20% weight)
    const bb = this.BollingerBands(prices);
    if (bb.percentB < 0.2) score += 20; // Oversold
    else if (bb.percentB > 0.8) score -= 20; // Overbought

    // Stochastic (15% weight)
    const stoch = this.Stochastic(data);
    if (stoch.signal === 'oversold') score += 15;
    else if (stoch.signal === 'overbought') score -= 15;

    // Moving Average Trend (15% weight)
    const sma50 = this.SMA(prices, Math.min(50, prices.length));
    const sma200 = this.SMA(prices, Math.min(200, prices.length));
    const currentPrice = prices[prices.length - 1];
    
    if (currentPrice > sma50 && sma50 > sma200) score += 15;
    else if (currentPrice < sma50 && sma50 < sma200) score -= 15;

    // Normalize score to 0-100
    const normalizedScore = 50 + score;

    // Determine signal
    let signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
    if (normalizedScore >= 80) signal = 'strong_buy';
    else if (normalizedScore >= 60) signal = 'buy';
    else if (normalizedScore <= 20) signal = 'strong_sell';
    else if (normalizedScore <= 40) signal = 'sell';
    else signal = 'neutral';

    return { score: normalizedScore, signal };
  }
}

export default TechnicalIndicators;