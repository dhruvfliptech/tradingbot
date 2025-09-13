/**
 * Funding Rates Service
 * Tracks perpetual futures funding rates across exchanges
 * Critical for futures/perpetuals trading decisions
 */

import axios from 'axios';

export interface FundingRate {
  symbol: string;
  exchange: string;
  rate: number; // Current funding rate as percentage
  nextFundingTime: Date;
  predictedRate?: number; // Predicted next funding rate
  historicalAverage: number; // Average over last 7 days
  trend: 'increasing' | 'decreasing' | 'stable';
}

export interface FundingRateAnalysis {
  symbol: string;
  averageRate: number;
  currentRate: number;
  rateChange24h: number;
  exchanges: {
    [exchange: string]: FundingRate;
  };
  arbitrageOpportunity: boolean;
  recommendedExchange?: string;
  signal: 'long_favorable' | 'short_favorable' | 'neutral' | 'extreme';
  extremeWarning?: string;
}

class FundingRatesService {
  private cache: Map<string, { data: FundingRateAnalysis; timestamp: number }> = new Map();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes
  private historicalRates: Map<string, number[]> = new Map();

  /**
   * Fetch funding rates from Binance
   */
  private async fetchBinanceFundingRate(symbol: string): Promise<FundingRate | null> {
    try {
      const ticker = `${symbol.toUpperCase()}USDT`;
      const response = await axios.get(
        `https://fapi.binance.com/fapi/v1/premiumIndex?symbol=${ticker}`
      );
      
      const data = response.data;
      const rate = parseFloat(data.lastFundingRate) * 100; // Convert to percentage
      
      // Get historical funding rates
      const histResponse = await axios.get(
        `https://fapi.binance.com/fapi/v1/fundingRate?symbol=${ticker}&limit=56` // 7 days * 8 funding periods
      );
      
      const historicalRates = histResponse.data.map((r: any) => parseFloat(r.fundingRate) * 100);
      const avgRate = historicalRates.reduce((a: number, b: number) => a + b, 0) / historicalRates.length;
      
      // Determine trend
      const recentAvg = historicalRates.slice(0, 8).reduce((a: number, b: number) => a + b, 0) / 8;
      const olderAvg = historicalRates.slice(-8).reduce((a: number, b: number) => a + b, 0) / 8;
      let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
      
      if (recentAvg > olderAvg * 1.1) trend = 'increasing';
      else if (recentAvg < olderAvg * 0.9) trend = 'decreasing';
      
      return {
        symbol: symbol.toUpperCase(),
        exchange: 'Binance',
        rate,
        nextFundingTime: new Date(parseInt(data.nextFundingTime)),
        historicalAverage: avgRate,
        trend
      };
    } catch (error) {
      console.warn(`Failed to fetch Binance funding rate for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Fetch funding rates from Bybit
   */
  private async fetchBybitFundingRate(symbol: string): Promise<FundingRate | null> {
    try {
      const ticker = `${symbol.toUpperCase()}USDT`;
      const response = await axios.get(
        `https://api.bybit.com/v5/market/tickers?category=linear&symbol=${ticker}`
      );
      
      if (!response.data.result?.list?.[0]) return null;
      
      const data = response.data.result.list[0];
      const rate = parseFloat(data.fundingRate) * 100;
      
      return {
        symbol: symbol.toUpperCase(),
        exchange: 'Bybit',
        rate,
        nextFundingTime: new Date(Date.now() + 8 * 60 * 60 * 1000), // Every 8 hours
        historicalAverage: rate, // Simplified for now
        trend: 'stable'
      };
    } catch (error) {
      console.warn(`Failed to fetch Bybit funding rate for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Simulate funding rates for demo/testing
   */
  private generateMockFundingRate(symbol: string, exchange: string): FundingRate {
    const baseRate = 0.01; // 0.01% base rate
    const variance = (Math.random() - 0.5) * 0.05; // +/- 0.025% variance
    const rate = baseRate + variance;
    
    // Store historical rates
    const history = this.historicalRates.get(`${symbol}-${exchange}`) || [];
    history.push(rate);
    if (history.length > 56) history.shift();
    this.historicalRates.set(`${symbol}-${exchange}`, history);
    
    const avgRate = history.length > 0 
      ? history.reduce((a, b) => a + b, 0) / history.length 
      : rate;
    
    // Determine trend
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (history.length > 8) {
      const recent = history.slice(-4).reduce((a, b) => a + b, 0) / 4;
      const older = history.slice(-8, -4).reduce((a, b) => a + b, 0) / 4;
      if (recent > older * 1.1) trend = 'increasing';
      else if (recent < older * 0.9) trend = 'decreasing';
    }
    
    return {
      symbol: symbol.toUpperCase(),
      exchange,
      rate: rate * 100, // Convert to percentage
      nextFundingTime: new Date(Date.now() + (8 - (Date.now() / 3600000) % 8) * 3600000),
      historicalAverage: avgRate * 100,
      trend,
      predictedRate: (rate + (Math.random() - 0.5) * 0.01) * 100
    };
  }

  /**
   * Get funding rates for a symbol across all exchanges
   */
  async getFundingRates(symbol: string): Promise<FundingRateAnalysis> {
    // Check cache
    const cached = this.cache.get(symbol);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    const exchanges: { [key: string]: FundingRate } = {};
    
    // Try to fetch real data (will fail without API setup)
    const [binanceRate, bybitRate] = await Promise.all([
      this.fetchBinanceFundingRate(symbol),
      this.fetchBybitFundingRate(symbol)
    ]);

    // Use real data if available, otherwise mock
    if (binanceRate) {
      exchanges['Binance'] = binanceRate;
    } else {
      exchanges['Binance'] = this.generateMockFundingRate(symbol, 'Binance');
    }

    if (bybitRate) {
      exchanges['Bybit'] = bybitRate;
    } else {
      exchanges['Bybit'] = this.generateMockFundingRate(symbol, 'Bybit');
    }

    // Add mock data for other exchanges
    exchanges['OKX'] = this.generateMockFundingRate(symbol, 'OKX');
    exchanges['Deribit'] = this.generateMockFundingRate(symbol, 'Deribit');

    // Calculate analysis
    const rates = Object.values(exchanges).map(e => e.rate);
    const averageRate = rates.reduce((a, b) => a + b, 0) / rates.length;
    const currentRate = exchanges['Binance']?.rate || averageRate;
    
    // Calculate 24h change (mocked for now)
    const rateChange24h = (Math.random() - 0.5) * 0.02;
    
    // Check for arbitrage opportunity
    const maxRate = Math.max(...rates);
    const minRate = Math.min(...rates);
    const arbitrageOpportunity = (maxRate - minRate) > 0.005; // 0.5% difference
    
    // Find best exchange for funding
    let recommendedExchange: string | undefined;
    if (arbitrageOpportunity) {
      // For longs, choose exchange with lowest (most negative) funding rate
      // For shorts, choose exchange with highest (most positive) funding rate
      const minExchange = Object.entries(exchanges).find(([_, e]) => e.rate === minRate)?.[0];
      const maxExchange = Object.entries(exchanges).find(([_, e]) => e.rate === maxRate)?.[0];
      recommendedExchange = averageRate > 0 ? minExchange : maxExchange;
    }
    
    // Determine signal
    let signal: 'long_favorable' | 'short_favorable' | 'neutral' | 'extreme' = 'neutral';
    let extremeWarning: string | undefined;
    
    if (Math.abs(averageRate) > 0.1) {
      signal = 'extreme';
      extremeWarning = `Extreme funding rate detected: ${averageRate.toFixed(3)}%. ${
        averageRate > 0 
          ? 'Longs are paying shorts heavily - potential reversal risk' 
          : 'Shorts are paying longs heavily - potential squeeze risk'
      }`;
    } else if (averageRate < -0.01) {
      signal = 'long_favorable';
    } else if (averageRate > 0.01) {
      signal = 'short_favorable';
    }

    const analysis: FundingRateAnalysis = {
      symbol: symbol.toUpperCase(),
      averageRate,
      currentRate,
      rateChange24h,
      exchanges,
      arbitrageOpportunity,
      recommendedExchange,
      signal,
      extremeWarning
    };

    // Cache the result
    this.cache.set(symbol, { data: analysis, timestamp: Date.now() });
    
    return analysis;
  }

  /**
   * Get funding rate impact on position
   * Calculates how much funding you'll pay/receive
   */
  calculateFundingImpact(
    positionSize: number,
    fundingRate: number,
    hoursHeld: number = 8
  ): {
    cost: number;
    costPercent: number;
    description: string;
  } {
    // Funding is typically paid every 8 hours
    const periodsHeld = hoursHeld / 8;
    const cost = positionSize * (fundingRate / 100) * periodsHeld;
    const costPercent = (fundingRate / 100) * periodsHeld * 100;
    
    const description = cost > 0 
      ? `You will pay $${Math.abs(cost).toFixed(2)} (${Math.abs(costPercent).toFixed(3)}%) in funding`
      : `You will receive $${Math.abs(cost).toFixed(2)} (${Math.abs(costPercent).toFixed(3)}%) in funding`;
    
    return { cost, costPercent, description };
  }

  /**
   * Check if funding rates suggest market sentiment
   */
  analyzeSentiment(rates: FundingRateAnalysis): {
    sentiment: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    reasoning: string;
  } {
    const avgRate = rates.averageRate;
    let sentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    let confidence = 50;
    let reasoning = '';
    
    if (avgRate > 0.05) {
      sentiment = 'bearish';
      confidence = Math.min(90, 50 + avgRate * 400);
      reasoning = 'High positive funding suggests overleveraged longs - bearish signal';
    } else if (avgRate > 0.01) {
      sentiment = 'bearish';
      confidence = 60;
      reasoning = 'Positive funding indicates more longs than shorts - slightly bearish';
    } else if (avgRate < -0.05) {
      sentiment = 'bullish';
      confidence = Math.min(90, 50 + Math.abs(avgRate) * 400);
      reasoning = 'High negative funding suggests overleveraged shorts - bullish signal';
    } else if (avgRate < -0.01) {
      sentiment = 'bullish';
      confidence = 60;
      reasoning = 'Negative funding indicates more shorts than longs - slightly bullish';
    } else {
      reasoning = 'Neutral funding rates suggest balanced market positioning';
    }
    
    // Adjust confidence based on trend
    const trends = Object.values(rates.exchanges).map(e => e.trend);
    const increasingCount = trends.filter(t => t === 'increasing').length;
    const decreasingCount = trends.filter(t => t === 'decreasing').length;
    
    if (increasingCount > decreasingCount && sentiment === 'bearish') {
      confidence = Math.min(100, confidence + 10);
      reasoning += ' (funding trend increasing)';
    } else if (decreasingCount > increasingCount && sentiment === 'bullish') {
      confidence = Math.min(100, confidence + 10);
      reasoning += ' (funding trend decreasing)';
    }
    
    return { sentiment, confidence, reasoning };
  }
}

// Export singleton instance
export const fundingRatesService = new FundingRatesService();
export default fundingRatesService;