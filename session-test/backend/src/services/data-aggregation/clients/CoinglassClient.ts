import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface CoinglassConfig {
  endpoint: string;
  rateLimit: number;
}

export interface CoinglasFundingRate {
  symbol: string;
  rate: number;
  nextFundingTime: number;
  markPrice: number;
  indexPrice: number;
  estimatedRate: number;
  timestamp: number;
  exchange: string;
}

export interface CoinglassLiquidation {
  symbol: string;
  side: 'long' | 'short';
  amount: number;
  amountUsd: number;
  price: number;
  timestamp: number;
  exchange: string;
  percentage: number;
}

export interface CoinglassOpenInterest {
  symbol: string;
  openInterest: number;
  openInterestUsd: number;
  change24h: number;
  changePercent24h: number;
  timestamp: number;
  exchange: string;
}

export interface CoinglassLongShortRatio {
  symbol: string;
  longRatio: number;
  shortRatio: number;
  longAccount: number;
  shortAccount: number;
  timestamp: number;
  exchange: string;
}

export class CoinglassClient {
  private config: CoinglassConfig;
  private baseUrl: string;

  constructor(config: CoinglassConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://fapi.coinglass.com';
  }

  async testConnection(): Promise<void> {
    try {
      // Test with a simple endpoint
      const response = await this.makeRequest('/api/futures/funding/v2', {
        symbol: 'BTC',
        type: 'C'
      });

      if (!response || response.code !== 0) {
        throw new Error(`Coinglass API test failed: ${response?.msg || 'Unknown error'}`);
      }

      logger.debug('Coinglass API connection test successful');
    } catch (error) {
      logger.error('Coinglass API connection test failed:', error);
      throw error;
    }
  }

  async getFundingRate(symbol: string): Promise<CoinglasFundingRate | null> {
    try {
      logger.debug(`Fetching funding rate for ${symbol} from Coinglass`);

      const response = await this.makeRequest('/api/futures/funding/v2', {
        symbol: symbol.replace('USDT', '').replace('USD', ''),
        type: 'C' // Current funding rate
      });

      if (!response || response.code !== 0) {
        logger.warn(`Failed to get funding rate for ${symbol}: ${response?.msg}`);
        return null;
      }

      const data = response.data;
      if (!data || !data.dataMap) {
        return null;
      }

      // Get average across exchanges
      const exchanges = Object.keys(data.dataMap);
      let totalRate = 0;
      let validExchanges = 0;
      let exchangeData: any = null;

      for (const exchange of exchanges) {
        const exchangeInfo = data.dataMap[exchange];
        if (exchangeInfo && exchangeInfo.rate !== null && exchangeInfo.rate !== undefined) {
          totalRate += parseFloat(exchangeInfo.rate);
          validExchanges++;
          if (!exchangeData) exchangeData = exchangeInfo; // Use first valid exchange for other data
        }
      }

      if (validExchanges === 0) {
        return null;
      }

      const avgRate = totalRate / validExchanges;

      return {
        symbol,
        rate: avgRate,
        nextFundingTime: exchangeData.nextFundingTime || Date.now() + 8 * 60 * 60 * 1000, // Default to 8 hours
        markPrice: exchangeData.markPrice || 0,
        indexPrice: exchangeData.indexPrice || 0,
        estimatedRate: exchangeData.estimatedRate || avgRate,
        timestamp: Date.now(),
        exchange: 'average'
      };

    } catch (error) {
      logger.error(`Error fetching funding rate for ${symbol}:`, error);
      return null;
    }
  }

  async getAllFundingRates(): Promise<CoinglasFundingRate[]> {
    try {
      const response = await this.makeRequest('/api/futures/funding/v2', {
        type: 'C'
      });

      if (!response || response.code !== 0) {
        logger.warn(`Failed to get all funding rates: ${response?.msg}`);
        return [];
      }

      const results: CoinglasFundingRate[] = [];
      const symbolsData = response.data;

      for (const [symbol, data] of Object.entries(symbolsData)) {
        if (typeof data === 'object' && data !== null && 'dataMap' in data) {
          const symbolData = data as any;
          const exchanges = Object.keys(symbolData.dataMap);
          
          let totalRate = 0;
          let validExchanges = 0;
          let exchangeData: any = null;

          for (const exchange of exchanges) {
            const exchangeInfo = symbolData.dataMap[exchange];
            if (exchangeInfo && exchangeInfo.rate !== null) {
              totalRate += parseFloat(exchangeInfo.rate);
              validExchanges++;
              if (!exchangeData) exchangeData = exchangeInfo;
            }
          }

          if (validExchanges > 0) {
            results.push({
              symbol,
              rate: totalRate / validExchanges,
              nextFundingTime: exchangeData.nextFundingTime || Date.now() + 8 * 60 * 60 * 1000,
              markPrice: exchangeData.markPrice || 0,
              indexPrice: exchangeData.indexPrice || 0,
              estimatedRate: exchangeData.estimatedRate || (totalRate / validExchanges),
              timestamp: Date.now(),
              exchange: 'average'
            });
          }
        }
      }

      return results;

    } catch (error) {
      logger.error('Error fetching all funding rates:', error);
      return [];
    }
  }

  async getLiquidations(symbol: string, timeRange: '5m' | '15m' | '1h' | '4h' | '12h' | '24h' = '1h'): Promise<CoinglassLiquidation[]> {
    try {
      logger.debug(`Fetching liquidations for ${symbol} with range ${timeRange}`);

      const response = await this.makeRequest('/api/futures/liquidation_v2', {
        symbol: symbol.replace('USDT', '').replace('USD', ''),
        timeType: this.convertTimeRange(timeRange),
        dataType: 0 // 0 for liquidation data
      });

      if (!response || response.code !== 0) {
        logger.warn(`Failed to get liquidations for ${symbol}: ${response?.msg}`);
        return [];
      }

      const liquidations: CoinglassLiquidation[] = [];
      const data = response.data;

      if (!data || !data.dataMap) {
        return [];
      }

      // Process liquidation data from all exchanges
      for (const [exchange, exchangeData] of Object.entries(data.dataMap)) {
        if (typeof exchangeData === 'object' && exchangeData !== null && 'data' in exchangeData) {
          const liqData = (exchangeData as any).data;
          
          if (Array.isArray(liqData)) {
            for (const item of liqData) {
              // Long liquidations
              if (item.longLiqUsd && parseFloat(item.longLiqUsd) > 0) {
                liquidations.push({
                  symbol,
                  side: 'long',
                  amount: parseFloat(item.longLiq || '0'),
                  amountUsd: parseFloat(item.longLiqUsd),
                  price: parseFloat(item.price || '0'),
                  timestamp: item.createTime || Date.now(),
                  exchange,
                  percentage: parseFloat(item.longLiqUsd) / (parseFloat(item.longLiqUsd) + parseFloat(item.shortLiqUsd || '0')) * 100
                });
              }

              // Short liquidations
              if (item.shortLiqUsd && parseFloat(item.shortLiqUsd) > 0) {
                liquidations.push({
                  symbol,
                  side: 'short',
                  amount: parseFloat(item.shortLiq || '0'),
                  amountUsd: parseFloat(item.shortLiqUsd),
                  price: parseFloat(item.price || '0'),
                  timestamp: item.createTime || Date.now(),
                  exchange,
                  percentage: parseFloat(item.shortLiqUsd) / (parseFloat(item.longLiqUsd || '0') + parseFloat(item.shortLiqUsd)) * 100
                });
              }
            }
          }
        }
      }

      return liquidations.sort((a, b) => b.timestamp - a.timestamp);

    } catch (error) {
      logger.error(`Error fetching liquidations for ${symbol}:`, error);
      return [];
    }
  }

  async getOpenInterest(symbol: string): Promise<CoinglassOpenInterest[]> {
    try {
      logger.debug(`Fetching open interest for ${symbol}`);

      const response = await this.makeRequest('/api/futures/openInterest/v2', {
        symbol: symbol.replace('USDT', '').replace('USD', ''),
        timeType: 'h1' // 1 hour intervals
      });

      if (!response || response.code !== 0) {
        logger.warn(`Failed to get open interest for ${symbol}: ${response?.msg}`);
        return [];
      }

      const results: CoinglassOpenInterest[] = [];
      const data = response.data;

      if (!data || !data.dataMap) {
        return [];
      }

      for (const [exchange, exchangeData] of Object.entries(data.dataMap)) {
        if (typeof exchangeData === 'object' && exchangeData !== null && 'data' in exchangeData) {
          const oiData = (exchangeData as any).data;
          
          if (Array.isArray(oiData) && oiData.length > 0) {
            const latest = oiData[oiData.length - 1];
            const previous = oiData.length > 1 ? oiData[oiData.length - 2] : null;
            
            const currentOI = parseFloat(latest.openInterest || '0');
            const previousOI = previous ? parseFloat(previous.openInterest || '0') : currentOI;
            const change24h = currentOI - previousOI;
            const changePercent24h = previousOI > 0 ? (change24h / previousOI) * 100 : 0;

            results.push({
              symbol,
              openInterest: currentOI,
              openInterestUsd: parseFloat(latest.openInterestUsd || '0'),
              change24h,
              changePercent24h,
              timestamp: latest.createTime || Date.now(),
              exchange
            });
          }
        }
      }

      return results;

    } catch (error) {
      logger.error(`Error fetching open interest for ${symbol}:`, error);
      return [];
    }
  }

  async getLongShortRatio(symbol: string): Promise<CoinglassLongShortRatio[]> {
    try {
      logger.debug(`Fetching long/short ratio for ${symbol}`);

      const response = await this.makeRequest('/api/futures/longShortRatio', {
        symbol: symbol.replace('USDT', '').replace('USD', ''),
        timeType: 'h1'
      });

      if (!response || response.code !== 0) {
        logger.warn(`Failed to get long/short ratio for ${symbol}: ${response?.msg}`);
        return [];
      }

      const results: CoinglassLongShortRatio[] = [];
      const data = response.data;

      if (!data || !data.dataMap) {
        return [];
      }

      for (const [exchange, exchangeData] of Object.entries(data.dataMap)) {
        if (typeof exchangeData === 'object' && exchangeData !== null && 'data' in exchangeData) {
          const ratioData = (exchangeData as any).data;
          
          if (Array.isArray(ratioData) && ratioData.length > 0) {
            const latest = ratioData[ratioData.length - 1];
            
            results.push({
              symbol,
              longRatio: parseFloat(latest.longRate || '0'),
              shortRatio: parseFloat(latest.shortRate || '0'),
              longAccount: parseFloat(latest.longAccount || '0'),
              shortAccount: parseFloat(latest.shortAccount || '0'),
              timestamp: latest.createTime || Date.now(),
              exchange
            });
          }
        }
      }

      return results;

    } catch (error) {
      logger.error(`Error fetching long/short ratio for ${symbol}:`, error);
      return [];
    }
  }

  async getMarketOverview(): Promise<{
    totalMarketCap: number;
    totalVolume24h: number;
    btcDominance: number;
    fearGreedIndex: number;
    topGainers: Array<{
      symbol: string;
      change24h: number;
      volume24h: number;
    }>;
    topLosers: Array<{
      symbol: string;
      change24h: number;
      volume24h: number;
    }>;
  }> {
    try {
      // Coinglass might not have all this data, so we'll provide a basic structure
      return {
        totalMarketCap: 0,
        totalVolume24h: 0,
        btcDominance: 0,
        fearGreedIndex: 50, // Neutral
        topGainers: [],
        topLosers: []
      };

    } catch (error) {
      logger.error('Error fetching market overview:', error);
      return {
        totalMarketCap: 0,
        totalVolume24h: 0,
        btcDominance: 0,
        fearGreedIndex: 50,
        topGainers: [],
        topLosers: []
      };
    }
  }

  async getExchangeInfo(): Promise<Array<{
    name: string;
    volume24h: number;
    marketShare: number;
    status: 'online' | 'offline' | 'maintenance';
  }>> {
    try {
      // This would fetch exchange information
      // For now, return some common exchanges
      return [
        { name: 'Binance', volume24h: 0, marketShare: 0, status: 'online' },
        { name: 'OKX', volume24h: 0, marketShare: 0, status: 'online' },
        { name: 'Bybit', volume24h: 0, marketShare: 0, status: 'online' },
        { name: 'BitMEX', volume24h: 0, marketShare: 0, status: 'online' }
      ];

    } catch (error) {
      logger.error('Error fetching exchange info:', error);
      return [];
    }
  }

  private async makeRequest(endpoint: string, params: Record<string, any> = {}): Promise<any> {
    const url = new URL(`${this.baseUrl}${endpoint}`);
    
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value.toString());
    });

    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'User-Agent': 'TradingBot/1.0',
          'Accept': 'application/json'
        },
        timeout: 10000
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error(`Coinglass API request failed: ${url.toString()}`, error);
      throw error;
    }
  }

  private convertTimeRange(timeRange: string): string {
    const timeMap: Record<string, string> = {
      '5m': 'm5',
      '15m': 'm15',
      '1h': 'h1',
      '4h': 'h4',
      '12h': 'h12',
      '24h': 'h24'
    };

    return timeMap[timeRange] || 'h1';
  }

  // Advanced analytics methods

  async getFundingRateHeatmap(): Promise<Array<{
    symbol: string;
    rate: number;
    percentile: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  }>> {
    try {
      const allRates = await this.getAllFundingRates();
      const sorted = allRates.sort((a, b) => a.rate - b.rate);
      
      return sorted.map((rate, index) => ({
        symbol: rate.symbol,
        rate: rate.rate,
        percentile: (index / sorted.length) * 100,
        trend: this.determineTrend(rate) // This would need historical data
      }));

    } catch (error) {
      logger.error('Error creating funding rate heatmap:', error);
      return [];
    }
  }

  async getLiquidationHeatmap(timeRange: '1h' | '4h' | '24h' = '1h'): Promise<Array<{
    symbol: string;
    totalLiquidationsUsd: number;
    longLiquidationPercent: number;
    shortLiquidationPercent: number;
    severity: 'low' | 'medium' | 'high';
  }>> {
    // This would aggregate liquidation data across symbols
    // For now, return empty array as it requires multiple API calls
    return [];
  }

  private determineTrend(rate: CoinglasFundingRate): 'increasing' | 'decreasing' | 'stable' {
    // This would require historical data comparison
    // For now, use a simple heuristic based on estimated vs current rate
    const diff = rate.estimatedRate - rate.rate;
    if (Math.abs(diff) < 0.0001) return 'stable';
    return diff > 0 ? 'increasing' : 'decreasing';
  }
}