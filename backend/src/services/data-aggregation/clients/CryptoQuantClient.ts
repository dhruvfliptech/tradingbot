import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface CryptoQuantConfig {
  apiKey?: string;
  endpoint: string;
  rateLimit: number;
}

export interface CryptoQuantMetric {
  date: string;
  value: number;
  symbol?: string;
  exchange?: string;
}

export interface CryptoQuantExchangeFlow {
  symbol: string;
  exchange: string;
  inflow: number;
  outflow: number;
  netFlow: number;
  timestamp: Date;
}

export interface CryptoQuantSmartMoney {
  address: string;
  label: string;
  action: 'accumulation' | 'distribution' | 'neutral';
  amount: number;
  confidence: number;
  timestamp: Date;
}

export interface CryptoQuantMinerMetrics {
  hashRate: number;
  difficulty: number;
  minerRevenue: number;
  feeRate: number;
  timestamp: Date;
}

export class CryptoQuantClient {
  private config: CryptoQuantConfig;
  private baseUrl: string;

  constructor(config: CryptoQuantConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://api.cryptoquant.com';
  }

  async testConnection(): Promise<void> {
    try {
      // Test with a simple metrics endpoint (might require API key)
      if (!this.config.apiKey) {
        logger.warn('CryptoQuant API key not provided, using free tier endpoints only');
        return;
      }

      const response = await this.makeRequest('/v1/btc/indicator/exchange-flows');
      
      if (response.error) {
        throw new Error(`CryptoQuant API test failed: ${response.error}`);
      }

      logger.debug('CryptoQuant API connection test successful');
    } catch (error) {
      logger.error('CryptoQuant API connection test failed:', error);
      throw error;
    }
  }

  async getExchangeFlows(symbol: string = 'BTC', exchange?: string): Promise<CryptoQuantExchangeFlow[]> {
    try {
      logger.debug(`Fetching exchange flows for ${symbol}${exchange ? ` on ${exchange}` : ''}`);

      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/exchange-flows`;
      const params: any = {};
      
      if (exchange) {
        params.exchange = exchange;
      }

      const response = await this.makeRequest(endpoint, params);

      if (response.error) {
        logger.warn(`Failed to get exchange flows: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        symbol: symbol.toUpperCase(),
        exchange: item.exchange || 'all',
        inflow: parseFloat(item.inflow || '0'),
        outflow: parseFloat(item.outflow || '0'),
        netFlow: parseFloat(item.net_flow || '0'),
        timestamp: new Date(item.datetime)
      }));

    } catch (error) {
      logger.error(`Error fetching exchange flows for ${symbol}:`, error);
      return [];
    }
  }

  async getExchangeReserves(symbol: string = 'BTC', exchange?: string): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/exchange-reserve`;
      const params: any = {};
      
      if (exchange) {
        params.exchange = exchange;
      }

      const response = await this.makeRequest(endpoint, params);

      if (response.error) {
        logger.warn(`Failed to get exchange reserves: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase(),
        exchange: item.exchange
      }));

    } catch (error) {
      logger.error(`Error fetching exchange reserves for ${symbol}:`, error);
      return [];
    }
  }

  async getWhaleTransactions(symbol: string = 'BTC', threshold: number = 100): Promise<Array<{
    hash: string;
    from: string;
    to: string;
    amount: number;
    amountUsd: number;
    timestamp: Date;
    type: 'exchange_inflow' | 'exchange_outflow' | 'whale_transfer';
  }>> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/large-transactions`;
      const params = {
        threshold: threshold.toString()
      };

      const response = await this.makeRequest(endpoint, params);

      if (response.error) {
        logger.warn(`Failed to get whale transactions: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        hash: item.tx_hash,
        from: item.from_address,
        to: item.to_address,
        amount: parseFloat(item.amount),
        amountUsd: parseFloat(item.amount_usd || '0'),
        timestamp: new Date(item.datetime),
        type: this.determineTransactionType(item)
      }));

    } catch (error) {
      logger.error(`Error fetching whale transactions for ${symbol}:`, error);
      return [];
    }
  }

  async getSmartMoneyFlows(symbol: string = 'BTC'): Promise<CryptoQuantSmartMoney[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/smart-money`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get smart money flows: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        address: item.address,
        label: item.label || 'Unknown Smart Money',
        action: this.determineSmartMoneyAction(item),
        amount: parseFloat(item.amount || '0'),
        confidence: this.calculateSmartMoneyConfidence(item),
        timestamp: new Date(item.datetime)
      }));

    } catch (error) {
      logger.error(`Error fetching smart money flows for ${symbol}:`, error);
      return [];
    }
  }

  async getMinerMetrics(symbol: string = 'BTC'): Promise<CryptoQuantMinerMetrics[]> {
    try {
      const endpoints = [
        `/v1/${symbol.toLowerCase()}/indicator/hash-rate`,
        `/v1/${symbol.toLowerCase()}/indicator/difficulty`,
        `/v1/${symbol.toLowerCase()}/indicator/miner-revenue`
      ];

      const responses = await Promise.allSettled(
        endpoints.map(endpoint => this.makeRequest(endpoint))
      );

      const metrics: CryptoQuantMinerMetrics[] = [];
      
      // Process each response and combine data by timestamp
      responses.forEach((result, index) => {
        if (result.status === 'fulfilled' && !result.value.error) {
          const data = result.value.result?.data || [];
          // This is a simplified version - in reality you'd need to align timestamps
          data.forEach((item: any, i: number) => {
            if (i < 10) { // Limit to recent data
              const existingMetric = metrics[i];
              if (existingMetric) {
                // Update existing metric
                switch (index) {
                  case 0: existingMetric.hashRate = parseFloat(item.value || '0'); break;
                  case 1: existingMetric.difficulty = parseFloat(item.value || '0'); break;
                  case 2: existingMetric.minerRevenue = parseFloat(item.value || '0'); break;
                }
              } else {
                // Create new metric
                const newMetric: CryptoQuantMinerMetrics = {
                  hashRate: index === 0 ? parseFloat(item.value || '0') : 0,
                  difficulty: index === 1 ? parseFloat(item.value || '0') : 0,
                  minerRevenue: index === 2 ? parseFloat(item.value || '0') : 0,
                  feeRate: 0,
                  timestamp: new Date(item.datetime)
                };
                metrics.push(newMetric);
              }
            }
          });
        }
      });

      return metrics;

    } catch (error) {
      logger.error(`Error fetching miner metrics for ${symbol}:`, error);
      return [];
    }
  }

  async getNVT(symbol: string = 'BTC'): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/nvt`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get NVT: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase()
      }));

    } catch (error) {
      logger.error(`Error fetching NVT for ${symbol}:`, error);
      return [];
    }
  }

  async getMVRV(symbol: string = 'BTC'): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/mvrv`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get MVRV: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase()
      }));

    } catch (error) {
      logger.error(`Error fetching MVRV for ${symbol}:`, error);
      return [];
    }
  }

  async getSOPR(symbol: string = 'BTC'): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/sopr`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get SOPR: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase()
      }));

    } catch (error) {
      logger.error(`Error fetching SOPR for ${symbol}:`, error);
      return [];
    }
  }

  async getStablecoinSupply(symbol: string = 'USDT'): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/supply`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get stablecoin supply: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase()
      }));

    } catch (error) {
      logger.error(`Error fetching stablecoin supply for ${symbol}:`, error);
      return [];
    }
  }

  async getLongTermHolderSupply(symbol: string = 'BTC'): Promise<CryptoQuantMetric[]> {
    try {
      const endpoint = `/v1/${symbol.toLowerCase()}/indicator/lth-supply`;
      const response = await this.makeRequest(endpoint);

      if (response.error) {
        logger.warn(`Failed to get LTH supply: ${response.error}`);
        return [];
      }

      const data = response.result?.data || [];
      
      return data.map((item: any) => ({
        date: item.datetime,
        value: parseFloat(item.value || '0'),
        symbol: symbol.toUpperCase()
      }));

    } catch (error) {
      logger.error(`Error fetching LTH supply for ${symbol}:`, error);
      return [];
    }
  }

  async getFearGreedIndex(): Promise<{
    value: number;
    classification: string;
    timestamp: Date;
    components: {
      volatility: number;
      volume: number;
      socialMedia: number;
      surveys: number;
      dominance: number;
      trends: number;
    };
  } | null> {
    try {
      // This might not be available in CryptoQuant free tier
      // Return a placeholder implementation
      return {
        value: 50,
        classification: 'Neutral',
        timestamp: new Date(),
        components: {
          volatility: 50,
          volume: 50,
          socialMedia: 50,
          surveys: 50,
          dominance: 50,
          trends: 50
        }
      };

    } catch (error) {
      logger.error('Error fetching fear & greed index:', error);
      return null;
    }
  }

  private async makeRequest(endpoint: string, params: Record<string, any> = {}): Promise<any> {
    const url = new URL(`${this.baseUrl}${endpoint}`);
    
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value.toString());
    });

    const headers: Record<string, string> = {
      'User-Agent': 'TradingBot/1.0',
      'Accept': 'application/json'
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers,
        timeout: 15000
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error(`CryptoQuant API request failed: ${url.toString()}`, error);
      throw error;
    }
  }

  private determineTransactionType(item: any): 'exchange_inflow' | 'exchange_outflow' | 'whale_transfer' {
    // This would use logic to determine transaction type based on addresses
    // For now, return a default value
    if (item.to_label && item.to_label.includes('exchange')) {
      return 'exchange_inflow';
    } else if (item.from_label && item.from_label.includes('exchange')) {
      return 'exchange_outflow';
    }
    return 'whale_transfer';
  }

  private determineSmartMoneyAction(item: any): 'accumulation' | 'distribution' | 'neutral' {
    // Logic to determine smart money action based on transaction patterns
    const amount = parseFloat(item.amount || '0');
    const flow = parseFloat(item.net_flow || '0');
    
    if (flow > amount * 0.1) return 'accumulation';
    if (flow < -amount * 0.1) return 'distribution';
    return 'neutral';
  }

  private calculateSmartMoneyConfidence(item: any): number {
    // Calculate confidence based on various factors
    let confidence = 50; // Base confidence
    
    // Increase confidence for larger amounts
    const amount = parseFloat(item.amount || '0');
    if (amount > 1000) confidence += 20;
    else if (amount > 100) confidence += 10;
    
    // Increase confidence for known labels
    if (item.label && item.label !== 'Unknown') {
      confidence += 15;
    }
    
    // Increase confidence for consistent patterns
    if (item.pattern_score && parseFloat(item.pattern_score) > 0.7) {
      confidence += 15;
    }
    
    return Math.min(100, confidence);
  }

  // Market structure analysis methods

  async getMarketStructure(symbol: string = 'BTC'): Promise<{
    trend: 'bullish' | 'bearish' | 'sideways';
    strength: number;
    support: number[];
    resistance: number[];
    sentiment: 'extreme_fear' | 'fear' | 'neutral' | 'greed' | 'extreme_greed';
  }> {
    try {
      // This would combine multiple indicators to determine market structure
      // For now, return a basic structure
      return {
        trend: 'sideways',
        strength: 50,
        support: [],
        resistance: [],
        sentiment: 'neutral'
      };

    } catch (error) {
      logger.error(`Error analyzing market structure for ${symbol}:`, error);
      return {
        trend: 'sideways',
        strength: 0,
        support: [],
        resistance: [],
        sentiment: 'neutral'
      };
    }
  }

  async getCorrelationMatrix(symbols: string[] = ['BTC', 'ETH']): Promise<Record<string, Record<string, number>>> {
    try {
      // This would calculate correlation between different assets
      // For now, return empty correlation matrix
      const matrix: Record<string, Record<string, number>> = {};
      
      symbols.forEach(symbol1 => {
        matrix[symbol1] = {};
        symbols.forEach(symbol2 => {
          matrix[symbol1][symbol2] = symbol1 === symbol2 ? 1 : 0;
        });
      });

      return matrix;

    } catch (error) {
      logger.error('Error calculating correlation matrix:', error);
      return {};
    }
  }
}