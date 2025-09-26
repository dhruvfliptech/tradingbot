import { EventEmitter } from 'events';
import axios from 'axios';
import logger from '../../utils/logger';
import { MarketData } from '../../types/trading';
import { StateManager } from './StateManager';

export class MarketDataService extends EventEmitter {
  private stateManager: StateManager;
  private dataProviders: Map<string, any> = new Map();
  private marketDataCache: Map<string, MarketData> = new Map();
  private subscriptions: Set<string> = new Set();
  private updateInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.stateManager = new StateManager();
    this.initializeProviders();
  }

  private initializeProviders(): void {
    // Initialize data provider connections
    this.dataProviders.set('coingecko', {
      url: process.env.COINGECKO_API_URL || 'https://api.coingecko.com/api/v3',
      apiKey: process.env.COINGECKO_API_KEY
    });

    this.dataProviders.set('binance', {
      url: process.env.BINANCE_API_URL || 'https://api.binance.com/api/v3',
      apiKey: process.env.BINANCE_API_KEY
    });
  }

  async initialize(): Promise<void> {
    try {
      // Start market data update loop
      this.startMarketDataUpdates();
      logger.info('Market Data Service initialized');
    } catch (error) {
      logger.error('Failed to initialize Market Data Service:', error);
      throw error;
    }
  }

  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    const marketData: MarketData[] = [];

    for (const symbol of symbols) {
      try {
        // Check cache first
        const cached = await this.stateManager.getCachedMarketData(symbol);
        if (cached) {
          marketData.push(cached);
          continue;
        }

        // Fetch fresh data
        const data = await this.fetchMarketData(symbol);
        if (data) {
          // Cache the data
          await this.stateManager.cacheMarketData(symbol, data);
          marketData.push(data);
        }
      } catch (error) {
        logger.error(`Failed to get market data for ${symbol}:`, error);
      }
    }

    return marketData;
  }

  private async fetchMarketData(symbol: string): Promise<MarketData | null> {
    try {
      // Try multiple data sources
      const [coinGeckoData, binanceData] = await Promise.allSettled([
        this.fetchFromCoinGecko(symbol),
        this.fetchFromBinance(symbol)
      ]);

      // Prefer Binance for real-time data
      if (binanceData.status === 'fulfilled' && binanceData.value) {
        return binanceData.value;
      }

      // Fallback to CoinGecko
      if (coinGeckoData.status === 'fulfilled' && coinGeckoData.value) {
        return coinGeckoData.value;
      }

      return null;
    } catch (error) {
      logger.error(`Error fetching market data for ${symbol}:`, error);
      return null;
    }
  }

  private async fetchFromCoinGecko(symbol: string): Promise<MarketData | null> {
    try {
      const provider = this.dataProviders.get('coingecko');
      const response = await axios.get(
        `${provider.url}/simple/price`,
        {
          params: {
            ids: symbol,
            vs_currencies: 'usd',
            include_market_cap: true,
            include_24hr_vol: true,
            include_24hr_change: true
          },
          timeout: 5000
        }
      );

      if (response.data[symbol]) {
        const data = response.data[symbol];
        return {
          symbol,
          price: data.usd,
          volume: data.usd_24h_vol,
          changePercent: data.usd_24h_change,
          marketCap: data.usd_market_cap,
          bid: data.usd * 0.999, // Approximate bid
          ask: data.usd * 1.001, // Approximate ask
          timestamp: new Date()
        };
      }

      return null;
    } catch (error) {
      logger.debug(`CoinGecko fetch failed for ${symbol}:`, error);
      return null;
    }
  }

  private async fetchFromBinance(symbol: string): Promise<MarketData | null> {
    try {
      const provider = this.dataProviders.get('binance');
      const tradingPair = this.symbolToTradingPair(symbol);

      // Get ticker data
      const tickerResponse = await axios.get(
        `${provider.url}/ticker/24hr`,
        {
          params: { symbol: tradingPair },
          timeout: 5000
        }
      );

      // Get order book for bid/ask
      const depthResponse = await axios.get(
        `${provider.url}/depth`,
        {
          params: { symbol: tradingPair, limit: 5 },
          timeout: 5000
        }
      );

      const ticker = tickerResponse.data;
      const depth = depthResponse.data;

      return {
        symbol,
        price: parseFloat(ticker.lastPrice),
        volume: parseFloat(ticker.volume),
        changePercent: parseFloat(ticker.priceChangePercent),
        marketCap: 0, // Not available from Binance
        bid: parseFloat(depth.bids[0][0]),
        ask: parseFloat(depth.asks[0][0]),
        volatility: this.calculateVolatility(ticker),
        timestamp: new Date()
      };
    } catch (error) {
      logger.debug(`Binance fetch failed for ${symbol}:`, error);
      return null;
    }
  }

  private symbolToTradingPair(symbol: string): string {
    const mapping: Record<string, string> = {
      'bitcoin': 'BTCUSDT',
      'ethereum': 'ETHUSDT',
      'binancecoin': 'BNBUSDT',
      'cardano': 'ADAUSDT',
      'solana': 'SOLUSDT',
      'polkadot': 'DOTUSDT',
      'avalanche': 'AVAXUSDT',
      'chainlink': 'LINKUSDT',
      'polygon': 'MATICUSDT',
      'uniswap': 'UNIUSDT'
    };

    return mapping[symbol.toLowerCase()] || `${symbol.toUpperCase()}USDT`;
  }

  private calculateVolatility(ticker: any): number {
    // Simple volatility calculation based on high/low range
    const high = parseFloat(ticker.highPrice);
    const low = parseFloat(ticker.lowPrice);
    const current = parseFloat(ticker.lastPrice);

    if (current > 0) {
      return ((high - low) / current) * 100;
    }

    return 0;
  }

  async subscribeToRealtime(symbols: string[]): Promise<void> {
    for (const symbol of symbols) {
      this.subscriptions.add(symbol);
    }

    // Restart update loop with new symbols
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    this.startMarketDataUpdates();
  }

  async unsubscribeFromRealtime(symbols: string[]): Promise<void> {
    for (const symbol of symbols) {
      this.subscriptions.delete(symbol);
    }
  }

  private startMarketDataUpdates(): void {
    // Update market data every 30 seconds
    this.updateInterval = setInterval(async () => {
      if (this.subscriptions.size === 0) return;

      const symbols = Array.from(this.subscriptions);
      const marketData = await this.getMarketData(symbols);

      // Emit updates
      for (const data of marketData) {
        this.emit('marketDataUpdate', data);

        // Also publish to Redis for other services
        await this.stateManager.publishEvent('trading:market_data', {
          type: 'price_update',
          data
        });
      }
    }, 30000);
  }

  async getHistoricalData(symbol: string, period: string = '7d'): Promise<any> {
    try {
      const provider = this.dataProviders.get('coingecko');
      const days = this.periodToDays(period);

      const response = await axios.get(
        `${provider.url}/coins/${symbol}/market_chart`,
        {
          params: {
            vs_currency: 'usd',
            days: days,
            interval: days > 30 ? 'daily' : 'hourly'
          },
          timeout: 10000
        }
      );

      return {
        prices: response.data.prices,
        volumes: response.data.total_volumes,
        marketCaps: response.data.market_caps
      };
    } catch (error) {
      logger.error(`Failed to get historical data for ${symbol}:`, error);
      return null;
    }
  }

  private periodToDays(period: string): number {
    const mapping: Record<string, number> = {
      '24h': 1,
      '7d': 7,
      '30d': 30,
      '90d': 90,
      '1y': 365
    };

    return mapping[period] || 7;
  }

  async getOrderBook(symbol: string, limit: number = 20): Promise<any> {
    try {
      const provider = this.dataProviders.get('binance');
      const tradingPair = this.symbolToTradingPair(symbol);

      const response = await axios.get(
        `${provider.url}/depth`,
        {
          params: { symbol: tradingPair, limit },
          timeout: 5000
        }
      );

      return {
        symbol,
        bids: response.data.bids.map((b: any) => ({
          price: parseFloat(b[0]),
          quantity: parseFloat(b[1])
        })),
        asks: response.data.asks.map((a: any) => ({
          price: parseFloat(a[0]),
          quantity: parseFloat(a[1])
        })),
        timestamp: new Date()
      };
    } catch (error) {
      logger.error(`Failed to get order book for ${symbol}:`, error);
      return null;
    }
  }

  async getTrades(symbol: string, limit: number = 50): Promise<any> {
    try {
      const provider = this.dataProviders.get('binance');
      const tradingPair = this.symbolToTradingPair(symbol);

      const response = await axios.get(
        `${provider.url}/trades`,
        {
          params: { symbol: tradingPair, limit },
          timeout: 5000
        }
      );

      return response.data.map((trade: any) => ({
        id: trade.id,
        price: parseFloat(trade.price),
        quantity: parseFloat(trade.qty),
        time: new Date(trade.time),
        isBuyerMaker: trade.isBuyerMaker
      }));
    } catch (error) {
      logger.error(`Failed to get trades for ${symbol}:`, error);
      return [];
    }
  }

  async getMarketStats(): Promise<any> {
    try {
      const provider = this.dataProviders.get('coingecko');

      const response = await axios.get(
        `${provider.url}/global`,
        { timeout: 5000 }
      );

      return {
        totalMarketCap: response.data.data.total_market_cap.usd,
        totalVolume: response.data.data.total_volume.usd,
        btcDominance: response.data.data.market_cap_percentage.btc,
        activeCoins: response.data.data.active_cryptocurrencies,
        markets: response.data.data.markets,
        marketCapChange24h: response.data.data.market_cap_change_percentage_24h_usd
      };
    } catch (error) {
      logger.error('Failed to get market stats:', error);
      return null;
    }
  }

  cleanup(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    this.subscriptions.clear();
    this.marketDataCache.clear();
  }
}

export default MarketDataService;