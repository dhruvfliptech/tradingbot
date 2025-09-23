import fetch from 'node-fetch';
import logger from '../../../utils/logger';

export interface BinanceConfig {
  endpoint: string;
  rateLimit: number;
}

export interface BinanceFundingRate {
  symbol: string;
  fundingRate: string;
  fundingTime: number;
  markPrice: string;
  indexPrice: string;
  estimatedSettlePrice: string;
  lastFundingRate: string;
  nextFundingTime: number;
  time: number;
}

export interface BinanceOpenInterest {
  symbol: string;
  openInterest: string;
  time: number;
}

export interface BinanceLongShortRatio {
  symbol: string;
  longShortRatio: string;
  longAccount: string;
  shortAccount: string;
  timestamp: number;
}

export interface BinanceKline {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteAssetVolume: string;
  numberOfTrades: number;
  takerBuyBaseAssetVolume: string;
  takerBuyQuoteAssetVolume: string;
}

export interface BinanceTicker24hr {
  symbol: string;
  priceChange: string;
  priceChangePercent: string;
  weightedAvgPrice: string;
  prevClosePrice: string;
  lastPrice: string;
  lastQty: string;
  bidPrice: string;
  askPrice: string;
  openPrice: string;
  highPrice: string;
  lowPrice: string;
  volume: string;
  quoteVolume: string;
  openTime: number;
  closeTime: number;
  firstId: number;
  lastId: number;
  count: number;
}

export class BinanceClient {
  private config: BinanceConfig;
  private baseUrl: string;
  private futuresUrl: string;

  constructor(config: BinanceConfig) {
    this.config = config;
    this.baseUrl = config.endpoint || 'https://api.binance.com';
    this.futuresUrl = 'https://fapi.binance.com';
  }

  async testConnection(): Promise<void> {
    try {
      // Test with a simple ping endpoint
      const response = await this.makeRequest(`${this.baseUrl}/api/v3/ping`);
      
      if (response && typeof response === 'object') {
        logger.debug('Binance API connection test successful');
      } else {
        throw new Error('Unexpected response from Binance API');
      }

    } catch (error) {
      logger.error('Binance API connection test failed:', error);
      throw error;
    }
  }

  async getFundingRate(symbol: string): Promise<BinanceFundingRate | null> {
    try {
      logger.debug(`Fetching funding rate for ${symbol} from Binance`);

      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.futuresUrl}/fapi/v1/premiumIndex`, {
        symbol: formattedSymbol
      });

      if (!response) {
        logger.warn(`No funding rate data found for ${symbol}`);
        return null;
      }

      return {
        symbol: formattedSymbol,
        fundingRate: response.lastFundingRate,
        fundingTime: response.time,
        markPrice: response.markPrice,
        indexPrice: response.indexPrice,
        estimatedSettlePrice: response.estimatedSettlePrice || '0',
        lastFundingRate: response.lastFundingRate,
        nextFundingTime: response.nextFundingTime,
        time: response.time
      };

    } catch (error) {
      logger.error(`Error fetching funding rate for ${symbol}:`, error);
      return null;
    }
  }

  async getAllFundingRates(): Promise<BinanceFundingRate[]> {
    try {
      logger.debug('Fetching all funding rates from Binance');

      const response = await this.makeRequest(`${this.futuresUrl}/fapi/v1/premiumIndex`);

      if (!Array.isArray(response)) {
        logger.warn('Invalid response format for all funding rates');
        return [];
      }

      return response.map(item => ({
        symbol: item.symbol,
        fundingRate: item.lastFundingRate,
        fundingTime: item.time,
        markPrice: item.markPrice,
        indexPrice: item.indexPrice,
        estimatedSettlePrice: item.estimatedSettlePrice || '0',
        lastFundingRate: item.lastFundingRate,
        nextFundingTime: item.nextFundingTime,
        time: item.time
      }));

    } catch (error) {
      logger.error('Error fetching all funding rates:', error);
      return [];
    }
  }

  async getOpenInterest(symbol: string): Promise<BinanceOpenInterest | null> {
    try {
      logger.debug(`Fetching open interest for ${symbol} from Binance`);

      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.futuresUrl}/fapi/v1/openInterest`, {
        symbol: formattedSymbol
      });

      if (!response) {
        logger.warn(`No open interest data found for ${symbol}`);
        return null;
      }

      return {
        symbol: formattedSymbol,
        openInterest: response.openInterest,
        time: response.time
      };

    } catch (error) {
      logger.error(`Error fetching open interest for ${symbol}:`, error);
      return null;
    }
  }

  async getLongShortRatio(symbol: string, period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d' = '1h'): Promise<BinanceLongShortRatio[]> {
    try {
      logger.debug(`Fetching long/short ratio for ${symbol} with period ${period}`);

      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.futuresUrl}/futures/data/globalLongShortAccountRatio`, {
        symbol: formattedSymbol,
        period: period,
        limit: 30
      });

      if (!Array.isArray(response)) {
        logger.warn(`No long/short ratio data found for ${symbol}`);
        return [];
      }

      return response.map(item => ({
        symbol: formattedSymbol,
        longShortRatio: item.longShortRatio,
        longAccount: item.longAccount,
        shortAccount: item.shortAccount,
        timestamp: item.timestamp
      }));

    } catch (error) {
      logger.error(`Error fetching long/short ratio for ${symbol}:`, error);
      return [];
    }
  }

  async get24hrTicker(symbol?: string): Promise<BinanceTicker24hr | BinanceTicker24hr[]> {
    try {
      const params = symbol ? { symbol: this.formatSymbol(symbol) } : {};
      const response = await this.makeRequest(`${this.baseUrl}/api/v3/ticker/24hr`, params);

      if (!response) {
        throw new Error('No ticker data received');
      }

      return response;

    } catch (error) {
      logger.error(`Error fetching 24hr ticker${symbol ? ` for ${symbol}` : ''}:`, error);
      if (symbol) {
        return null as any;
      }
      return [];
    }
  }

  async getKlines(
    symbol: string,
    interval: '1m' | '3m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '8h' | '12h' | '1d' | '3d' | '1w' | '1M',
    limit: number = 500,
    startTime?: number,
    endTime?: number
  ): Promise<BinanceKline[]> {
    try {
      logger.debug(`Fetching klines for ${symbol} with interval ${interval}`);

      const params: any = {
        symbol: this.formatSymbol(symbol),
        interval,
        limit
      };

      if (startTime) params.startTime = startTime;
      if (endTime) params.endTime = endTime;

      const response = await this.makeRequest(`${this.baseUrl}/api/v3/klines`, params);

      if (!Array.isArray(response)) {
        logger.warn(`No kline data found for ${symbol}`);
        return [];
      }

      return response.map(kline => ({
        openTime: kline[0],
        open: kline[1],
        high: kline[2],
        low: kline[3],
        close: kline[4],
        volume: kline[5],
        closeTime: kline[6],
        quoteAssetVolume: kline[7],
        numberOfTrades: kline[8],
        takerBuyBaseAssetVolume: kline[9],
        takerBuyQuoteAssetVolume: kline[10]
      }));

    } catch (error) {
      logger.error(`Error fetching klines for ${symbol}:`, error);
      return [];
    }
  }

  async getTopTradersLongShortRatio(symbol: string, period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d' = '1h'): Promise<Array<{
    symbol: string;
    longShortRatio: number;
    longAccount: number;
    shortAccount: number;
    timestamp: number;
  }>> {
    try {
      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.futuresUrl}/futures/data/topLongShortAccountRatio`, {
        symbol: formattedSymbol,
        period: period,
        limit: 30
      });

      if (!Array.isArray(response)) {
        return [];
      }

      return response.map(item => ({
        symbol: formattedSymbol,
        longShortRatio: parseFloat(item.longShortRatio),
        longAccount: parseFloat(item.longAccount),
        shortAccount: parseFloat(item.shortAccount),
        timestamp: item.timestamp
      }));

    } catch (error) {
      logger.error(`Error fetching top traders long/short ratio for ${symbol}:`, error);
      return [];
    }
  }

  async getTradingVolume(symbol: string, period: '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '6h' | '12h' | '1d' = '1h'): Promise<Array<{
    symbol: string;
    buySellRatio: number;
    buyVol: number;
    sellVol: number;
    timestamp: number;
  }>> {
    try {
      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.futuresUrl}/futures/data/takerlongshortRatio`, {
        symbol: formattedSymbol,
        period: period,
        limit: 30
      });

      if (!Array.isArray(response)) {
        return [];
      }

      return response.map(item => ({
        symbol: formattedSymbol,
        buySellRatio: parseFloat(item.buySellRatio),
        buyVol: parseFloat(item.buyVol),
        sellVol: parseFloat(item.sellVol),
        timestamp: item.timestamp
      }));

    } catch (error) {
      logger.error(`Error fetching trading volume for ${symbol}:`, error);
      return [];
    }
  }

  async getOrderBookDepth(symbol: string, limit: 5 | 10 | 20 | 50 | 100 | 500 | 1000 = 20): Promise<{
    lastUpdateId: number;
    bids: Array<[string, string]>;
    asks: Array<[string, string]>;
  } | null> {
    try {
      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.baseUrl}/api/v3/depth`, {
        symbol: formattedSymbol,
        limit
      });

      if (!response) {
        return null;
      }

      return {
        lastUpdateId: response.lastUpdateId,
        bids: response.bids,
        asks: response.asks
      };

    } catch (error) {
      logger.error(`Error fetching order book depth for ${symbol}:`, error);
      return null;
    }
  }

  async getRecentTrades(symbol: string, limit: number = 500): Promise<Array<{
    id: number;
    price: string;
    qty: string;
    quoteQty: string;
    time: number;
    isBuyerMaker: boolean;
  }>> {
    try {
      const formattedSymbol = this.formatSymbol(symbol);
      const response = await this.makeRequest(`${this.baseUrl}/api/v3/trades`, {
        symbol: formattedSymbol,
        limit
      });

      if (!Array.isArray(response)) {
        return [];
      }

      return response.map(trade => ({
        id: trade.id,
        price: trade.price,
        qty: trade.qty,
        quoteQty: trade.quoteQty,
        time: trade.time,
        isBuyerMaker: trade.isBuyerMaker
      }));

    } catch (error) {
      logger.error(`Error fetching recent trades for ${symbol}:`, error);
      return [];
    }
  }

  async getExchangeInfo(): Promise<{
    timezone: string;
    serverTime: number;
    symbols: Array<{
      symbol: string;
      status: string;
      baseAsset: string;
      quoteAsset: string;
      baseAssetPrecision: number;
      quotePrecision: number;
      orderTypes: string[];
      filters: any[];
    }>;
  } | null> {
    try {
      const response = await this.makeRequest(`${this.baseUrl}/api/v3/exchangeInfo`);
      
      if (!response) {
        return null;
      }

      return {
        timezone: response.timezone,
        serverTime: response.serverTime,
        symbols: response.symbols.map((symbol: any) => ({
          symbol: symbol.symbol,
          status: symbol.status,
          baseAsset: symbol.baseAsset,
          quoteAsset: symbol.quoteAsset,
          baseAssetPrecision: symbol.baseAssetPrecision,
          quotePrecision: symbol.quotePrecision,
          orderTypes: symbol.orderTypes,
          filters: symbol.filters
        }))
      };

    } catch (error) {
      logger.error('Error fetching exchange info:', error);
      return null;
    }
  }

  private async makeRequest(url: string, params: Record<string, any> = {}): Promise<any> {
    const urlObj = new URL(url);
    
    Object.entries(params).forEach(([key, value]) => {
      urlObj.searchParams.set(key, value.toString());
    });

    try {
      const response = await fetch(urlObj.toString(), {
        method: 'GET',
        headers: {
          'User-Agent': 'TradingBot/1.0'
        },
        timeout: 10000
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      return data;

    } catch (error) {
      logger.error(`Binance API request failed: ${urlObj.toString()}`, error);
      throw error;
    }
  }

  private formatSymbol(symbol: string): string {
    // Ensure symbol is in correct format for Binance (e.g., BTCUSDT)
    const cleaned = symbol.toUpperCase().replace(/[-_\/]/g, '');
    
    // If it doesn't end with USDT and doesn't already have a quote currency, add USDT
    if (!cleaned.includes('USDT') && !cleaned.includes('BUSD') && !cleaned.includes('BTC') && !cleaned.includes('ETH')) {
      return cleaned + 'USDT';
    }
    
    return cleaned;
  }

  // Utility methods for analysis

  async getVWAP(symbol: string, interval: string = '1h', periods: number = 24): Promise<number> {
    try {
      const klines = await this.getKlines(symbol, interval as any, periods);
      
      if (klines.length === 0) {
        return 0;
      }

      let totalVolume = 0;
      let totalPriceVolume = 0;

      for (const kline of klines) {
        const volume = parseFloat(kline.volume);
        const avgPrice = (parseFloat(kline.high) + parseFloat(kline.low) + parseFloat(kline.close)) / 3;
        
        totalVolume += volume;
        totalPriceVolume += avgPrice * volume;
      }

      return totalVolume > 0 ? totalPriceVolume / totalVolume : 0;

    } catch (error) {
      logger.error(`Error calculating VWAP for ${symbol}:`, error);
      return 0;
    }
  }

  async getVolatility(symbol: string, periods: number = 20): Promise<number> {
    try {
      const klines = await this.getKlines(symbol, '1d', periods);
      
      if (klines.length < 2) {
        return 0;
      }

      const returns = [];
      for (let i = 1; i < klines.length; i++) {
        const prevClose = parseFloat(klines[i - 1].close);
        const currentClose = parseFloat(klines[i].close);
        const return_ = Math.log(currentClose / prevClose);
        returns.push(return_);
      }

      // Calculate standard deviation
      const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
      const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
      const volatility = Math.sqrt(variance) * Math.sqrt(365); // Annualized

      return volatility;

    } catch (error) {
      logger.error(`Error calculating volatility for ${symbol}:`, error);
      return 0;
    }
  }
}