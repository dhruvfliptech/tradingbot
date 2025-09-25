import { Request, Response, NextFunction } from 'express';
import axios, { AxiosRequestConfig } from 'axios';

export class MarketDataController {
  // CoinGecko API proxy
  async proxyCoinGecko(req: Request, res: Response, next: NextFunction) {
    try {
      const path = req.path.replace('/proxy/coingecko', '');
      const baseUrl = 'https://api.coingecko.com/api/v3';
      const url = `${baseUrl}${path}`;

      // Get query parameters
      const queryParams = req.query;
      const config: AxiosRequestConfig = {
        method: req.method as any,
        url,
        params: queryParams,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'TradingBot/1.0',
        },
        timeout: 30000,
      };

      // Add API key if available
      const apiKey = process.env.COINGECKO_API_KEY;
      if (apiKey) {
        const usePro = process.env.USE_COINGECKO_PRO === 'true';
        if (usePro) {
          config.headers!['x-cg-pro-api-key'] = apiKey;
        }
      }

      console.log(`📡 Proxying CoinGecko request: ${req.method} ${url}`);

      const response = await axios(config);

      // Set CORS headers
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

      res.json(response.data);
    } catch (error: any) {
      console.error('CoinGecko proxy error:', error.message);

      // Handle specific error cases
      if (error.response) {
        res.status(error.response.status).json({
          success: false,
          error: {
            code: 'COINGECKO_ERROR',
            message: error.response.data?.error || error.message
          }
        });
      } else {
        res.status(500).json({
          success: false,
          error: {
            code: 'PROXY_ERROR',
            message: 'Failed to proxy request to CoinGecko'
          }
        });
      }
    }
  }

  // Binance API proxy
  async proxyBinance(req: Request, res: Response, next: NextFunction) {
    try {
      const path = req.path.replace('/proxy/binance', '');
      const baseUrl = 'https://api.binance.com';
      const url = `${baseUrl}${path}`;

      // Get query parameters
      const queryParams = req.query;
      const config: AxiosRequestConfig = {
        method: req.method as any,
        url,
        params: queryParams,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'TradingBot/1.0',
        },
        timeout: 30000,
      };

      // Add API key for authenticated requests
      if (req.headers.authorization || process.env.BINANCE_API_KEY) {
        const apiKey = req.headers.authorization?.replace('Bearer ', '') || process.env.BINANCE_API_KEY;
        const secretKey = process.env.BINANCE_SECRET_KEY;

        if (apiKey) {
          config.headers!['X-MBX-APIKEY'] = apiKey;

          // If this is a signed request and we have the secret key, add signature
          if (secretKey && req.method !== 'GET' && req.method !== 'DELETE') {
            // Note: For simplicity, we're not implementing full signature logic here
            // In a production environment, you'd want to properly sign requests
            console.log('⚠️ Binance signed requests not fully implemented in proxy');
          }
        }
      }

      console.log(`📡 Proxying Binance request: ${req.method} ${url}`);

      const response = await axios(config);

      // Set CORS headers
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, X-MBX-APIKEY');

      res.json(response.data);
    } catch (error: any) {
      console.error('Binance proxy error:', error.message);

      // Handle specific error cases
      if (error.response) {
        res.status(error.response.status).json({
          success: false,
          error: {
            code: 'BINANCE_ERROR',
            message: error.response.data?.msg || error.message
          }
        });
      } else {
        res.status(500).json({
          success: false,
          error: {
            code: 'PROXY_ERROR',
            message: 'Failed to proxy request to Binance'
          }
        });
      }
    }
  }

  // Get market prices (main endpoint)
  async getPrices(req: Request, res: Response, next: NextFunction) {
    try {
      const { symbols = 'BTC,ETH,SOL,ADA', include_indicators = 'true' } = req.query;

      // Parse symbols
      const symbolList = String(symbols).split(',').map(s => s.trim().toUpperCase());

      // Fetch from multiple sources for redundancy
      const promises = [
        this.fetchFromCoinGecko(symbolList),
        this.fetchFromBinance(symbolList)
      ];

      const results = await Promise.allSettled(promises);
      const validResults = results
        .filter((result): result is PromiseFulfilledResult<any> => result.status === 'fulfilled')
        .map(result => result.value)
        .filter(Boolean);

      if (validResults.length === 0) {
        return res.status(503).json({
          success: false,
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'All price sources are currently unavailable'
          }
        });
      }

      // Merge results from different sources
      const mergedData = this.mergePriceData(validResults);

      res.json({
        success: true,
        data: mergedData,
        sources: validResults.map((r, i) => ({
          name: i === 0 ? 'CoinGecko' : 'Binance',
          available: true
        }))
      });
    } catch (error: any) {
      console.error('Market prices error:', error);
      next(error);
    }
  }

  // Get user's watchlist with live data
  async getWatchlist(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: {
            code: 'UNAUTHORIZED',
            message: 'User authentication required'
          }
        });
      }

      // For now, return default watchlist
      // TODO: Implement actual user watchlist from database
      const defaultSymbols = ['BTC', 'ETH', 'SOL', 'ADA', 'BNB'];
      const prices = await this.getPrices({ query: { symbols: defaultSymbols.join(',') } } as Request, res, next);

      res.json({
        success: true,
        data: {
          symbols: defaultSymbols,
          prices: prices // This will be handled by the response
        }
      });
    } catch (error: any) {
      console.error('Watchlist error:', error);
      next(error);
    }
  }

  // Update user's watchlist
  async updateWatchlist(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: {
            code: 'UNAUTHORIZED',
            message: 'User authentication required'
          }
        });
      }

      const { symbols } = req.body;

      if (!Array.isArray(symbols) || symbols.length === 0) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_REQUEST',
            message: 'Symbols array is required'
          }
        });
      }

      // TODO: Save watchlist to database
      console.log(`📝 Updating watchlist for user ${userId}:`, symbols);

      res.json({
        success: true,
        message: 'Watchlist updated successfully',
        data: { symbols }
      });
    } catch (error: any) {
      console.error('Update watchlist error:', error);
      next(error);
    }
  }

  // Helper methods
  private async fetchFromCoinGecko(symbols: string[]): Promise<any[]> {
    try {
      const ids = symbols.map(s => s.toLowerCase()).join(',');
      const url = `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=${ids}&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=24h`;

      const response = await axios.get(url, { timeout: 15000 });

      return response.data.map((coin: any) => ({
        symbol: coin.symbol.toUpperCase(),
        price: coin.current_price,
        change_24h: coin.price_change_24h,
        change_percent_24h: coin.price_change_percentage_24h,
        volume_24h: coin.total_volume,
        market_cap: coin.market_cap,
        source: 'coingecko'
      }));
    } catch (error) {
      console.warn('CoinGecko fetch failed:', error);
      return [];
    }
  }

  private async fetchFromBinance(symbols: string[]): Promise<any[]> {
    try {
      const binanceSymbols = symbols.map(s => `${s}USDT`);
      const url = 'https://api.binance.com/api/v3/ticker/24hr';

      const response = await axios.get(url, {
        params: { symbols: JSON.stringify(binanceSymbols) },
        timeout: 15000
      });

      return response.data.map((ticker: any) => ({
        symbol: ticker.symbol.replace('USDT', ''),
        price: parseFloat(ticker.lastPrice),
        change_24h: parseFloat(ticker.priceChange),
        change_percent_24h: parseFloat(ticker.priceChangePercent),
        volume_24h: parseFloat(ticker.volume),
        source: 'binance'
      }));
    } catch (error) {
      console.warn('Binance fetch failed:', error);
      return [];
    }
  }

  private mergePriceData(dataArrays: any[][]): any[] {
    const merged = new Map();

    for (const dataArray of dataArrays) {
      for (const item of dataArray) {
        const existing = merged.get(item.symbol);
        if (existing) {
          // Average the prices if we have multiple sources
          existing.price = (existing.price + item.price) / 2;
          existing.change_24h = (existing.change_24h + item.change_24h) / 2;
          existing.change_percent_24h = (existing.change_percent_24h + item.change_percent_24h) / 2;
        } else {
          merged.set(item.symbol, { ...item });
        }
      }
    }

    return Array.from(merged.values());
  }

  // Simple in-memory cache for price data
  private priceCache = new Map();
  private cacheExpiry = new Map();

  private getCachedPrice(symbol: string): any | null {
    const cached = this.priceCache.get(symbol);
    const expiry = this.cacheExpiry.get(symbol);

    if (cached && expiry && Date.now() < expiry) {
      return cached;
    }

    return null;
  }

  private setCachedPrice(symbol: string, data: any): void {
    this.priceCache.set(symbol, data);
    // Cache for 30 seconds
    this.cacheExpiry.set(symbol, Date.now() + 30000);
  }
}
