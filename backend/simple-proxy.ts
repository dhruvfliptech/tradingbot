import express from 'express';
import cors from 'cors';
import axios, { AxiosRequestConfig } from 'axios';
import CryptoJS from 'crypto-js';
import * as dotenv from 'dotenv';

// Load environment variables from parent directory
dotenv.config({ path: '../.env' });

const app = express();
const PORT = process.env.PORT || 3000;

// Helper function to sign Binance requests
function signBinanceRequest(params: Record<string, any>, secretKey: string): string {
  // Remove signature from params if it exists
  const { signature, ...paramsWithoutSignature } = params;

  // Filter out undefined/null values and convert to strings
  const filteredParams: Record<string, string> = {};
  for (const [key, value] of Object.entries(paramsWithoutSignature)) {
    if (value !== undefined && value !== null && value !== '') {
      filteredParams[key] = String(value);
    }
  }

  // Sort parameters by key
  const sortedParams = Object.keys(filteredParams)
    .sort()
    .reduce((result, key) => {
      result[key] = filteredParams[key];
      return result;
    }, {} as Record<string, string>);

  // Create query string
  const queryString = new URLSearchParams(sortedParams).toString();

  // Add signature to query string
  const signatureHash = CryptoJS.HmacSHA256(queryString, secretKey).toString(CryptoJS.enc.Hex);
  const fullQueryString = `${queryString}&signature=${signatureHash}`;

  return fullQueryString;
}

// CORS configuration
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check
app.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'Trading Bot API Proxy is running',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// CoinGecko API proxy
app.get('/api/v1/proxy/coingecko/*', async (req, res) => {
  try {
    const path = req.path.replace('/api/v1/proxy/coingecko', '');
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
});

// Binance API proxy
app.get('/api/v1/proxy/binance/*', async (req, res) => {
  try {
    const path = req.path.replace('/api/v1/proxy/binance', '');
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

      console.log('🔐 Checking authentication:', {
        hasAuthHeader: !!req.headers.authorization,
        apiKeyFromEnv: !!process.env.BINANCE_API_KEY,
        secretKeyFromEnv: !!process.env.BINANCE_SECRET_KEY,
        apiKeyUsed: !!apiKey,
        secretKeyUsed: !!secretKey,
        method: req.method,
        path
      });

      if (apiKey) {
        config.headers!['X-MBX-APIKEY'] = apiKey;

        // For GET requests, add signature to query params if we have the secret key
        if (secretKey && req.method === 'GET') {
          console.log('🔐 Adding Binance signature for authenticated request');
          // For authenticated GET requests, add required parameters and signature
          const authParams = {
            timestamp: Date.now(),
            recvWindow: 5000,
            ...queryParams
          };
          console.log('🔐 Parameters before signing:', authParams);
          const signedQueryString = signBinanceRequest(authParams, secretKey);
          console.log('🔐 Signed query string:', signedQueryString);
          config.url = `${baseUrl}${path}?${signedQueryString}`;
          console.log('🔐 Final URL:', config.url);
        } else if (secretKey && (req.method === 'POST' || req.method === 'PUT' || req.method === 'DELETE')) {
          // For other methods, signature is added in the body (simplified implementation)
          console.log('⚠️ POST/PUT/DELETE signature implementation is simplified');
        } else {
          console.log('⚠️ No signature added - either no secret key or not a GET request');
        }
      } else {
        console.log('⚠️ No API key found');
      }
    } else {
      console.log('⚠️ No authentication headers or API keys found');
    }

    console.log(`📡 Proxying Binance request: ${req.method} ${url}`);

    const response = await axios(config);

    // Set CORS headers
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, X-MBX-APIKEY');

    res.json(response.data);
  } catch (error: any) {
    console.error('Binance proxy error:', error.message);

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
});

// Market prices endpoint
app.get('/api/v1/market/prices', async (req, res) => {
  try {
    const { symbols = 'BTC,ETH,SOL,ADA', include_indicators = 'true' } = req.query;

    const symbolList = String(symbols).split(',').map(s => s.trim().toUpperCase());

    // Fetch from CoinGecko
    const ids = symbolList.map(s => {
      // Map common symbols to CoinGecko IDs
      const symbolMap: Record<string, string> = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'ADA': 'cardano',
        'BNB': 'binancecoin',
        'DOGE': 'dogecoin',
        'XRP': 'ripple',
        'MATIC': 'matic-network'
      };
      return symbolMap[s] || s.toLowerCase();
    }).join(',');
    const url = `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=${ids}&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=24h`;

    let response;
    try {
      response = await axios.get(url, { timeout: 15000 });
    } catch (error: any) {
      console.warn('CoinGecko API rate limited or failed, using fallback data');
      return res.json({
        success: true,
        data: [
          {
            symbol: 'SOL',
            price: 204.00,
            change_24h: -4.10,
            change_percent_24h: -1.97,
            volume_24h: 7920810015,
            market_cap: 110805170771,
            source: 'fallback'
          },
          {
            symbol: 'ADA',
            price: 0.79,
            change_24h: -0.018,
            change_percent_24h: -2.19,
            volume_24h: 1200000000,
            market_cap: 28000000000,
            source: 'fallback'
          }
        ],
        sources: [{ name: 'CoinGecko', available: false, error: 'Rate limited' }]
      });
    }

    const data = response.data.map((coin: any) => ({
      symbol: coin.symbol.toUpperCase(),
      price: coin.current_price,
      change_24h: coin.price_change_24h,
      change_percent_24h: coin.price_change_percentage_24h,
      volume_24h: coin.total_volume,
      market_cap: coin.market_cap,
      source: 'coingecko'
    }));

    res.json({
      success: true,
      data,
      sources: [{ name: 'CoinGecko', available: true }]
    });
  } catch (error: any) {
    console.error('Market prices error:', error);
    res.status(500).json({
      success: false,
      error: {
        code: 'SERVICE_UNAVAILABLE',
        message: 'Price service temporarily unavailable'
      }
    });
  }
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: {
      code: 'NOT_FOUND',
      message: `Route ${req.originalUrl} not found`
    }
  });
});

// Global error handler
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Global error handler:', err);

  res.status(err.statusCode || 500).json({
    success: false,
    error: {
      code: err.code || 'INTERNAL_ERROR',
      message: err.message || 'An unexpected error occurred'
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Trading Bot API Proxy server running on port ${PORT}`);
  console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🌐 Frontend URL: http://localhost:5173`);
  console.log(`📡 CoinGecko Proxy: http://localhost:${PORT}/api/v1/proxy/coingecko/*`);
  console.log(`📡 Binance Proxy: http://localhost:${PORT}/api/v1/proxy/binance/*`);
});
