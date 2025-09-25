import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { config } from 'dotenv';

// Load environment variables
config();

import { TradingController } from './src/controllers/TradingController';
import { BacktestingController } from './src/controllers/BacktestingController';
import { MarketDataController } from './src/controllers/MarketDataController';

class App {
  private app: express.Application;
  private server: any;
  private tradingController: TradingController;
  private backtestingController: BacktestingController;
  private marketDataController: MarketDataController;

  constructor() {
    this.app = express();
    this.tradingController = new TradingController();
    this.backtestingController = new BacktestingController();
    this.marketDataController = new MarketDataController();

    this.configureMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  private configureMiddleware(): void {
    // CORS configuration
    this.app.use(cors({
      origin: process.env.FRONTEND_URL || 'http://localhost:5173',
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }));

    // Body parsing middleware
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
      next();
    });
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        success: true,
        message: 'Trading Bot API is running',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    });

    // API routes
    this.setupApiRoutes();

    // Serve static files from frontend build
    if (process.env.NODE_ENV === 'production') {
      this.app.use(express.static('../dist'));
    }
  }

  private setupApiRoutes(): void {
    const apiRouter = express.Router();

    // Trading routes
    apiRouter.get('/trading/status', this.tradingController.getStatus.bind(this.tradingController));
    apiRouter.post('/trading/start', this.tradingController.startTrading.bind(this.tradingController));
    apiRouter.post('/trading/stop', this.tradingController.stopTrading.bind(this.tradingController));
    apiRouter.get('/trading/signals', this.tradingController.getSignals.bind(this.tradingController));
    apiRouter.post('/trading/orders', this.tradingController.placeOrder.bind(this.tradingController));
    apiRouter.get('/trading/orders', this.tradingController.getOrders.bind(this.tradingController));
    apiRouter.delete('/trading/orders/:orderId', this.tradingController.cancelOrder.bind(this.tradingController));
    apiRouter.get('/trading/positions', this.tradingController.getPositions.bind(this.tradingController));
    apiRouter.post('/trading/positions/:symbol/close', this.tradingController.closePosition.bind(this.tradingController));
    apiRouter.put('/trading/settings', this.tradingController.updateSettings.bind(this.tradingController));
    apiRouter.get('/trading/performance', this.tradingController.getPerformance.bind(this.tradingController));

    // Backtesting routes
    apiRouter.post('/backtesting/run', this.backtestingController.runBacktest.bind(this.backtestingController));
    apiRouter.get('/backtesting/results', this.backtestingController.getResults.bind(this.backtestingController));

    // Market data routes (proxy endpoints)
    apiRouter.get('/market/prices', this.marketDataController.getPrices.bind(this.marketDataController));
    apiRouter.get('/market/watchlist', this.marketDataController.getWatchlist.bind(this.marketDataController));
    apiRouter.put('/market/watchlist', this.marketDataController.updateWatchlist.bind(this.marketDataController));

    // External API proxies
    apiRouter.get('/proxy/coingecko/*', this.marketDataController.proxyCoinGecko.bind(this.marketDataController));
    apiRouter.get('/proxy/binance/*', this.marketDataController.proxyBinance.bind(this.marketDataController));

    this.app.use('/api/v1', apiRouter);
  }

  private setupErrorHandling(): void {
    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        error: {
          code: 'NOT_FOUND',
          message: `Route ${req.originalUrl} not found`
        }
      });
    });

    // Global error handler
    this.app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
      console.error('Global error handler:', err);

      res.status(err.statusCode || 500).json({
        success: false,
        error: {
          code: err.code || 'INTERNAL_ERROR',
          message: err.message || 'An unexpected error occurred'
        }
      });
    });
  }

  public async start(): Promise<void> {
    const port = process.env.PORT || 3000;

    try {
      this.server = this.app.listen(port, () => {
        console.log(`🚀 Trading Bot API server running on port ${port}`);
        console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
        console.log(`🌐 Frontend URL: ${process.env.FRONTEND_URL || 'http://localhost:5173'}`);
      });
    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }

  public async stop(): Promise<void> {
    if (this.server) {
      this.server.close();
      console.log('🛑 Server stopped');
    }
  }

  public getApp(): express.Application {
    return this.app;
  }
}

// Export for testing
export { App };

// Start server if this file is run directly
if (require.main === module) {
  const app = new App();
  app.start();
}
