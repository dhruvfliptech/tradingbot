import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { config } from 'dotenv';

// Load environment variables
config();

import { TradingController } from './src/controllers/TradingController';
import { BacktestingController } from './src/controllers/BacktestingController';
import { MarketDataController } from './src/controllers/MarketDataController';
import { AutoTradingController } from './src/controllers/AutoTradingController';
import { BrokerController } from './src/controllers/BrokerController';
import { DataAggregatorController } from './src/controllers/DataAggregatorController';
import { MetricsController } from './src/controllers/MetricsController';
import { MonitoringController } from './src/controllers/MonitoringController';
import { BinanceProxyController } from './src/controllers/BinanceProxyController';
import { SocketIOServer } from './src/websocket/SocketIOServer';
import { TradingEngineService } from './src/services/trading/TradingEngineService';
import { DataAggregatorService } from './src/services/data-aggregation/DataAggregatorService';
import { PerformanceMetricsService } from './src/services/metrics/PerformanceMetricsService';
import monitoringService from './src/services/monitoring/UnifiedMonitoringService';

// Middleware imports
import { authenticate, optionalAuth, requireTier, restrictInDemo } from './src/middleware/auth';
import {
  apiLimiter,
  tradingLimiter,
  orderLimiter,
  botControlLimiter,
  marketDataLimiter
} from './src/middleware/rateLimiter';
import {
  sanitizeInput,
  validateStartBot,
  validateStopBot,
  validateUpdateConfig,
  validatePlaceOrder,
  validateSetActiveBroker,
  validateInitializeBroker,
  validateGetMarketData,
  validateSymbolWhitelist
} from './src/middleware/validation';
import { trackApiMetrics, trackTradingMetrics } from './src/middleware/monitoring';

import logger from './src/utils/logger';

class App {
  private app: express.Application;
  private server: any;
  private socketServer: SocketIOServer | null = null;
  private tradingController: TradingController;
  private backtestingController: BacktestingController;
  private marketDataController: MarketDataController;
  private autoTradingController: AutoTradingController;
  private brokerController: BrokerController;
  private dataAggregatorController: DataAggregatorController;
  private metricsController: MetricsController;
  private monitoringController: MonitoringController;
  private binanceProxyController: BinanceProxyController;
  private tradingEngine: TradingEngineService;
  private dataAggregator: DataAggregatorService | null = null;
  private performanceMetrics: PerformanceMetricsService | null = null;

  constructor() {
    this.app = express();
    this.server = createServer(this.app);

    // Controllers will be initialized after services
    this.tradingController = new TradingController();
    this.backtestingController = new BacktestingController();
    this.marketDataController = new MarketDataController();
    // AutoTradingController will be initialized in initializeServices with SocketIO
    this.autoTradingController = null as any; // Temporary placeholder
    this.brokerController = new BrokerController();
    this.dataAggregatorController = new DataAggregatorController();
    this.metricsController = new MetricsController();
    this.monitoringController = new MonitoringController();
    this.binanceProxyController = new BinanceProxyController();

    // Initialize trading engine
    this.tradingEngine = TradingEngineService.getInstance();

    this.configureMiddleware();
    // Routes will be set up after services are initialized
    // Error handling must be set up AFTER routes
  }

  private async initializeServices(): Promise<void> {
    try {
      // Initialize Monitoring Service first
      await monitoringService.initialize();
      logger.info('Monitoring Service initialized successfully');

      // Initialize Trading Engine
      await this.tradingEngine.initialize();
      logger.info('Trading Engine initialized successfully');

      // Initialize Socket.IO Server
      this.socketServer = new SocketIOServer(this.server);
      logger.info('Socket.IO Server initialized successfully');

      // Now initialize AutoTradingController with SocketIO
      this.autoTradingController = new AutoTradingController(this.socketServer);
      logger.info('AutoTradingController initialized with Socket.IO support');

      // Initialize Data Aggregator Service
      try {
        const dataAggConfig = {
          apis: {
            etherscan: {
              apiKey: process.env.ETHERSCAN_API_KEY || '',
              endpoint: 'https://api.etherscan.io/api',
              rateLimit: 5
            },
            bitquery: {
              apiKey: process.env.BITQUERY_API_KEY,
              endpoint: 'https://graphql.bitquery.io',
              rateLimit: 10
            },
            covalent: {
              apiKey: process.env.COVALENT_API_KEY || '',
              endpoint: 'https://api.covalenthq.com/v1',
              rateLimit: 5
            },
            coinglass: {
              endpoint: 'https://open-api.coinglass.com',
              rateLimit: 10
            },
            binance: {
              endpoint: 'https://api.binance.com',
              rateLimit: 20
            },
            cryptoquant: {
              apiKey: process.env.CRYPTOQUANT_API_KEY,
              endpoint: 'https://api.cryptoquant.com',
              rateLimit: 5
            }
          },
          cache: {
            ttl: {
              onchain: 300,
              funding: 60,
              whale: 120,
              market: 30
            },
            maxSize: 1000
          },
          fallback: {
            maxRetries: 3,
            retryDelay: 1000,
            timeout: 10000
          }
        };

        this.dataAggregator = new DataAggregatorService(dataAggConfig);
        await this.dataAggregator.initialize();
        logger.info('DataAggregatorService initialized successfully');

        // Wire up controller
        this.dataAggregatorController.setDataAggregator(this.dataAggregator);
      } catch (error) {
        logger.warn('DataAggregatorService initialization failed (non-critical):', error);
      }

      // Initialize Performance Metrics Service
      try {
        this.performanceMetrics = new PerformanceMetricsService();
        await this.performanceMetrics.initialize();
        logger.info('PerformanceMetricsService initialized successfully');

        // Wire up controller
        this.metricsController.setMetricsService(this.performanceMetrics);

        // Connect to trading engine for performance tracking
        if (this.tradingEngine) {
          this.tradingEngine.on('trade_executed', (trade) => {
            this.performanceMetrics?.recordTrade(trade);
          });

          this.tradingEngine.on('position_closed', (position) => {
            this.performanceMetrics?.recordPosition(position);
          });
        }
      } catch (error) {
        logger.warn('PerformanceMetricsService initialization failed (non-critical):', error);
      }

      // Now setup routes after services are initialized
      this.setupRoutes();
      // Setup error handling AFTER routes
      this.setupErrorHandling();

    } catch (error) {
      logger.error('Failed to initialize services:', error);
      // Don't exit process, allow the server to start but log the error
    }
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
        version: '1.0.0',
        services: {
          tradingEngine: 'initialized',
          socketIO: this.socketServer ? 'running' : 'not started',
          redis: process.env.REDIS_HOST ? 'configured' : 'not configured'
        }
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

    // Apply general middleware
    apiRouter.use(sanitizeInput);
    apiRouter.use(trackApiMetrics);
    apiRouter.use(apiLimiter);
    apiRouter.use(optionalAuth);

    // Auto-Trading Bot routes (protected)
    apiRouter.post('/trading/bot/start',
      authenticate,
      botControlLimiter,
      validateStartBot,
      this.autoTradingController.startBot.bind(this.autoTradingController)
    );

    apiRouter.post('/trading/bot/stop',
      authenticate,
      botControlLimiter,
      validateStopBot,
      this.autoTradingController.stopBot.bind(this.autoTradingController)
    );

    apiRouter.get('/trading/bot/status',
      authenticate,
      this.autoTradingController.getBotStatus.bind(this.autoTradingController)
    );

    apiRouter.put('/trading/bot/config',
      authenticate,
      botControlLimiter,
      validateUpdateConfig,
      this.autoTradingController.updateBotConfig.bind(this.autoTradingController)
    );

    apiRouter.post('/trading/bot/emergency-stop',
      authenticate,
      this.autoTradingController.emergencyStop.bind(this.autoTradingController)
    );

    apiRouter.get('/trading/bot/performance',
      authenticate,
      this.autoTradingController.getBotPerformance.bind(this.autoTradingController)
    );

    apiRouter.post('/trading/manual/order',
      authenticate,
      orderLimiter,
      validatePlaceOrder,
      validateSymbolWhitelist,
      restrictInDemo,
      this.autoTradingController.placeManualOrder.bind(this.autoTradingController)
    );

    // Broker routes (protected)
    apiRouter.get('/brokers',
      authenticate,
      this.brokerController.getBrokers.bind(this.brokerController)
    );

    apiRouter.post('/brokers/active',
      authenticate,
      validateSetActiveBroker,
      this.brokerController.setActiveBroker.bind(this.brokerController)
    );

    apiRouter.get('/brokers/test',
      authenticate,
      this.brokerController.testConnections.bind(this.brokerController)
    );

    apiRouter.get('/brokers/account',
      authenticate,
      this.brokerController.getAccount.bind(this.brokerController)
    );

    apiRouter.get('/brokers/accounts',
      authenticate,
      this.brokerController.getAggregatedAccounts.bind(this.brokerController)
    );

    apiRouter.get('/brokers/positions',
      authenticate,
      tradingLimiter,
      this.brokerController.getPositions.bind(this.brokerController)
    );

    apiRouter.get('/brokers/orders',
      authenticate,
      tradingLimiter,
      this.brokerController.getOrders.bind(this.brokerController)
    );

    apiRouter.post('/brokers/orders',
      authenticate,
      orderLimiter,
      validatePlaceOrder,
      validateSymbolWhitelist,
      restrictInDemo,
      this.brokerController.placeOrder.bind(this.brokerController)
    );

    apiRouter.delete('/brokers/orders/:orderId',
      authenticate,
      orderLimiter,
      this.brokerController.cancelOrder.bind(this.brokerController)
    );

    apiRouter.get('/brokers/market-data',
      marketDataLimiter,
      validateGetMarketData,
      this.brokerController.getMarketData.bind(this.brokerController)
    );

    apiRouter.post('/brokers/initialize',
      authenticate,
      requireTier('premium'),
      validateInitializeBroker,
      this.brokerController.initializeBroker.bind(this.brokerController)
    );

    // Binance Proxy Routes (Secure - API keys on backend only)
    apiRouter.get('/binance/account',
      authenticate,
      marketDataLimiter,
      this.binanceProxyController.getAccount.bind(this.binanceProxyController)
    );

    apiRouter.post('/binance/order',
      authenticate,
      orderLimiter,
      this.binanceProxyController.placeOrder.bind(this.binanceProxyController)
    );

    apiRouter.get('/binance/orders',
      authenticate,
      marketDataLimiter,
      this.binanceProxyController.getOrders.bind(this.binanceProxyController)
    );

    apiRouter.delete('/binance/orders/:orderId',
      authenticate,
      orderLimiter,
      this.binanceProxyController.cancelOrder.bind(this.binanceProxyController)
    );

    apiRouter.get('/binance/positions',
      authenticate,
      marketDataLimiter,
      this.binanceProxyController.getPositions.bind(this.binanceProxyController)
    );

    apiRouter.get('/binance/market/:symbol',
      marketDataLimiter,
      this.binanceProxyController.getMarketData.bind(this.binanceProxyController)
    );

    apiRouter.get('/binance/test',
      marketDataLimiter,
      this.binanceProxyController.testConnection.bind(this.binanceProxyController)
    );

    // Trading routes (existing)
    apiRouter.get('/trading/status', this.tradingController.getStatus.bind(this.tradingController));
    apiRouter.post('/trading/start', this.tradingController.startTrading.bind(this.tradingController));
    apiRouter.post('/trading/stop', this.tradingController.stopTrading.bind(this.tradingController));
    apiRouter.post('/trading/pause', this.tradingController.pauseTrading.bind(this.tradingController));
    apiRouter.post('/trading/resume', this.tradingController.resumeTrading.bind(this.tradingController));
    apiRouter.get('/trading/signals', this.autoTradingController.getSignals.bind(this.autoTradingController));
    apiRouter.post('/trading/orders', this.tradingController.placeOrder.bind(this.tradingController));
    apiRouter.get('/trading/orders', this.tradingController.getOrders.bind(this.tradingController));
    apiRouter.delete('/trading/orders/:orderId', this.tradingController.cancelOrder.bind(this.tradingController));
    apiRouter.get('/trading/positions', this.tradingController.getPositions.bind(this.tradingController));
    apiRouter.post('/trading/positions/:positionId/close', this.tradingController.closePosition.bind(this.tradingController));
    apiRouter.put('/trading/settings', this.tradingController.updateSettings.bind(this.tradingController));
    apiRouter.get('/trading/performance', this.tradingController.getPerformance.bind(this.tradingController));
    apiRouter.post('/trading/emergency-stop', this.tradingController.emergencyStop.bind(this.tradingController));
    apiRouter.get('/trading/risk-metrics', this.tradingController.getRiskMetrics.bind(this.tradingController));
    apiRouter.get('/trading/market-data/:symbol', this.tradingController.getMarketData.bind(this.tradingController));
    apiRouter.get('/trading/health', this.tradingController.healthCheck.bind(this.tradingController));

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

    // Data Aggregator routes
    apiRouter.get('/data/aggregated-market',
      authenticate,
      marketDataLimiter,
      this.dataAggregatorController.getAggregatedMarketData.bind(this.dataAggregatorController)
    );

    apiRouter.get('/data/funding-rates',
      marketDataLimiter,
      this.dataAggregatorController.getFundingRates.bind(this.dataAggregatorController)
    );

    apiRouter.get('/data/whale-alerts',
      authenticate,
      marketDataLimiter,
      this.dataAggregatorController.getWhaleAlerts.bind(this.dataAggregatorController)
    );

    apiRouter.get('/data/on-chain',
      authenticate,
      requireTier('premium'),
      marketDataLimiter,
      this.dataAggregatorController.getOnChainData.bind(this.dataAggregatorController)
    );

    apiRouter.get('/data/sentiment',
      authenticate,
      marketDataLimiter,
      this.dataAggregatorController.getSentimentAnalysis.bind(this.dataAggregatorController)
    );

    // Performance Metrics routes
    apiRouter.get('/metrics/performance',
      authenticate,
      this.metricsController.getPerformanceMetrics.bind(this.metricsController)
    );

    apiRouter.get('/metrics/sharpe-ratio',
      authenticate,
      this.metricsController.getSharpeRatio.bind(this.metricsController)
    );

    apiRouter.get('/metrics/drawdown',
      authenticate,
      this.metricsController.getDrawdownAnalysis.bind(this.metricsController)
    );

    apiRouter.get('/metrics/pnl',
      authenticate,
      this.metricsController.getPnLAnalysis.bind(this.metricsController)
    );

    apiRouter.get('/metrics/win-rate',
      authenticate,
      this.metricsController.getWinRateAnalysis.bind(this.metricsController)
    );

    apiRouter.get('/metrics/risk',
      authenticate,
      this.metricsController.getRiskMetrics.bind(this.metricsController)
    );

    // Monitoring routes
    apiRouter.get('/monitoring/dashboard',
      authenticate,
      this.monitoringController.getDashboard.bind(this.monitoringController)
    );

    apiRouter.get('/monitoring/metrics',
      authenticate,
      this.monitoringController.getMetrics.bind(this.monitoringController)
    );

    apiRouter.get('/monitoring/alerts',
      authenticate,
      this.monitoringController.getAlerts.bind(this.monitoringController)
    );

    apiRouter.get('/monitoring/health',
      this.monitoringController.getHealth.bind(this.monitoringController)
    );

    apiRouter.post('/monitoring/test-alert',
      authenticate,
      requireTier('premium'),
      this.monitoringController.createTestAlert.bind(this.monitoringController)
    );

    this.app.use('/api/v1', apiRouter);
  }

  private setupErrorHandling(): void {
    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.path
      });
    });

    // Error handler
    this.app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
      const status = err.status || 500;
      const message = err.message || 'Internal server error';

      logger.error('API Error:', {
        status,
        message,
        path: req.path,
        method: req.method,
        error: err
      });

      res.status(status).json({
        success: false,
        error: message,
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
      });
    });
  }

  public async start(port: number = 3001): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(port, () => {
        console.log(`\n🚀 Trading Bot Backend is running on port ${port}`);
        console.log(`📡 Socket.IO Server is running on ws://localhost:${port}`);
        console.log(`🔗 API available at http://localhost:${port}/api/v1`);
        console.log(`📊 Health check at http://localhost:${port}/health\n`);

        if (process.env.REDIS_HOST) {
          console.log(`🗄️  Redis connected at ${process.env.REDIS_HOST}:${process.env.REDIS_PORT || 6379}`);
        } else {
          console.log('⚠️  Redis not configured - using in-memory state (not recommended for production)');
        }

        console.log('\nEnvironment:', process.env.NODE_ENV || 'development');
        console.log('Frontend URL:', process.env.FRONTEND_URL || 'http://localhost:5173');

        resolve();
      });
    });
  }

  public async stop(): Promise<void> {
    logger.info('Shutting down server...');

    // Stop trading engine
    await this.tradingEngine.emergencyStopAll();

    // Shutdown Socket.IO server
    if (this.socketServer) {
      await this.socketServer.shutdown();
    }

    // Close HTTP server
    return new Promise((resolve) => {
      this.server.close(() => {
        logger.info('Server shutdown complete');
        resolve();
      });
    });
  }
}

// Start the application
const app = new App();
const port = parseInt(process.env.PORT || '3001', 10);

app.start(port).catch((error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Received SIGINT, shutting down gracefully...');
  await app.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
  await app.stop();
  process.exit(0);
});

export default app;