import express from 'express';
import cors from 'cors';
import { BacktestingController } from '../../controllers/BacktestingController';
import { TradingController } from '../../controllers/TradingController';

export async function createTestApp() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true }));

  // Mock authentication middleware for testing
  app.use((req: any, res, next) => {
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      req.user = {
        id: 'test-user-id',
        email: 'test@example.com'
      };
    }
    next();
  });

  // Controllers
  const backtestingController = new BacktestingController();
  const tradingController = new TradingController();

  // Backtesting routes
  app.post('/api/v1/backtesting/strategies', backtestingController.createStrategy.bind(backtestingController));
  app.get('/api/v1/backtesting/strategies', backtestingController.getStrategies.bind(backtestingController));
  app.get('/api/v1/backtesting/strategies/:strategyId', backtestingController.getStrategy.bind(backtestingController));
  app.post('/api/v1/backtesting/strategies/:strategyId/validate', backtestingController.validateStrategy.bind(backtestingController));
  app.post('/api/v1/backtesting/run', backtestingController.runBacktest.bind(backtestingController));
  app.get('/api/v1/backtesting/results', backtestingController.getBacktestResults.bind(backtestingController));
  app.get('/api/v1/backtesting/results/:backtestId', backtestingController.getBacktestResult.bind(backtestingController));
  app.delete('/api/v1/backtesting/results/:backtestId', backtestingController.cancelBacktest.bind(backtestingController));
  app.post('/api/v1/backtesting/optimize', backtestingController.optimizeStrategy.bind(backtestingController));
  app.get('/api/v1/backtesting/historical-data', backtestingController.getHistoricalData.bind(backtestingController));
  app.get('/api/v1/backtesting/market-regimes', backtestingController.getMarketRegimes.bind(backtestingController));
  app.post('/api/v1/backtesting/adaptive-threshold/train', backtestingController.trainAdaptiveThreshold.bind(backtestingController));
  app.get('/api/v1/backtesting/available-strategies', backtestingController.getAvailableStrategies.bind(backtestingController));

  // Trading routes
  app.get('/api/v1/trading/status', tradingController.getStatus.bind(tradingController));
  app.post('/api/v1/trading/start', tradingController.startTrading.bind(tradingController));
  app.post('/api/v1/trading/stop', tradingController.stopTrading.bind(tradingController));

  // Mock auth routes
  app.post('/api/v1/auth/register', (req, res) => {
    res.status(201).json({
      success: true,
      data: {
        user: {
          id: 'test-user-id',
          email: req.body.email,
          username: req.body.username
        },
        token: 'test-jwt-token'
      }
    });
  });

  // Error handling middleware
  app.use((error: any, req: any, res: any, next: any) => {
    console.error('Test app error:', error);
    res.status(error.status || 500).json({
      success: false,
      message: error.message || 'Internal server error',
      ...(process.env.NODE_ENV === 'test' && { stack: error.stack })
    });
  });

  // 404 handler
  app.use((req, res) => {
    res.status(404).json({
      success: false,
      message: 'Route not found'
    });
  });

  return app;
}