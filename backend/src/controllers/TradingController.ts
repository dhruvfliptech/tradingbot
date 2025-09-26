import { Request, Response, NextFunction } from 'express';
import { TradingEngineService } from '../services/trading/TradingEngineService';
import logger from '../utils/logger';

// Extend Request type to include user
interface AuthRequest extends Request {
  user?: {
    id: string;
    email?: string;
  };
}

export class TradingController {
  private tradingEngine: TradingEngineService;

  constructor() {
    this.tradingEngine = TradingEngineService.getInstance();

    // Initialize the trading engine
    this.initializeTradingEngine();
  }

  private async initializeTradingEngine(): Promise<void> {
    try {
      await this.tradingEngine.initialize();
      logger.info('Trading Engine initialized in controller');
    } catch (error) {
      logger.error('Failed to initialize trading engine:', error);
    }
  }

  /**
   * GET /api/v1/trading/status
   * Get current trading session status
   */
  async getStatus(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const session = await this.tradingEngine.getSession(userId);

      res.json({
        success: true,
        data: session ? {
          active: session.status === 'active',
          status: session.status,
          startedAt: session.startedAt,
          lastActivity: session.lastActivity,
          totalPnL: session.totalPnL,
          weeklyPnL: session.currentWeekPnL,
          activeTrades: session.activeTrades,
          settings: session.settings
        } : {
          active: false,
          status: 'stopped'
        }
      });
    } catch (error) {
      logger.error('Error getting trading status:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/start
   * Start the trading bot for a user
   */
  async startTrading(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { settings } = req.body;

      // Validate settings
      if (!settings?.watchlist || settings.watchlist.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'Watchlist is required'
        });
      }

      const session = await this.tradingEngine.startSession(userId, settings);

      res.json({
        success: true,
        message: 'Trading bot started successfully',
        data: {
          sessionId: session.id,
          status: session.status
        }
      });
    } catch (error) {
      logger.error('Error starting trading:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/stop
   * Stop the trading bot for a user
   */
  async stopTrading(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      await this.tradingEngine.stopSession(userId);

      res.json({
        success: true,
        message: 'Trading bot stopped successfully'
      });
    } catch (error) {
      logger.error('Error stopping trading:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/pause
   * Pause the trading bot for a user
   */
  async pauseTrading(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      await this.tradingEngine.pauseSession(userId);

      res.json({
        success: true,
        message: 'Trading bot paused successfully'
      });
    } catch (error) {
      logger.error('Error pausing trading:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/resume
   * Resume the trading bot for a user
   */
  async resumeTrading(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      await this.tradingEngine.resumeSession(userId);

      res.json({
        success: true,
        message: 'Trading bot resumed successfully'
      });
    } catch (error) {
      logger.error('Error resuming trading:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/signals
   * Get current trading signals (for informational purposes)
   */
  async getSignals(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      // This would fetch recent signals from the database or cache
      // For now, return empty array as signals are processed internally
      res.json({
        success: true,
        data: [],
        message: 'Signals are processed internally by the trading engine'
      });
    } catch (error) {
      logger.error('Error getting signals:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/orders
   * Place a manual trading order (optional feature)
   */
  async placeOrder(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { symbol, side, type, quantity, price } = req.body;

      // Validate order data
      if (!symbol || !side || !type || !quantity) {
        return res.status(400).json({
          success: false,
          error: 'Invalid order data'
        });
      }

      // This would integrate with OrderExecutor directly for manual orders
      // For now, return error as manual trading might not be desired
      res.status(501).json({
        success: false,
        error: 'Manual order placement not implemented. Trading is automated.'
      });
    } catch (error) {
      logger.error('Error placing order:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/orders
   * Get order history
   */
  async getOrders(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { status, limit = 50, offset = 0 } = req.query;

      const orders = await this.tradingEngine.getOrders(userId, status as string);

      // Apply pagination
      const startIndex = Number(offset);
      const endIndex = startIndex + Number(limit);
      const paginatedOrders = orders.slice(startIndex, endIndex);

      res.json({
        success: true,
        data: paginatedOrders,
        pagination: {
          total: orders.length,
          limit: Number(limit),
          offset: Number(offset)
        }
      });
    } catch (error) {
      logger.error('Error getting orders:', error);
      next(error);
    }
  }

  /**
   * DELETE /api/v1/trading/orders/:orderId
   * Cancel a pending order
   */
  async cancelOrder(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { orderId } = req.params;

      // This would integrate with OrderExecutor to cancel the order
      res.status(501).json({
        success: false,
        error: 'Order cancellation not implemented for automated trading'
      });
    } catch (error) {
      logger.error('Error cancelling order:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/positions
   * Get current positions
   */
  async getPositions(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const positions = await this.tradingEngine.getPositions(userId);

      res.json({
        success: true,
        data: positions
      });
    } catch (error) {
      logger.error('Error getting positions:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/positions/:positionId/close
   * Close a specific position
   */
  async closePosition(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { positionId } = req.params;
      const { reason = 'Manual close' } = req.body;

      // This would integrate with PositionManager to close the position
      res.status(501).json({
        success: false,
        error: 'Manual position closing not implemented for automated trading'
      });
    } catch (error) {
      logger.error('Error closing position:', error);
      next(error);
    }
  }

  /**
   * PUT /api/v1/trading/settings
   * Update trading settings
   */
  async updateSettings(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const settings = req.body;

      // Validate settings
      if (!settings || typeof settings !== 'object') {
        return res.status(400).json({
          success: false,
          error: 'Invalid settings data'
        });
      }

      await this.tradingEngine.updateSettings(userId, settings);

      res.json({
        success: true,
        message: 'Trading settings updated successfully'
      });
    } catch (error) {
      logger.error('Error updating settings:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/performance
   * Get trading performance metrics
   */
  async getPerformance(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { period = '24h' } = req.query;

      const performance = await this.tradingEngine.getPerformanceMetrics(userId, period as string);

      res.json({
        success: true,
        data: performance
      });
    } catch (error) {
      logger.error('Error getting performance:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/emergency-stop
   * Emergency stop all trading activities
   */
  async emergencyStop(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';
      const { stopAll = false } = req.body;

      if (stopAll) {
        // Stop all users' trading (admin only)
        await this.tradingEngine.emergencyStopAll();
        res.json({
          success: true,
          message: 'All trading activities stopped'
        });
      } else {
        // Stop specific user's trading
        await this.tradingEngine.stopSession(userId);
        res.json({
          success: true,
          message: 'Trading stopped for user'
        });
      }
    } catch (error) {
      logger.error('Error in emergency stop:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/risk-metrics
   * Get current risk metrics
   */
  async getRiskMetrics(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id || 'demo';

      // This would integrate with RiskManager to get metrics
      // For now, return basic metrics from session
      const session = await this.tradingEngine.getSession(userId);

      res.json({
        success: true,
        data: {
          maxDrawdown: session?.settings?.maxDrawdown || 0.15,
          currentDrawdown: 0, // Would be calculated
          exposure: 0, // Would be calculated
          openPositions: session?.activeTrades || 0,
          dailyPnL: 0, // Would be calculated
          weeklyPnL: session?.currentWeekPnL || 0
        }
      });
    } catch (error) {
      logger.error('Error getting risk metrics:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/market-data/:symbol
   * Get market data for a symbol
   */
  async getMarketData(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      const { symbol } = req.params;

      // This would integrate with MarketDataService
      res.status(501).json({
        success: false,
        error: 'Market data endpoint not implemented. Use WebSocket for real-time data.'
      });
    } catch (error) {
      logger.error('Error getting market data:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/health
   * Health check endpoint
   */
  async healthCheck(req: Request, res: Response, next: NextFunction) {
    try {
      res.json({
        success: true,
        data: {
          status: 'healthy',
          service: 'trading-engine',
          timestamp: new Date()
        }
      });
    } catch (error) {
      logger.error('Error in health check:', error);
      next(error);
    }
  }
}