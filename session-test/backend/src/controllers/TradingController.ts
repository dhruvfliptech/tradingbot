import { Request, Response, NextFunction } from 'express';
import { TradingService } from '../services/trading/TradingService';
import { OrderManager } from '../services/trading/OrderManager';
import { PositionManager } from '../services/trading/PositionManager';

export class TradingController {
  private tradingService: TradingService;
  private orderManager: OrderManager;
  private positionManager: PositionManager;

  constructor() {
    this.tradingService = new TradingService();
    this.orderManager = new OrderManager();
    this.positionManager = new PositionManager();
  }

  /**
   * GET /api/v1/trading/status
   * Get current trading bot status
   */
  async getStatus(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const status = await this.tradingService.getStatus(userId);
      
      res.json({
        success: true,
        data: status
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/start
   * Start the trading bot for a user
   */
  async startTrading(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { watchlist, settings } = req.body;
      
      await this.tradingService.start(userId, watchlist, settings);
      
      res.json({
        success: true,
        message: 'Trading bot started successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/stop
   * Stop the trading bot for a user
   */
  async stopTrading(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      await this.tradingService.stop(userId);
      
      res.json({
        success: true,
        message: 'Trading bot stopped successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/signals
   * Get current trading signals
   */
  async getSignals(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { limit = 10, symbol } = req.query;
      
      const signals = await this.tradingService.getSignals(userId, {
        limit: Number(limit),
        symbol: symbol as string
      });
      
      res.json({
        success: true,
        data: signals
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/orders
   * Place a manual trading order
   */
  async placeOrder(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const orderData = req.body;
      
      const order = await this.orderManager.placeOrder(userId, orderData);
      
      res.json({
        success: true,
        data: order
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/orders
   * Get order history
   */
  async getOrders(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { status, symbol, limit = 50, offset = 0 } = req.query;
      
      const orders = await this.orderManager.getOrders(userId, {
        status: status as string,
        symbol: symbol as string,
        limit: Number(limit),
        offset: Number(offset)
      });
      
      res.json({
        success: true,
        data: orders
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * DELETE /api/v1/trading/orders/:orderId
   * Cancel a pending order
   */
  async cancelOrder(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { orderId } = req.params;
      
      await this.orderManager.cancelOrder(userId, orderId);
      
      res.json({
        success: true,
        message: 'Order cancelled successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/positions
   * Get current positions
   */
  async getPositions(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const positions = await this.positionManager.getPositions(userId);
      
      res.json({
        success: true,
        data: positions
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/v1/trading/positions/:symbol/close
   * Close a specific position
   */
  async closePosition(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { symbol } = req.params;
      const { percentage = 100 } = req.body;
      
      const result = await this.positionManager.closePosition(userId, symbol, percentage);
      
      res.json({
        success: true,
        data: result
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * PUT /api/v1/trading/settings
   * Update trading settings
   */
  async updateSettings(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const settings = req.body;
      
      await this.tradingService.updateSettings(userId, settings);
      
      res.json({
        success: true,
        message: 'Trading settings updated successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/v1/trading/performance
   * Get trading performance metrics
   */
  async getPerformance(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { period = '24h' } = req.query;
      
      const performance = await this.tradingService.getPerformance(userId, period as string);
      
      res.json({
        success: true,
        data: performance
      });
    } catch (error) {
      next(error);
    }
  }
}