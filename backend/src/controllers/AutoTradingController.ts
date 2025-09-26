/**
 * AutoTradingController - API endpoints for automated trading bot control
 * Provides REST API for frontend to control trading bot
 */

import { Request, Response } from 'express';
import logger from '../utils/logger';
import { AutoTradingService, AutoTradingConfig } from '../services/trading/AutoTradingService';
import { StateManager } from '../services/trading/StateManager';
import SocketIOServer from '../websocket/SocketIOServer';
import jwt from 'jsonwebtoken';

interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
  };
}

export class AutoTradingController {
  private tradingBots: Map<string, AutoTradingService> = new Map();
  private stateManager: StateManager;
  private socketServer: SocketIOServer | null = null;

  constructor(socketServer?: SocketIOServer) {
    this.stateManager = new StateManager();
    this.socketServer = socketServer || null;
  }

  /**
   * Start the auto-trading bot for a user
   * POST /api/v1/trading/bot/start
   */
  async startBot(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';
      const {
        watchlist = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA'],
        cycleIntervalMs = 45000,
        cooldownMinutes = 5,
        maxOpenPositions = 10,
        riskBudgetUsd = 100,
        confidenceThreshold = 0.65,
        settings = {}
      } = req.body;

      // Check if bot already exists
      let bot = this.tradingBots.get(userId);

      if (bot) {
        const status = bot.getStatus();
        if (status.isActive) {
          res.status(400).json({
            success: false,
            error: 'Trading bot is already running'
          });
          return;
        }
      } else {
        // Create new bot instance
        const config: AutoTradingConfig = {
          userId,
          settings: {
            ...settings,
            maxPositions: maxOpenPositions,
            riskPerTrade: riskBudgetUsd / 50000, // Convert to percentage
            stopLossPercent: settings.stopLossPercent || 2,
            takeProfitPercent: settings.takeProfitPercent || 5,
            emergencyCloseAll: settings.emergencyCloseAll || false,
            enabledStrategies: settings.enabledStrategies || ['momentum', 'meanReversion']
          },
          watchlist,
          cycleIntervalMs,
          cooldownMinutes,
          maxOpenPositions,
          riskBudgetUsd,
          confidenceThreshold
        };

        bot = new AutoTradingService(config);

        // Subscribe to events for real-time updates
        bot.on('tradingEvent', (event) => {
          // Forward to WebSocket for real-time updates
          this.broadcastEvent(userId, 'trading_event', event);
        });

        this.tradingBots.set(userId, bot);
      }

      // Start the bot
      await bot.start();

      // Log to state manager
      await this.stateManager.logActivity(userId, 'bot_started', {
        watchlist,
        settings
      });

      res.json({
        success: true,
        data: {
          status: 'active',
          config: bot.getStatus().config,
          message: 'Trading bot started successfully'
        }
      });

    } catch (error) {
      logger.error('Error starting trading bot:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start trading bot'
      });
    }
  }

  /**
   * Stop the auto-trading bot for a user
   * POST /api/v1/trading/bot/stop
   */
  async stopBot(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';

      const bot = this.tradingBots.get(userId);
      if (!bot) {
        res.status(404).json({
          success: false,
          error: 'Trading bot not found'
        });
        return;
      }

      await bot.stop();

      // Log to state manager
      await this.stateManager.logActivity(userId, 'bot_stopped', {
        reason: req.body.reason || 'User requested'
      });

      res.json({
        success: true,
        data: {
          status: 'stopped',
          message: 'Trading bot stopped successfully'
        }
      });

    } catch (error) {
      logger.error('Error stopping trading bot:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop trading bot'
      });
    }
  }

  /**
   * Get the trading bot status for a user
   * GET /api/v1/trading/bot/status
   */
  async getBotStatus(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';

      const bot = this.tradingBots.get(userId);
      if (!bot) {
        res.json({
          success: true,
          data: {
            status: 'not_initialized',
            isActive: false,
            message: 'Trading bot not initialized'
          }
        });
        return;
      }

      const status = bot.getStatus();

      res.json({
        success: true,
        data: {
          status: status.isActive ? 'active' : 'stopped',
          isActive: status.isActive,
          cyclesExecuted: status.cyclesExecuted,
          signalsGenerated: status.signalsGenerated,
          tradesExecuted: status.tradesExecuted,
          errors: status.errors,
          lastError: status.lastError,
          positions: status.positions.length,
          account: status.account,
          config: status.config
        }
      });

    } catch (error) {
      logger.error('Error getting bot status:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get bot status'
      });
    }
  }

  /**
   * Update bot configuration
   * PUT /api/v1/trading/bot/config
   */
  async updateBotConfig(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';

      const bot = this.tradingBots.get(userId);
      if (!bot) {
        res.status(404).json({
          success: false,
          error: 'Trading bot not found'
        });
        return;
      }

      const updates = req.body;
      bot.updateConfig(updates);

      // Log to state manager
      await this.stateManager.logActivity(userId, 'config_updated', updates);

      res.json({
        success: true,
        data: {
          message: 'Configuration updated successfully',
          config: bot.getStatus().config
        }
      });

    } catch (error) {
      logger.error('Error updating bot config:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update configuration'
      });
    }
  }

  /**
   * Emergency stop the trading bot
   * POST /api/v1/trading/bot/emergency-stop
   */
  async emergencyStop(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';

      const bot = this.tradingBots.get(userId);
      if (!bot) {
        res.status(404).json({
          success: false,
          error: 'Trading bot not found'
        });
        return;
      }

      await bot.emergencyStop();

      // Log to state manager
      await this.stateManager.logActivity(userId, 'emergency_stop', {
        reason: req.body.reason || 'Emergency stop triggered'
      });

      res.json({
        success: true,
        data: {
          status: 'emergency_stopped',
          message: 'Emergency stop executed successfully'
        }
      });

    } catch (error) {
      logger.error('Error during emergency stop:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to execute emergency stop'
      });
    }
  }

  /**
   * Place a manual order (bypasses bot logic)
   * POST /api/v1/trading/manual/order
   */
  async placeManualOrder(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';
      const {
        symbol,
        side,
        quantity,
        orderType = 'market',
        limitPrice
      } = req.body;

      // Validate input
      if (!symbol || !side || !quantity) {
        res.status(400).json({
          success: false,
          error: 'Missing required fields: symbol, side, quantity'
        });
        return;
      }

      // Use OrderExecutor directly for manual orders
      const { OrderExecutor } = await import('../services/trading/OrderExecutor');
      const orderExecutor = new OrderExecutor();

      const order = await orderExecutor.executeOrder({
        userId,
        symbol,
        side,
        quantity,
        type: orderType,
        limitPrice,
        metadata: {
          source: 'manual',
          placedBy: req.user?.email || 'unknown'
        }
      });

      // Log to state manager
      await this.stateManager.logActivity(userId, 'manual_order', {
        symbol,
        side,
        quantity,
        orderType,
        orderId: order.id
      });

      res.json({
        success: true,
        data: {
          order,
          message: 'Manual order placed successfully'
        }
      });

    } catch (error) {
      logger.error('Error placing manual order:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to place manual order'
      });
    }
  }

  /**
   * Get recent trading signals
   * GET /api/v1/trading/signals
   */
  async getSignals(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';
      const limit = parseInt(req.query.limit as string) || 10;

      // Get signals from state manager
      const signals = await this.stateManager.getRecentData(
        `signals:${userId}`,
        limit
      );

      res.json({
        success: true,
        data: {
          signals,
          count: signals.length
        }
      });

    } catch (error) {
      logger.error('Error getting signals:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get signals'
      });
    }
  }

  /**
   * Get bot performance metrics
   * GET /api/v1/trading/bot/performance
   */
  async getBotPerformance(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id || 'demo-user';
      const period = req.query.period as string || 'daily';

      // Get performance data from state manager
      const performance = await this.stateManager.getRecentData(
        `performance:${userId}:${period}`,
        1
      );

      const bot = this.tradingBots.get(userId);
      const status = bot ? bot.getStatus() : null;

      res.json({
        success: true,
        data: {
          performance: performance[0] || {
            totalReturn: 0,
            winRate: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            totalTrades: status?.tradesExecuted || 0
          },
          botMetrics: status ? {
            cyclesExecuted: status.cyclesExecuted,
            signalsGenerated: status.signalsGenerated,
            tradesExecuted: status.tradesExecuted,
            errors: status.errors
          } : null
        }
      });

    } catch (error) {
      logger.error('Error getting bot performance:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get performance'
      });
    }
  }

  /**
   * Broadcast event to WebSocket clients
   */
  private broadcastEvent(userId: string, eventName: string, data: any): void {
    if (this.socketServer) {
      // Emit to user-specific room
      this.socketServer.emitToUser(userId, eventName, {
        ...data,
        timestamp: new Date()
      });

      // Also emit specific event types to appropriate channels
      switch (data.type) {
        case 'order_submitted':
        case 'order_error':
          this.socketServer.emitToRoom(`user:${userId}:orders`, 'order_update', data);
          break;
        case 'analysis':
        case 'decision':
          this.socketServer.emitToRoom(`user:${userId}:signals`, 'signal_update', data);
          break;
        case 'status':
        case 'cycle_complete':
          this.socketServer.emitToRoom(`user:${userId}:bot`, 'bot_status', data);
          break;
        case 'emergency_stop':
        case 'error':
          this.socketServer.emitToRoom(`user:${userId}:alerts`, 'alert', {
            id: `alert_${Date.now()}`,
            userId,
            type: 'system',
            severity: data.type === 'emergency_stop' ? 'critical' : 'warning',
            title: data.type === 'emergency_stop' ? 'Emergency Stop' : 'Bot Error',
            message: data.data.message || data.data.reason,
            timestamp: new Date()
          });
          break;
      }
    }

    logger.debug(`Broadcasting event for user ${userId}:`, eventName, data.type);
  }

  /**
   * Cleanup resources on shutdown
   */
  async shutdown(): Promise<void> {
    logger.info('Shutting down auto-trading controllers...');

    // Stop all bots
    for (const [userId, bot] of this.tradingBots) {
      try {
        await bot.stop();
        logger.info(`Stopped trading bot for user ${userId}`);
      } catch (error) {
        logger.error(`Error stopping bot for user ${userId}:`, error);
      }
    }

    this.tradingBots.clear();
  }
}