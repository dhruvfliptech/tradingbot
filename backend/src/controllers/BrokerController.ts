/**
 * Broker Controller - API endpoints for broker operations
 * Provides secure backend access to trading brokers
 */

import { Request, Response } from 'express';
import { BrokerManager, BrokerType } from '../services/brokers/BrokerManager';
import logger from '../utils/logger';

interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
  };
}

export class BrokerController {
  private brokerManager: BrokerManager;

  constructor() {
    this.brokerManager = BrokerManager.getInstance();
  }

  /**
   * Get available brokers
   * GET /api/v1/brokers
   */
  async getBrokers(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const brokers = this.brokerManager.getAvailableBrokers();

      res.json({
        success: true,
        data: {
          brokers,
          active: this.brokerManager.getActiveBroker().getBrokerName().toLowerCase()
        }
      });
    } catch (error) {
      logger.error('Error getting brokers:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get brokers'
      });
    }
  }

  /**
   * Set active broker
   * POST /api/v1/brokers/active
   */
  async setActiveBroker(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker } = req.body;

      if (!broker || !['alpaca', 'binance'].includes(broker)) {
        res.status(400).json({
          success: false,
          error: 'Invalid broker type. Must be "alpaca" or "binance"'
        });
        return;
      }

      this.brokerManager.setActiveBroker(broker as BrokerType);

      res.json({
        success: true,
        data: {
          active: broker,
          message: `Active broker set to ${broker}`
        }
      });
    } catch (error) {
      logger.error('Error setting active broker:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to set active broker'
      });
    }
  }

  /**
   * Test broker connections
   * GET /api/v1/brokers/test
   */
  async testConnections(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const results = await this.brokerManager.testAllConnections();
      const connections: any = {};

      for (const [broker, connected] of results) {
        connections[broker] = connected;
      }

      res.json({
        success: true,
        data: {
          connections
        }
      });
    } catch (error) {
      logger.error('Error testing broker connections:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to test connections'
      });
    }
  }

  /**
   * Get account information
   * GET /api/v1/brokers/account
   */
  async getAccount(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker } = req.query;

      if (broker) {
        // Get specific broker account
        const brokerInstance = this.brokerManager.getBroker(broker as BrokerType);
        if (!brokerInstance) {
          res.status(404).json({
            success: false,
            error: `Broker ${broker} not found`
          });
          return;
        }

        const account = await brokerInstance.getAccount();
        res.json({
          success: true,
          data: { account, broker }
        });
      } else {
        // Get active broker account
        const activeBroker = this.brokerManager.getActiveBroker();
        const account = await activeBroker.getAccount();

        res.json({
          success: true,
          data: {
            account,
            broker: activeBroker.getBrokerName().toLowerCase()
          }
        });
      }
    } catch (error) {
      logger.error('Error getting account:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get account'
      });
    }
  }

  /**
   * Get aggregated accounts from all brokers
   * GET /api/v1/brokers/accounts
   */
  async getAggregatedAccounts(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const aggregated = await this.brokerManager.getAggregatedAccount();

      res.json({
        success: true,
        data: aggregated
      });
    } catch (error) {
      logger.error('Error getting aggregated accounts:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get aggregated accounts'
      });
    }
  }

  /**
   * Get positions
   * GET /api/v1/brokers/positions
   */
  async getPositions(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker } = req.query;

      const brokerInstance = broker
        ? this.brokerManager.getBroker(broker as BrokerType)
        : this.brokerManager.getActiveBroker();

      if (!brokerInstance) {
        res.status(404).json({
          success: false,
          error: `Broker ${broker} not found`
        });
        return;
      }

      const positions = await brokerInstance.getPositions();

      res.json({
        success: true,
        data: {
          positions,
          broker: brokerInstance.getBrokerName().toLowerCase()
        }
      });
    } catch (error) {
      logger.error('Error getting positions:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get positions'
      });
    }
  }

  /**
   * Get orders
   * GET /api/v1/brokers/orders
   */
  async getOrders(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker, status } = req.query;

      const brokerInstance = broker
        ? this.brokerManager.getBroker(broker as BrokerType)
        : this.brokerManager.getActiveBroker();

      if (!brokerInstance) {
        res.status(404).json({
          success: false,
          error: `Broker ${broker} not found`
        });
        return;
      }

      const orderStatus = status as 'all' | 'open' | 'closed' | undefined;
      const orders = await brokerInstance.getOrders(orderStatus);

      res.json({
        success: true,
        data: {
          orders,
          broker: brokerInstance.getBrokerName().toLowerCase()
        }
      });
    } catch (error) {
      logger.error('Error getting orders:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get orders'
      });
    }
  }

  /**
   * Place an order
   * POST /api/v1/brokers/orders
   */
  async placeOrder(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker, ...orderParams } = req.body;

      const brokerInstance = broker
        ? this.brokerManager.getBroker(broker as BrokerType)
        : this.brokerManager.getActiveBroker();

      if (!brokerInstance) {
        res.status(404).json({
          success: false,
          error: `Broker ${broker} not found`
        });
        return;
      }

      // Validate order parameters
      if (!orderParams.symbol || !orderParams.qty || !orderParams.side) {
        res.status(400).json({
          success: false,
          error: 'Missing required fields: symbol, qty, side'
        });
        return;
      }

      const order = await brokerInstance.placeOrder(orderParams);

      // Log the order
      logger.info('Order placed:', {
        user: req.user?.id || 'unknown',
        broker: brokerInstance.getBrokerName(),
        order
      });

      res.json({
        success: true,
        data: {
          order,
          broker: brokerInstance.getBrokerName().toLowerCase()
        }
      });
    } catch (error) {
      logger.error('Error placing order:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to place order'
      });
    }
  }

  /**
   * Cancel an order
   * DELETE /api/v1/brokers/orders/:orderId
   */
  async cancelOrder(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { orderId } = req.params;
      const { broker } = req.query;

      const brokerInstance = broker
        ? this.brokerManager.getBroker(broker as BrokerType)
        : this.brokerManager.getActiveBroker();

      if (!brokerInstance) {
        res.status(404).json({
          success: false,
          error: `Broker ${broker} not found`
        });
        return;
      }

      const success = await brokerInstance.cancelOrder(orderId);

      if (success) {
        res.json({
          success: true,
          data: {
            message: `Order ${orderId} canceled successfully`,
            broker: brokerInstance.getBrokerName().toLowerCase()
          }
        });
      } else {
        res.status(400).json({
          success: false,
          error: `Failed to cancel order ${orderId}`
        });
      }
    } catch (error) {
      logger.error('Error canceling order:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to cancel order'
      });
    }
  }

  /**
   * Get market data
   * GET /api/v1/brokers/market-data
   */
  async getMarketData(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { symbols, broker } = req.query;

      if (!symbols || typeof symbols !== 'string') {
        res.status(400).json({
          success: false,
          error: 'Symbols parameter is required'
        });
        return;
      }

      const brokerInstance = broker
        ? this.brokerManager.getBroker(broker as BrokerType)
        : this.brokerManager.getActiveBroker();

      if (!brokerInstance) {
        res.status(404).json({
          success: false,
          error: `Broker ${broker} not found`
        });
        return;
      }

      const symbolList = symbols.split(',').map(s => s.trim());
      const marketData = await brokerInstance.getMarketData(symbolList);

      res.json({
        success: true,
        data: {
          marketData,
          broker: brokerInstance.getBrokerName().toLowerCase()
        }
      });
    } catch (error) {
      logger.error('Error getting market data:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get market data'
      });
    }
  }

  /**
   * Initialize broker with API credentials
   * POST /api/v1/brokers/initialize
   */
  async initializeBroker(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { broker, apiKey, secretKey, baseUrl } = req.body;

      if (!broker || !apiKey || !secretKey) {
        res.status(400).json({
          success: false,
          error: 'Missing required fields: broker, apiKey, secretKey'
        });
        return;
      }

      const success = await this.brokerManager.initializeBrokerWithCredentials(
        broker as BrokerType,
        apiKey,
        secretKey,
        baseUrl
      );

      if (success) {
        res.json({
          success: true,
          data: {
            message: `Broker ${broker} initialized successfully`,
            broker
          }
        });
      } else {
        res.status(400).json({
          success: false,
          error: `Failed to initialize broker ${broker}. Check your credentials.`
        });
      }
    } catch (error) {
      logger.error('Error initializing broker:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to initialize broker'
      });
    }
  }
}