import { Request, Response } from 'express';
import { BinanceBroker } from '../services/brokers/BinanceBroker';
import logger from '../utils/logger';

/**
 * Proxy controller to handle all Binance API calls from frontend
 * This ensures API keys stay on backend only
 */
export class BinanceProxyController {
  private binanceBroker: BinanceBroker;

  constructor() {
    this.binanceBroker = new BinanceBroker();
  }

  /**
   * Proxy for account information
   */
  async getAccount(req: Request, res: Response): Promise<void> {
    try {
      const account = await this.binanceBroker.getAccount();
      res.json({ success: true, data: account });
    } catch (error: any) {
      logger.error('Error fetching Binance account:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch account'
      });
    }
  }

  /**
   * Proxy for placing orders
   */
  async placeOrder(req: Request, res: Response): Promise<void> {
    try {
      const { symbol, side, quantity, price } = req.body;

      // Validate required fields
      if (!symbol || !side || !quantity) {
        res.status(400).json({
          success: false,
          error: 'Missing required fields: symbol, side, quantity'
        });
        return;
      }

      const order = await this.binanceBroker.placeOrder({
        symbol,
        side,
        quantity,
        price
      });

      res.json({ success: true, data: order });
    } catch (error: any) {
      logger.error('Error placing Binance order:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to place order'
      });
    }
  }

  /**
   * Proxy for getting orders (open, closed, or all)
   */
  async getOrders(req: Request, res: Response): Promise<void> {
    try {
      const { status } = req.query;
      const orderStatus = (status as 'all' | 'open' | 'closed') || 'all';
      const orders = await this.binanceBroker.getOrders(orderStatus);
      res.json({ success: true, data: orders });
    } catch (error: any) {
      logger.error('Error fetching orders:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch orders'
      });
    }
  }

  /**
   * Proxy for canceling orders
   */
  async cancelOrder(req: Request, res: Response): Promise<void> {
    try {
      const { orderId } = req.params;
      const result = await this.binanceBroker.cancelOrder(orderId);
      res.json({ success: true, data: result });
    } catch (error: any) {
      logger.error('Error canceling order:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to cancel order'
      });
    }
  }

  /**
   * Proxy for getting positions (balances)
   */
  async getPositions(req: Request, res: Response): Promise<void> {
    try {
      const positions = await this.binanceBroker.getPositions();
      res.json({ success: true, data: positions });
    } catch (error: any) {
      logger.error('Error fetching positions:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch positions'
      });
    }
  }

  /**
   * Proxy for market data (prices) - expects array of symbols
   */
  async getMarketData(req: Request, res: Response): Promise<void> {
    try {
      const { symbol } = req.params;

      if (!symbol) {
        res.status(400).json({
          success: false,
          error: 'Symbol is required'
        });
        return;
      }

      // BinanceBroker expects an array of symbols
      const symbols = [symbol];
      const marketData = await this.binanceBroker.getMarketData(symbols);

      // Return the first result since we only requested one symbol
      res.json({ success: true, data: marketData[0] || null });
    } catch (error: any) {
      logger.error('Error fetching market data:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch market data'
      });
    }
  }

  /**
   * Test connection to Binance
   */
  async testConnection(req: Request, res: Response): Promise<void> {
    try {
      const result = await this.binanceBroker.testConnection();
      res.json({ success: true, data: { connected: result } });
    } catch (error: any) {
      logger.error('Error testing Binance connection:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to test connection'
      });
    }
  }
}