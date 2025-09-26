import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { Order, ExecutedOrder } from '../../types/trading';
import { StateManager } from './StateManager';
import { BrokerManager } from '../brokers/BrokerManager';
import { PlaceOrderParams } from '../brokers/BrokerService';

export class OrderExecutor extends EventEmitter {
  private stateManager: StateManager;
  private brokerManager: BrokerManager;
  private pendingOrders: Map<string, Order> = new Map();
  private orderHistory: Map<string, Order[]> = new Map();
  private maxRetries: number = 3;
  private retryDelay: number = 1000; // 1 second

  constructor() {
    super();
    this.stateManager = new StateManager();
    this.brokerManager = BrokerManager.getInstance();
  }

  async initialize(): Promise<void> {
    try {
      // Test broker connections
      await this.brokerManager.testAllConnections();

      // Restore pending orders from state
      await this.restorePendingOrders();

      logger.info('Order Executor initialized');
    } catch (error) {
      logger.error('Failed to initialize Order Executor:', error);
      throw error;
    }
  }

  /**
   * Get account information from active broker
   */
  async getAccount(userId: string): Promise<any> {
    try {
      const broker = this.brokerManager.getActiveBroker();
      const account = await broker.getAccount();
      return account;
    } catch (error) {
      logger.error(`Failed to get account for user ${userId}:`, error);
      throw error;
    }
  }

  async executeOrder(order: Order): Promise<ExecutedOrder> {
    const startTime = Date.now();

    try {
      // Validate order
      this.validateOrder(order);

      // Store pending order
      this.pendingOrders.set(order.id, order);
      await this.stateManager.savePendingOrder(order);

      // Execute with retry logic
      const executedOrder = await this.executeWithRetry(order);

      // Calculate execution time
      const executionTime = Date.now() - startTime;
      logger.info(`Order ${order.id} executed in ${executionTime}ms`);

      // Update order status
      executedOrder.status = 'filled';
      executedOrder.filledAt = new Date();
      executedOrder.executionTime = executionTime;

      // Remove from pending
      this.pendingOrders.delete(order.id);
      await this.stateManager.removePendingOrder(order.id);

      // Store in history
      this.addToHistory(order.userId, executedOrder);
      await this.stateManager.saveExecutedOrder(executedOrder);

      // Emit success event
      this.emit('orderFilled', executedOrder);

      return executedOrder;

    } catch (error: any) {
      const executionTime = Date.now() - startTime;
      logger.error(`Order ${order.id} failed after ${executionTime}ms:`, error);

      // Update order status
      order.status = 'failed';
      order.error = error.message;

      // Remove from pending
      this.pendingOrders.delete(order.id);
      await this.stateManager.removePendingOrder(order.id);

      // Store failed order
      this.addToHistory(order.userId, order);

      // Emit failure event
      this.emit('orderFailed', order, error);

      throw error;
    }
  }

  private async executeWithRetry(order: Order, attempt: number = 1): Promise<ExecutedOrder> {
    try {
      // Get active broker
      const broker = this.brokerManager.getActiveBroker();

      // Prepare order parameters
      const orderParams: PlaceOrderParams = {
        symbol: order.symbol,
        qty: order.quantity,
        side: order.side as 'buy' | 'sell',
        order_type: order.type as 'market' | 'limit',
        limit_price: order.limitPrice,
        time_in_force: (order.timeInForce as 'day' | 'gtc' | 'ioc' | 'fok') || 'gtc',
        client_order_id: order.id
      };

      // Execute order through broker
      const brokerOrder = await broker.placeOrder(orderParams);

      // Convert broker order to ExecutedOrder
      return {
        ...order,
        id: brokerOrder.id,
        status: brokerOrder.status === 'filled' ? 'filled' : 'pending',
        price: brokerOrder.filled_avg_price || order.limitPrice || 0,
        filledQuantity: Number(brokerOrder.filled_qty) || 0,
        filledAt: brokerOrder.filled_at ? new Date(brokerOrder.filled_at) : undefined,
        commission: 0, // Broker doesn't provide commission info
        commissionAsset: 'USD'
      } as ExecutedOrder;

    } catch (error: any) {
      if (attempt < this.maxRetries) {
        logger.warn(`Order ${order.id} attempt ${attempt} failed, retrying...`);
        await this.delay(this.retryDelay * attempt);
        return this.executeWithRetry(order, attempt + 1);
      }
      throw error;
    }
  }

  private prepareBrokerOrder(order: Order): any {
    // Convert internal order format to broker API format
    return {
      symbol: this.formatSymbol(order.symbol),
      side: order.side.toUpperCase(),
      type: order.type.toUpperCase(),
      quantity: order.quantity,
      price: order.limitPrice,
      timeInForce: order.timeInForce || 'GTC',
      reduceOnly: order.reduceOnly || false,
      postOnly: order.postOnly || false,
      clientOrderId: order.id,
    };
  }

  private formatSymbol(symbol: string): string {
    // Convert symbol format (e.g., 'bitcoin' -> 'BTCUSDT')
    const symbolMap: Record<string, string> = {
      'bitcoin': 'BTCUSDT',
      'ethereum': 'ETHUSDT',
      'binancecoin': 'BNBUSDT',
      'cardano': 'ADAUSDT',
      'solana': 'SOLUSDT',
      'polkadot': 'DOTUSDT',
      'avalanche': 'AVAXUSDT',
      'chainlink': 'LINKUSDT',
      'polygon': 'MATICUSDT',
      'uniswap': 'UNIUSDT',
    };

    return symbolMap[symbol.toLowerCase()] || symbol.toUpperCase();
  }

  async cancelOrder(orderId: string): Promise<void> {
    try {
      const order = this.pendingOrders.get(orderId);
      if (!order) {
        throw new Error(`Order ${orderId} not found`);
      }

      // Get active broker
      const broker = this.brokerManager.getActiveBroker();

      // Cancel order through broker
      const success = await broker.cancelOrder(orderId);

      if (success) {
        // Update order status
        order.status = 'cancelled';
        order.cancelledAt = new Date();

        // Remove from pending
        this.pendingOrders.delete(orderId);
        await this.stateManager.removePendingOrder(orderId);

        // Store in history
        this.addToHistory(order.userId, order);

        // Emit event
        this.emit('orderCancelled', order);

        logger.info(`Order ${orderId} cancelled successfully`);
      } else {
        throw new Error(response.data.error || 'Order cancellation failed');
      }

    } catch (error) {
      logger.error(`Failed to cancel order ${orderId}:`, error);
      throw error;
    }
  }

  async getOrderStatus(orderId: string): Promise<Order | null> {
    // Check pending orders
    const pendingOrder = this.pendingOrders.get(orderId);
    if (pendingOrder) return pendingOrder;

    // Check executed orders in state
    return await this.stateManager.getOrder(orderId);
  }

  async getPendingOrders(userId: string): Promise<Order[]> {
    const userOrders: Order[] = [];
    for (const [_, order] of this.pendingOrders) {
      if (order.userId === userId) {
        userOrders.push(order);
      }
    }
    return userOrders;
  }

  async getOrders(userId: string, status?: string): Promise<Order[]> {
    if (status === 'pending') {
      return this.getPendingOrders(userId);
    }

    const history = this.orderHistory.get(userId) || [];
    if (status) {
      return history.filter(order => order.status === status);
    }
    return history;
  }

  async getExecutionMetrics(): Promise<any> {
    const metrics = {
      pendingOrders: this.pendingOrders.size,
      averageExecutionTime: 0,
      successRate: 0,
      totalVolume: 0,
    };

    // Calculate metrics from order history
    let totalExecutionTime = 0;
    let successCount = 0;
    let totalCount = 0;

    for (const [_, orders] of this.orderHistory) {
      for (const order of orders) {
        totalCount++;
        if (order.status === 'filled') {
          successCount++;
          if ((order as ExecutedOrder).executionTime) {
            totalExecutionTime += (order as ExecutedOrder).executionTime;
          }
        }
        if ((order as ExecutedOrder).filledQuantity) {
          metrics.totalVolume += (order as ExecutedOrder).filledQuantity * ((order as ExecutedOrder).price || 0);
        }
      }
    }

    if (successCount > 0) {
      metrics.averageExecutionTime = totalExecutionTime / successCount;
    }
    if (totalCount > 0) {
      metrics.successRate = (successCount / totalCount) * 100;
    }

    return metrics;
  }

  private validateOrder(order: Order): void {
    if (!order.userId) throw new Error('Order userId is required');
    if (!order.symbol) throw new Error('Order symbol is required');
    if (!order.side) throw new Error('Order side is required');
    if (!order.quantity || order.quantity <= 0) throw new Error('Order quantity must be positive');
    if (!order.type) throw new Error('Order type is required');

    if (order.type === 'limit' && !order.limitPrice) {
      throw new Error('Limit price is required for limit orders');
    }
  }

  private addToHistory(userId: string, order: Order): void {
    if (!this.orderHistory.has(userId)) {
      this.orderHistory.set(userId, []);
    }
    const history = this.orderHistory.get(userId)!;
    history.push(order);

    // Keep only last 100 orders per user in memory
    if (history.length > 100) {
      history.shift();
    }
  }

  private async restorePendingOrders(): Promise<void> {
    try {
      const pendingOrders = await this.stateManager.getPendingOrders();
      for (const order of pendingOrders) {
        this.pendingOrders.set(order.id, order);
      }
      logger.info(`Restored ${pendingOrders.length} pending orders`);
    } catch (error) {
      logger.error('Failed to restore pending orders:', error);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async cleanup(): Promise<void> {
    // Cancel all pending orders
    for (const [orderId] of this.pendingOrders) {
      try {
        await this.cancelOrder(orderId);
      } catch (error) {
        logger.error(`Failed to cancel order ${orderId} during cleanup:`, error);
      }
    }

    this.pendingOrders.clear();
    this.orderHistory.clear();
  }
}

export default OrderExecutor;