import { BrokerAdapter, BrokerOrderRequest } from '../../core/adapters';

export class NoopBrokerAdapter implements BrokerAdapter {
  async getAccount(): Promise<any> {
    return {
      portfolio_value: 0,
      balance_usd: 0,
      balance_btc: 0,
      total_trades: 0,
    };
  }

  async getPositions(): Promise<any[]> {
    return [];
  }

  async getOrders(): Promise<any[]> {
    return [];
  }

  async placeOrder(order: BrokerOrderRequest): Promise<any> {
    console.log('[NoopBrokerAdapter] placeOrder called', order);
    return {
      status: 'acked',
      id: `noop-${Date.now()}`,
      ...order,
    };
  }

  async cancelOrder(symbol: string, orderId: string): Promise<void> {
    console.log('[NoopBrokerAdapter] cancelOrder', { symbol, orderId });
  }
}
