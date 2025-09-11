import { supabase } from '../lib/supabase';
import { Account, Position, Order } from '../types/trading';
import { coinGeckoService } from './coinGeckoService';
import { User } from '@supabase/supabase-js';

class PortfolioService {
  async initializeUserPortfolio(user: User): Promise<void> {
    // Check if user already has a portfolio
    const { data: existingPortfolios } = await supabase
      .from('portfolios')
      .select('id')
      .eq('user_id', user.id);

    if (!existingPortfolios || existingPortfolios.length === 0) {
      await this.createInitialPortfolio(user.id);
    }
  }

  private async createInitialPortfolio(userId: string): Promise<void> {
    // Create initial portfolio for new user
    const { data: portfolio, error: portfolioError } = await supabase
      .from('portfolios')
      .insert({
        user_id: userId,
        balance_usd: 10000, // Starting balance
        balance_btc: 0,
        total_trades: 0,
      })
      .select()
      .maybeSingle();

    if (portfolioError) {
      console.error('Error creating initial portfolio:', portfolioError);
      return;
    }

    if (!portfolio) return;

    // Get current crypto prices for realistic positions
    const cryptoData = await coinGeckoService.getCryptoData(['bitcoin', 'ethereum']);
    const btcData = cryptoData.find(c => c.symbol === 'BTC');
    const ethData = cryptoData.find(c => c.symbol === 'ETH');

    if (!btcData || !ethData) return;

    // Create initial demo positions
    const positions = [
      {
        portfolio_id: portfolio.id,
        symbol: 'BTC',
        name: btcData.name,
        image: btcData.image,
        quantity: 0.5,
        avg_cost_basis: btcData.price * 0.95, // Slightly below current price
        side: 'long',
      },
      {
        portfolio_id: portfolio.id,
        symbol: 'ETH',
        name: ethData.name,
        image: ethData.image,
        quantity: 2.5,
        avg_cost_basis: ethData.price * 0.92, // Slightly below current price
        side: 'long',
      },
    ];

    const { error: positionsError } = await supabase
      .from('positions')
      .insert(positions);

    if (positionsError) {
      console.error('Error creating demo positions:', positionsError);
    }

    // Create sample order history
    const orders = [
      {
        portfolio_id: portfolio.id,
        symbol: 'BTC',
        quantity: 0.1,
        side: 'buy',
        order_type: 'limit',
        status: 'filled',
        filled_quantity: 0.1,
        filled_avg_price: btcData.price * 0.99,
        limit_price: btcData.price * 0.99,
        submitted_at: new Date(Date.now() - 3600000).toISOString(),
        filled_at: new Date(Date.now() - 3500000).toISOString(),
      },
      {
        portfolio_id: portfolio.id,
        symbol: 'ETH',
        quantity: 1.0,
        side: 'buy',
        order_type: 'market',
        status: 'pending',
        filled_quantity: 0,
        filled_avg_price: 0,
        submitted_at: new Date(Date.now() - 300000).toISOString(),
      },
    ];

    const { error: ordersError } = await supabase
      .from('orders')
      .insert(orders);

    if (ordersError) {
      console.error('Error creating demo orders:', ordersError);
    }
  }

  async getAccount(user: User): Promise<Account> {
    if (!user) {
      throw new Error('User not authenticated');
    }

    // Initialize portfolio if needed
    await this.initializeUserPortfolio(user);
    const userId = user.id;

    const { data: portfolios, error } = await supabase
      .from('portfolios')
      .select('*')
      .eq('user_id', userId);

    if (error) {
      console.error('Error fetching portfolio:', error);
      throw new Error('Failed to fetch portfolio data');
    }

    if (!portfolios || portfolios.length === 0) {
      throw new Error('No portfolio found for user');
    }

    const portfolio = portfolios[0];

    // Calculate total portfolio value
    const positions = await this.getPositions();
    let portfolioValue = portfolio.balance_usd;

    for (const position of positions) {
      portfolioValue += position.market_value;
    }

    return {
      id: portfolio.id,
      balance_usd: portfolio.balance_usd,
      balance_btc: portfolio.balance_btc,
      portfolio_value: portfolioValue,
      available_balance: portfolio.balance_usd,
      total_trades: portfolio.total_trades,
    };
  }

  async getPositions(user?: User): Promise<Position[]> {
    const { data: { user: currentUser } } = await supabase.auth.getUser();
    if (!currentUser) {
      throw new Error('User not authenticated');
    }
    const userId = currentUser.id;

    // Get portfolio ID
    const { data: portfolios } = await supabase
      .from('portfolios')
      .select('id')
      .eq('user_id', userId);

    if (!portfolios || portfolios.length === 0) return [];
    const portfolio = portfolios[0];

    // Get positions
    const { data: positions, error } = await supabase
      .from('positions')
      .select('*')
      .eq('portfolio_id', portfolio.id);

    if (error) {
      console.error('Error fetching positions:', error);
      return [];
    }

    // Get current prices for all symbols
    const symbols = positions.map(p => p.symbol.toLowerCase());
    const cryptoData = await coinGeckoService.getCryptoData(symbols);

    return positions.map(position => {
      const crypto = cryptoData.find(c => c.symbol === position.symbol);
      const currentPrice = crypto?.price || 0;
      const marketValue = position.quantity * currentPrice;
      const costBasis = position.quantity * position.avg_cost_basis;
      const unrealizedPL = marketValue - costBasis;
      const unrealizedPLPC = costBasis > 0 ? unrealizedPL / costBasis : 0;

      return {
        symbol: position.symbol,
        qty: position.quantity.toString(),
        market_value: marketValue,
        cost_basis: costBasis,
        unrealized_pl: unrealizedPL,
        unrealized_plpc: unrealizedPLPC,
        side: position.side as 'long' | 'short',
        name: position.name,
        image: position.image,
      };
    });
  }

  async getOrders(user?: User): Promise<Order[]> {
    const { data: { user: currentUser } } = await supabase.auth.getUser();
    if (!currentUser) {
      throw new Error('User not authenticated');
    }
    const userId = currentUser.id;

    // Get portfolio ID
    const { data: portfolios } = await supabase
      .from('portfolios')
      .select('id')
      .eq('user_id', userId);

    if (!portfolios || portfolios.length === 0) return [];
    const portfolio = portfolios[0];

    // Get orders
    const { data: orders, error } = await supabase
      .from('orders')
      .select('*')
      .eq('portfolio_id', portfolio.id)
      .order('submitted_at', { ascending: false })
      .limit(10);

    if (error) {
      console.error('Error fetching orders:', error);
      return [];
    }

    return orders.map(order => ({
      id: order.id,
      symbol: order.symbol,
      qty: order.quantity.toString(),
      side: order.side as 'buy' | 'sell',
      order_type: order.order_type as 'market' | 'limit',
      status: order.status,
      filled_qty: order.filled_quantity.toString(),
      filled_avg_price: order.filled_avg_price,
      submitted_at: order.submitted_at,
      filled_at: order.filled_at,
      limit_price: order.limit_price,
    }));
  }

  async placeOrder(orderData: Partial<Order>, user?: User): Promise<Order> {
    const { data: { user: currentUser } } = await supabase.auth.getUser();
    if (!currentUser) {
      throw new Error('User not authenticated');
    }
    const userId = currentUser.id;

    // Get portfolio ID
    const { data: portfolios } = await supabase
      .from('portfolios')
      .select('id')
      .eq('user_id', userId);

    if (!portfolios || portfolios.length === 0) {
      throw new Error('Portfolio not found');
    }
    const portfolio = portfolios[0];

    // Get current price for the symbol
    const cryptoData = await coinGeckoService.getCryptoData([orderData.symbol?.toLowerCase() || 'bitcoin']);
    const currentPrice = cryptoData[0]?.price || 0;

    const orderToInsert = {
      portfolio_id: portfolio.id,
      symbol: orderData.symbol || '',
      quantity: parseFloat(orderData.qty || '0'),
      side: orderData.side || 'buy',
      order_type: orderData.order_type || 'market',
      status: orderData.order_type === 'market' ? 'filled' : 'pending',
      filled_quantity: orderData.order_type === 'market' ? parseFloat(orderData.qty || '0') : 0,
      filled_avg_price: orderData.order_type === 'market' ? currentPrice : 0,
      limit_price: orderData.limit_price || (orderData.order_type === 'market' ? currentPrice : null),
      submitted_at: new Date().toISOString(),
      filled_at: orderData.order_type === 'market' ? new Date().toISOString() : null,
    };

    const { data: order, error } = await supabase
      .from('orders')
      .insert(orderToInsert)
      .select()
      .single();

    if (error) {
      console.error('Error placing order:', error);
      throw new Error('Failed to place order');
    }

    // If market order, update positions and portfolio
    if (orderData.order_type === 'market') {
      await this.updatePositionFromOrder(order);
      await this.updatePortfolioFromOrder(order);
    }

    return {
      id: order.id,
      symbol: order.symbol,
      qty: order.quantity.toString(),
      side: order.side as 'buy' | 'sell',
      order_type: order.order_type as 'market' | 'limit',
      status: order.status,
      filled_qty: order.filled_quantity.toString(),
      filled_avg_price: order.filled_avg_price,
      submitted_at: order.submitted_at,
      filled_at: order.filled_at,
      limit_price: order.limit_price,
    };
  }

  private async updatePositionFromOrder(order: any): Promise<void> {
    // Get existing position
    const { data: existingPosition } = await supabase
      .from('positions')
      .select('*')
      .eq('portfolio_id', order.portfolio_id)
      .eq('symbol', order.symbol)
      .single();

    if (existingPosition) {
      // Update existing position
      const newQuantity = order.side === 'buy' 
        ? existingPosition.quantity + order.filled_quantity
        : existingPosition.quantity - order.filled_quantity;

      const newAvgCost = order.side === 'buy'
        ? ((existingPosition.quantity * existingPosition.avg_cost_basis) + (order.filled_quantity * order.filled_avg_price)) / newQuantity
        : existingPosition.avg_cost_basis;

      await supabase
        .from('positions')
        .update({
          quantity: newQuantity,
          avg_cost_basis: newAvgCost,
        })
        .eq('id', existingPosition.id);
    } else if (order.side === 'buy') {
      // Create new position for buy orders
      const cryptoData = await coinGeckoService.getCryptoData([order.symbol.toLowerCase()]);
      const crypto = cryptoData[0];

      await supabase
        .from('positions')
        .insert({
          portfolio_id: order.portfolio_id,
          symbol: order.symbol,
          name: crypto?.name || order.symbol,
          image: crypto?.image || '',
          quantity: order.filled_quantity,
          avg_cost_basis: order.filled_avg_price,
          side: 'long',
        });
    }
  }

  private async updatePortfolioFromOrder(order: any): Promise<void> {
    const orderValue = order.filled_quantity * order.filled_avg_price;
    
    if (order.side === 'buy') {
      // Decrease USD balance for buy orders
      await supabase
        .from('portfolios')
        .update({
          balance_usd: supabase.raw(`balance_usd - ${orderValue}`),
          total_trades: supabase.raw('total_trades + 1'),
        })
        .eq('id', order.portfolio_id);
    } else {
      // Increase USD balance for sell orders
      await supabase
        .from('portfolios')
        .update({
          balance_usd: supabase.raw(`balance_usd + ${orderValue}`),
          total_trades: supabase.raw('total_trades + 1'),
        })
        .eq('id', order.portfolio_id);
    }
  }
}

export const portfolioService = new PortfolioService();