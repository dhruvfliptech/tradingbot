import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { Position, Order, ExecutedOrder } from '../../types/trading';
import { StateManager } from './StateManager';
import { BrokerManager } from '../brokers/BrokerManager';

export class PositionManager extends EventEmitter {
  private stateManager: StateManager;
  private brokerManager: BrokerManager;
  private positions: Map<string, Position[]> = new Map(); // userId -> positions
  private positionById: Map<string, Position> = new Map(); // positionId -> position

  constructor() {
    super();
    this.stateManager = new StateManager();
    this.brokerManager = BrokerManager.getInstance();
  }

  async initialize(): Promise<void> {
    try {
      // Restore open positions from state
      await this.restorePositions();

      // Start position monitoring
      this.startPositionMonitoring();

      logger.info('Position Manager initialized');
    } catch (error) {
      logger.error('Failed to initialize Position Manager:', error);
      throw error;
    }
  }

  async openPosition(order: ExecutedOrder): Promise<Position> {
    try {
      const position: Position = {
        id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        userId: order.userId,
        symbol: order.symbol,
        side: order.side === 'buy' ? 'long' : 'short',
        entryPrice: order.price!,
        quantity: order.filledQuantity!,
        currentPrice: order.price!,
        unrealizedPnL: 0,
        realizedPnL: 0,
        status: 'open',
        openedAt: new Date(),
        stopLoss: this.calculateStopLoss(order.price!, order.side),
        takeProfit: this.calculateTakeProfit(order.price!, order.side),
        trailingStop: null,
        highWaterMark: order.price!,
        commission: order.commission || 0,
        orderId: order.id,
      };

      // Store position
      this.addPosition(position);
      await this.stateManager.savePosition(position);

      // Emit event
      this.emit('positionOpened', position);

      logger.info(`Position opened: ${position.id} for user ${position.userId}`);
      return position;

    } catch (error) {
      logger.error('Failed to open position:', error);
      throw error;
    }
  }

  async closePosition(positionId: string, closePrice: number): Promise<Position> {
    try {
      const position = this.positionById.get(positionId);
      if (!position) {
        throw new Error(`Position ${positionId} not found`);
      }

      // Calculate final PnL
      const pnl = this.calculatePnL(position, closePrice);

      // Update position
      position.status = 'closed';
      position.closedAt = new Date();
      position.closePrice = closePrice;
      position.realizedPnL = pnl;
      position.unrealizedPnL = 0;

      // Update state
      await this.stateManager.savePosition(position);

      // Remove from active positions
      this.removePosition(position);

      // Emit event
      this.emit('positionClosed', position);

      logger.info(`Position closed: ${positionId}, PnL: ${pnl.toFixed(2)}`);
      return position;

    } catch (error) {
      logger.error(`Failed to close position ${positionId}:`, error);
      throw error;
    }
  }

  async updatePosition(positionId: string, updates: Partial<Position>): Promise<Position> {
    const position = this.positionById.get(positionId);
    if (!position) {
      throw new Error(`Position ${positionId} not found`);
    }

    // Update position fields
    Object.assign(position, updates);

    // Update unrealized PnL if current price changed
    if (updates.currentPrice) {
      position.unrealizedPnL = this.calculatePnL(position, updates.currentPrice);

      // Update high water mark for trailing stop
      if (position.side === 'long' && updates.currentPrice > position.highWaterMark!) {
        position.highWaterMark = updates.currentPrice;
      } else if (position.side === 'short' && updates.currentPrice < position.highWaterMark!) {
        position.highWaterMark = updates.currentPrice;
      }

      // Check trailing stop
      if (position.trailingStop) {
        this.updateTrailingStop(position);
      }
    }

    // Save to state
    await this.stateManager.savePosition(position);

    return position;
  }

  async getPositions(userId: string): Promise<Position[]> {
    // First try to get positions from broker for real-time data
    try {
      const broker = this.brokerManager.getActiveBroker();
      const brokerPositions = await broker.getPositions();

      // Convert broker positions to internal Position format
      const positions: Position[] = brokerPositions.map(bp => ({
        id: `pos_${bp.symbol}_${Date.now()}`,
        userId,
        symbol: bp.symbol,
        side: bp.side,
        entryPrice: bp.cost_basis / Number(bp.qty),
        quantity: Number(bp.qty),
        currentPrice: bp.market_value / Number(bp.qty),
        unrealizedPnL: bp.unrealized_pl,
        realizedPnL: 0,
        status: 'open' as const,
        openedAt: new Date(),
        stopLoss: null,
        takeProfit: null,
        trailingStop: null,
        highWaterMark: bp.market_value / Number(bp.qty),
        commission: 0,
        orderId: ''
      }));

      // Update cache
      this.positions.set(userId, positions);
      return positions;
    } catch (error) {
      logger.warn('Failed to get positions from broker, using cache:', error);
      return this.positions.get(userId) || [];
    }
  }

  async getPosition(positionId: string): Promise<Position | null> {
    return this.positionById.get(positionId) || null;
  }

  async getOpenPositions(userId: string): Promise<Position[]> {
    const positions = await this.getPositions(userId);
    return positions.filter(p => p.status === 'open');
  }

  async calculateTotalPnL(userId: string): Promise<number> {
    const positions = await this.getPositions(userId);
    let totalPnL = 0;

    for (const position of positions) {
      if (position.status === 'open') {
        totalPnL += position.unrealizedPnL;
      } else {
        totalPnL += position.realizedPnL;
      }
    }

    return totalPnL;
  }

  async calculateWeeklyPnL(userId: string): Promise<number> {
    const positions = await this.getPositions(userId);
    const weekStart = new Date();
    weekStart.setDate(weekStart.getDate() - 7);

    let weeklyPnL = 0;
    for (const position of positions) {
      if (position.openedAt >= weekStart) {
        if (position.status === 'open') {
          weeklyPnL += position.unrealizedPnL;
        } else {
          weeklyPnL += position.realizedPnL;
        }
      }
    }

    return weeklyPnL;
  }

  async getPerformanceMetrics(userId: string, period: string = '24h'): Promise<any> {
    const positions = await this.getPositions(userId);
    const now = new Date();
    let periodStart: Date;

    switch (period) {
      case '24h':
        periodStart = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        break;
      case '7d':
        periodStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case '30d':
        periodStart = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      default:
        periodStart = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    }

    const periodPositions = positions.filter(p => p.openedAt >= periodStart);

    let totalPnL = 0;
    let winCount = 0;
    let lossCount = 0;
    let totalWinAmount = 0;
    let totalLossAmount = 0;
    let openPositions = 0;
    let closedPositions = 0;

    for (const position of periodPositions) {
      const pnl = position.status === 'open' ? position.unrealizedPnL : position.realizedPnL;
      totalPnL += pnl;

      if (position.status === 'open') {
        openPositions++;
      } else {
        closedPositions++;
        if (pnl > 0) {
          winCount++;
          totalWinAmount += pnl;
        } else if (pnl < 0) {
          lossCount++;
          totalLossAmount += Math.abs(pnl);
        }
      }
    }

    const totalTrades = winCount + lossCount;
    const winRate = totalTrades > 0 ? (winCount / totalTrades) * 100 : 0;
    const avgWin = winCount > 0 ? totalWinAmount / winCount : 0;
    const avgLoss = lossCount > 0 ? totalLossAmount / lossCount : 0;
    const profitFactor = totalLossAmount > 0 ? totalWinAmount / totalLossAmount : 0;

    // Calculate Sharpe ratio (simplified)
    const returns = periodPositions.map(p =>
      p.status === 'open' ? p.unrealizedPnL : p.realizedPnL
    );
    const sharpeRatio = this.calculateSharpeRatio(returns);

    // Calculate max drawdown
    const maxDrawdown = this.calculateMaxDrawdown(periodPositions);

    return {
      period,
      totalPnL: totalPnL.toFixed(2),
      totalPnLPercent: 0, // Will be calculated with portfolio value
      winRate: winRate.toFixed(2),
      totalTrades,
      winningTrades: winCount,
      losingTrades: lossCount,
      avgWin: avgWin.toFixed(2),
      avgLoss: avgLoss.toFixed(2),
      profitFactor: profitFactor.toFixed(2),
      sharpeRatio: sharpeRatio.toFixed(2),
      maxDrawdown: maxDrawdown.toFixed(2),
      openPositions,
      closedPositions,
      bestTrade: this.getBestTrade(periodPositions),
      worstTrade: this.getWorstTrade(periodPositions),
    };
  }

  private calculatePnL(position: Position, currentPrice: number): number {
    const quantity = position.quantity;
    const entryPrice = position.entryPrice;

    let pnl: number;
    if (position.side === 'long') {
      pnl = (currentPrice - entryPrice) * quantity;
    } else {
      pnl = (entryPrice - currentPrice) * quantity;
    }

    // Subtract commission
    pnl -= position.commission || 0;

    return pnl;
  }

  private calculateStopLoss(entryPrice: number, side: 'buy' | 'sell', percentage: number = 0.05): number {
    if (side === 'buy') {
      return entryPrice * (1 - percentage);
    } else {
      return entryPrice * (1 + percentage);
    }
  }

  private calculateTakeProfit(entryPrice: number, side: 'buy' | 'sell', percentage: number = 0.15): number {
    if (side === 'buy') {
      return entryPrice * (1 + percentage);
    } else {
      return entryPrice * (1 - percentage);
    }
  }

  private updateTrailingStop(position: Position): void {
    if (!position.trailingStop || !position.highWaterMark) return;

    const trailPercent = position.trailingStop;

    if (position.side === 'long') {
      const newStop = position.highWaterMark * (1 - trailPercent);
      if (!position.stopLoss || newStop > position.stopLoss) {
        position.stopLoss = newStop;
      }
    } else {
      const newStop = position.highWaterMark * (1 + trailPercent);
      if (!position.stopLoss || newStop < position.stopLoss) {
        position.stopLoss = newStop;
      }
    }
  }

  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;

    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return 0;

    // Annualized Sharpe ratio (assuming daily returns)
    const riskFreeRate = 0.02 / 365; // 2% annual risk-free rate
    return Math.sqrt(365) * (avgReturn - riskFreeRate) / stdDev;
  }

  private calculateMaxDrawdown(positions: Position[]): number {
    if (positions.length === 0) return 0;

    let peak = 0;
    let maxDrawdown = 0;
    let runningPnL = 0;

    const sortedPositions = positions.sort((a, b) => a.openedAt.getTime() - b.openedAt.getTime());

    for (const position of sortedPositions) {
      const pnl = position.status === 'open' ? position.unrealizedPnL : position.realizedPnL;
      runningPnL += pnl;

      if (runningPnL > peak) {
        peak = runningPnL;
      }

      const drawdown = peak - runningPnL;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return peak > 0 ? (maxDrawdown / peak) * 100 : 0;
  }

  private getBestTrade(positions: Position[]): number {
    let best = 0;
    for (const position of positions) {
      const pnl = position.status === 'open' ? position.unrealizedPnL : position.realizedPnL;
      if (pnl > best) best = pnl;
    }
    return best;
  }

  private getWorstTrade(positions: Position[]): number {
    let worst = 0;
    for (const position of positions) {
      const pnl = position.status === 'open' ? position.unrealizedPnL : position.realizedPnL;
      if (pnl < worst) worst = pnl;
    }
    return worst;
  }

  private addPosition(position: Position): void {
    const userId = position.userId;
    if (!this.positions.has(userId)) {
      this.positions.set(userId, []);
    }
    this.positions.get(userId)!.push(position);
    this.positionById.set(position.id, position);
  }

  private removePosition(position: Position): void {
    const userId = position.userId;
    const userPositions = this.positions.get(userId);
    if (userPositions) {
      const index = userPositions.findIndex(p => p.id === position.id);
      if (index !== -1) {
        userPositions.splice(index, 1);
      }
    }
    this.positionById.delete(position.id);
  }

  private async restorePositions(): Promise<void> {
    try {
      const positions = await this.stateManager.getAllPositions();
      for (const position of positions) {
        if (position.status === 'open') {
          this.addPosition(position);
        }
      }
      logger.info(`Restored ${positions.length} positions`);
    } catch (error) {
      logger.error('Failed to restore positions:', error);
    }
  }

  private startPositionMonitoring(): void {
    // Monitor positions every 10 seconds
    setInterval(() => {
      this.monitorPositions();
    }, 10000);
  }

  private async monitorPositions(): Promise<void> {
    for (const [userId, positions] of this.positions) {
      for (const position of positions) {
        if (position.status === 'open') {
          // Emit position update events for real-time monitoring
          this.emit('positionUpdate', {
            userId,
            position,
            unrealizedPnL: position.unrealizedPnL,
          });
        }
      }
    }
  }

  async cleanup(): Promise<void> {
    // Save all positions before cleanup
    for (const [_, positions] of this.positions) {
      for (const position of positions) {
        await this.stateManager.savePosition(position);
      }
    }

    this.positions.clear();
    this.positionById.clear();
  }
}

export default PositionManager;