/**
 * Advanced Position Management Service
 * Handles partial profit taking, DCA entries, pyramid positioning, and risk management
 */

import { alpacaService } from './alpacaService';
import { tradeHistoryService } from './persistence/tradeHistoryService';

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  stop_loss?: number;
  take_profit?: number;
  risk_amount: number;
  created_at: string;
  strategy: string;
}

export interface PositionRule {
  id: string;
  position_id: string;
  rule_type: 'take_profit' | 'stop_loss' | 'trailing_stop' | 'time_exit';
  trigger_price?: number;
  trigger_percentage?: number;
  quantity_percentage: number; // Percentage of position to close
  is_active: boolean;
  created_at: string;
}

export interface ProfitTakingLevel {
  r_multiple: number; // Risk multiple (1R = 1x risk amount)
  quantity_percentage: number; // Percentage of position to close
  executed: boolean;
}

export interface DCAEntry {
  id: string;
  symbol: string;
  target_price: number;
  quantity: number;
  max_entries: number;
  current_entries: number;
  price_step_percentage: number;
  is_active: boolean;
}

class PositionManager {
  private positions: Map<string, Position> = new Map();
  private positionRules: Map<string, PositionRule[]> = new Map();
  private dcaOrders: Map<string, DCAEntry> = new Map();
  private readonly CHECK_INTERVAL = 30000; // 30 seconds
  private checkHandle: number | null = null;

  /**
   * Start position monitoring
   */
  start(): void {
    if (this.checkHandle) return;
    
    this.loadPositions();
    this.checkHandle = window.setInterval(() => {
      this.checkPositions();
    }, this.CHECK_INTERVAL);
  }

  /**
   * Stop position monitoring
   */
  stop(): void {
    if (this.checkHandle) {
      clearInterval(this.checkHandle);
      this.checkHandle = null;
    }
  }

  /**
   * Load current positions from Alpaca
   */
  async loadPositions(): Promise<void> {
    try {
      const alpacaPositions = await alpacaService.getPositions();
      
      for (const alpacaPos of alpacaPositions) {
        const position: Position = {
          id: alpacaPos.asset_id || alpacaPos.symbol,
          symbol: alpacaPos.symbol,
          side: parseFloat(alpacaPos.qty) > 0 ? 'long' : 'short',
          quantity: Math.abs(parseFloat(alpacaPos.qty)),
          entry_price: parseFloat(alpacaPos.avg_entry_price || '0'),
          current_price: parseFloat(alpacaPos.market_value || '0') / Math.abs(parseFloat(alpacaPos.qty)),
          unrealized_pnl: parseFloat(alpacaPos.unrealized_pl || '0'),
          risk_amount: 0, // Will be calculated
          created_at: new Date().toISOString(),
          strategy: 'unknown'
        };

        this.positions.set(position.id, position);
      }
    } catch (error) {
      console.error('Error loading positions:', error);
    }
  }

  /**
   * Set up automatic profit taking levels
   */
  setupProfitTaking(
    positionId: string,
    levels: ProfitTakingLevel[]
  ): void {
    const position = this.positions.get(positionId);
    if (!position) return;

    const rules: PositionRule[] = [];
    
    for (const level of levels) {
      const riskAmount = position.risk_amount || (position.quantity * position.entry_price * 0.02); // 2% default risk
      const targetPrice = position.side === 'long' 
        ? position.entry_price + (riskAmount * level.r_multiple / position.quantity)
        : position.entry_price - (riskAmount * level.r_multiple / position.quantity);

      const rule: PositionRule = {
        id: `${positionId}_tp_${level.r_multiple}R`,
        position_id: positionId,
        rule_type: 'take_profit',
        trigger_price: targetPrice,
        quantity_percentage: level.quantity_percentage,
        is_active: true,
        created_at: new Date().toISOString()
      };

      rules.push(rule);
    }

    this.positionRules.set(positionId, rules);
    this.saveRulesToStorage(positionId, rules);
  }

  /**
   * Set up Dollar-Cost Averaging (DCA) entries
   */
  setupDCAEntry(
    symbol: string,
    initialPrice: number,
    totalQuantity: number,
    maxEntries: number = 3,
    priceStepPercentage: number = 5
  ): string {
    const dcaId = `dca_${symbol}_${Date.now()}`;
    const quantityPerEntry = totalQuantity / maxEntries;

    const dcaEntry: DCAEntry = {
      id: dcaId,
      symbol,
      target_price: initialPrice,
      quantity: quantityPerEntry,
      max_entries: maxEntries,
      current_entries: 0,
      price_step_percentage: priceStepPercentage,
      is_active: true
    };

    this.dcaOrders.set(dcaId, dcaEntry);
    this.saveDCAToStorage(dcaId, dcaEntry);

    return dcaId;
  }

  /**
   * Add to winning position (pyramid)
   */
  async addToPosition(
    positionId: string,
    additionalQuantity: number,
    maxPyramidLevels: number = 2
  ): Promise<boolean> {
    const position = this.positions.get(positionId);
    if (!position) return false;

    // Check if position is profitable
    if (position.unrealized_pnl <= 0) {
      console.log('Position not profitable, skipping pyramid add');
      return false;
    }

    // Get existing pyramid levels from trade history
    const existingTrades = await tradeHistoryService.getTradesForSymbol(position.symbol);
    const pyramidLevels = existingTrades.filter(trade => 
      trade.symbol === position.symbol && 
      trade.execution_status === 'filled'
    ).length;

    if (pyramidLevels >= maxPyramidLevels) {
      console.log('Maximum pyramid levels reached');
      return false;
    }

    try {
      // Place additional order
      const order = await alpacaService.placeOrder({
        symbol: position.symbol,
        qty: additionalQuantity,
        side: position.side === 'long' ? 'buy' : 'sell',
        type: 'market',
        time_in_force: 'day'
      });

      // Record the pyramid addition
      await tradeHistoryService.recordTrade({
        symbol: position.symbol,
        side: position.side === 'long' ? 'buy' : 'sell',
        quantity: additionalQuantity,
        entry_price: position.current_price,
        execution_status: 'filled',
        confidence_score: 80, // High confidence for pyramid adds
        risk_reward_ratio: 2.0,
        position_size_percent: 0, // Part of existing position
        risk_amount: 0,
        strategy_attribution: 'pyramid'
      });

      console.log(`Added ${additionalQuantity} to ${position.symbol} position (pyramid level ${pyramidLevels + 1})`);
      return true;

    } catch (error) {
      console.error('Error adding to position:', error);
      return false;
    }
  }

  /**
   * Set trailing stop loss
   */
  setTrailingStop(
    positionId: string,
    trailPercentage: number
  ): void {
    const position = this.positions.get(positionId);
    if (!position) return;

    const rule: PositionRule = {
      id: `${positionId}_trailing_stop`,
      position_id: positionId,
      rule_type: 'trailing_stop',
      trigger_percentage: trailPercentage,
      quantity_percentage: 100, // Close entire position
      is_active: true,
      created_at: new Date().toISOString()
    };

    const existingRules = this.positionRules.get(positionId) || [];
    // Remove existing trailing stops
    const filteredRules = existingRules.filter(r => r.rule_type !== 'trailing_stop');
    filteredRules.push(rule);

    this.positionRules.set(positionId, filteredRules);
    this.saveRulesToStorage(positionId, filteredRules);
  }

  /**
   * Check all positions and execute rules
   */
  private async checkPositions(): Promise<void> {
    try {
      // Reload current positions
      await this.loadPositions();

      // Check position rules
      for (const [positionId, position] of this.positions) {
        const rules = this.positionRules.get(positionId) || [];
        await this.checkPositionRules(position, rules);
      }

      // Check DCA orders
      for (const [dcaId, dcaEntry] of this.dcaOrders) {
        await this.checkDCAEntry(dcaEntry);
      }

    } catch (error) {
      console.error('Error checking positions:', error);
    }
  }

  /**
   * Check and execute position rules
   */
  private async checkPositionRules(position: Position, rules: PositionRule[]): Promise<void> {
    for (const rule of rules) {
      if (!rule.is_active) continue;

      let shouldTrigger = false;

      switch (rule.rule_type) {
        case 'take_profit':
          if (position.side === 'long' && position.current_price >= (rule.trigger_price || 0)) {
            shouldTrigger = true;
          } else if (position.side === 'short' && position.current_price <= (rule.trigger_price || 0)) {
            shouldTrigger = true;
          }
          break;

        case 'stop_loss':
          if (position.side === 'long' && position.current_price <= (rule.trigger_price || 0)) {
            shouldTrigger = true;
          } else if (position.side === 'short' && position.current_price >= (rule.trigger_price || 0)) {
            shouldTrigger = true;
          }
          break;

        case 'trailing_stop':
          // Implement trailing stop logic
          shouldTrigger = await this.checkTrailingStop(position, rule);
          break;
      }

      if (shouldTrigger) {
        await this.executeRule(position, rule);
      }
    }
  }

  /**
   * Check trailing stop condition
   */
  private async checkTrailingStop(position: Position, rule: PositionRule): Promise<boolean> {
    const trailPercentage = rule.trigger_percentage || 5;
    
    // Get the highest price since position opened (for long) or lowest (for short)
    const storedHigh = localStorage.getItem(`trail_high_${position.id}`);
    const storedLow = localStorage.getItem(`trail_low_${position.id}`);

    if (position.side === 'long') {
      const currentHigh = storedHigh ? Math.max(parseFloat(storedHigh), position.current_price) : position.current_price;
      localStorage.setItem(`trail_high_${position.id}`, currentHigh.toString());
      
      const trailStop = currentHigh * (1 - trailPercentage / 100);
      return position.current_price <= trailStop;
    } else {
      const currentLow = storedLow ? Math.min(parseFloat(storedLow), position.current_price) : position.current_price;
      localStorage.setItem(`trail_low_${position.id}`, currentLow.toString());
      
      const trailStop = currentLow * (1 + trailPercentage / 100);
      return position.current_price >= trailStop;
    }
  }

  /**
   * Execute a position rule
   */
  private async executeRule(position: Position, rule: PositionRule): Promise<void> {
    try {
      const quantityToClose = position.quantity * (rule.quantity_percentage / 100);

      // Place order to close portion of position
      const order = await alpacaService.placeOrder({
        symbol: position.symbol,
        qty: quantityToClose,
        side: position.side === 'long' ? 'sell' : 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      // Update trade history
      await tradeHistoryService.recordTrade({
        symbol: position.symbol,
        side: position.side === 'long' ? 'sell' : 'buy',
        quantity: quantityToClose,
        entry_price: position.current_price,
        execution_status: 'filled',
        confidence_score: 100, // Rule-based execution
        risk_reward_ratio: 0,
        position_size_percent: 0,
        risk_amount: 0,
        strategy_attribution: rule.rule_type
      });

      // Deactivate the rule
      rule.is_active = false;
      this.saveRulesToStorage(position.id, this.positionRules.get(position.id) || []);

      console.log(`Executed ${rule.rule_type} rule for ${position.symbol}: closed ${quantityToClose} shares at $${position.current_price}`);

    } catch (error) {
      console.error(`Error executing rule ${rule.id}:`, error);
    }
  }

  /**
   * Check DCA entry conditions
   */
  private async checkDCAEntry(dcaEntry: DCAEntry): Promise<void> {
    if (!dcaEntry.is_active || dcaEntry.current_entries >= dcaEntry.max_entries) {
      return;
    }

    try {
      // Get current price
      const currentPrice = await this.getCurrentPrice(dcaEntry.symbol);
      
      // Calculate next entry price
      const nextEntryPrice = dcaEntry.target_price * (1 - (dcaEntry.price_step_percentage * (dcaEntry.current_entries + 1)) / 100);

      if (currentPrice <= nextEntryPrice) {
        // Execute DCA entry
        const order = await alpacaService.placeOrder({
          symbol: dcaEntry.symbol,
          qty: dcaEntry.quantity,
          side: 'buy',
          type: 'market',
          time_in_force: 'day'
        });

        // Update DCA entry
        dcaEntry.current_entries++;
        if (dcaEntry.current_entries >= dcaEntry.max_entries) {
          dcaEntry.is_active = false;
        }

        this.saveDCAToStorage(dcaEntry.id, dcaEntry);

        // Record trade
        await tradeHistoryService.recordTrade({
          symbol: dcaEntry.symbol,
          side: 'buy',
          quantity: dcaEntry.quantity,
          entry_price: currentPrice,
          execution_status: 'filled',
          confidence_score: 70,
          risk_reward_ratio: 2.0,
          position_size_percent: 0,
          risk_amount: 0,
          strategy_attribution: 'dca'
        });

        console.log(`DCA entry ${dcaEntry.current_entries}/${dcaEntry.max_entries} executed for ${dcaEntry.symbol} at $${currentPrice}`);
      }

    } catch (error) {
      console.error('Error checking DCA entry:', error);
    }
  }

  /**
   * Get current price for a symbol
   */
  private async getCurrentPrice(symbol: string): Promise<number> {
    // In a real implementation, you'd fetch this from a price service
    // For now, return a simulated price
    return 50000; // Placeholder
  }

  /**
   * Save rules to localStorage
   */
  private saveRulesToStorage(positionId: string, rules: PositionRule[]): void {
    localStorage.setItem(`position_rules_${positionId}`, JSON.stringify(rules));
  }

  /**
   * Save DCA to localStorage
   */
  private saveDCAToStorage(dcaId: string, dcaEntry: DCAEntry): void {
    localStorage.setItem(`dca_${dcaId}`, JSON.stringify(dcaEntry));
  }

  /**
   * Get position with rules
   */
  getPositionWithRules(positionId: string): { position: Position | null; rules: PositionRule[] } {
    return {
      position: this.positions.get(positionId) || null,
      rules: this.positionRules.get(positionId) || []
    };
  }

  /**
   * Cancel a position rule
   */
  cancelRule(ruleId: string): void {
    for (const [positionId, rules] of this.positionRules) {
      const rule = rules.find(r => r.id === ruleId);
      if (rule) {
        rule.is_active = false;
        this.saveRulesToStorage(positionId, rules);
        break;
      }
    }
  }

  /**
   * Cancel DCA order
   */
  cancelDCAOrder(dcaId: string): void {
    const dcaEntry = this.dcaOrders.get(dcaId);
    if (dcaEntry) {
      dcaEntry.is_active = false;
      this.saveDCAToStorage(dcaId, dcaEntry);
    }
  }

  /**
   * Get all active positions
   */
  getAllPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get all active DCA orders
   */
  getAllDCAOrders(): DCAEntry[] {
    return Array.from(this.dcaOrders.values()).filter(dca => dca.is_active);
  }
}

export const positionManager = new PositionManager();
export default positionManager;