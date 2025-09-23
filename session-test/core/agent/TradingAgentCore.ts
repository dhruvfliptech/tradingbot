import { AgentContext } from '../adapters';

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: string;
}

export class TradingAgentCore {
  private context: AgentContext;
  private isActive = false;
  private lastPrices: Record<string, number> = {};
  private lastCycleAt: number | null = null;

  constructor(context: AgentContext) {
    this.context = context;
  }

  start(): void {
    if (this.isActive) return;
    this.context.logger?.info?.('TradingAgentCore starting');
    this.isActive = true;
    void this.bootstrapState();
  }

  stop(): void {
    if (!this.isActive) return;
    this.context.logger?.info?.('TradingAgentCore stopping');
    this.isActive = false;
  }

  private async bootstrapState() {
    try {
      const storedPrices = await this.context.persistence.loadState<Record<string, number>>('lastPrices');
      if (storedPrices) {
        this.lastPrices = storedPrices;
      }
    } catch (error) {
      this.context.logger?.warn?.('Failed to bootstrap last prices', error);
    }
  }

  async runCycle(): Promise<void> {
    if (!this.isActive) {
      this.context.logger?.debug?.('Skipping cycle; agent inactive');
      return;
    }

    const settings = await this.context.settings.loadSettings();
    const watchlist = (settings.watchlist as string[]) || ['bitcoin', 'ethereum'];

    const prices = await this.context.marketData.fetchWatchlistPrices(watchlist);
    const signals: TradingSignal[] = [];
    const nowIso = new Date().toISOString();

    for (const priceInfo of prices) {
      const previousPrice = this.lastPrices[priceInfo.symbol];
      let action: TradingSignal['action'] = 'HOLD';
      let confidence = 0;

      if (previousPrice) {
        const pctChange = ((priceInfo.price - previousPrice) / previousPrice) * 100;
        const buyThreshold = Number(settings.buyThresholdPct ?? 1);
        const sellThreshold = Number(settings.sellThresholdPct ?? -1);

        if (pctChange >= buyThreshold) {
          action = 'BUY';
          confidence = Math.min(100, Math.round(pctChange * 10));
        } else if (pctChange <= sellThreshold) {
          action = 'SELL';
          confidence = Math.min(100, Math.round(Math.abs(pctChange) * 10));
        }

        this.context.logger?.debug?.('Price delta', {
          symbol: priceInfo.symbol,
          previousPrice,
          currentPrice: priceInfo.price,
          pctChange,
          action,
          confidence,
        });
      } else {
        this.context.logger?.debug?.('First observation for symbol', {
          symbol: priceInfo.symbol,
          price: priceInfo.price,
        });
      }

      this.lastPrices[priceInfo.symbol] = priceInfo.price;

      signals.push({
        symbol: priceInfo.symbol,
        action,
        confidence,
        price: priceInfo.price,
        timestamp: nowIso,
      });
    }

    await this.context.persistence.saveState('lastPrices', this.lastPrices);
    await this.context.persistence.saveState('lastSignals', signals);

    if (this.context.events) {
      this.context.events.emit('analysis', { signals, timestamp: nowIso });
    }

    await this.evaluateSignals(signals, settings);

    this.lastCycleAt = Date.now();
  }

  private async evaluateSignals(signals: TradingSignal[], settings: Record<string, any>) {
    for (const signal of signals) {
      if (signal.action === 'HOLD') continue;

      const minConfidence = Number(settings.minConfidence ?? 70);
      if (signal.confidence < minConfidence) {
        this.context.logger?.info?.('Signal below confidence threshold', signal);
        continue;
      }

      await this.context.persistence.appendAuditLog({
        event_type: 'ai_decision',
        symbol: signal.symbol,
        action: signal.action.toLowerCase(),
        confidence_score: signal.confidence,
        created_at: signal.timestamp,
      });

      try {
        await this.context.broker.placeOrder({
          symbol: signal.symbol,
          quantity: Number(settings.orderQuantity ?? 0.01),
          side: signal.action === 'BUY' ? 'buy' : 'sell',
          orderType: 'market',
        });

        await this.context.persistence.recordTrade({
          symbol: signal.symbol,
          side: signal.action === 'BUY' ? 'buy' : 'sell',
          quantity: Number(settings.orderQuantity ?? 0.01),
          entry_price: signal.price,
          confidence_score: signal.confidence,
          execution_status: 'filled',
        });

        if (this.context.events) {
          this.context.events.emit('order_submitted', {
            symbol: signal.symbol,
            side: signal.action.toLowerCase(),
            quantity: settings.orderQuantity ?? 0.01,
            timestamp: signal.timestamp,
          });
        }
      } catch (error) {
        this.context.logger?.error?.('Failed to place order', error);
        await this.context.persistence.appendAuditLog({
          event_type: 'system_alert',
          event_category: 'system',
          symbol: signal.symbol,
          user_reason: 'Order placement failed',
          new_value: { error: (error as Error).message },
        });
      }
    }

    if (this.context.logger && signals.length) {
      this.context.logger.info('Cycle complete', {
        signals,
        nextRunInMs: this.lastCycleAt ? undefined : null,
      });
    }
  }
}
