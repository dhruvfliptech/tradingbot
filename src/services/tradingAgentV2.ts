import { TradingSignal, CryptoData } from '../types/trading';
import { riskManager } from './riskManager';
import { coinGeckoService } from './coinGeckoService';
import { alpacaService } from './alpacaService';
import { groqService } from './groqService';
import { tradeHistoryService } from './persistence/tradeHistoryService';
import { auditLogService } from './persistence/auditLogService';
import { statePersistenceService } from './persistence/statePersistenceService';
import { agentSettingsService, AgentSettings } from './agentSettingsService';
import { ValidatorSystem } from './validatorSystem';
import { TechnicalIndicators, PriceData } from './technicalIndicators';
import { riskManagerV2 } from './riskManagerV2';
import { fundingRatesService } from './fundingRatesService';
import { portfolioAnalytics } from './portfolioAnalytics';

export type AgentEvent =
  | { type: 'status'; active: boolean; timestamp: string }
  | { type: 'analysis'; evaluated: number; top: Array<{ symbol: string; action: 'BUY' | 'SELL' | 'HOLD'; confidence: number }>; note?: string; timestamp: string }
  | { type: 'decision'; symbol: string; action: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reason: string; price: number; timestamp: string }
  | { type: 'analysis_detail'; symbol: string; price: number; indicators: { rsi: number; macd: number; ma20: number; ma50: number; ma_trend: 'bullish' | 'bearish' | 'neutral' }; fundamentals: { market_cap_rank: number; ath_change_percentage: number }; momentum: { changePercent: number; volume: number }; decision: 'BUY' | 'SELL' | 'HOLD'; confidence: number; reasoning: string; breakdown: Array<{ name: string; score: number; weight: number; contribution: number; detail: string }>; timestamp: string }
  | { type: 'market_sentiment'; overall: 'bullish' | 'bearish' | 'neutral'; confidence: number; reasoning: string; timestamp: string }
  | { type: 'no_trade'; reason: string; timestamp: string }
  | { type: 'order_submitted'; symbol: string; side: 'buy' | 'sell'; qty: number; order_type: 'market' | 'limit'; limit_price?: number | null; timestamp: string }
  | { type: 'order_error'; symbol: string; message: string; timestamp: string }
  | { type: 'threshold_block'; symbol: string; confidence: number; threshold: number; timestamp: string };

class TradingAgentV2 {
  private isActive: boolean = false;
  private signals: TradingSignal[] = [];
  private watchlist: string[] = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana'];
  private tickHandle: number | null = null;
  private subscribers: Array<(e: AgentEvent) => void> = [];
  private lastTradeAtBySymbol: Record<string, number> = {};
  private lastAccountSnapshot: { account: any; positions: any[] } | null = null;
  private settings: AgentSettings | null = null;
  private validatorSystem: ValidatorSystem;
  private priceHistory: Map<string, PriceData[]> = new Map();
  private fearGreedIndex: number = 50; // Default neutral

  constructor() {
    // Load persisted state on initialization
    const persistedState = statePersistenceService.getState();
    this.isActive = persistedState.agentActive || false;
    this.lastTradeAtBySymbol = persistedState.lastPrices ? 
      Object.keys(persistedState.lastPrices).reduce((acc, key) => {
        acc[key] = Date.now() - 3600000; // Set to 1 hour ago
        return acc;
      }, {} as Record<string, number>) : {};
    
    // Initialize validator system
    this.validatorSystem = new ValidatorSystem();
  }

  async start(): Promise<void> {
    if (this.isActive) return;
    
    // Load user settings first
    await this.loadSettings();
    
    this.isActive = true;
    statePersistenceService.setValue('agentActive', true);
    
    this.emit({ type: 'status', active: true, timestamp: new Date().toISOString() });
    
    // Log agent start
    await auditLogService.logAgentControl('resume', 'Agent started by user');
    
    this.generateDefaultSignals();
    // kick off trading loop
    this.runCycle();
    this.tickHandle = window.setInterval(() => this.runCycle(), 45 * 1000);
  }

  async stop(): Promise<void> {
    this.isActive = false;
    statePersistenceService.setValue('agentActive', false);
    
    if (this.tickHandle) {
      clearInterval(this.tickHandle);
      this.tickHandle = null;
    }
    
    this.emit({ type: 'status', active: false, timestamp: new Date().toISOString() });
    
    // Log agent stop
    await auditLogService.logAgentControl('pause', 'Agent stopped by user');
  }

  subscribe(handler: (e: AgentEvent) => void): () => void {
    this.subscribers.push(handler);
    return () => {
      this.subscribers = this.subscribers.filter(h => h !== handler);
    };
  }

  private emit(event: AgentEvent) {
    try {
      for (const h of this.subscribers) h(event);
    } catch {}
  }

  private async loadSettings() {
    try {
      // Load settings from agentSettingsService which handles both Supabase and localStorage
      this.settings = await agentSettingsService.getSettings();
      console.log('✅ Trading settings loaded:', this.settings);
    } catch (error) {
      console.error('Failed to load settings:', error);
      // Settings service already returns defaults on error
      this.settings = await agentSettingsService.getSettings();
    }
  }

  private async runCycle(): Promise<void> {
    if (!this.isActive) return;
    
    try {
      // Reload settings each cycle to pick up changes
      await this.loadSettings();
      
      // snapshot account/positions for risk manager
      try {
        const [accountSnap, positionsSnap] = await Promise.all([
          alpacaService.getAccount(),
          alpacaService.getPositions(),
        ]);
        this.lastAccountSnapshot = { account: accountSnap, positions: positionsSnap };
      } catch {}
      
      // 1) Fetch latest watchlist prices
      const data = await coinGeckoService.getCryptoData(this.watchlist);
      
      // Update persisted prices
      const priceMap: Record<string, number> = {};
      data.forEach(d => {
        priceMap[d.symbol.toUpperCase()] = d.price;
      });
      statePersistenceService.setValue('lastPrices', priceMap);
      
      // 2) Generate signals with validation
      const signals = await this.analyzeCryptoData(data);
      this.signals = signals;

      // Emit analysis snapshot
      const top = [...signals]
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3)
        .map(s => ({ symbol: s.symbol.toUpperCase(), action: s.action, confidence: s.confidence }));
      
      this.emit({ 
        type: 'analysis', 
        evaluated: signals.length, 
        top, 
        note: `threshold=${this.settings?.confidenceThreshold || 0.75}`, 
        timestamp: new Date().toISOString() 
      });

      // Emit per-coin detailed analysis and check for trades
      for (const signal of signals) {
        const cryptoData = data.find(d => d.symbol === signal.symbol);
        if (!cryptoData) continue;

        // Log AI decision for audit
        await auditLogService.logAIDecision(
          signal.symbol,
          signal.action.toLowerCase() as 'buy' | 'sell' | 'hold',
          signal.confidence,
          {
            signals: {
              technical: {
                rsi: this.calculateRSI(cryptoData),
                macd: this.calculateMACD(cryptoData) > 0 ? 'bullish' : 'bearish',
                ma_trend: signal.maIndicator || 'neutral'
              },
              momentum: {
                volume: cryptoData.volume,
                change_24h: cryptoData.changePercent
              }
            },
            confidence: signal.confidence,
            expected_risk_reward: 3.0 // Default risk/reward ratio
          },
          {
            btc_price: data.find(d => d.symbol === 'bitcoin')?.price,
            eth_price: data.find(d => d.symbol === 'ethereum')?.price,
            total_market_cap: data.reduce((sum, d) => sum + (d.marketCap || 0), 0)
          },
          this.lastAccountSnapshot ? {
            total_value: parseFloat(this.lastAccountSnapshot.account?.portfolio_value || 0),
            cash_balance: parseFloat(this.lastAccountSnapshot.account?.cash || 0),
            positions_count: this.lastAccountSnapshot.positions?.length || 0
          } : undefined
        );

        // CRITICAL: Check confidence threshold BEFORE attempting trade
        const threshold = (this.settings?.confidenceThreshold || 0.78) * 100; // Convert to percentage
        if (signal.confidence < threshold) {
          console.log(`🚫 Signal blocked: ${signal.symbol} confidence ${signal.confidence}% < threshold ${threshold}%`);
          
          // Emit threshold block event
          this.emit({
            type: 'threshold_block',
            symbol: signal.symbol,
            confidence: signal.confidence,
            threshold: threshold,
            timestamp: new Date().toISOString()
          });
          
          continue; // Skip this signal
        }

        // Check if we should act on this signal
        if (signal.action === 'BUY' && this.shouldExecuteTrade(signal)) {
          await this.executeTrade(signal, cryptoData);
        }
      }

    } catch (error) {
      console.error('Trading cycle error:', error);
      await auditLogService.logSystemAlert('cycle_error', 'Trading cycle failed', { error: error.message });
    }
  }

  private shouldExecuteTrade(signal: TradingSignal): boolean {
    if (!this.settings) return false;
    
    // DOUBLE-CHECK: Confidence threshold (redundant but safe)
    const threshold = this.settings.confidenceThreshold * 100;
    if (signal.confidence < threshold) {
      console.log(`Confidence ${signal.confidence}% below threshold ${threshold}%`);
      return false;
    }

    // Check cooldown period using settings
    const lastTrade = this.lastTradeAtBySymbol[signal.symbol] || 0;
    const cooldownMs = (this.settings.cooldownMinutes || 5) * 60 * 1000;
    if (Date.now() - lastTrade < cooldownMs) {
      console.log(`Cooldown active for ${signal.symbol} (${this.settings.cooldownMinutes} min)`);
      return false;
    }

    // Check if we already have a position in this symbol
    if (this.lastAccountSnapshot?.positions) {
      const hasPosition = this.lastAccountSnapshot.positions.some(
        p => p.symbol.toLowerCase() === signal.symbol.toLowerCase()
      );
      if (hasPosition) {
        console.log(`Already have position in ${signal.symbol}`);
        return false;
      }
      
      // Check max open positions limit
      const openPositions = this.lastAccountSnapshot.positions.length;
      if (openPositions >= (this.settings.maxOpenPositions || 10)) {
        console.log(`Max open positions reached: ${openPositions}/${this.settings.maxOpenPositions}`);
        return false;
      }
    }

    return true;
  }

  private async executeTrade(signal: TradingSignal, cryptoData: CryptoData) {
    try {
      // Calculate position size based on risk settings
      const accountValue = parseFloat(this.lastAccountSnapshot?.account?.portfolio_value || '50000');
      const riskBudgetUsd = this.settings?.riskBudgetUsd || 100;
      
      // Use the risk budget directly as position size
      const positionValue = Math.min(riskBudgetUsd, accountValue * 0.1); // Cap at 10% of portfolio
      const quantity = Math.floor((positionValue / cryptoData.price) * 100) / 100; // Round down to 2 decimals

      if (quantity < 0.01) {
        console.log(`Position size too small for ${signal.symbol}`);
        return;
      }

      // Record trade intent in history
      const tradeRecord = await tradeHistoryService.recordTrade({
        symbol: signal.symbol.toUpperCase(),
        side: 'buy',
        quantity,
        entry_price: cryptoData.price,
        execution_status: 'pending',
        confidence_score: signal.confidence,
        risk_reward_ratio: 3.0,
        position_size_percent: (positionValue / accountValue) * 100,
        risk_amount: riskBudgetUsd
      });

      // Execute via Alpaca (or demo mode)
      const order = await alpacaService.placeOrder({
        symbol: signal.symbol.toUpperCase(),
        qty: quantity,
        side: 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      // Update trade record with order details
      if (tradeRecord?.id && order) {
        await tradeHistoryService.updateTrade(tradeRecord.id, {
          alpaca_order_id: order.id,
          execution_status: 'filled',
          filled_at: new Date().toISOString()
        });
      }

      // Log trade execution
      await auditLogService.logTradeExecution(
        signal.symbol,
        'buy',
        quantity,
        cryptoData.price,
        order?.id
      );

      // Update cooldown
      this.lastTradeAtBySymbol[signal.symbol] = Date.now();

      // Emit success event
      this.emit({
        type: 'order_submitted',
        symbol: signal.symbol,
        side: 'buy',
        qty: quantity,
        order_type: 'market',
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error(`Failed to execute trade for ${signal.symbol}:`, error);
      
      this.emit({
        type: 'order_error',
        symbol: signal.symbol,
        message: error.message,
        timestamp: new Date().toISOString()
      });
      
      await auditLogService.logSystemAlert('trade_error', `Trade execution failed for ${signal.symbol}`, { error: error.message });
    }
  }

  // ... rest of the technical indicator methods remain the same ...
  
  private generateDefaultSignals(): void {
    // Placeholder for immediate UI feedback
    this.signals = this.watchlist.map(symbol => ({
      symbol,
      action: 'HOLD' as const,
      confidence: 50,
      price: 0,
      timestamp: new Date().toISOString()
    }));
  }

  private async analyzeCryptoData(data: CryptoData[]): Promise<TradingSignal[]> {
    const signals: TradingSignal[] = [];
    
    for (const crypto of data) {
      // Update price history for technical analysis
      this.updatePriceHistory(crypto);
      const priceData = this.priceHistory.get(crypto.symbol) || [];
      
      if (priceData.length < 26) {
        // Not enough data for technical analysis
        signals.push({
          symbol: crypto.symbol,
          action: 'HOLD',
          confidence: 0,
          price: crypto.price,
          timestamp: new Date().toISOString()
        });
        continue;
      }
      
      // Calculate real technical indicators
      const prices = priceData.map(p => p.close);
      const rsiResult = TechnicalIndicators.RSI(prices);
      const macdResult = TechnicalIndicators.MACD(prices);
      const bbResult = TechnicalIndicators.BollingerBands(prices);
      const composite = TechnicalIndicators.getCompositeScore(priceData);
      
      // Determine initial action based on technical indicators
      let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      let baseConfidence = 50;
      
      if (composite.signal === 'strong_buy') {
        action = 'BUY';
        baseConfidence = 80;
      } else if (composite.signal === 'buy') {
        action = 'BUY';
        baseConfidence = 65;
      } else if (composite.signal === 'strong_sell') {
        action = 'SELL';
        baseConfidence = 80;
      } else if (composite.signal === 'sell') {
        action = 'SELL';
        baseConfidence = 65;
      }
      
      // Create initial signal
      const signal: TradingSignal = {
        symbol: crypto.symbol,
        action,
        confidence: baseConfidence,
        price: crypto.price,
        timestamp: new Date().toISOString(),
        maIndicator: macdResult.trend,
        rsi: rsiResult.value,
        technicalScore: composite.score
      };
      
      // Run through validator system if not HOLD
      if (action !== 'HOLD') {
        const validation = await this.validatorSystem.validate(
          signal,
          priceData,
          crypto,
          {
            portfolioValue: parseFloat(this.lastAccountSnapshot?.account?.portfolio_value || '50000'),
            existingPositions: this.lastAccountSnapshot?.positions?.length || 0,
            fearGreedIndex: this.fearGreedIndex,
            settings: this.settings
          }
        );
        
        // Update signal based on validation
        if (validation.passed) {
          signal.confidence = Math.round(validation.finalScore);
          signal.validationDetails = validation;
        } else {
          // If validation fails, reduce to HOLD
          signal.action = 'HOLD';
          signal.confidence = 30;
          signal.validationDetails = validation;
        }
      }
      
      signals.push(signal);
    }
    
    return signals;
  }
  
  private updatePriceHistory(crypto: CryptoData) {
    const history = this.priceHistory.get(crypto.symbol) || [];
    
    // Add new price data point
    history.push({
      timestamp: Date.now(),
      open: crypto.price * (1 - crypto.changePercent / 200), // Approximate
      high: crypto.high24h || crypto.price * 1.01,
      low: crypto.low24h || crypto.price * 0.99,
      close: crypto.price,
      volume: crypto.volume || 0
    });
    
    // Keep only last 200 data points
    if (history.length > 200) {
      history.shift();
    }
    
    this.priceHistory.set(crypto.symbol, history);
  }

  private calculateRSI(crypto: CryptoData): number {
    // Simplified RSI calculation based on 24h change
    const change = crypto.changePercent;
    return 50 + (change * 2.5); // Maps -20% to +20% change to 0-100 RSI
  }

  private calculateMACD(crypto: CryptoData): number {
    // Simplified MACD based on recent momentum
    return crypto.changePercent * 0.5;
  }

  private calculateMA20(crypto: CryptoData): number {
    // Approximate 20-day MA
    return crypto.price * (1 - crypto.changePercent / 200);
  }

  private calculateMA50(crypto: CryptoData): number {
    // Approximate 50-day MA
    return crypto.price * (1 - crypto.changePercent / 100);
  }

  isRunning(): boolean {
    return this.isActive;
  }

  getSignals(): TradingSignal[] {
    return this.signals;
  }
}

// Export singleton instance
export const tradingAgentV2 = new TradingAgentV2();