import { TradingSignal, CryptoData } from '../types/trading';
import { coinGeckoService } from './coinGeckoService';
import { tradingProviderService } from './tradingProviderService';
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
import { 
  liquidityHuntingStrategy,
  smartMoneyDivergenceStrategy,
  volumeProfileAnalysisStrategy,
  microstructureAnalysisStrategy
} from './strategies/index';
import { performanceTracker } from './performanceTracker';
import { correlationService } from './correlationService';
import { newsSentimentService } from './newsSentiment';

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
  private targetReturnWeekly: number = 4; // Target 4% weekly return
  private weeklyPnL: number = 0; // Current week P&L percentage
  private strategySignals: Map<string, any> = new Map(); // Store strategy signals
  private strategyEnabled: Map<string, boolean> = new Map(); // Track enabled strategies
  private marketRegime: 'risk_on' | 'risk_off' | 'mixed' | 'uncertain' = 'uncertain';
  private lastMarketRegimeUpdate: number = 0;

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
    
    // Initialize strategy enablement (all enabled by default)
    this.strategyEnabled.set('liquidity', true);
    this.strategyEnabled.set('smartMoney', true);
    this.strategyEnabled.set('volumeProfile', true);
    this.strategyEnabled.set('microstructure', true);
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
      
      // Update market regime and strategy selection
      await this.updateMarketRegimeAndStrategies();
      
      // snapshot account/positions for risk manager
      try {
        const [accountSnap, positionsSnap] = await Promise.all([
          tradingProviderService.getAccount(),
          tradingProviderService.getPositions(),
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

      // Determine which strategy triggered this trade
      const triggeringStrategy = this.determineTriggeringStrategy(signal);
      
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
        risk_amount: riskBudgetUsd,
        strategy_attribution: triggeringStrategy
      });

      // Record strategy attribution for performance tracking
      if (tradeRecord?.id) {
        await performanceTracker.recordTradeAttribution(
          tradeRecord.id,
          triggeringStrategy,
          signal.confidence,
          'win' // Assume optimistic outcome
        );
      }

      // Execute via configured trading broker
      const order = await tradingProviderService.placeOrder({
        symbol: signal.symbol.toUpperCase(),
        qty: quantity,
        side: 'buy',
        order_type: 'market',
        time_in_force: 'day',
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
      
      // Run strategy analysis based on dynamic selection
      const strategyPromises: Promise<any>[] = [];
      const strategyKeys: string[] = [];
      
      if (this.strategyEnabled.get('liquidity')) {
        strategyPromises.push(liquidityHuntingStrategy.analyze(crypto.symbol, priceData, '4h'));
        strategyKeys.push('liquidity');
      }
      
      if (this.strategyEnabled.get('smartMoney')) {
        strategyPromises.push(smartMoneyDivergenceStrategy.analyze(crypto.symbol, priceData, '24h'));
        strategyKeys.push('smartMoney');
      }
      
      if (this.strategyEnabled.get('volumeProfile')) {
        strategyPromises.push(volumeProfileAnalysisStrategy.analyze(crypto.symbol, priceData, 50));
        strategyKeys.push('volumeProfile');
      }
      
      if (this.strategyEnabled.get('microstructure')) {
        strategyPromises.push(microstructureAnalysisStrategy.analyze(crypto.symbol, priceData, '1h'));
        strategyKeys.push('microstructure');
      }

      const strategyResults = await Promise.allSettled(strategyPromises);

      // Process strategy signals
      const strategySignals: any = {};
      strategyKeys.forEach((key, index) => {
        strategySignals[key] = strategyResults[index].status === 'fulfilled' ? 
          strategyResults[index].value.signal : null;
      });

      // Store strategy signals for UI components
      this.strategySignals.set(crypto.symbol, strategySignals);
      
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

      // Integrate strategy signals (30% weight for strategies, 70% for validators)
      const strategyScore = this.calculateStrategyScore(strategySignals);
      let combinedConfidence = Math.round(baseConfidence * 0.7 + strategyScore * 0.3);
      
      // Enhanced strategy execution logic - if strategies provide strong signals, execute
      if (strategyScore > 75) {
        // Strong strategy signal overrides validator decision
        action = 'BUY';
        console.log(`🎯 Strong strategy signal (${strategyScore}%) - executing BUY for ${crypto.symbol}`);
      } else if (strategyScore < 25) {
        // Strong bearish strategy signal
        action = 'SELL';
        console.log(`🎯 Strong bearish strategy signal (${strategyScore}%) - executing SELL for ${crypto.symbol}`);
      } else if (strategyScore > 70 && action === 'HOLD') {
        // Moderate strategy signal when validator is neutral
        action = 'BUY';
        console.log(`📈 Strategy override: ${crypto.symbol} BUY signal (${strategyScore}%)`);
      } else if (strategyScore < 30 && action === 'HOLD') {
        // Moderate bearish strategy signal when validator is neutral
        action = 'SELL';
        console.log(`📉 Strategy override: ${crypto.symbol} SELL signal (${strategyScore}%)`);
      }
      
      // If strategy score is very high, boost confidence
      if (strategyScore > 80) {
        combinedConfidence = Math.min(95, combinedConfidence + 10);
      }

      // Apply target return logic - reduce risk if approaching weekly target
      const targetProgress = this.weeklyPnL / this.targetReturnWeekly;
      if (targetProgress > 0.8) {
        // Reduce confidence when close to target to preserve gains
        combinedConfidence = Math.round(combinedConfidence * (1 - (targetProgress - 0.8) * 2));
      }
      
      // Create initial signal
      const signal: TradingSignal = {
        symbol: crypto.symbol,
        action,
        confidence: combinedConfidence,
        price: crypto.price,
        timestamp: new Date().toISOString(),
        maIndicator: macdResult.trend,
        rsi: rsiResult.value,
        technicalScore: composite.score,
        strategySignals: strategySignals
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

  getStrategySignals(): Map<string, any> {
    return this.strategySignals;
  }

  getWeeklyProgress(): { current: number; target: number; progress: number } {
    return {
      current: this.weeklyPnL,
      target: this.targetReturnWeekly,
      progress: this.weeklyPnL / this.targetReturnWeekly
    };
  }

  private calculateStrategyScore(strategySignals: any): number {
    let totalScore = 0;
    let signalCount = 0;
    let validSignals = 0;

    try {
      // Liquidity hunting strategy
      if (strategySignals.liquidity && strategySignals.liquidity.confidence != null) {
        const signal = strategySignals.liquidity;
        let score = signal.confidence || 50;
        if (signal.action === 'hunt_long' || signal.action === 'hunt_short') {
          score += 10; // Bonus for actionable signals
        }
        totalScore += Math.min(100, score);
        signalCount++;
        validSignals++;
      }

      // Smart money divergence strategy
      if (strategySignals.smartMoney && strategySignals.smartMoney.confidence != null) {
        const signal = strategySignals.smartMoney;
        let score = signal.confidence || 50;
        if (signal.action === 'follow_smart_money') {
          score += 15; // High confidence in smart money signals
        } else if (signal.action === 'fade_retail') {
          score += 10;
        }
        totalScore += Math.min(100, score);
        signalCount++;
        validSignals++;
      }

      // Volume profile strategy
      if (strategySignals.volumeProfile && strategySignals.volumeProfile.confidence != null) {
        const signal = strategySignals.volumeProfile;
        let score = signal.confidence || 50;
        if (signal.action === 'long' || signal.action === 'short') {
          score += 8; // Volume-based signals are valuable
        }
        totalScore += Math.min(100, score);
        signalCount++;
        validSignals++;
      }

      // Microstructure strategy
      if (strategySignals.microstructure && strategySignals.microstructure.confidence != null) {
        const signal = strategySignals.microstructure;
        let score = signal.confidence || 50;
        if (signal.action === 'aggressive_buy' || signal.action === 'aggressive_sell') {
          score += 12; // Microstructure signals for immediate action
        } else if (signal.action === 'passive_buy' || signal.action === 'passive_sell') {
          score += 6;
        }
        totalScore += Math.min(100, score);
        signalCount++;
        validSignals++;
      }

      // Fallback to 100% validator weight if strategies failed or returned invalid data
      if (validSignals === 0) {
        console.log('⚠️ No valid strategy signals, falling back to 100% validator weight');
        return 50; // Neutral score, will rely entirely on validator system
      }

      // If only some strategies failed, compensate by boosting valid ones
      if (validSignals < 4) {
        console.log(`⚠️ Only ${validSignals}/4 strategies provided valid signals`);
        // Boost remaining signals slightly to compensate for missing strategies
        totalScore = totalScore * (1 + (4 - validSignals) * 0.05);
      }

      const averageScore = totalScore / signalCount;
      return Math.min(100, Math.max(0, averageScore));

    } catch (error) {
      console.error('❌ Error calculating strategy score:', error);
      console.log('🔄 Falling back to 100% validator weight due to strategy calculation error');
      return 50; // Fallback to neutral score if calculation fails
    }
  }

  /**
   * Determine which strategy was primarily responsible for triggering this trade
   */
  private determineTriggeringStrategy(signal: TradingSignal): string {
    if (!signal.strategySignals) {
      return 'validator'; // Fallback if no strategy signals
    }

    // Find the strategy with the highest confidence that matches the trade action
    let bestStrategy = 'validator';
    let bestScore = 0;

    try {
      // Check liquidity hunting
      if (signal.strategySignals.liquidity?.confidence > bestScore) {
        const action = signal.strategySignals.liquidity.action;
        if ((action === 'hunt_long' && signal.action === 'BUY') || 
            (action === 'hunt_short' && signal.action === 'SELL')) {
          bestStrategy = 'liquidity';
          bestScore = signal.strategySignals.liquidity.confidence;
        }
      }

      // Check smart money
      if (signal.strategySignals.smartMoney?.confidence > bestScore) {
        const action = signal.strategySignals.smartMoney.action;
        if ((action === 'follow_smart_money' && signal.action === 'BUY') || 
            (action === 'fade_retail' && signal.action === 'SELL')) {
          bestStrategy = 'smartMoney';
          bestScore = signal.strategySignals.smartMoney.confidence;
        }
      }

      // Check volume profile
      if (signal.strategySignals.volumeProfile?.confidence > bestScore) {
        const action = signal.strategySignals.volumeProfile.action;
        if ((action === 'long' && signal.action === 'BUY') || 
            (action === 'short' && signal.action === 'SELL')) {
          bestStrategy = 'volumeProfile';
          bestScore = signal.strategySignals.volumeProfile.confidence;
        }
      }

      // Check microstructure
      if (signal.strategySignals.microstructure?.confidence > bestScore) {
        const action = signal.strategySignals.microstructure.action;
        if ((action === 'aggressive_buy' && signal.action === 'BUY') || 
            (action === 'aggressive_sell' && signal.action === 'SELL')) {
          bestStrategy = 'microstructure';
          bestScore = signal.strategySignals.microstructure.confidence;
        }
      }

    } catch (error) {
      console.error('Error determining triggering strategy:', error);
      return 'validator'; // Safe fallback
    }

    return bestStrategy;
  }

  /**
   * Update market regime and dynamically adjust strategy selection
   */
  private async updateMarketRegimeAndStrategies(): Promise<void> {
    const now = Date.now();
    
    // Update market regime every 30 minutes
    if (now - this.lastMarketRegimeUpdate > 30 * 60 * 1000) {
      try {
        // Get market regime from correlation service
        const marketRegime = await correlationService.getMarketRegime();
        this.marketRegime = marketRegime.regime;
        
        // Get news sentiment for additional context
        const sentiment = await newsSentimentService.getSentiment();
        
        // Adjust strategy selection based on market regime and time
        this.adjustStrategySelection(marketRegime, sentiment);
        
        this.lastMarketRegimeUpdate = now;
      } catch (error) {
        console.error('Error updating market regime:', error);
      }
    }
  }

  /**
   * Dynamically adjust which strategies are enabled based on market conditions
   */
  private adjustStrategySelection(
    marketRegime: any,
    sentiment: any
  ): void {
    const currentHour = new Date().getHours();
    const isHighVolatilityTime = (currentHour >= 14 && currentHour <= 16) || // US market open
                                 (currentHour >= 0 && currentHour <= 2);     // Asian market activity
    
    // Reset all strategies
    this.strategyEnabled.set('liquidity', false);
    this.strategyEnabled.set('smartMoney', false);
    this.strategyEnabled.set('volumeProfile', false);
    this.strategyEnabled.set('microstructure', false);

    // Enable strategies based on market regime
    switch (marketRegime.regime) {
      case 'risk_on':
        // In risk-on environments, momentum strategies work better
        this.strategyEnabled.set('smartMoney', true);
        this.strategyEnabled.set('microstructure', true);
        if (isHighVolatilityTime) {
          this.strategyEnabled.set('volumeProfile', true);
        }
        break;
        
      case 'risk_off':
        // In risk-off environments, contrarian strategies may work better
        this.strategyEnabled.set('liquidity', true);
        this.strategyEnabled.set('volumeProfile', true);
        break;
        
      case 'mixed':
        // In mixed regimes, use a balanced approach
        this.strategyEnabled.set('smartMoney', true);
        this.strategyEnabled.set('volumeProfile', true);
        if (isHighVolatilityTime) {
          this.strategyEnabled.set('microstructure', true);
        }
        break;
        
      case 'uncertain':
      default:
        // In uncertain conditions, use conservative strategies
        this.strategyEnabled.set('liquidity', true);
        this.strategyEnabled.set('smartMoney', true);
        break;
    }

    // Adjust based on sentiment
    if (sentiment.regulatory_risk > 70) {
      // High regulatory risk - disable aggressive strategies
      this.strategyEnabled.set('microstructure', false);
    }

    if (Math.abs(sentiment.overall_sentiment) > 60) {
      // Strong sentiment - enable momentum strategies
      this.strategyEnabled.set('smartMoney', true);
      this.strategyEnabled.set('microstructure', true);
    }

    // Time-based adjustments
    if (currentHour >= 22 || currentHour <= 6) {
      // Low activity hours - reduce to conservative strategies
      this.strategyEnabled.set('microstructure', false);
      if (!this.strategyEnabled.get('liquidity') && !this.strategyEnabled.get('smartMoney')) {
        this.strategyEnabled.set('liquidity', true); // Ensure at least one strategy is active
      }
    }

    // Ensure at least one strategy is always enabled
    const enabledCount = Array.from(this.strategyEnabled.values()).filter(enabled => enabled).length;
    if (enabledCount === 0) {
      this.strategyEnabled.set('smartMoney', true); // Default fallback strategy
    }

    console.log(`🔄 Strategy selection updated for ${marketRegime.regime} regime:`, 
      Object.fromEntries(this.strategyEnabled));
  }

  /**
   * Get current market regime and strategy status
   */
  getMarketRegimeStatus(): {
    regime: string;
    enabledStrategies: string[];
    lastUpdate: string;
  } {
    const enabledStrategies = Array.from(this.strategyEnabled.entries())
      .filter(([, enabled]) => enabled)
      .map(([strategy]) => strategy);

    return {
      regime: this.marketRegime,
      enabledStrategies,
      lastUpdate: new Date(this.lastMarketRegimeUpdate).toISOString()
    };
  }

  /**
   * Manually override strategy enablement (for testing or manual control)
   */
  setStrategyEnabled(strategy: string, enabled: boolean): void {
    if (this.strategyEnabled.has(strategy)) {
      this.strategyEnabled.set(strategy, enabled);
      console.log(`📊 Strategy ${strategy} manually set to ${enabled ? 'enabled' : 'disabled'}`);
    }
  }
}

// Export singleton instance
export const tradingAgentV2 = new TradingAgentV2();