import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { TradingSignal, MarketData, TradingSettings } from '../../types/trading';
import axios from 'axios';

interface ValidationResult {
  isValid: boolean;
  reason?: string;
  confidence: number;
}

interface StrategySignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strategy: string;
  reasoning: string;
  indicators: any;
}

export class SignalProcessor extends EventEmitter {
  private mlServiceUrl: string;
  private rlServiceUrl: string;
  private validatorThreshold: number = 0.6;
  private strategyWeights: Map<string, number> = new Map();

  constructor() {
    super();
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:5001';
    this.rlServiceUrl = process.env.RL_SERVICE_URL || 'http://localhost:8000';

    // Initialize strategy weights
    this.strategyWeights.set('liquidity', 0.25);
    this.strategyWeights.set('smartMoney', 0.25);
    this.strategyWeights.set('volumeProfile', 0.25);
    this.strategyWeights.set('microstructure', 0.25);
  }

  async generateSignals(
    marketData: MarketData[],
    settings: TradingSettings,
    marketRegime: any
  ): Promise<TradingSignal[]> {
    try {
      const signals: TradingSignal[] = [];

      for (const data of marketData) {
        // Skip if no significant price movement
        if (Math.abs(data.changePercent || 0) < 0.5) continue;

        // Get signals from different sources
        const [strategySignals, mlSignal, rlSignal] = await Promise.all([
          this.getStrategySignals(data, settings),
          this.getMLSignal(data, marketRegime),
          this.getRLSignal(data, marketRegime)
        ]);

        // Combine signals with weighted average
        const combinedSignal = this.combineSignals(
          data.symbol,
          strategySignals,
          mlSignal,
          rlSignal,
          settings
        );

        if (combinedSignal && combinedSignal.action !== 'HOLD') {
          signals.push(combinedSignal);
        }
      }

      // Sort by confidence
      signals.sort((a, b) => b.confidence - a.confidence);

      // Emit signal generation event
      this.emit('signalsGenerated', {
        count: signals.length,
        topSignals: signals.slice(0, 3)
      });

      return signals;

    } catch (error) {
      logger.error('Error generating signals:', error);
      return [];
    }
  }

  private async getStrategySignals(
    data: MarketData,
    settings: TradingSettings
  ): Promise<StrategySignal[]> {
    const signals: StrategySignal[] = [];

    // Calculate technical indicators
    const indicators = this.calculateIndicators(data);

    // Check each enabled strategy
    if (settings.enabledStrategies?.includes('liquidity')) {
      const signal = this.liquidityHuntingSignal(data, indicators);
      if (signal) signals.push(signal);
    }

    if (settings.enabledStrategies?.includes('smartMoney')) {
      const signal = this.smartMoneyDivergenceSignal(data, indicators);
      if (signal) signals.push(signal);
    }

    if (settings.enabledStrategies?.includes('volumeProfile')) {
      const signal = this.volumeProfileSignal(data, indicators);
      if (signal) signals.push(signal);
    }

    if (settings.enabledStrategies?.includes('microstructure')) {
      const signal = this.microstructureSignal(data, indicators);
      if (signal) signals.push(signal);
    }

    return signals;
  }

  private liquidityHuntingSignal(data: MarketData, indicators: any): StrategySignal | null {
    // Simplified liquidity hunting logic
    const volumeSpike = (data.volume || 0) > (indicators.avgVolume * 2);
    const priceReversal = indicators.rsi < 30 || indicators.rsi > 70;

    if (volumeSpike && priceReversal) {
      return {
        symbol: data.symbol,
        action: indicators.rsi < 30 ? 'BUY' : 'SELL',
        confidence: 0.7,
        strategy: 'liquidity',
        reasoning: 'Liquidity hunt detected with volume spike and price reversal',
        indicators: {
          volumeRatio: data.volume / indicators.avgVolume,
          rsi: indicators.rsi
        }
      };
    }

    return null;
  }

  private smartMoneyDivergenceSignal(data: MarketData, indicators: any): StrategySignal | null {
    // Simplified smart money divergence logic
    const priceTrend = indicators.ma20 > indicators.ma50 ? 'up' : 'down';
    const volumeTrend = data.volume > indicators.avgVolume ? 'up' : 'down';
    const divergence = priceTrend !== volumeTrend;

    if (divergence && Math.abs(indicators.macd) > 0.5) {
      return {
        symbol: data.symbol,
        action: volumeTrend === 'up' && priceTrend === 'down' ? 'BUY' : 'SELL',
        confidence: 0.65,
        strategy: 'smartMoney',
        reasoning: 'Smart money divergence detected between price and volume',
        indicators: {
          priceTrend,
          volumeTrend,
          macd: indicators.macd
        }
      };
    }

    return null;
  }

  private volumeProfileSignal(data: MarketData, indicators: any): StrategySignal | null {
    // Simplified volume profile logic
    const volumeNode = this.identifyVolumeNode(data, indicators);

    if (volumeNode) {
      const nearSupport = Math.abs(data.price - volumeNode.support) / data.price < 0.02;
      const nearResistance = Math.abs(data.price - volumeNode.resistance) / data.price < 0.02;

      if (nearSupport || nearResistance) {
        return {
          symbol: data.symbol,
          action: nearSupport ? 'BUY' : 'SELL',
          confidence: 0.6,
          strategy: 'volumeProfile',
          reasoning: `Price near volume ${nearSupport ? 'support' : 'resistance'} level`,
          indicators: {
            volumeNode,
            priceLevel: data.price
          }
        };
      }
    }

    return null;
  }

  private microstructureSignal(data: MarketData, indicators: any): StrategySignal | null {
    // Simplified microstructure analysis
    const orderFlowImbalance = this.calculateOrderFlowImbalance(data);
    const bidAskSpread = (data.ask - data.bid) / data.price;

    if (Math.abs(orderFlowImbalance) > 0.3 && bidAskSpread < 0.002) {
      return {
        symbol: data.symbol,
        action: orderFlowImbalance > 0 ? 'BUY' : 'SELL',
        confidence: 0.55,
        strategy: 'microstructure',
        reasoning: 'Order flow imbalance detected with tight spread',
        indicators: {
          orderFlowImbalance,
          bidAskSpread
        }
      };
    }

    return null;
  }

  private async getMLSignal(data: MarketData, marketRegime: any): Promise<StrategySignal | null> {
    try {
      const response = await axios.post(
        `${this.mlServiceUrl}/predict`,
        {
          symbol: data.symbol,
          features: {
            price: data.price,
            volume: data.volume,
            changePercent: data.changePercent,
            marketCap: data.marketCap,
            regime: marketRegime.type
          }
        },
        { timeout: 2000 }
      );

      if (response.data.success) {
        return {
          symbol: data.symbol,
          action: response.data.prediction.action,
          confidence: response.data.prediction.confidence,
          strategy: 'ml_ensemble',
          reasoning: response.data.prediction.reasoning,
          indicators: response.data.prediction.features
        };
      }
    } catch (error) {
      logger.debug('ML service unavailable or error:', error);
    }

    return null;
  }

  private async getRLSignal(data: MarketData, marketRegime: any): Promise<StrategySignal | null> {
    try {
      const response = await axios.post(
        `${this.rlServiceUrl}/api/signal`,
        {
          symbol: data.symbol,
          state: {
            price: data.price,
            volume: data.volume,
            volatility: data.volatility || 0,
            regime: marketRegime.type
          }
        },
        { timeout: 2000 }
      );

      if (response.data) {
        return {
          symbol: data.symbol,
          action: response.data.action,
          confidence: response.data.confidence || 0.5,
          strategy: 'rl_agent',
          reasoning: response.data.reasoning || 'RL agent decision',
          indicators: response.data.state
        };
      }
    } catch (error) {
      logger.debug('RL service unavailable or error:', error);
    }

    return null;
  }

  private combineSignals(
    symbol: string,
    strategySignals: StrategySignal[],
    mlSignal: StrategySignal | null,
    rlSignal: StrategySignal | null,
    settings: TradingSettings
  ): TradingSignal | null {
    const allSignals = [...strategySignals];
    if (mlSignal) allSignals.push(mlSignal);
    if (rlSignal) allSignals.push(rlSignal);

    if (allSignals.length === 0) return null;

    // Count votes for each action
    const votes = { BUY: 0, SELL: 0, HOLD: 0 };
    const confidences = { BUY: [], SELL: [], HOLD: [] } as Record<string, number[]>;
    const reasons: string[] = [];

    for (const signal of allSignals) {
      const weight = this.strategyWeights.get(signal.strategy) || 0.2;
      votes[signal.action] += weight;
      confidences[signal.action].push(signal.confidence * weight);
      reasons.push(`${signal.strategy}: ${signal.reasoning}`);
    }

    // Determine winning action
    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let maxVotes = 0;

    for (const [act, voteCount] of Object.entries(votes)) {
      if (voteCount > maxVotes) {
        maxVotes = voteCount;
        action = act as 'BUY' | 'SELL' | 'HOLD';
      }
    }

    // Calculate weighted confidence
    const actionConfidences = confidences[action];
    const avgConfidence = actionConfidences.length > 0
      ? actionConfidences.reduce((a, b) => a + b, 0) / actionConfidences.length
      : 0;

    // Apply strategy vs validator balance
    const finalConfidence = avgConfidence * (settings.strategyWeightBalance || 0.5);

    return {
      symbol,
      action,
      confidence: finalConfidence,
      price: allSignals[0]?.indicators?.price || 0,
      reasoning: reasons.join('; '),
      timestamp: new Date(),
      metadata: {
        signalCount: allSignals.length,
        strategies: allSignals.map(s => s.strategy),
        votes
      }
    };
  }

  async validateSignal(signal: TradingSignal, settings: TradingSettings): Promise<ValidationResult> {
    try {
      // Run validation checks
      const checks = await Promise.all([
        this.validateRiskReward(signal),
        this.validateMarketConditions(signal),
        this.validateTechnicalSetup(signal),
        this.validateFundamentals(signal)
      ]);

      // Count passed validations
      const passedChecks = checks.filter(c => c.isValid).length;
      const totalChecks = checks.length;
      const validationScore = passedChecks / totalChecks;

      // Check if validators are enabled and meet threshold
      if (settings.validatorEnabled && validationScore < this.validatorThreshold) {
        return {
          isValid: false,
          reason: `Validation score ${validationScore.toFixed(2)} below threshold ${this.validatorThreshold}`,
          confidence: validationScore
        };
      }

      return {
        isValid: true,
        confidence: validationScore
      };

    } catch (error) {
      logger.error('Error validating signal:', error);
      return {
        isValid: false,
        reason: 'Validation error',
        confidence: 0
      };
    }
  }

  private async validateRiskReward(signal: TradingSignal): Promise<ValidationResult> {
    // Simple risk/reward validation
    const minRiskReward = 2; // Require 2:1 risk/reward ratio

    // This would normally calculate based on stop loss and take profit levels
    const estimatedRiskReward = signal.confidence * 3; // Simplified calculation

    return {
      isValid: estimatedRiskReward >= minRiskReward,
      reason: `Risk/reward ratio: ${estimatedRiskReward.toFixed(2)}`,
      confidence: Math.min(estimatedRiskReward / minRiskReward, 1)
    };
  }

  private async validateMarketConditions(signal: TradingSignal): Promise<ValidationResult> {
    // Check if market conditions are favorable
    // This would normally check volatility, liquidity, etc.
    return {
      isValid: true,
      confidence: 0.8
    };
  }

  private async validateTechnicalSetup(signal: TradingSignal): Promise<ValidationResult> {
    // Validate technical indicators alignment
    return {
      isValid: signal.confidence > 0.6,
      confidence: signal.confidence
    };
  }

  private async validateFundamentals(signal: TradingSignal): Promise<ValidationResult> {
    // Check fundamental factors
    // This would normally check news, sentiment, etc.
    return {
      isValid: true,
      confidence: 0.7
    };
  }

  private calculateIndicators(data: MarketData): any {
    // Simplified indicator calculations
    // In production, these would use proper technical analysis libraries
    return {
      rsi: 50 + (Math.random() * 50 - 25), // Mock RSI
      macd: Math.random() * 2 - 1, // Mock MACD
      ma20: data.price * 0.98, // Mock 20-day MA
      ma50: data.price * 0.95, // Mock 50-day MA
      avgVolume: data.volume * 0.8, // Mock average volume
      volatility: Math.abs(data.changePercent || 0) / 100
    };
  }

  private identifyVolumeNode(data: MarketData, indicators: any): any {
    // Simplified volume node identification
    return {
      support: data.price * 0.95,
      resistance: data.price * 1.05,
      volume: data.volume
    };
  }

  private calculateOrderFlowImbalance(data: MarketData): number {
    // Simplified order flow imbalance calculation
    // In production, this would use real order book data
    return (Math.random() * 2 - 1) * 0.5;
  }

  updateStrategyWeights(weights: Record<string, number>): void {
    for (const [strategy, weight] of Object.entries(weights)) {
      this.strategyWeights.set(strategy, weight);
    }
  }

  setValidatorThreshold(threshold: number): void {
    this.validatorThreshold = Math.max(0, Math.min(1, threshold));
  }
}

export default SignalProcessor;