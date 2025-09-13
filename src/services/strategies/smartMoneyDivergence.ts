/**
 * Smart Money Divergence Strategy
 * Analyzes whale movements vs retail behavior to identify divergences between institutional and retail positioning
 * Uses whaleAlertService for whale movements and etherscanService for wallet analysis
 */

import { whaleAlertService, WhaleTransaction, WhaleMovement } from '../whaleAlertService';
import { etherscanService, ActiveAddress, WalletBalance } from '../etherscanService';
import { TechnicalIndicators, PriceData } from '../technicalIndicators';
import { ValidationResult } from '../validatorSystem';

export interface SmartMoneyMetrics {
  whaleActivity: {
    totalVolume: number;
    totalTransactions: number;
    netFlow: number; // Positive = accumulation, Negative = distribution
    avgTransactionSize: number;
    exchangeFlows: {
      inflows: number;
      outflows: number;
      netFlow: number;
    };
  };
  retailActivity: {
    activeAddresses: number;
    avgBalance: number;
    transactionCount: number;
    avgTransactionSize: number;
  };
  divergenceMetrics: {
    volumeDivergence: number; // -100 to 100
    flowDivergence: number; // -100 to 100
    behaviorDivergence: number; // -100 to 100
    timeframeDivergence: number; // -100 to 100
  };
}

export interface SmartMoneySignal {
  symbol: string;
  action: 'follow_smart_money' | 'fade_retail' | 'wait' | 'neutral';
  divergenceScore: number; // 0-100, higher = stronger divergence
  smartMoneyBias: 'bullish' | 'bearish' | 'neutral';
  retailBias: 'bullish' | 'bearish' | 'neutral';
  confidence: number; // 0-100
  timeframe: string;
  timestamp: string;
  
  signals: {
    whaleAccumulation: boolean;
    exchangeOutflows: boolean;
    retailCapitulation: boolean;
    smartMoneyReversal: boolean;
  };
  
  metrics: SmartMoneyMetrics;
  reasoning: string[];
  
  riskFactors: {
    manipulationRisk: number; // 0-100
    falseSignalRisk: number; // 0-100
    liquidityRisk: number; // 0-100
  };
  
  actionPlan: {
    entry: {
      condition: string;
      price?: number;
      timing: 'immediate' | 'on_confirmation' | 'wait_for_setup';
    };
    exit: {
      target: number;
      stopLoss: number;
      timeLimit: string;
    };
  };
}

export interface DivergenceAnalysisResult {
  signal: SmartMoneySignal | null;
  metrics: SmartMoneyMetrics;
  marketPhase: 'accumulation' | 'distribution' | 'trending' | 'consolidation';
  divergenceStrength: 'weak' | 'moderate' | 'strong' | 'extreme';
  keyLevels: {
    smartMoneySupport: number[];
    smartMoneyResistance: number[];
    retailSupport: number[];
    retailResistance: number[];
  };
}

class SmartMoneyDivergenceStrategy {
  private cache: Map<string, { data: any; timestamp: number }>;
  private readonly CACHE_DURATION = 300000; // 5 minutes
  private readonly WHALE_THRESHOLD = 1000000; // $1M+ transactions considered "whale"
  private readonly RETAIL_THRESHOLD = 10000; // $10K- transactions considered "retail"

  constructor() {
    this.cache = new Map();
  }

  private getCacheKey(symbol: string, timeframe: string): string {
    return `smart_money_${symbol}_${timeframe}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
    
    // Clean old cache entries
    if (this.cache.size > 50) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < 10; i++) {
        this.cache.delete(keys[i]);
      }
    }
  }

  /**
   * Main analysis function
   */
  async analyze(
    symbol: string,
    priceData: PriceData[],
    timeframe: '1h' | '4h' | '24h' | '7d' = '24h'
  ): Promise<DivergenceAnalysisResult> {
    try {
      console.log(`🧠 Analyzing smart money divergence for ${symbol} (${timeframe})`);

      const cacheKey = this.getCacheKey(symbol, timeframe);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.log('📋 Using cached smart money divergence analysis');
        return cached;
      }

      // Step 1: Gather whale activity data
      const whaleActivity = await this.analyzeWhaleActivity(symbol, timeframe);

      // Step 2: Gather retail activity data  
      const retailActivity = await this.analyzeRetailActivity(timeframe);

      // Step 3: Calculate divergence metrics
      const divergenceMetrics = this.calculateDivergenceMetrics(whaleActivity, retailActivity, priceData);

      // Step 4: Determine market phase
      const marketPhase = this.determineMarketPhase(whaleActivity, retailActivity, priceData);

      // Step 5: Assess divergence strength
      const divergenceStrength = this.assessDivergenceStrength(divergenceMetrics);

      // Step 6: Identify key levels
      const keyLevels = await this.identifyKeyLevels(symbol, whaleActivity, priceData);

      // Step 7: Generate signal
      const signal = await this.generateSmartMoneySignal(
        symbol,
        { whaleActivity, retailActivity, divergenceMetrics },
        marketPhase,
        divergenceStrength,
        priceData,
        timeframe
      );

      const result: DivergenceAnalysisResult = {
        signal,
        metrics: { whaleActivity, retailActivity, divergenceMetrics },
        marketPhase,
        divergenceStrength,
        keyLevels
      };

      this.setCache(cacheKey, result);
      console.log(`✅ Smart money divergence analysis completed for ${symbol}`);
      
      return result;
    } catch (error) {
      console.error('❌ Smart money divergence analysis failed:', error);
      return {
        signal: null,
        metrics: this.getDefaultMetrics(),
        marketPhase: 'consolidation',
        divergenceStrength: 'weak',
        keyLevels: {
          smartMoneySupport: [],
          smartMoneyResistance: [],
          retailSupport: [],
          retailResistance: []
        }
      };
    }
  }

  /**
   * Analyze whale activity patterns
   */
  private async analyzeWhaleActivity(symbol: string, timeframe: string) {
    const whaleTransactions = await whaleAlertService.getLargeTransactions(this.WHALE_THRESHOLD);
    const whaleMovements = await whaleAlertService.getWhaleMovements(timeframe as any);

    // Filter for relevant symbol/blockchain
    const relevantTransactions = whaleTransactions.filter(tx => 
      tx.symbol.toLowerCase() === symbol.toLowerCase() || 
      tx.blockchain === 'ethereum' // Fallback for ETH-based tokens
    );

    const relevantMovements = whaleMovements.filter(mv =>
      mv.symbol.toLowerCase() === symbol.toLowerCase()
    );

    // Calculate metrics
    const totalVolume = relevantTransactions.reduce((sum, tx) => sum + tx.amount_usd, 0);
    const totalTransactions = relevantTransactions.length;
    const avgTransactionSize = totalVolume / Math.max(1, totalTransactions);

    // Calculate net flow (exchange flows)
    const exchangeInflows = relevantMovements
      .filter(mv => mv.classification === 'whale_to_exchange')
      .reduce((sum, mv) => sum + mv.amount_usd, 0);

    const exchangeOutflows = relevantMovements
      .filter(mv => mv.classification === 'exchange_to_whale')
      .reduce((sum, mv) => sum + mv.amount_usd, 0);

    const netExchangeFlow = exchangeOutflows - exchangeInflows; // Positive = accumulation
    const netFlow = this.calculateNetFlow(relevantTransactions);

    return {
      totalVolume,
      totalTransactions,
      netFlow,
      avgTransactionSize,
      exchangeFlows: {
        inflows: exchangeInflows,
        outflows: exchangeOutflows,
        netFlow: netExchangeFlow
      }
    };
  }

  /**
   * Analyze retail activity patterns
   */
  private async analyzeRetailActivity(timeframe: string) {
    const activeAddresses = await etherscanService.getActiveAddresses(timeframe as any);
    const largeTransactions = await etherscanService.getLargeTransactions(1); // 1+ ETH

    // Filter for retail-sized transactions (< $10K equivalent)
    const retailTransactions = largeTransactions.filter(tx => {
      const ethValue = parseFloat(tx.value);
      const usdValue = ethValue * 3000; // Approximate ETH price
      return usdValue < this.RETAIL_THRESHOLD;
    });

    // Calculate retail metrics
    const totalRetailAddresses = activeAddresses.length;
    const avgBalance = activeAddresses.reduce((sum, addr) => sum + addr.transactionCount, 0) / Math.max(1, totalRetailAddresses);
    const transactionCount = retailTransactions.length;
    const avgTransactionSize = retailTransactions.reduce((sum, tx) => sum + parseFloat(tx.value), 0) / Math.max(1, transactionCount);

    return {
      activeAddresses: totalRetailAddresses,
      avgBalance,
      transactionCount,
      avgTransactionSize
    };
  }

  /**
   * Calculate divergence metrics between smart money and retail
   */
  private calculateDivergenceMetrics(
    whaleActivity: any,
    retailActivity: any,
    priceData: PriceData[]
  ) {
    const currentPrice = priceData[priceData.length - 1].close;
    const priceChange = priceData.length > 1 ? 
      (currentPrice - priceData[priceData.length - 2].close) / priceData[priceData.length - 2].close * 100 : 0;

    // Volume divergence: Compare whale vs retail volume trends
    const whaleVolumeIntensity = whaleActivity.totalVolume / Math.max(1, whaleActivity.totalTransactions);
    const retailVolumeIntensity = retailActivity.avgTransactionSize * retailActivity.transactionCount;
    
    const volumeDivergence = this.calculateNormalizedDivergence(
      whaleVolumeIntensity,
      retailVolumeIntensity,
      'volume'
    );

    // Flow divergence: Net flows vs price action
    const whaleFlowDirection = whaleActivity.netFlow > 0 ? 1 : -1;
    const priceDirection = priceChange > 0 ? 1 : -1;
    
    const flowDivergence = whaleFlowDirection === priceDirection ? 0 : 
      Math.abs(whaleActivity.netFlow) / 1000000 * 50; // Scale to 0-100

    // Behavior divergence: Exchange flows vs price
    const exchangeFlowDirection = whaleActivity.exchangeFlows.netFlow > 0 ? 1 : -1; // Positive = accumulation
    const behaviorDivergence = exchangeFlowDirection === priceDirection ? 0 :
      Math.abs(whaleActivity.exchangeFlows.netFlow) / 1000000 * 30;

    // Timeframe divergence: Short vs long term behavior
    const recentPriceChange = priceChange;
    const longerTermChange = priceData.length > 5 ?
      (currentPrice - priceData[priceData.length - 6].close) / priceData[priceData.length - 6].close * 100 : 0;
    
    const timeframeDivergence = Math.abs(recentPriceChange - longerTermChange) * 2;

    return {
      volumeDivergence: Math.min(100, Math.max(-100, volumeDivergence)),
      flowDivergence: Math.min(100, Math.max(-100, flowDivergence)),
      behaviorDivergence: Math.min(100, Math.max(-100, behaviorDivergence)),
      timeframeDivergence: Math.min(100, Math.max(0, timeframeDivergence))
    };
  }

  /**
   * Determine current market phase based on smart money vs retail behavior
   */
  private determineMarketPhase(whaleActivity: any, retailActivity: any, priceData: PriceData[]): 
    'accumulation' | 'distribution' | 'trending' | 'consolidation' {
    
    const netWhaleFlow = whaleActivity.exchangeFlows.netFlow;
    const priceVolatility = TechnicalIndicators.ATR(priceData, 14);
    const currentPrice = priceData[priceData.length - 1].close;
    const volatilityPercent = (priceVolatility / currentPrice) * 100;

    // Accumulation: Whales buying, low volatility
    if (netWhaleFlow > 0 && volatilityPercent < 3 && whaleActivity.totalVolume > retailActivity.transactionCount * 1000) {
      return 'accumulation';
    }

    // Distribution: Whales selling, increasing retail activity
    if (netWhaleFlow < 0 && retailActivity.activeAddresses > 100 && whaleActivity.exchangeFlows.inflows > whaleActivity.exchangeFlows.outflows) {
      return 'distribution';
    }

    // Trending: Consistent directional movement
    if (volatilityPercent > 5 && Math.abs(netWhaleFlow) > 1000000) {
      return 'trending';
    }

    // Default to consolidation
    return 'consolidation';
  }

  /**
   * Assess the strength of the divergence
   */
  private assessDivergenceStrength(divergenceMetrics: any): 'weak' | 'moderate' | 'strong' | 'extreme' {
    const avgDivergence = (
      Math.abs(divergenceMetrics.volumeDivergence) +
      Math.abs(divergenceMetrics.flowDivergence) +
      Math.abs(divergenceMetrics.behaviorDivergence) +
      Math.abs(divergenceMetrics.timeframeDivergence)
    ) / 4;

    if (avgDivergence > 75) return 'extreme';
    if (avgDivergence > 50) return 'strong';
    if (avgDivergence > 25) return 'moderate';
    return 'weak';
  }

  /**
   * Identify key support/resistance levels for smart money vs retail
   */
  private async identifyKeyLevels(symbol: string, whaleActivity: any, priceData: PriceData[]) {
    // Get large whale transactions and their price levels
    const whaleTransactions = await whaleAlertService.getLargeTransactions(this.WHALE_THRESHOLD);
    const relevantTransactions = whaleTransactions.filter(tx => 
      tx.symbol.toLowerCase() === symbol.toLowerCase()
    );

    // Extract price levels from whale activity (simplified - in real implementation, 
    // you'd need price data at transaction times)
    const smartMoneyLevels = relevantTransactions
      .map(tx => tx.amount_usd / tx.amount) // Approximate price
      .filter(price => price > 0 && price < 1000000) // Reasonable price range
      .sort((a, b) => a - b);

    // Calculate support/resistance from technical analysis
    const supportResistance = TechnicalIndicators.SupportResistance(priceData, 20);

    // Split smart money levels into support/resistance
    const currentPrice = priceData[priceData.length - 1].close;
    const smartMoneySupport = smartMoneyLevels.filter(level => level < currentPrice).slice(-3);
    const smartMoneyResistance = smartMoneyLevels.filter(level => level > currentPrice).slice(0, 3);

    return {
      smartMoneySupport,
      smartMoneyResistance,
      retailSupport: supportResistance.support.slice(0, 3),
      retailResistance: supportResistance.resistance.slice(0, 3)
    };
  }

  /**
   * Generate smart money signal based on analysis
   */
  private async generateSmartMoneySignal(
    symbol: string,
    metrics: { whaleActivity: any; retailActivity: any; divergenceMetrics: any },
    marketPhase: string,
    divergenceStrength: string,
    priceData: PriceData[],
    timeframe: string
  ): Promise<SmartMoneySignal | null> {

    const { whaleActivity, retailActivity, divergenceMetrics } = metrics;
    
    // Don't generate signals for weak divergences
    if (divergenceStrength === 'weak') {
      return null;
    }

    // Calculate divergence score
    const divergenceScore = (
      Math.abs(divergenceMetrics.volumeDivergence) * 0.3 +
      Math.abs(divergenceMetrics.flowDivergence) * 0.4 +
      Math.abs(divergenceMetrics.behaviorDivergence) * 0.2 +
      Math.abs(divergenceMetrics.timeframeDivergence) * 0.1
    );

    // Determine biases
    const smartMoneyBias = whaleActivity.netFlow > 0 ? 'bullish' : 
                          whaleActivity.netFlow < 0 ? 'bearish' : 'neutral';
    
    const retailBias = retailActivity.activeAddresses > 150 ? 'bullish' : 
                       retailActivity.activeAddresses < 50 ? 'bearish' : 'neutral';

    // Determine action based on divergence and market phase
    let action: SmartMoneySignal['action'] = 'neutral';
    let confidence = 50;

    if (marketPhase === 'accumulation' && smartMoneyBias === 'bullish' && retailBias !== 'bullish') {
      action = 'follow_smart_money';
      confidence = 75;
    } else if (marketPhase === 'distribution' && smartMoneyBias === 'bearish' && retailBias === 'bullish') {
      action = 'follow_smart_money';
      confidence = 70;
    } else if (divergenceScore > 60 && smartMoneyBias !== retailBias) {
      action = 'fade_retail';
      confidence = 65;
    } else if (divergenceScore > 40) {
      action = 'wait';
      confidence = 55;
    }

    // Don't generate low-confidence signals
    if (confidence < 60) {
      return null;
    }

    // Calculate risk factors
    const manipulationRisk = Math.min(100, whaleActivity.totalVolume / 10000000 * 30); // Higher volume = higher manipulation risk
    const falseSignalRisk = divergenceStrength === 'extreme' ? 60 : 
                           divergenceStrength === 'strong' ? 40 : 30;
    const liquidityRisk = Math.max(0, 50 - retailActivity.activeAddresses);

    // Identify specific signals
    const signals = {
      whaleAccumulation: whaleActivity.exchangeFlows.netFlow > 1000000,
      exchangeOutflows: whaleActivity.exchangeFlows.outflows > whaleActivity.exchangeFlows.inflows * 1.5,
      retailCapitulation: retailActivity.activeAddresses < 50 && smartMoneyBias === 'bullish',
      smartMoneyReversal: Math.abs(divergenceMetrics.flowDivergence) > 70
    };

    const currentPrice = priceData[priceData.length - 1].close;
    const atr = TechnicalIndicators.ATR(priceData, 14);

    // Create action plan
    const actionPlan = {
      entry: {
        condition: action === 'follow_smart_money' ? 
          'Enter when price confirms smart money direction' : 
          'Wait for retail sentiment reversal',
        timing: divergenceStrength === 'extreme' ? 'immediate' : 'on_confirmation' as const
      },
      exit: {
        target: action === 'follow_smart_money' && smartMoneyBias === 'bullish' ? 
          currentPrice + (atr * 2) : 
          currentPrice - (atr * 2),
        stopLoss: action === 'follow_smart_money' && smartMoneyBias === 'bullish' ?
          currentPrice - (atr * 1.5) :
          currentPrice + (atr * 1.5),
        timeLimit: `${timeframe}_period`
      }
    };

    const reasoning = [
      `Smart money ${smartMoneyBias} vs retail ${retailBias} creates ${divergenceScore.toFixed(0)}% divergence`,
      `Market phase: ${marketPhase} with ${divergenceStrength} divergence strength`,
      `Whale net flow: ${whaleActivity.netFlow > 0 ? '+' : ''}${(whaleActivity.netFlow / 1000000).toFixed(2)}M`,
      `Exchange flows: ${whaleActivity.exchangeFlows.netFlow > 0 ? 'Accumulation' : 'Distribution'}`,
      `Retail activity: ${retailActivity.activeAddresses} active addresses`
    ];

    return {
      symbol,
      action,
      divergenceScore,
      smartMoneyBias,
      retailBias,
      confidence,
      timeframe,
      timestamp: new Date().toISOString(),
      signals,
      metrics: { whaleActivity, retailActivity, divergenceMetrics },
      reasoning,
      riskFactors: {
        manipulationRisk,
        falseSignalRisk,
        liquidityRisk
      },
      actionPlan
    };
  }

  // Helper methods
  private calculateNetFlow(transactions: WhaleTransaction[]): number {
    return transactions.reduce((net, tx) => {
      // Simplified: assume transactions to exchanges are selling, from exchanges are buying
      const isToExchange = tx.to.owner_type === 'exchange';
      const isFromExchange = tx.from.owner_type === 'exchange';
      
      if (isFromExchange && !isToExchange) {
        return net + tx.amount_usd; // Buying pressure
      } else if (isToExchange && !isFromExchange) {
        return net - tx.amount_usd; // Selling pressure
      }
      return net;
    }, 0);
  }

  private calculateNormalizedDivergence(value1: number, value2: number, type: string): number {
    if (value1 === 0 && value2 === 0) return 0;
    
    const ratio = value1 / Math.max(value2, 1);
    
    // Convert ratio to divergence score (-100 to 100)
    if (ratio > 1) {
      return Math.min(100, (ratio - 1) * 50);
    } else {
      return Math.max(-100, (ratio - 1) * 50);
    }
  }

  private getDefaultMetrics(): SmartMoneyMetrics {
    return {
      whaleActivity: {
        totalVolume: 0,
        totalTransactions: 0,
        netFlow: 0,
        avgTransactionSize: 0,
        exchangeFlows: {
          inflows: 0,
          outflows: 0,
          netFlow: 0
        }
      },
      retailActivity: {
        activeAddresses: 0,
        avgBalance: 0,
        transactionCount: 0,
        avgTransactionSize: 0
      },
      divergenceMetrics: {
        volumeDivergence: 0,
        flowDivergence: 0,
        behaviorDivergence: 0,
        timeframeDivergence: 0
      }
    };
  }
}

export const smartMoneyDivergenceStrategy = new SmartMoneyDivergenceStrategy();
export default SmartMoneyDivergenceStrategy;