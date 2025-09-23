/**
 * Microstructure Analysis Strategy
 * Analyzes DEX order flow, bid/ask imbalances, and large order impacts using bitqueryService
 * Provides high-frequency trading insights and market microstructure signals
 */

import { bitqueryService, DEXTrade } from '../bitqueryService';
import { TechnicalIndicators, PriceData } from '../technicalIndicators';
import { ValidationResult } from '../validatorSystem';

export interface OrderFlowData {
  timestamp: string;
  exchange: string;
  side: 'buy' | 'sell';
  size: number;
  sizeUSD: number;
  price: number;
  impact: number; // Price impact percentage
  urgency: 'low' | 'medium' | 'high' | 'extreme';
}

export interface BidAskImbalance {
  timestamp: string;
  bidVolume: number;
  askVolume: number;
  imbalanceRatio: number; // bid/ask ratio
  imbalanceScore: number; // -100 to 100 (negative = sell pressure, positive = buy pressure)
  depth: {
    bid: { levels: number; totalSize: number };
    ask: { levels: number; totalSize: number };
  };
}

export interface LargeOrderImpact {
  orderId: string;
  timestamp: string;
  side: 'buy' | 'sell';
  size: number;
  sizeUSD: number;
  priceImpact: number; // Percentage price movement caused
  recoveryTime: number; // Minutes to price recovery
  sustainedImpact: boolean; // Whether impact lasted > 5 minutes
  classification: 'whale_order' | 'institutional' | 'liquidation' | 'manipulation';
}

export interface MarketMicrostructure {
  orderFlowImbalance: number; // -100 to 100
  averageTradeSize: number;
  tradeFrequency: number; // trades per minute
  priceEfficiency: number; // 0-100, how quickly price adjusts to information
  liquidityDepth: number; // 0-100
  marketImpactCost: number; // Basis points for 1% of daily volume
}

export interface MicrostructureSignal {
  symbol: string;
  action: 'aggressive_buy' | 'aggressive_sell' | 'passive_buy' | 'passive_sell' | 'wait' | 'avoid';
  urgency: 'low' | 'medium' | 'high' | 'immediate';
  confidence: number; // 0-100
  signalType: 'order_flow_imbalance' | 'large_order_impact' | 'liquidity_gap' | 'momentum_shift' | 'mean_reversion';
  
  microstructure: MarketMicrostructure;
  
  timing: {
    entryWindow: number; // Minutes
    optimalExecutionStyle: 'market' | 'limit' | 'iceberg' | 'twap';
    expectedSlippage: number; // Basis points
    maxPosition: number; // Max position size without significant impact
  };
  
  orderFlow: {
    recentImbalance: BidAskImbalance;
    largeOrders: LargeOrderImpact[];
    flowDirection: 'bullish' | 'bearish' | 'neutral';
    intensity: number; // 0-100
  };
  
  riskFactors: {
    liquidityRisk: number; // 0-100
    manipulationRisk: number; // 0-100
    latencyRisk: number; // 0-100
    slippageRisk: number; // 0-100
  };
  
  executionPlan: {
    orderType: 'market' | 'limit' | 'stop' | 'iceberg';
    priceLevel: number;
    maxQuantity: number;
    timeLimit: number; // minutes
    conditions: string[];
  };
  
  reasoning: string[];
  timestamp: string;
}

export interface MicrostructureAnalysisResult {
  signal: MicrostructureSignal | null;
  microstructure: MarketMicrostructure;
  orderFlow: OrderFlowData[];
  imbalances: BidAskImbalance[];
  largeOrderImpacts: LargeOrderImpact[];
  marketState: {
    regime: 'normal' | 'stressed' | 'manipulation' | 'illiquid';
    volatility: 'low' | 'medium' | 'high' | 'extreme';
    efficiency: 'efficient' | 'inefficient' | 'very_inefficient';
  };
}

class MicrostructureAnalysisStrategy {
  private cache: Map<string, { data: any; timestamp: number }>;
  private readonly CACHE_DURATION = 60000; // 1 minute (high frequency)
  private readonly LARGE_ORDER_THRESHOLD = 50000; // $50K+ considered large
  private readonly WHALE_ORDER_THRESHOLD = 500000; // $500K+ considered whale

  constructor() {
    this.cache = new Map();
  }

  private getCacheKey(symbol: string, timeframe: string): string {
    return `microstructure_${symbol}_${timeframe}`;
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
    
    if (this.cache.size > 100) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < 20; i++) {
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
    timeframe: '1h' | '4h' = '1h'
  ): Promise<MicrostructureAnalysisResult> {
    try {
      console.log(`⚡ Analyzing market microstructure for ${symbol} (${timeframe})`);

      const cacheKey = this.getCacheKey(symbol, timeframe);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.log('📋 Using cached microstructure analysis');
        return cached;
      }

      // Step 1: Fetch DEX trade data
      const dexTrades = await bitqueryService.getDEXTrades(symbol, timeframe);
      console.log(`📊 Retrieved ${dexTrades.length} DEX trades`);

      // Step 2: Analyze order flow
      const orderFlow = this.analyzeOrderFlow(dexTrades);

      // Step 3: Calculate bid/ask imbalances
      const imbalances = this.calculateBidAskImbalances(dexTrades, priceData);

      // Step 4: Identify large order impacts
      const largeOrderImpacts = this.identifyLargeOrderImpacts(dexTrades, priceData);

      // Step 5: Calculate market microstructure metrics
      const microstructure = this.calculateMicrostructureMetrics(dexTrades, priceData);

      // Step 6: Determine market state
      const marketState = this.determineMarketState(microstructure, largeOrderImpacts, priceData);

      // Step 7: Generate signal
      const signal = await this.generateMicrostructureSignal(
        symbol,
        microstructure,
        orderFlow,
        imbalances,
        largeOrderImpacts,
        marketState,
        priceData
      );

      const result: MicrostructureAnalysisResult = {
        signal,
        microstructure,
        orderFlow,
        imbalances,
        largeOrderImpacts,
        marketState
      };

      this.setCache(cacheKey, result);
      console.log(`✅ Microstructure analysis completed for ${symbol}`);
      
      return result;
    } catch (error) {
      console.error('❌ Microstructure analysis failed:', error);
      return this.getDefaultResult();
    }
  }

  /**
   * Analyze order flow patterns
   */
  private analyzeOrderFlow(dexTrades: DEXTrade[]): OrderFlowData[] {
    const orderFlow: OrderFlowData[] = [];

    for (const trade of dexTrades) {
      // Calculate price impact (simplified)
      const impact = this.calculatePriceImpact(trade, dexTrades);
      
      // Determine urgency based on size and impact
      let urgency: OrderFlowData['urgency'] = 'low';
      if (trade.tradeAmount > this.WHALE_ORDER_THRESHOLD) {
        urgency = 'extreme';
      } else if (trade.tradeAmount > this.LARGE_ORDER_THRESHOLD * 5) {
        urgency = 'high';
      } else if (trade.tradeAmount > this.LARGE_ORDER_THRESHOLD) {
        urgency = 'medium';
      }

      orderFlow.push({
        timestamp: trade.block.timestamp.iso8601,
        exchange: trade.exchange.name,
        side: trade.side.toLowerCase() as 'buy' | 'sell',
        size: trade.tradeAmount || 0,
        sizeUSD: trade.quoteAmount || 0,
        price: trade.price || 0,
        impact,
        urgency
      });
    }

    // Sort by timestamp (most recent first)
    return orderFlow.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  /**
   * Calculate bid/ask imbalances from trade data
   */
  private calculateBidAskImbalances(dexTrades: DEXTrade[], priceData: PriceData[]): BidAskImbalance[] {
    const imbalances: BidAskImbalance[] = [];
    const timeWindows = this.createTimeWindows(dexTrades, 5); // 5-minute windows

    for (const window of timeWindows) {
      const buys = window.trades.filter(t => t.side === 'BUY');
      const sells = window.trades.filter(t => t.side === 'SELL');

      const bidVolume = buys.reduce((sum, t) => sum + (t.tradeAmount || 0), 0);
      const askVolume = sells.reduce((sum, t) => sum + (t.tradeAmount || 0), 0);

      const imbalanceRatio = askVolume > 0 ? bidVolume / askVolume : bidVolume > 0 ? 10 : 1;
      
      // Convert ratio to score (-100 to 100)
      let imbalanceScore = 0;
      if (imbalanceRatio > 1) {
        imbalanceScore = Math.min(100, (imbalanceRatio - 1) * 50);
      } else if (imbalanceRatio < 1) {
        imbalanceScore = Math.max(-100, (imbalanceRatio - 1) * 50);
      }

      // Estimate depth (simplified)
      const bidLevels = new Set(buys.map(t => Math.round((t.price || 0) * 100) / 100)).size;
      const askLevels = new Set(sells.map(t => Math.round((t.price || 0) * 100) / 100)).size;

      imbalances.push({
        timestamp: window.start,
        bidVolume,
        askVolume,
        imbalanceRatio,
        imbalanceScore,
        depth: {
          bid: { levels: bidLevels, totalSize: bidVolume },
          ask: { levels: askLevels, totalSize: askVolume }
        }
      });
    }

    return imbalances.slice(-20); // Keep last 20 imbalances
  }

  /**
   * Identify large order impacts
   */
  private identifyLargeOrderImpacts(dexTrades: DEXTrade[], priceData: PriceData[]): LargeOrderImpact[] {
    const largeOrders: LargeOrderImpact[] = [];
    
    const largeTrades = dexTrades.filter(trade => 
      (trade.tradeAmount || 0) > this.LARGE_ORDER_THRESHOLD
    );

    for (const trade of largeTrades) {
      const impact = this.calculateDetailedPriceImpact(trade, dexTrades, priceData);
      const classification = this.classifyOrder(trade);
      
      largeOrders.push({
        orderId: trade.transaction.hash,
        timestamp: trade.block.timestamp.iso8601,
        side: trade.side.toLowerCase() as 'buy' | 'sell',
        size: trade.tradeAmount || 0,
        sizeUSD: trade.quoteAmount || 0,
        priceImpact: impact.immediate,
        recoveryTime: impact.recoveryTime,
        sustainedImpact: impact.sustained,
        classification
      });
    }

    return largeOrders.sort((a, b) => b.sizeUSD - a.sizeUSD).slice(0, 10);
  }

  /**
   * Calculate comprehensive microstructure metrics
   */
  private calculateMicrostructureMetrics(dexTrades: DEXTrade[], priceData: PriceData[]): MarketMicrostructure {
    if (dexTrades.length === 0) {
      return this.getDefaultMicrostructure();
    }

    // Order flow imbalance
    const buys = dexTrades.filter(t => t.side === 'BUY');
    const sells = dexTrades.filter(t => t.side === 'SELL');
    const buyVolume = buys.reduce((sum, t) => sum + (t.tradeAmount || 0), 0);
    const sellVolume = sells.reduce((sum, t) => sum + (t.tradeAmount || 0), 0);
    
    const totalVolume = buyVolume + sellVolume;
    const orderFlowImbalance = totalVolume > 0 ? ((buyVolume - sellVolume) / totalVolume) * 100 : 0;

    // Average trade size
    const averageTradeSize = totalVolume / dexTrades.length;

    // Trade frequency (trades per minute)
    const timeSpan = this.getTimeSpanMinutes(dexTrades);
    const tradeFrequency = timeSpan > 0 ? dexTrades.length / timeSpan : 0;

    // Price efficiency (how quickly price adjusts)
    const priceEfficiency = this.calculatePriceEfficiency(dexTrades, priceData);

    // Liquidity depth estimation
    const liquidityDepth = this.estimateLiquidityDepth(dexTrades);

    // Market impact cost
    const marketImpactCost = this.calculateMarketImpactCost(dexTrades);

    return {
      orderFlowImbalance,
      averageTradeSize,
      tradeFrequency,
      priceEfficiency,
      liquidityDepth,
      marketImpactCost
    };
  }

  /**
   * Determine overall market state
   */
  private determineMarketState(
    microstructure: MarketMicrostructure,
    largeOrders: LargeOrderImpact[],
    priceData: PriceData[]
  ) {
    // Market regime
    let regime: 'normal' | 'stressed' | 'manipulation' | 'illiquid' = 'normal';
    
    if (microstructure.liquidityDepth < 30) {
      regime = 'illiquid';
    } else if (largeOrders.some(o => o.classification === 'manipulation')) {
      regime = 'manipulation';
    } else if (microstructure.marketImpactCost > 100 || microstructure.priceEfficiency < 40) {
      regime = 'stressed';
    }

    // Volatility assessment
    const atr = TechnicalIndicators.ATR(priceData, 14);
    const currentPrice = priceData[priceData.length - 1].close;
    const volatilityPercent = (atr / currentPrice) * 100;
    
    let volatility: 'low' | 'medium' | 'high' | 'extreme';
    if (volatilityPercent > 8) {
      volatility = 'extreme';
    } else if (volatilityPercent > 5) {
      volatility = 'high';
    } else if (volatilityPercent > 2) {
      volatility = 'medium';
    } else {
      volatility = 'low';
    }

    // Efficiency assessment
    let efficiency: 'efficient' | 'inefficient' | 'very_inefficient';
    if (microstructure.priceEfficiency > 70) {
      efficiency = 'efficient';
    } else if (microstructure.priceEfficiency > 40) {
      efficiency = 'inefficient';
    } else {
      efficiency = 'very_inefficient';
    }

    return { regime, volatility, efficiency };
  }

  /**
   * Generate microstructure-based trading signal
   */
  private async generateMicrostructureSignal(
    symbol: string,
    microstructure: MarketMicrostructure,
    orderFlow: OrderFlowData[],
    imbalances: BidAskImbalance[],
    largeOrders: LargeOrderImpact[],
    marketState: any,
    priceData: PriceData[]
  ): Promise<MicrostructureSignal | null> {

    // Don't trade in poor market conditions
    if (marketState.regime === 'manipulation' || marketState.efficiency === 'very_inefficient') {
      return null;
    }

    const currentPrice = priceData[priceData.length - 1].close;
    const recentImbalance = imbalances[imbalances.length - 1];
    const recentFlow = orderFlow.slice(0, 10); // Last 10 orders

    let action: MicrostructureSignal['action'] = 'wait';
    let signalType: MicrostructureSignal['signalType'] = 'order_flow_imbalance';
    let confidence = 50;
    let urgency: MicrostructureSignal['urgency'] = 'low';

    // Order flow imbalance signal
    if (Math.abs(microstructure.orderFlowImbalance) > 30) {
      signalType = 'order_flow_imbalance';
      confidence = Math.min(90, Math.abs(microstructure.orderFlowImbalance) * 2);
      
      if (microstructure.orderFlowImbalance > 30) {
        action = microstructure.priceEfficiency > 60 ? 'aggressive_buy' : 'passive_buy';
        urgency = microstructure.orderFlowImbalance > 60 ? 'high' : 'medium';
      } else {
        action = microstructure.priceEfficiency > 60 ? 'aggressive_sell' : 'passive_sell';
        urgency = microstructure.orderFlowImbalance < -60 ? 'high' : 'medium';
      }
    }

    // Large order impact signal
    const recentLargeOrders = largeOrders.filter(o => {
      const orderTime = new Date(o.timestamp).getTime();
      const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
      return orderTime > fiveMinutesAgo;
    });

    if (recentLargeOrders.length > 0) {
      const dominantSide = recentLargeOrders.reduce((sum, o) => 
        sum + (o.side === 'buy' ? o.sizeUSD : -o.sizeUSD), 0
      );
      
      if (Math.abs(dominantSide) > this.LARGE_ORDER_THRESHOLD * 2) {
        signalType = 'large_order_impact';
        confidence = 75;
        urgency = 'high';
        
        if (dominantSide > 0) {
          action = 'aggressive_buy';
        } else {
          action = 'aggressive_sell';
        }
      }
    }

    // Liquidity gap signal
    if (microstructure.liquidityDepth < 40 && Math.abs(microstructure.orderFlowImbalance) > 20) {
      signalType = 'liquidity_gap';
      action = 'avoid';
      confidence = 80;
      urgency = 'immediate';
    }

    // Don't generate low-confidence signals
    if (confidence < 60 || action === 'wait') {
      return null;
    }

    // Calculate risk factors
    const riskFactors = {
      liquidityRisk: Math.max(0, 100 - microstructure.liquidityDepth),
      manipulationRisk: largeOrders.filter(o => o.classification === 'manipulation').length * 20,
      latencyRisk: microstructure.tradeFrequency > 100 ? 80 : 20, // High frequency = high latency risk
      slippageRisk: Math.min(100, microstructure.marketImpactCost)
    };

    // Determine optimal execution parameters
    const timing = this.calculateOptimalTiming(microstructure, marketState, action);
    const executionPlan = this.createExecutionPlan(action, currentPrice, microstructure, timing);

    // Assess flow direction and intensity
    const flowDirection = microstructure.orderFlowImbalance > 10 ? 'bullish' : 
                         microstructure.orderFlowImbalance < -10 ? 'bearish' : 'neutral';
    const intensity = Math.abs(microstructure.orderFlowImbalance);

    const reasoning = [
      `Order flow imbalance: ${microstructure.orderFlowImbalance.toFixed(1)}%`,
      `Trade frequency: ${microstructure.tradeFrequency.toFixed(1)} trades/min`,
      `Price efficiency: ${microstructure.priceEfficiency.toFixed(0)}%`,
      `Liquidity depth: ${microstructure.liquidityDepth.toFixed(0)}%`,
      `Market impact cost: ${microstructure.marketImpactCost.toFixed(0)} bps`,
      `Recent large orders: ${recentLargeOrders.length}`,
      `Market regime: ${marketState.regime}`
    ];

    return {
      symbol,
      action,
      urgency,
      confidence,
      signalType,
      microstructure,
      timing,
      orderFlow: {
        recentImbalance,
        largeOrders: recentLargeOrders,
        flowDirection,
        intensity
      },
      riskFactors,
      executionPlan,
      reasoning,
      timestamp: new Date().toISOString()
    };
  }

  // Helper methods
  private calculatePriceImpact(trade: DEXTrade, allTrades: DEXTrade[]): number {
    // Find trades around the same time
    const tradeTime = new Date(trade.block.timestamp.iso8601).getTime();
    const nearbyTrades = allTrades.filter(t => {
      const tTime = new Date(t.block.timestamp.iso8601).getTime();
      return Math.abs(tTime - tradeTime) < 60000; // Within 1 minute
    });

    if (nearbyTrades.length < 2) return 0;

    // Calculate price movement
    const beforePrice = nearbyTrades[0].price || 0;
    const afterPrice = trade.price || 0;
    
    return beforePrice > 0 ? ((afterPrice - beforePrice) / beforePrice) * 100 : 0;
  }

  private calculateDetailedPriceImpact(
    trade: DEXTrade,
    allTrades: DEXTrade[],
    priceData: PriceData[]
  ): { immediate: number; recoveryTime: number; sustained: boolean } {
    const immediate = this.calculatePriceImpact(trade, allTrades);
    
    // Simplified recovery calculation
    const recoveryTime = Math.random() * 10; // Random 0-10 minutes
    const sustained = Math.abs(immediate) > 2 && recoveryTime > 5;

    return { immediate, recoveryTime, sustained };
  }

  private classifyOrder(trade: DEXTrade): LargeOrderImpact['classification'] {
    const size = trade.tradeAmount || 0;
    
    if (size > this.WHALE_ORDER_THRESHOLD) {
      return Math.random() > 0.7 ? 'whale_order' : 'institutional';
    } else if (size > this.LARGE_ORDER_THRESHOLD * 3) {
      return Math.random() > 0.5 ? 'institutional' : 'liquidation';
    }
    
    return 'whale_order';
  }

  private createTimeWindows(trades: DEXTrade[], windowMinutes: number): Array<{ start: string; trades: DEXTrade[] }> {
    if (trades.length === 0) return [];

    const windows: Array<{ start: string; trades: DEXTrade[] }> = [];
    const windowMs = windowMinutes * 60 * 1000;
    
    const sortedTrades = [...trades].sort((a, b) => 
      new Date(a.block.timestamp.iso8601).getTime() - new Date(b.block.timestamp.iso8601).getTime()
    );

    let windowStart = new Date(sortedTrades[0].block.timestamp.iso8601).getTime();
    let currentWindow: DEXTrade[] = [];

    for (const trade of sortedTrades) {
      const tradeTime = new Date(trade.block.timestamp.iso8601).getTime();
      
      if (tradeTime - windowStart > windowMs) {
        if (currentWindow.length > 0) {
          windows.push({
            start: new Date(windowStart).toISOString(),
            trades: [...currentWindow]
          });
        }
        windowStart = tradeTime;
        currentWindow = [trade];
      } else {
        currentWindow.push(trade);
      }
    }

    if (currentWindow.length > 0) {
      windows.push({
        start: new Date(windowStart).toISOString(),
        trades: currentWindow
      });
    }

    return windows;
  }

  private getTimeSpanMinutes(trades: DEXTrade[]): number {
    if (trades.length < 2) return 0;
    
    const times = trades.map(t => new Date(t.block.timestamp.iso8601).getTime());
    const earliest = Math.min(...times);
    const latest = Math.max(...times);
    
    return (latest - earliest) / (1000 * 60); // Convert to minutes
  }

  private calculatePriceEfficiency(trades: DEXTrade[], priceData: PriceData[]): number {
    // Simplified efficiency calculation
    if (trades.length < 10) return 50;
    
    const priceVariance = this.calculatePriceVariance(trades);
    const volumeNormalized = priceVariance / Math.max(1, trades.length);
    
    return Math.max(0, Math.min(100, 100 - volumeNormalized * 1000));
  }

  private calculatePriceVariance(trades: DEXTrade[]): number {
    const prices = trades.map(t => t.price || 0).filter(p => p > 0);
    if (prices.length < 2) return 0;
    
    const mean = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const variance = prices.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / prices.length;
    
    return Math.sqrt(variance);
  }

  private estimateLiquidityDepth(trades: DEXTrade[]): number {
    // Simplified liquidity estimation based on trade distribution
    const totalVolume = trades.reduce((sum, t) => sum + (t.tradeAmount || 0), 0);
    const uniquePriceLevels = new Set(trades.map(t => Math.round((t.price || 0) * 100) / 100)).size;
    
    const depth = uniquePriceLevels > 0 ? Math.min(100, (totalVolume / 1000000) * (uniquePriceLevels / 10)) : 0;
    return Math.max(0, depth);
  }

  private calculateMarketImpactCost(trades: DEXTrade[]): number {
    // Simplified market impact cost in basis points
    const avgTradeSize = trades.reduce((sum, t) => sum + (t.tradeAmount || 0), 0) / Math.max(1, trades.length);
    const impactCost = Math.min(200, avgTradeSize / 10000); // Very simplified
    
    return Math.max(10, impactCost);
  }

  private calculateOptimalTiming(
    microstructure: MarketMicrostructure,
    marketState: any,
    action: string
  ): MicrostructureSignal['timing'] {
    let entryWindow = 5; // Default 5 minutes
    let executionStyle: 'market' | 'limit' | 'iceberg' | 'twap' = 'limit';
    let expectedSlippage = 20; // Default 20 bps
    let maxPosition = 50000; // Default $50K

    // Adjust based on market conditions
    if (marketState.regime === 'stressed') {
      entryWindow = 2;
      expectedSlippage = 50;
      maxPosition = 25000;
    } else if (marketState.efficiency === 'efficient') {
      executionStyle = 'market';
      expectedSlippage = 10;
      maxPosition = 100000;
    }

    // Adjust for urgency
    if (action.includes('aggressive')) {
      executionStyle = 'market';
      entryWindow = 1;
      expectedSlippage *= 1.5;
    } else if (microstructure.liquidityDepth < 40) {
      executionStyle = 'iceberg';
      maxPosition *= 0.5;
    }

    return {
      entryWindow,
      optimalExecutionStyle: executionStyle,
      expectedSlippage,
      maxPosition
    };
  }

  private createExecutionPlan(
    action: string,
    currentPrice: number,
    microstructure: MarketMicrostructure,
    timing: any
  ): MicrostructureSignal['executionPlan'] {
    let orderType: 'market' | 'limit' | 'stop' | 'iceberg' = timing.optimalExecutionStyle;
    let priceLevel = currentPrice;
    let maxQuantity = timing.maxPosition / currentPrice;
    let timeLimit = timing.entryWindow;

    const conditions = [];

    // Adjust price level for limit orders
    if (orderType === 'limit') {
      if (action.includes('buy')) {
        priceLevel = currentPrice * 0.999; // Slightly below market
        conditions.push('Price remains stable');
      } else {
        priceLevel = currentPrice * 1.001; // Slightly above market
        conditions.push('Price remains stable');
      }
    }

    // Add liquidity conditions
    if (microstructure.liquidityDepth < 50) {
      conditions.push('Monitor liquidity depth');
      maxQuantity *= 0.7; // Reduce size in low liquidity
    }

    // Add flow conditions
    if (Math.abs(microstructure.orderFlowImbalance) > 50) {
      conditions.push('Strong order flow continues');
      timeLimit = Math.min(timeLimit, 3); // Shorter window for strong flows
    }

    return {
      orderType,
      priceLevel,
      maxQuantity,
      timeLimit,
      conditions
    };
  }

  private getDefaultMicrostructure(): MarketMicrostructure {
    return {
      orderFlowImbalance: 0,
      averageTradeSize: 0,
      tradeFrequency: 0,
      priceEfficiency: 50,
      liquidityDepth: 50,
      marketImpactCost: 50
    };
  }

  private getDefaultResult(): MicrostructureAnalysisResult {
    return {
      signal: null,
      microstructure: this.getDefaultMicrostructure(),
      orderFlow: [],
      imbalances: [],
      largeOrderImpacts: [],
      marketState: {
        regime: 'normal',
        volatility: 'medium',
        efficiency: 'efficient'
      }
    };
  }
}

export const microstructureAnalysisStrategy = new MicrostructureAnalysisStrategy();
export default MicrostructureAnalysisStrategy;