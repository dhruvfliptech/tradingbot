/**
 * Liquidity Hunting Strategy
 * Identifies liquidity pools and high-volume zones where stop losses are likely to cluster
 * Uses bitqueryService to analyze on-chain liquidity and coinGeckoService for support/resistance levels
 */

import { bitqueryService, LiquidityPool } from '../bitqueryService';
import { coinGeckoService } from '../coinGeckoService';
import { TechnicalIndicators, PriceData } from '../technicalIndicators';
import { ValidationResult } from '../validatorSystem';

export interface LiquidityZone {
  priceLevel: number;
  liquidityAmount: number;
  liquidityUSD: number;
  type: 'support' | 'resistance' | 'neutral';
  confidence: number; // 0-100
  poolCount: number;
  avgVolume: number;
  lastActivity: string;
}

export interface LiquidityHuntingSignal {
  symbol: string;
  action: 'hunt_long' | 'hunt_short' | 'avoid' | 'watch';
  targetZone: LiquidityZone;
  currentPrice: number;
  distanceToZone: number;
  distancePercent: number;
  confidence: number; // 0-100
  timestamp: string;
  entry: {
    price: number;
    zone: 'below_support' | 'above_resistance' | 'within_zone';
  };
  exit: {
    price: number;
    reason: 'liquidity_grabbed' | 'reversal_signal' | 'stop_loss';
  };
  riskReward: number;
  reasoning: string[];
  liquidityData: {
    totalLiquidity: number;
    majorPools: LiquidityPool[];
    volumeProfile: VolumeLevel[];
  };
}

export interface VolumeLevel {
  priceLevel: number;
  volume: number;
  transactions: number;
  avgSize: number;
  timeSpent: number; // minutes spent at this level
}

export interface LiquidityAnalysisResult {
  signal: LiquidityHuntingSignal | null;
  zones: LiquidityZone[];
  marketStructure: {
    trend: 'bullish' | 'bearish' | 'ranging';
    liquidityImbalance: 'buy_side' | 'sell_side' | 'balanced';
    huntProbability: number; // 0-100
  };
  riskAssessment: {
    stopHuntRisk: number; // 0-100
    liquidityRisk: number; // 0-100
    marketRisk: number; // 0-100
  };
}

class LiquidityHuntingStrategy {
  private cache: Map<string, { data: any; timestamp: number }>;
  private readonly CACHE_DURATION = 300000; // 5 minutes

  constructor() {
    this.cache = new Map();
  }

  private getCacheKey(symbol: string, timeframe: string): string {
    return `liquidity_hunting_${symbol}_${timeframe}`;
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
    timeframe: '1h' | '4h' | '1d' = '4h'
  ): Promise<LiquidityAnalysisResult> {
    try {
      console.log(`🏊 Analyzing liquidity hunting opportunities for ${symbol}`);

      const cacheKey = this.getCacheKey(symbol, timeframe);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.log('📋 Using cached liquidity hunting analysis');
        return cached;
      }

      // Get current market data
      const cryptoData = await coinGeckoService.getCryptoData([symbol.toLowerCase()]);
      const currentData = cryptoData[0];
      
      if (!currentData) {
        throw new Error(`No price data found for ${symbol}`);
      }

      const currentPrice = currentData.price;

      // Step 1: Identify liquidity pools and zones
      const liquidityZones = await this.identifyLiquidityZones(symbol, priceData, currentPrice);

      // Step 2: Analyze volume profile
      const volumeProfile = this.analyzeVolumeProfile(priceData);

      // Step 3: Calculate market structure
      const marketStructure = this.analyzeMarketStructure(priceData, liquidityZones);

      // Step 4: Assess risks
      const riskAssessment = this.assessRisks(priceData, liquidityZones, currentPrice);

      // Step 5: Generate signal if conditions are met
      const signal = await this.generateSignal(
        symbol,
        currentPrice,
        liquidityZones,
        volumeProfile,
        marketStructure,
        riskAssessment
      );

      const result: LiquidityAnalysisResult = {
        signal,
        zones: liquidityZones,
        marketStructure,
        riskAssessment
      };

      this.setCache(cacheKey, result);
      console.log(`✅ Liquidity hunting analysis completed for ${symbol}`);
      
      return result;
    } catch (error) {
      console.error('❌ Liquidity hunting analysis failed:', error);
      return {
        signal: null,
        zones: [],
        marketStructure: {
          trend: 'ranging',
          liquidityImbalance: 'balanced',
          huntProbability: 0
        },
        riskAssessment: {
          stopHuntRisk: 50,
          liquidityRisk: 50,
          marketRisk: 50
        }
      };
    }
  }

  /**
   * Identify liquidity zones using on-chain data and technical analysis
   */
  private async identifyLiquidityZones(
    symbol: string,
    priceData: PriceData[],
    currentPrice: number
  ): Promise<LiquidityZone[]> {
    const zones: LiquidityZone[] = [];

    try {
      // Get liquidity pools from Bitquery
      const liquidityPools = await bitqueryService.getLiquidityPools(symbol);
      console.log(`📊 Found ${liquidityPools.length} liquidity pools for ${symbol}`);

      // Calculate support and resistance levels
      const supportResistance = TechnicalIndicators.SupportResistance(priceData, 20);
      
      // Combine technical levels with liquidity data
      const allLevels = [
        ...supportResistance.support.map(level => ({ price: level, type: 'support' as const })),
        ...supportResistance.resistance.map(level => ({ price: level, type: 'resistance' as const }))
      ];

      for (const level of allLevels) {
        // Find nearby liquidity pools (within 2% of the level)
        const nearbyPools = liquidityPools.filter(pool => {
          const poolPrice = this.estimatePoolPrice(pool);
          return Math.abs(poolPrice - level.price) / level.price <= 0.02;
        });

        if (nearbyPools.length > 0) {
          const totalLiquidity = nearbyPools.reduce((sum, pool) => sum + pool.liquidityUSD, 0);
          const avgVolume = this.calculateVolumeAtLevel(priceData, level.price);
          
          zones.push({
            priceLevel: level.price,
            liquidityAmount: nearbyPools.reduce((sum, pool) => sum + pool.totalSupply, 0),
            liquidityUSD: totalLiquidity,
            type: level.type,
            confidence: this.calculateZoneConfidence(nearbyPools, avgVolume, currentPrice, level.price),
            poolCount: nearbyPools.length,
            avgVolume,
            lastActivity: new Date().toISOString()
          });
        }
      }

      // Add high-volume zones that might not be traditional S/R
      const volumeZones = this.identifyHighVolumeZones(priceData);
      for (const volumeZone of volumeZones) {
        if (!zones.find(z => Math.abs(z.priceLevel - volumeZone.priceLevel) / volumeZone.priceLevel <= 0.01)) {
          zones.push({
            priceLevel: volumeZone.priceLevel,
            liquidityAmount: volumeZone.volume,
            liquidityUSD: volumeZone.volume * volumeZone.priceLevel,
            type: 'neutral',
            confidence: Math.min(90, volumeZone.volume / 1000000 * 50 + 40),
            poolCount: 0,
            avgVolume: volumeZone.volume,
            lastActivity: new Date().toISOString()
          });
        }
      }

      // Sort by liquidity amount (descending)
      zones.sort((a, b) => b.liquidityUSD - a.liquidityUSD);
      
      console.log(`🎯 Identified ${zones.length} liquidity zones`);
      return zones.slice(0, 10); // Return top 10 zones

    } catch (error) {
      console.error('❌ Failed to identify liquidity zones:', error);
      
      // Fallback to technical analysis only
      const supportResistance = TechnicalIndicators.SupportResistance(priceData, 20);
      const allLevels = [...supportResistance.support, ...supportResistance.resistance];
      
      return allLevels.slice(0, 5).map((level, index) => ({
        priceLevel: level,
        liquidityAmount: 0,
        liquidityUSD: 0,
        type: index < supportResistance.support.length ? 'support' : 'resistance',
        confidence: 60,
        poolCount: 0,
        avgVolume: this.calculateVolumeAtLevel(priceData, level),
        lastActivity: new Date().toISOString()
      }));
    }
  }

  /**
   * Analyze volume profile to identify high-activity zones
   */
  private analyzeVolumeProfile(priceData: PriceData[]): VolumeLevel[] {
    const volumeLevels: VolumeLevel[] = [];
    const priceRange = {
      min: Math.min(...priceData.map(d => d.low)),
      max: Math.max(...priceData.map(d => d.high))
    };

    const levelCount = 50; // Divide price range into 50 levels
    const levelSize = (priceRange.max - priceRange.min) / levelCount;

    for (let i = 0; i < levelCount; i++) {
      const levelPrice = priceRange.min + (i * levelSize);
      const levelData = this.calculateVolumeProfileLevel(priceData, levelPrice, levelSize);
      
      if (levelData.volume > 0) {
        volumeLevels.push({
          priceLevel: levelPrice,
          volume: levelData.volume,
          transactions: levelData.transactions,
          avgSize: levelData.volume / Math.max(1, levelData.transactions),
          timeSpent: levelData.timeSpent
        });
      }
    }

    // Sort by volume (descending)
    volumeLevels.sort((a, b) => b.volume - a.volume);
    
    return volumeLevels.slice(0, 20); // Return top 20 volume levels
  }

  /**
   * Analyze market structure for liquidity hunting opportunities
   */
  private analyzeMarketStructure(
    priceData: PriceData[],
    liquidityZones: LiquidityZone[]
  ): LiquidityAnalysisResult['marketStructure'] {
    // Determine trend
    const prices = priceData.map(d => d.close);
    const sma20 = TechnicalIndicators.SMA(prices, 20);
    const sma50 = TechnicalIndicators.SMA(prices, Math.min(50, prices.length));
    const currentPrice = prices[prices.length - 1];

    let trend: 'bullish' | 'bearish' | 'ranging';
    if (currentPrice > sma20 && sma20 > sma50) {
      trend = 'bullish';
    } else if (currentPrice < sma20 && sma20 < sma50) {
      trend = 'bearish';
    } else {
      trend = 'ranging';
    }

    // Analyze liquidity imbalance
    const supportLiquidity = liquidityZones
      .filter(z => z.type === 'support')
      .reduce((sum, z) => sum + z.liquidityUSD, 0);
    
    const resistanceLiquidity = liquidityZones
      .filter(z => z.type === 'resistance')
      .reduce((sum, z) => sum + z.liquidityUSD, 0);

    let liquidityImbalance: 'buy_side' | 'sell_side' | 'balanced';
    const imbalanceRatio = supportLiquidity / Math.max(1, resistanceLiquidity);
    
    if (imbalanceRatio > 1.5) {
      liquidityImbalance = 'buy_side';
    } else if (imbalanceRatio < 0.67) {
      liquidityImbalance = 'sell_side';
    } else {
      liquidityImbalance = 'balanced';
    }

    // Calculate hunt probability
    let huntProbability = 50; // Base probability

    // Increase probability if there's significant liquidity imbalance
    if (liquidityImbalance !== 'balanced') {
      huntProbability += 20;
    }

    // Increase probability if price is near major liquidity zones
    const nearbyZones = liquidityZones.filter(z => 
      Math.abs(z.priceLevel - currentPrice) / currentPrice <= 0.05
    );
    huntProbability += nearbyZones.length * 5;

    // Adjust for market conditions
    if (trend === 'ranging') {
      huntProbability += 15; // More likely in ranging markets
    }

    huntProbability = Math.min(100, Math.max(0, huntProbability));

    return {
      trend,
      liquidityImbalance,
      huntProbability
    };
  }

  /**
   * Assess various risk factors
   */
  private assessRisks(
    priceData: PriceData[],
    liquidityZones: LiquidityZone[],
    currentPrice: number
  ): LiquidityAnalysisResult['riskAssessment'] {
    // Stop hunt risk: Risk of being caught in a liquidity grab
    let stopHuntRisk = 30; // Base risk
    
    const nearbyZones = liquidityZones.filter(z => 
      Math.abs(z.priceLevel - currentPrice) / currentPrice <= 0.03
    );
    
    if (nearbyZones.length > 0) {
      stopHuntRisk += nearbyZones.length * 10;
      stopHuntRisk += nearbyZones.reduce((sum, z) => sum + z.confidence, 0) / nearbyZones.length * 0.3;
    }

    // Liquidity risk: Risk of insufficient liquidity
    const totalLiquidity = liquidityZones.reduce((sum, z) => sum + z.liquidityUSD, 0);
    let liquidityRisk = 50;
    
    if (totalLiquidity > 10000000) { // $10M+
      liquidityRisk = 20;
    } else if (totalLiquidity > 1000000) { // $1M+
      liquidityRisk = 35;
    } else if (totalLiquidity < 100000) { // <$100K
      liquidityRisk = 80;
    }

    // Market risk: General market volatility risk
    const atr = TechnicalIndicators.ATR(priceData, 14);
    const atrPercent = (atr / currentPrice) * 100;
    
    let marketRisk = 40;
    if (atrPercent > 5) {
      marketRisk = 70;
    } else if (atrPercent > 3) {
      marketRisk = 55;
    } else if (atrPercent < 1) {
      marketRisk = 25;
    }

    return {
      stopHuntRisk: Math.min(100, Math.max(0, stopHuntRisk)),
      liquidityRisk: Math.min(100, Math.max(0, liquidityRisk)),
      marketRisk: Math.min(100, Math.max(0, marketRisk))
    };
  }

  /**
   * Generate trading signal based on analysis
   */
  private async generateSignal(
    symbol: string,
    currentPrice: number,
    liquidityZones: LiquidityZone[],
    volumeProfile: VolumeLevel[],
    marketStructure: LiquidityAnalysisResult['marketStructure'],
    riskAssessment: LiquidityAnalysisResult['riskAssessment']
  ): Promise<LiquidityHuntingSignal | null> {
    // Don't generate signals if risks are too high
    if (riskAssessment.liquidityRisk > 70 || riskAssessment.marketRisk > 80) {
      return null;
    }

    // Find the most promising zone to hunt
    const targetZone = this.findBestHuntingZone(liquidityZones, currentPrice, marketStructure);
    
    if (!targetZone) {
      return null;
    }

    const distanceToZone = Math.abs(targetZone.priceLevel - currentPrice);
    const distancePercent = (distanceToZone / currentPrice) * 100;

    // Only generate signals if zone is within reasonable distance (< 5%)
    if (distancePercent > 5) {
      return null;
    }

    // Determine action based on market structure and zone type
    let action: LiquidityHuntingSignal['action'];
    let entryZone: 'below_support' | 'above_resistance' | 'within_zone';
    let exitPrice: number;
    let exitReason: 'liquidity_grabbed' | 'reversal_signal' | 'stop_loss';

    if (targetZone.type === 'support' && marketStructure.trend !== 'bearish') {
      action = 'hunt_long';
      entryZone = 'below_support';
      exitPrice = targetZone.priceLevel * 1.02; // 2% above support
      exitReason = 'liquidity_grabbed';
    } else if (targetZone.type === 'resistance' && marketStructure.trend !== 'bullish') {
      action = 'hunt_short';
      entryZone = 'above_resistance';
      exitPrice = targetZone.priceLevel * 0.98; // 2% below resistance
      exitReason = 'liquidity_grabbed';
    } else if (distancePercent < 1) {
      action = 'watch';
      entryZone = 'within_zone';
      exitPrice = currentPrice;
      exitReason = 'reversal_signal';
    } else {
      return null;
    }

    // Calculate risk/reward
    const entryPrice = action === 'hunt_long' 
      ? targetZone.priceLevel * 0.995 
      : targetZone.priceLevel * 1.005;
    
    const stopLoss = action === 'hunt_long'
      ? targetZone.priceLevel * 0.985
      : targetZone.priceLevel * 1.015;

    const risk = Math.abs(entryPrice - stopLoss);
    const reward = Math.abs(exitPrice - entryPrice);
    const riskReward = reward / Math.max(risk, 0.001);

    // Calculate confidence
    let confidence = targetZone.confidence * 0.4;
    confidence += marketStructure.huntProbability * 0.3;
    confidence += (100 - riskAssessment.stopHuntRisk) * 0.2;
    confidence += Math.min(20, riskReward * 5); // Bonus for good R/R
    confidence = Math.min(100, Math.max(0, confidence));

    // Don't generate low-confidence signals
    if (confidence < 60) {
      return null;
    }

    const reasoning = [
      `Target zone at $${targetZone.priceLevel.toFixed(4)} with $${(targetZone.liquidityUSD / 1000000).toFixed(2)}M liquidity`,
      `Market structure: ${marketStructure.trend} trend with ${marketStructure.liquidityImbalance} liquidity imbalance`,
      `Hunt probability: ${marketStructure.huntProbability}%`,
      `Distance to zone: ${distancePercent.toFixed(2)}%`,
      `Risk/Reward ratio: ${riskReward.toFixed(2)}:1`
    ];

    return {
      symbol,
      action,
      targetZone,
      currentPrice,
      distanceToZone,
      distancePercent,
      confidence,
      timestamp: new Date().toISOString(),
      entry: {
        price: entryPrice,
        zone: entryZone
      },
      exit: {
        price: exitPrice,
        reason: exitReason
      },
      riskReward,
      reasoning,
      liquidityData: {
        totalLiquidity: liquidityZones.reduce((sum, z) => sum + z.liquidityUSD, 0),
        majorPools: [], // Would be populated with actual pool data
        volumeProfile
      }
    };
  }

  // Helper methods
  private estimatePoolPrice(pool: LiquidityPool): number {
    // Simplified price estimation from pool reserves
    if (pool.reserve0 > 0 && pool.reserve1 > 0) {
      return pool.reserve1 / pool.reserve0;
    }
    return 0;
  }

  private calculateVolumeAtLevel(priceData: PriceData[], level: number): number {
    const tolerance = level * 0.005; // 0.5% tolerance
    return priceData
      .filter(d => d.low <= level + tolerance && d.high >= level - tolerance)
      .reduce((sum, d) => sum + d.volume, 0);
  }

  private calculateZoneConfidence(
    pools: LiquidityPool[],
    volume: number,
    currentPrice: number,
    zonePrice: number
  ): number {
    let confidence = 40; // Base confidence

    // More pools = higher confidence
    confidence += Math.min(30, pools.length * 5);

    // Higher volume = higher confidence
    confidence += Math.min(20, volume / 1000000 * 10);

    // Closer to current price = higher confidence for immediate relevance
    const distance = Math.abs(currentPrice - zonePrice) / currentPrice;
    confidence += Math.max(0, 20 - distance * 400);

    return Math.min(100, Math.max(0, confidence));
  }

  private identifyHighVolumeZones(priceData: PriceData[]): VolumeLevel[] {
    const zones: VolumeLevel[] = [];
    const volumeThreshold = priceData.reduce((sum, d) => sum + d.volume, 0) / priceData.length * 2;

    for (let i = 1; i < priceData.length - 1; i++) {
      if (priceData[i].volume > volumeThreshold) {
        zones.push({
          priceLevel: (priceData[i].high + priceData[i].low) / 2,
          volume: priceData[i].volume,
          transactions: 1,
          avgSize: priceData[i].volume,
          timeSpent: 1
        });
      }
    }

    return zones;
  }

  private calculateVolumeProfileLevel(
    priceData: PriceData[],
    levelPrice: number,
    levelSize: number
  ): { volume: number; transactions: number; timeSpent: number } {
    let volume = 0;
    let transactions = 0;
    let timeSpent = 0;

    for (const data of priceData) {
      if (data.low <= levelPrice + levelSize && data.high >= levelPrice - levelSize) {
        volume += data.volume;
        transactions += 1;
        timeSpent += 1; // Simplified time calculation
      }
    }

    return { volume, transactions, timeSpent };
  }

  private findBestHuntingZone(
    zones: LiquidityZone[],
    currentPrice: number,
    marketStructure: LiquidityAnalysisResult['marketStructure']
  ): LiquidityZone | null {
    // Filter zones based on distance and relevance
    const relevantZones = zones.filter(zone => {
      const distance = Math.abs(zone.priceLevel - currentPrice) / currentPrice;
      return distance <= 0.05 && zone.confidence > 60; // Within 5% and high confidence
    });

    if (relevantZones.length === 0) {
      return null;
    }

    // Score zones based on multiple factors
    const scoredZones = relevantZones.map(zone => {
      let score = zone.confidence * 0.4;
      score += (zone.liquidityUSD / 1000000) * 10; // Liquidity in millions
      score += zone.poolCount * 2;
      
      // Prefer zones aligned with market structure
      if (marketStructure.trend === 'bullish' && zone.type === 'support') {
        score += 15;
      } else if (marketStructure.trend === 'bearish' && zone.type === 'resistance') {
        score += 15;
      }

      // Prefer closer zones
      const distance = Math.abs(zone.priceLevel - currentPrice) / currentPrice;
      score += Math.max(0, 10 - distance * 200);

      return { zone, score };
    });

    // Return the highest scored zone
    scoredZones.sort((a, b) => b.score - a.score);
    return scoredZones[0]?.zone || null;
  }
}

export const liquidityHuntingStrategy = new LiquidityHuntingStrategy();
export default LiquidityHuntingStrategy;