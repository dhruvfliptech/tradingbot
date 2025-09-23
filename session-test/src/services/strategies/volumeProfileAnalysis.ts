/**
 * Volume Profile Analysis Strategy
 * Enhanced volume analysis using CoinGecko data with Point of Control (POC) and Value Area calculations
 * Provides volume-weighted support/resistance levels and integrates with technicalIndicators.ts
 */

import { coinGeckoService } from '../coinGeckoService';
import { TechnicalIndicators, PriceData } from '../technicalIndicators';
import { ValidationResult } from '../validatorSystem';

export interface VolumeNode {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  transactions: number;
  timeSpent: number; // minutes
  volumePercent: number; // percentage of total volume
}

export interface VolumeProfile {
  nodes: VolumeNode[];
  pointOfControl: VolumeNode; // Highest volume node
  valueAreaHigh: number;
  valueAreaLow: number;
  valueArea: {
    volume: number;
    volumePercent: number; // Should be ~70% of total
    priceRange: number;
  };
  volumeWeightedAveragePrice: number; // VWAP
  profileType: 'normal' | 'b_shape' | 'd_shape' | 'p_shape' | 'b_shape';
}

export interface VolumeSupport {
  price: number;
  volume: number;
  strength: number; // 0-100
  type: 'major' | 'minor' | 'weak';
  bounces: number;
  lastTest: string;
}

export interface VolumeResistance {
  price: number;
  volume: number;
  strength: number; // 0-100
  type: 'major' | 'minor' | 'weak';
  rejections: number;
  lastTest: string;
}

export interface VolumeProfileSignal {
  symbol: string;
  action: 'long' | 'short' | 'neutral' | 'watch';
  confidence: number; // 0-100
  signalType: 'poc_retest' | 'value_area_breakout' | 'volume_support' | 'volume_resistance' | 'vwap_cross';
  currentPrice: number;
  targetLevel: number;
  stopLoss: number;
  riskReward: number;
  timestamp: string;
  
  volumeContext: {
    profile: VolumeProfile;
    currentPosition: 'above_vah' | 'within_value_area' | 'below_val' | 'at_poc';
    volumeTrend: 'increasing' | 'decreasing' | 'stable';
    profileBalance: 'balanced' | 'top_heavy' | 'bottom_heavy';
  };

  technicalAlignment: {
    withTrend: boolean;
    withMomentum: boolean;
    withVolume: boolean;
    overallScore: number; // 0-100
  };

  reasoning: string[];
}

export interface VolumeProfileAnalysisResult {
  signal: VolumeProfileSignal | null;
  profile: VolumeProfile;
  supports: VolumeSupport[];
  resistances: VolumeResistance[];
  marketStructure: {
    phase: 'accumulation' | 'distribution' | 'trending' | 'rotation';
    volumeBalance: 'buyer_controlled' | 'seller_controlled' | 'balanced';
    efficiency: number; // 0-100, higher = more efficient price discovery
  };
  riskAssessment: {
    volumeRisk: number; // 0-100
    supportStrength: number; // 0-100
    resistanceStrength: number; // 0-100
  };
}

class VolumeProfileAnalysisStrategy {
  private cache: Map<string, { data: any; timestamp: number }>;
  private readonly CACHE_DURATION = 600000; // 10 minutes
  private readonly VALUE_AREA_PERCENT = 0.7; // 70% of volume

  constructor() {
    this.cache = new Map();
  }

  private getCacheKey(symbol: string, period: number): string {
    return `volume_profile_${symbol}_${period}`;
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
    period: number = 50 // Number of bars to analyze
  ): Promise<VolumeProfileAnalysisResult> {
    try {
      console.log(`📊 Analyzing volume profile for ${symbol} (${period} periods)`);

      const cacheKey = this.getCacheKey(symbol, period);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.log('📋 Using cached volume profile analysis');
        return cached;
      }

      // Limit analysis to recent periods
      const recentData = priceData.slice(-period);
      
      if (recentData.length < 20) {
        throw new Error('Insufficient data for volume profile analysis');
      }

      // Step 1: Build volume profile
      const profile = this.buildVolumeProfile(recentData);

      // Step 2: Identify volume-based support and resistance
      const supports = this.identifyVolumeSupports(profile, recentData);
      const resistances = this.identifyVolumeResistances(profile, recentData);

      // Step 3: Analyze market structure
      const marketStructure = this.analyzeMarketStructure(profile, recentData);

      // Step 4: Assess risks
      const riskAssessment = this.assessVolumeRisks(profile, supports, resistances);

      // Step 5: Generate signal
      const signal = await this.generateVolumeSignal(
        symbol,
        profile,
        supports,
        resistances,
        marketStructure,
        recentData
      );

      const result: VolumeProfileAnalysisResult = {
        signal,
        profile,
        supports,
        resistances,
        marketStructure,
        riskAssessment
      };

      this.setCache(cacheKey, result);
      console.log(`✅ Volume profile analysis completed for ${symbol}`);
      
      return result;
    } catch (error) {
      console.error('❌ Volume profile analysis failed:', error);
      return this.getDefaultResult();
    }
  }

  /**
   * Build comprehensive volume profile
   */
  private buildVolumeProfile(priceData: PriceData[]): VolumeProfile {
    // Determine price range
    const priceRange = {
      min: Math.min(...priceData.map(d => d.low)),
      max: Math.max(...priceData.map(d => d.high))
    };

    const numberOfLevels = 100; // Divide price range into levels
    const levelSize = (priceRange.max - priceRange.min) / numberOfLevels;
    
    // Initialize volume nodes
    const nodes: VolumeNode[] = [];
    let totalVolume = 0;
    
    for (let i = 0; i < numberOfLevels; i++) {
      const levelPrice = priceRange.min + (i * levelSize);
      const levelData = this.calculateVolumeAtLevel(priceData, levelPrice, levelSize);
      
      if (levelData.volume > 0) {
        nodes.push({
          price: levelPrice,
          volume: levelData.volume,
          buyVolume: levelData.buyVolume,
          sellVolume: levelData.sellVolume,
          transactions: levelData.transactions,
          timeSpent: levelData.timeSpent,
          volumePercent: 0 // Will be calculated after total volume
        });
        
        totalVolume += levelData.volume;
      }
    }

    // Calculate volume percentages and sort by volume
    nodes.forEach(node => {
      node.volumePercent = (node.volume / totalVolume) * 100;
    });
    nodes.sort((a, b) => b.volume - a.volume);

    // Find Point of Control (highest volume node)
    const pointOfControl = nodes[0];

    // Calculate Value Area (70% of volume)
    const { valueAreaHigh, valueAreaLow, valueAreaVolume } = this.calculateValueArea(nodes, totalVolume);

    // Calculate VWAP
    const vwap = this.calculateVWAP(priceData);

    // Determine profile type
    const profileType = this.determineProfileType(nodes, pointOfControl);

    return {
      nodes: nodes.slice(0, 50), // Keep top 50 nodes
      pointOfControl,
      valueAreaHigh,
      valueAreaLow,
      valueArea: {
        volume: valueAreaVolume,
        volumePercent: (valueAreaVolume / totalVolume) * 100,
        priceRange: valueAreaHigh - valueAreaLow
      },
      volumeWeightedAveragePrice: vwap,
      profileType
    };
  }

  /**
   * Calculate volume at a specific price level
   */
  private calculateVolumeAtLevel(
    priceData: PriceData[],
    levelPrice: number,
    levelSize: number
  ): {
    volume: number;
    buyVolume: number;
    sellVolume: number;
    transactions: number;
    timeSpent: number;
  } {
    let volume = 0;
    let buyVolume = 0;
    let sellVolume = 0;
    let transactions = 0;
    let timeSpent = 0;

    const levelHigh = levelPrice + levelSize / 2;
    const levelLow = levelPrice - levelSize / 2;

    for (let i = 0; i < priceData.length; i++) {
      const candle = priceData[i];
      
      // Check if price level intersects with this candle
      if (candle.low <= levelHigh && candle.high >= levelLow) {
        // Weight volume by how much of the candle's range intersects the level
        const intersection = Math.min(levelHigh, candle.high) - Math.max(levelLow, candle.low);
        const candleRange = candle.high - candle.low;
        const weight = candleRange > 0 ? intersection / candleRange : 1;
        
        const weightedVolume = candle.volume * weight;
        volume += weightedVolume;
        transactions += 1;
        timeSpent += 1; // Simplified: each candle = 1 time unit

        // Estimate buy/sell volume based on price action
        const isUpCandle = candle.close > candle.open;
        if (isUpCandle) {
          buyVolume += weightedVolume * 0.6; // 60% buy, 40% sell for up candles
          sellVolume += weightedVolume * 0.4;
        } else {
          buyVolume += weightedVolume * 0.4; // 40% buy, 60% sell for down candles
          sellVolume += weightedVolume * 0.6;
        }
      }
    }

    return { volume, buyVolume, sellVolume, transactions, timeSpent };
  }

  /**
   * Calculate Value Area (70% of volume around POC)
   */
  private calculateValueArea(nodes: VolumeNode[], totalVolume: number): {
    valueAreaHigh: number;
    valueAreaLow: number;
    valueAreaVolume: number;
  } {
    const targetVolume = totalVolume * this.VALUE_AREA_PERCENT;
    
    // Sort nodes by price to find contiguous value area
    const sortedByPrice = [...nodes].sort((a, b) => a.price - b.price);
    
    let bestValueArea = {
      high: 0,
      low: 0,
      volume: 0,
      range: Infinity
    };

    // Try different starting points to find optimal value area
    for (let i = 0; i < sortedByPrice.length; i++) {
      let currentVolume = 0;
      let j = i;
      
      // Expand upward until we reach target volume
      while (j < sortedByPrice.length && currentVolume < targetVolume) {
        currentVolume += sortedByPrice[j].volume;
        j++;
      }
      
      if (currentVolume >= targetVolume) {
        const range = sortedByPrice[j - 1].price - sortedByPrice[i].price;
        
        if (range < bestValueArea.range) {
          bestValueArea = {
            high: sortedByPrice[j - 1].price,
            low: sortedByPrice[i].price,
            volume: currentVolume,
            range
          };
        }
      }
    }

    return {
      valueAreaHigh: bestValueArea.high,
      valueAreaLow: bestValueArea.low,
      valueAreaVolume: bestValueArea.volume
    };
  }

  /**
   * Calculate Volume Weighted Average Price
   */
  private calculateVWAP(priceData: PriceData[]): number {
    let volumeSum = 0;
    let volumePriceSum = 0;

    for (const candle of priceData) {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      volumePriceSum += typicalPrice * candle.volume;
      volumeSum += candle.volume;
    }

    return volumeSum > 0 ? volumePriceSum / volumeSum : 0;
  }

  /**
   * Determine volume profile shape/type
   */
  private determineProfileType(nodes: VolumeNode[], poc: VolumeNode): VolumeProfile['profileType'] {
    const sortedByPrice = [...nodes].sort((a, b) => a.price - b.price);
    const pocIndex = sortedByPrice.findIndex(node => node.price === poc.price);
    
    if (pocIndex === -1) return 'normal';
    
    const totalNodes = sortedByPrice.length;
    const pocPosition = pocIndex / totalNodes;

    // Analyze volume distribution
    const topHalfVolume = sortedByPrice.slice(pocIndex).reduce((sum, node) => sum + node.volume, 0);
    const bottomHalfVolume = sortedByPrice.slice(0, pocIndex + 1).reduce((sum, node) => sum + node.volume, 0);
    const totalVolume = topHalfVolume + bottomHalfVolume;

    const topHeavy = topHalfVolume / totalVolume > 0.6;
    const bottomHeavy = bottomHalfVolume / totalVolume > 0.6;

    // D-shape: POC at top
    if (pocPosition > 0.7 && topHeavy) return 'd_shape';
    
    // b-shape: POC at bottom  
    if (pocPosition < 0.3 && bottomHeavy) return 'b_shape';
    
    // P-shape: POC in middle with extension on one side
    if (pocPosition > 0.3 && pocPosition < 0.7) {
      if (topHeavy) return 'p_shape';
      if (bottomHeavy) return 'b_shape';
    }

    return 'normal';
  }

  /**
   * Identify volume-based support levels
   */
  private identifyVolumeSupports(profile: VolumeProfile, priceData: PriceData[]): VolumeSupport[] {
    const currentPrice = priceData[priceData.length - 1].close;
    const supports: VolumeSupport[] = [];

    // Find high-volume nodes below current price
    const supportNodes = profile.nodes.filter(node => 
      node.price < currentPrice && 
      node.volume > profile.nodes[0].volume * 0.3 // At least 30% of max volume
    );

    for (const node of supportNodes.slice(0, 5)) { // Top 5 supports
      const strength = this.calculateSupportStrength(node, priceData, currentPrice);
      const bounces = this.countBounces(node.price, priceData);
      
      supports.push({
        price: node.price,
        volume: node.volume,
        strength,
        type: strength > 70 ? 'major' : strength > 40 ? 'minor' : 'weak',
        bounces,
        lastTest: this.findLastTest(node.price, priceData, 'support')
      });
    }

    // Add Value Area Low as support if below current price
    if (profile.valueAreaLow < currentPrice) {
      const valStrength = this.calculateSupportStrength(
        { price: profile.valueAreaLow, volume: profile.valueArea.volume } as VolumeNode,
        priceData,
        currentPrice
      );
      
      supports.push({
        price: profile.valueAreaLow,
        volume: profile.valueArea.volume,
        strength: valStrength,
        type: valStrength > 70 ? 'major' : 'minor',
        bounces: this.countBounces(profile.valueAreaLow, priceData),
        lastTest: this.findLastTest(profile.valueAreaLow, priceData, 'support')
      });
    }

    return supports.sort((a, b) => b.strength - a.strength);
  }

  /**
   * Identify volume-based resistance levels
   */
  private identifyVolumeResistances(profile: VolumeProfile, priceData: PriceData[]): VolumeResistance[] {
    const currentPrice = priceData[priceData.length - 1].close;
    const resistances: VolumeResistance[] = [];

    // Find high-volume nodes above current price
    const resistanceNodes = profile.nodes.filter(node => 
      node.price > currentPrice && 
      node.volume > profile.nodes[0].volume * 0.3
    );

    for (const node of resistanceNodes.slice(0, 5)) {
      const strength = this.calculateResistanceStrength(node, priceData, currentPrice);
      const rejections = this.countRejections(node.price, priceData);
      
      resistances.push({
        price: node.price,
        volume: node.volume,
        strength,
        type: strength > 70 ? 'major' : strength > 40 ? 'minor' : 'weak',
        rejections,
        lastTest: this.findLastTest(node.price, priceData, 'resistance')
      });
    }

    // Add Value Area High as resistance if above current price
    if (profile.valueAreaHigh > currentPrice) {
      const vahStrength = this.calculateResistanceStrength(
        { price: profile.valueAreaHigh, volume: profile.valueArea.volume } as VolumeNode,
        priceData,
        currentPrice
      );
      
      resistances.push({
        price: profile.valueAreaHigh,
        volume: profile.valueArea.volume,
        strength: vahStrength,
        type: vahStrength > 70 ? 'major' : 'minor',
        rejections: this.countRejections(profile.valueAreaHigh, priceData),
        lastTest: this.findLastTest(profile.valueAreaHigh, priceData, 'resistance')
      });
    }

    return resistances.sort((a, b) => b.strength - a.strength);
  }

  /**
   * Analyze market structure from volume perspective
   */
  private analyzeMarketStructure(profile: VolumeProfile, priceData: PriceData[]) {
    const currentPrice = priceData[priceData.length - 1].close;
    
    // Determine market phase
    let phase: 'accumulation' | 'distribution' | 'trending' | 'rotation';
    
    if (profile.profileType === 'b_shape' && currentPrice < profile.pointOfControl.price) {
      phase = 'accumulation';
    } else if (profile.profileType === 'd_shape' && currentPrice > profile.pointOfControl.price) {
      phase = 'distribution';
    } else if (currentPrice < profile.valueAreaLow || currentPrice > profile.valueAreaHigh) {
      phase = 'trending';
    } else {
      phase = 'rotation';
    }

    // Determine volume balance
    const buyVolumeTotal = profile.nodes.reduce((sum, node) => sum + node.buyVolume, 0);
    const sellVolumeTotal = profile.nodes.reduce((sum, node) => sum + node.sellVolume, 0);
    const totalVolume = buyVolumeTotal + sellVolumeTotal;
    
    let volumeBalance: 'buyer_controlled' | 'seller_controlled' | 'balanced';
    const buyPercent = (buyVolumeTotal / totalVolume) * 100;
    
    if (buyPercent > 55) {
      volumeBalance = 'buyer_controlled';
    } else if (buyPercent < 45) {
      volumeBalance = 'seller_controlled';
    } else {
      volumeBalance = 'balanced';
    }

    // Calculate price discovery efficiency
    const priceRange = Math.max(...priceData.map(d => d.high)) - Math.min(...priceData.map(d => d.low));
    const valueAreaRange = profile.valueArea.priceRange;
    const efficiency = Math.max(0, Math.min(100, (1 - (valueAreaRange / priceRange)) * 100));

    return {
      phase,
      volumeBalance,
      efficiency
    };
  }

  /**
   * Assess volume-related risks
   */
  private assessVolumeRisks(
    profile: VolumeProfile,
    supports: VolumeSupport[],
    resistances: VolumeResistance[]
  ) {
    // Volume risk: Low volume = higher risk
    const avgVolumeNode = profile.nodes.reduce((sum, node) => sum + node.volume, 0) / profile.nodes.length;
    const volumeRisk = Math.max(0, 100 - (avgVolumeNode / 1000000 * 50)); // Simplified

    // Support strength assessment
    const supportStrength = supports.length > 0 ? 
      supports.reduce((sum, s) => sum + s.strength, 0) / supports.length : 0;

    // Resistance strength assessment  
    const resistanceStrength = resistances.length > 0 ?
      resistances.reduce((sum, r) => sum + r.strength, 0) / resistances.length : 0;

    return {
      volumeRisk: Math.min(100, Math.max(0, volumeRisk)),
      supportStrength: Math.min(100, Math.max(0, supportStrength)),
      resistanceStrength: Math.min(100, Math.max(0, resistanceStrength))
    };
  }

  /**
   * Generate volume-based trading signal
   */
  private async generateVolumeSignal(
    symbol: string,
    profile: VolumeProfile,
    supports: VolumeSupport[],
    resistances: VolumeResistance[],
    marketStructure: any,
    priceData: PriceData[]
  ): Promise<VolumeProfileSignal | null> {

    const currentPrice = priceData[priceData.length - 1].close;
    const vwap = profile.volumeWeightedAveragePrice;
    const poc = profile.pointOfControl.price;

    // Determine current position relative to value area
    let currentPosition: VolumeProfileSignal['volumeContext']['currentPosition'];
    if (currentPrice > profile.valueAreaHigh) {
      currentPosition = 'above_vah';
    } else if (currentPrice < profile.valueAreaLow) {
      currentPosition = 'below_val';
    } else if (Math.abs(currentPrice - poc) / currentPrice < 0.005) {
      currentPosition = 'at_poc';
    } else {
      currentPosition = 'within_value_area';
    }

    // Check for signal conditions
    let signalType: VolumeProfileSignal['signalType'] | null = null;
    let action: VolumeProfileSignal['action'] = 'neutral';
    let confidence = 50;
    let targetLevel = currentPrice;
    let stopLoss = currentPrice;

    // POC retest signal
    if (Math.abs(currentPrice - poc) / currentPrice < 0.01) {
      signalType = 'poc_retest';
      confidence = 70;
      
      if (marketStructure.volumeBalance === 'buyer_controlled') {
        action = 'long';
        targetLevel = profile.valueAreaHigh;
        stopLoss = currentPrice * 0.98;
      } else if (marketStructure.volumeBalance === 'seller_controlled') {
        action = 'short';
        targetLevel = profile.valueAreaLow;
        stopLoss = currentPrice * 1.02;
      }
    }

    // Value area breakout
    else if (currentPosition === 'above_vah' || currentPosition === 'below_val') {
      signalType = 'value_area_breakout';
      confidence = 65;
      
      if (currentPosition === 'above_vah' && marketStructure.volumeBalance === 'buyer_controlled') {
        action = 'long';
        targetLevel = currentPrice * 1.05;
        stopLoss = profile.valueAreaHigh;
      } else if (currentPosition === 'below_val' && marketStructure.volumeBalance === 'seller_controlled') {
        action = 'short';
        targetLevel = currentPrice * 0.95;
        stopLoss = profile.valueAreaLow;
      }
    }

    // Volume support/resistance tests
    else if (supports.length > 0 || resistances.length > 0) {
      const nearbySupport = supports.find(s => Math.abs(s.price - currentPrice) / currentPrice < 0.02);
      const nearbyResistance = resistances.find(r => Math.abs(r.price - currentPrice) / currentPrice < 0.02);
      
      if (nearbySupport && nearbySupport.strength > 60) {
        signalType = 'volume_support';
        action = 'long';
        confidence = nearbySupport.strength;
        targetLevel = profile.valueAreaHigh;
        stopLoss = nearbySupport.price * 0.985;
      } else if (nearbyResistance && nearbyResistance.strength > 60) {
        signalType = 'volume_resistance';
        action = 'short';
        confidence = nearbyResistance.strength;
        targetLevel = profile.valueAreaLow;
        stopLoss = nearbyResistance.price * 1.015;
      }
    }

    // VWAP cross
    else if (Math.abs(currentPrice - vwap) / currentPrice < 0.005) {
      signalType = 'vwap_cross';
      confidence = 60;
      
      const recentTrend = this.analyzeTrend(priceData.slice(-10));
      if (recentTrend === 'bullish' && currentPrice > vwap) {
        action = 'long';
        targetLevel = profile.valueAreaHigh;
        stopLoss = vwap * 0.99;
      } else if (recentTrend === 'bearish' && currentPrice < vwap) {
        action = 'short';
        targetLevel = profile.valueAreaLow;
        stopLoss = vwap * 1.01;
      }
    }

    // Don't generate signals with low confidence
    if (!signalType || confidence < 60) {
      return null;
    }

    // Calculate technical alignment
    const technicalAlignment = this.assessTechnicalAlignment(priceData, action);

    // Determine volume trend
    const volumeTrend = this.analyzeVolumeTrend(priceData);

    // Calculate risk/reward
    const risk = Math.abs(currentPrice - stopLoss);
    const reward = Math.abs(targetLevel - currentPrice);
    const riskReward = reward / Math.max(risk, 0.001);

    // Adjust confidence based on technical alignment
    confidence = Math.min(100, confidence + technicalAlignment.overallScore * 0.2);

    const reasoning = [
      `${signalType.replace('_', ' ')} signal at $${currentPrice.toFixed(4)}`,
      `Current position: ${currentPosition.replace('_', ' ')}`,
      `Market phase: ${marketStructure.phase}`,
      `Volume balance: ${marketStructure.volumeBalance}`,
      `POC at $${poc.toFixed(4)}, VWAP at $${vwap.toFixed(4)}`,
      `Technical alignment score: ${technicalAlignment.overallScore}%`
    ];

    return {
      symbol,
      action,
      confidence,
      signalType,
      currentPrice,
      targetLevel,
      stopLoss,
      riskReward,
      timestamp: new Date().toISOString(),
      volumeContext: {
        profile,
        currentPosition,
        volumeTrend,
        profileBalance: marketStructure.volumeBalance === 'balanced' ? 'balanced' :
                       marketStructure.volumeBalance === 'buyer_controlled' ? 'top_heavy' : 'bottom_heavy'
      },
      technicalAlignment,
      reasoning
    };
  }

  // Helper methods
  private calculateSupportStrength(node: VolumeNode, priceData: PriceData[], currentPrice: number): number {
    let strength = Math.min(50, node.volume / 1000000 * 25); // Base volume strength
    
    // Distance penalty (closer = stronger)
    const distance = (currentPrice - node.price) / currentPrice;
    strength += Math.max(0, 30 - distance * 500);
    
    // Time factor (more recent = stronger)
    const bounces = this.countBounces(node.price, priceData);
    strength += Math.min(20, bounces * 5);
    
    return Math.min(100, Math.max(0, strength));
  }

  private calculateResistanceStrength(node: VolumeNode, priceData: PriceData[], currentPrice: number): number {
    let strength = Math.min(50, node.volume / 1000000 * 25);
    
    const distance = (node.price - currentPrice) / currentPrice;
    strength += Math.max(0, 30 - distance * 500);
    
    const rejections = this.countRejections(node.price, priceData);
    strength += Math.min(20, rejections * 5);
    
    return Math.min(100, Math.max(0, strength));
  }

  private countBounces(level: number, priceData: PriceData[]): number {
    let bounces = 0;
    const tolerance = level * 0.01; // 1% tolerance
    
    for (let i = 1; i < priceData.length; i++) {
      const prev = priceData[i - 1];
      const curr = priceData[i];
      
      if (prev.low <= level + tolerance && curr.close > level + tolerance) {
        bounces++;
      }
    }
    
    return bounces;
  }

  private countRejections(level: number, priceData: PriceData[]): number {
    let rejections = 0;
    const tolerance = level * 0.01;
    
    for (let i = 1; i < priceData.length; i++) {
      const prev = priceData[i - 1];
      const curr = priceData[i];
      
      if (prev.high >= level - tolerance && curr.close < level - tolerance) {
        rejections++;
      }
    }
    
    return rejections;
  }

  private findLastTest(level: number, priceData: PriceData[], type: 'support' | 'resistance'): string {
    const tolerance = level * 0.01;
    
    for (let i = priceData.length - 1; i >= 0; i--) {
      const candle = priceData[i];
      
      if (type === 'support' && candle.low <= level + tolerance) {
        return new Date(candle.timestamp).toISOString();
      } else if (type === 'resistance' && candle.high >= level - tolerance) {
        return new Date(candle.timestamp).toISOString();
      }
    }
    
    return new Date().toISOString();
  }

  private analyzeTrend(priceData: PriceData[]): 'bullish' | 'bearish' | 'neutral' {
    if (priceData.length < 3) return 'neutral';
    
    const start = priceData[0].close;
    const end = priceData[priceData.length - 1].close;
    const change = (end - start) / start;
    
    if (change > 0.02) return 'bullish';
    if (change < -0.02) return 'bearish';
    return 'neutral';
  }

  private analyzeVolumeTrend(priceData: PriceData[]): 'increasing' | 'decreasing' | 'stable' {
    if (priceData.length < 5) return 'stable';
    
    const recentVolume = priceData.slice(-3).reduce((sum, d) => sum + d.volume, 0) / 3;
    const olderVolume = priceData.slice(-6, -3).reduce((sum, d) => sum + d.volume, 0) / 3;
    
    const change = (recentVolume - olderVolume) / olderVolume;
    
    if (change > 0.2) return 'increasing';
    if (change < -0.2) return 'decreasing';
    return 'stable';
  }

  private assessTechnicalAlignment(priceData: PriceData[], action: string) {
    let score = 0;
    const prices = priceData.map(d => d.close);
    
    // Trend alignment
    const sma20 = TechnicalIndicators.SMA(prices, 20);
    const currentPrice = prices[prices.length - 1];
    const withTrend = (action === 'long' && currentPrice > sma20) || 
                     (action === 'short' && currentPrice < sma20);
    if (withTrend) score += 25;
    
    // Momentum alignment
    const rsi = TechnicalIndicators.RSI(prices);
    const withMomentum = (action === 'long' && rsi.value < 70) || 
                        (action === 'short' && rsi.value > 30);
    if (withMomentum) score += 25;
    
    // Volume confirmation
    const recentVolume = priceData.slice(-3).reduce((sum, d) => sum + d.volume, 0);
    const avgVolume = priceData.reduce((sum, d) => sum + d.volume, 0) / priceData.length;
    const withVolume = recentVolume > avgVolume * 1.2;
    if (withVolume) score += 25;
    
    // MACD alignment
    const macd = TechnicalIndicators.MACD(prices);
    const macdBullish = macd.histogram > 0;
    const withMacd = (action === 'long' && macdBullish) || (action === 'short' && !macdBullish);
    if (withMacd) score += 25;
    
    return {
      withTrend,
      withMomentum,
      withVolume,
      overallScore: score
    };
  }

  private getDefaultResult(): VolumeProfileAnalysisResult {
    return {
      signal: null,
      profile: {
        nodes: [],
        pointOfControl: { price: 0, volume: 0, buyVolume: 0, sellVolume: 0, transactions: 0, timeSpent: 0, volumePercent: 0 },
        valueAreaHigh: 0,
        valueAreaLow: 0,
        valueArea: { volume: 0, volumePercent: 0, priceRange: 0 },
        volumeWeightedAveragePrice: 0,
        profileType: 'normal'
      },
      supports: [],
      resistances: [],
      marketStructure: {
        phase: 'rotation',
        volumeBalance: 'balanced',
        efficiency: 50
      },
      riskAssessment: {
        volumeRisk: 50,
        supportStrength: 50,
        resistanceStrength: 50
      }
    };
  }
}

export const volumeProfileAnalysisStrategy = new VolumeProfileAnalysisStrategy();
export default VolumeProfileAnalysisStrategy;