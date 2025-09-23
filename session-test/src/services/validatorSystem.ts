/**
 * 6-Validator System for Trading Signal Validation
 * Implements comprehensive multi-factor validation for trading decisions
 */

import { TechnicalIndicators, PriceData } from './technicalIndicators';
import { CryptoData, TradingSignal } from '../types/trading';
import { groqService } from './groqService';
import { fundingRatesService } from './fundingRatesService';

export interface ValidationResult {
  passed: boolean;
  score: number; // 0-100
  reason: string;
  details?: any;
}

export interface ValidatorConfig {
  name: string;
  weight: number; // Weight in final score calculation
  enabled: boolean;
  threshold: number; // Minimum score to pass
}

export interface CompositeValidationResult {
  finalScore: number;
  passed: boolean;
  confidence: number;
  validators: { [key: string]: ValidationResult };
  recommendation: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';
  reasons: string[];
}

export class ValidatorSystem {
  private validators: ValidatorConfig[] = [
    { name: 'trend', weight: 0.20, enabled: true, threshold: 60 },
    { name: 'volume', weight: 0.15, enabled: true, threshold: 50 },
    { name: 'volatility', weight: 0.15, enabled: true, threshold: 40 },
    { name: 'riskReward', weight: 0.25, enabled: true, threshold: 70 },
    { name: 'sentiment', weight: 0.15, enabled: true, threshold: 50 },
    { name: 'positionSize', weight: 0.10, enabled: true, threshold: 60 }
  ];

  constructor(private config?: Partial<ValidatorConfig>[]) {
    if (config) {
      this.validators = this.validators.map(v => {
        const override = config.find(c => c.name === v.name);
        return override ? { ...v, ...override } : v;
      });
    }
  }

  /**
   * 1. TREND VALIDATOR
   * Confirms trend direction using multiple timeframes and indicators
   */
  private async validateTrend(
    signal: TradingSignal,
    priceData: PriceData[],
    cryptoData: CryptoData
  ): Promise<ValidationResult> {
    if (priceData.length < 50) {
      return {
        passed: false,
        score: 0,
        reason: 'Insufficient price data for trend analysis'
      };
    }

    const prices = priceData.map(p => p.close);
    let score = 0;
    const details: any = {};

    // Check moving average alignment
    const sma20 = TechnicalIndicators.SMA(prices, 20);
    const sma50 = TechnicalIndicators.SMA(prices, Math.min(50, prices.length));
    const sma200 = TechnicalIndicators.SMA(prices, Math.min(200, prices.length));
    const currentPrice = prices[prices.length - 1];

    // Bullish: Price > SMA20 > SMA50 > SMA200
    // Bearish: Price < SMA20 < SMA50 < SMA200
    if (signal.action === 'buy') {
      if (currentPrice > sma20) score += 25;
      if (sma20 > sma50) score += 25;
      if (sma50 > sma200) score += 25;
      
      // Check if making higher highs and higher lows
      const recentHighs = priceData.slice(-10).map(p => p.high);
      const recentLows = priceData.slice(-10).map(p => p.low);
      const trend = this.analyzeTrend(recentHighs, recentLows);
      if (trend === 'uptrend') score += 25;
      
      details.alignment = `Price: ${currentPrice.toFixed(2)}, SMA20: ${sma20.toFixed(2)}, SMA50: ${sma50.toFixed(2)}`;
    } else {
      if (currentPrice < sma20) score += 25;
      if (sma20 < sma50) score += 25;
      if (sma50 < sma200) score += 25;
      
      const recentHighs = priceData.slice(-10).map(p => p.high);
      const recentLows = priceData.slice(-10).map(p => p.low);
      const trend = this.analyzeTrend(recentHighs, recentLows);
      if (trend === 'downtrend') score += 25;
    }

    // MACD confirmation
    const macd = TechnicalIndicators.MACD(prices);
    if ((signal.action === 'buy' && macd.trend === 'bullish') ||
        (signal.action === 'sell' && macd.trend === 'bearish')) {
      score = Math.min(100, score + 20);
    }

    details.macdTrend = macd.trend;
    details.priceChange24h = cryptoData.changePercent;

    return {
      passed: score >= 60,
      score,
      reason: `Trend ${score >= 60 ? 'confirms' : 'does not confirm'} ${signal.action} signal`,
      details
    };
  }

  /**
   * 2. VOLUME VALIDATOR
   * Ensures sufficient trading volume and detects volume anomalies
   */
  private async validateVolume(
    signal: TradingSignal,
    priceData: PriceData[],
    cryptoData: CryptoData
  ): Promise<ValidationResult> {
    if (priceData.length < 20) {
      return {
        passed: false,
        score: 0,
        reason: 'Insufficient volume data'
      };
    }

    let score = 50; // Base score
    const details: any = {};

    // Calculate average volume
    const volumes = priceData.map(p => p.volume);
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    const currentVolume = volumes[volumes.length - 1];
    const volumeRatio = currentVolume / avgVolume;

    details.avgVolume = avgVolume;
    details.currentVolume = currentVolume;
    details.volumeRatio = volumeRatio;

    // Check if current volume is above average (good for breakouts)
    if (volumeRatio > 1.5) {
      score += 30;
      details.volumeSignal = 'High volume detected';
    } else if (volumeRatio > 1.0) {
      score += 15;
      details.volumeSignal = 'Above average volume';
    } else if (volumeRatio < 0.5) {
      score -= 20;
      details.volumeSignal = 'Low volume warning';
    }

    // Check 24h volume from crypto data
    if (cryptoData.volume24h && cryptoData.volume24h > 1000000) {
      score += 20;
      details.volume24h = cryptoData.volume24h;
    }

    // Volume trend analysis
    const recentVolumes = volumes.slice(-5);
    const oldVolumes = volumes.slice(-10, -5);
    const recentAvg = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length;
    const oldAvg = oldVolumes.reduce((a, b) => a + b, 0) / oldVolumes.length;
    
    if (recentAvg > oldAvg * 1.2) {
      score = Math.min(100, score + 15);
      details.volumeTrend = 'Increasing';
    } else if (recentAvg < oldAvg * 0.8) {
      score = Math.max(0, score - 15);
      details.volumeTrend = 'Decreasing';
    }

    return {
      passed: score >= 50,
      score,
      reason: `Volume ${score >= 50 ? 'sufficient' : 'insufficient'} for trading`,
      details
    };
  }

  /**
   * 3. VOLATILITY VALIDATOR
   * Checks if volatility is within acceptable range using ATR
   */
  private async validateVolatility(
    signal: TradingSignal,
    priceData: PriceData[],
    settings: { volatilityTolerance?: 'low' | 'medium' | 'high' }
  ): Promise<ValidationResult> {
    if (priceData.length < 14) {
      return {
        passed: false,
        score: 0,
        reason: 'Insufficient data for volatility analysis'
      };
    }

    const details: any = {};
    let score = 50;

    // Calculate ATR (Average True Range)
    const atr = TechnicalIndicators.ATR(priceData, 14);
    const currentPrice = priceData[priceData.length - 1].close;
    const atrPercent = (atr / currentPrice) * 100;

    details.atr = atr;
    details.atrPercent = atrPercent;

    // Define volatility thresholds based on tolerance
    const tolerance = settings.volatilityTolerance || 'medium';
    let maxVolatility: number;
    let minVolatility: number;

    switch (tolerance) {
      case 'low':
        maxVolatility = 3;
        minVolatility = 0.5;
        break;
      case 'high':
        maxVolatility = 10;
        minVolatility = 0.1;
        break;
      default: // medium
        maxVolatility = 5;
        minVolatility = 0.3;
    }

    // Score based on volatility level
    if (atrPercent < minVolatility) {
      score = 30;
      details.assessment = 'Too low volatility - limited profit potential';
    } else if (atrPercent > maxVolatility) {
      score = 20;
      details.assessment = 'Too high volatility - increased risk';
    } else {
      // Optimal volatility range
      const optimalMid = (maxVolatility + minVolatility) / 2;
      const distance = Math.abs(atrPercent - optimalMid);
      const range = (maxVolatility - minVolatility) / 2;
      score = 100 - (distance / range * 50);
      details.assessment = 'Volatility within acceptable range';
    }

    // Bollinger Bands width check
    const prices = priceData.map(p => p.close);
    const bb = TechnicalIndicators.BollingerBands(prices);
    details.bollingerBandwidth = bb.bandwidth;

    if (bb.bandwidth > 0.1) {
      score = Math.min(100, score + 10);
    } else if (bb.bandwidth < 0.02) {
      score = Math.max(0, score - 20);
      details.bbWarning = 'Bollinger Bands squeeze detected';
    }

    return {
      passed: score >= 40,
      score,
      reason: details.assessment || 'Volatility check completed',
      details
    };
  }

  /**
   * 4. RISK/REWARD VALIDATOR
   * Enforces minimum risk/reward ratio and calculates potential profit vs loss
   */
  private async validateRiskReward(
    signal: TradingSignal,
    priceData: PriceData[],
    minRatio: number = 2.0
  ): Promise<ValidationResult> {
    const currentPrice = priceData[priceData.length - 1].close;
    const details: any = {};
    let score = 0;

    // Calculate support and resistance levels
    const levels = TechnicalIndicators.SupportResistance(priceData, 20);
    
    let stopLoss: number;
    let takeProfit: number;
    
    if (signal.action === 'buy') {
      // For buy signals
      stopLoss = levels.support[0] || currentPrice * 0.98; // 2% default stop loss
      takeProfit = levels.resistance[0] || currentPrice * 1.06; // 6% default take profit
    } else {
      // For sell signals
      stopLoss = levels.resistance[0] || currentPrice * 1.02;
      takeProfit = levels.support[0] || currentPrice * 0.94;
    }

    const risk = Math.abs(currentPrice - stopLoss);
    const reward = Math.abs(takeProfit - currentPrice);
    const ratio = reward / risk;

    details.currentPrice = currentPrice;
    details.stopLoss = stopLoss;
    details.takeProfit = takeProfit;
    details.risk = risk;
    details.reward = reward;
    details.ratio = ratio;

    // Score based on risk/reward ratio
    if (ratio >= minRatio * 1.5) {
      score = 100;
      details.assessment = 'Excellent risk/reward ratio';
    } else if (ratio >= minRatio) {
      score = 70 + (ratio - minRatio) * 30 / (minRatio * 0.5);
      details.assessment = 'Good risk/reward ratio';
    } else if (ratio >= minRatio * 0.75) {
      score = 40 + (ratio - minRatio * 0.75) * 30 / (minRatio * 0.25);
      details.assessment = 'Marginal risk/reward ratio';
    } else {
      score = ratio / minRatio * 40;
      details.assessment = 'Poor risk/reward ratio';
    }

    // Additional scoring based on probability
    const technicalScore = TechnicalIndicators.getCompositeScore(priceData);
    const probabilityBonus = technicalScore.score / 100 * 20;
    score = Math.min(100, score + probabilityBonus);
    details.technicalScore = technicalScore;

    return {
      passed: ratio >= minRatio && score >= 70,
      score,
      reason: `R/R ratio ${ratio.toFixed(2)}:1 ${ratio >= minRatio ? 'meets' : 'fails'} minimum requirement`,
      details
    };
  }

  /**
   * 5. SENTIMENT VALIDATOR
   * Uses AI sentiment analysis, market indicators, and funding rates
   */
  private async validateSentiment(
    signal: TradingSignal,
    cryptoData: CryptoData,
    fearGreedIndex?: number
  ): Promise<ValidationResult> {
    let score = 50; // Base score
    const details: any = {};
    
    // Check funding rates for sentiment
    try {
      const fundingRates = await fundingRatesService.getFundingRates(signal.symbol);
      const fundingSentiment = fundingRatesService.analyzeSentiment(fundingRates);
      
      details.fundingRate = fundingRates.averageRate;
      details.fundingSentiment = fundingSentiment.sentiment;
      
      // Adjust score based on funding sentiment alignment
      if (signal.action === 'BUY' && fundingSentiment.sentiment === 'bullish') {
        score += 20;
        details.fundingSignal = 'Funding rates support long position';
      } else if (signal.action === 'SELL' && fundingSentiment.sentiment === 'bearish') {
        score += 20;
        details.fundingSignal = 'Funding rates support short position';
      } else if (fundingRates.signal === 'extreme') {
        score -= 15;
        details.fundingSignal = fundingRates.extremeWarning;
      }
      
      // Add funding confidence to overall score
      const fundingBonus = (fundingSentiment.confidence - 50) / 50 * 10;
      score = Math.max(0, Math.min(100, score + fundingBonus));
    } catch (error) {
      console.warn('Failed to get funding rates:', error);
    }

    // Fear & Greed Index analysis
    if (fearGreedIndex !== undefined) {
      details.fearGreedIndex = fearGreedIndex;
      
      if (signal.action === 'buy') {
        // For buy signals, extreme fear (< 20) is good, extreme greed (> 80) is bad
        if (fearGreedIndex < 20) {
          score += 30;
          details.fgiSignal = 'Extreme fear - contrarian buy opportunity';
        } else if (fearGreedIndex < 40) {
          score += 15;
          details.fgiSignal = 'Fear - potential buy opportunity';
        } else if (fearGreedIndex > 80) {
          score -= 20;
          details.fgiSignal = 'Extreme greed - caution advised';
        }
      } else {
        // For sell signals, extreme greed is good, extreme fear is bad
        if (fearGreedIndex > 80) {
          score += 30;
          details.fgiSignal = 'Extreme greed - good time to sell';
        } else if (fearGreedIndex > 60) {
          score += 15;
          details.fgiSignal = 'Greed - consider selling';
        } else if (fearGreedIndex < 20) {
          score -= 20;
          details.fgiSignal = 'Extreme fear - poor time to sell';
        }
      }
    }

    // Market momentum from 24h change
    const momentum = cryptoData.changePercent || 0;
    details.momentum24h = momentum;
    
    if (signal.action === 'buy') {
      if (momentum > 5) {
        score += 20;
        details.momentumSignal = 'Strong positive momentum';
      } else if (momentum > 0) {
        score += 10;
        details.momentumSignal = 'Positive momentum';
      } else if (momentum < -5) {
        score -= 15;
        details.momentumSignal = 'Strong negative momentum';
      }
    } else {
      if (momentum < -5) {
        score += 20;
        details.momentumSignal = 'Strong downward momentum';
      } else if (momentum > 5) {
        score -= 15;
        details.momentumSignal = 'Strong upward momentum against sell';
      }
    }

    // AI confidence from original signal
    if (signal.confidence) {
      const confidenceBonus = (signal.confidence - 50) / 50 * 20;
      score = Math.max(0, Math.min(100, score + confidenceBonus));
      details.aiConfidence = signal.confidence;
    }

    return {
      passed: score >= 50,
      score,
      reason: `Market sentiment ${score >= 50 ? 'supports' : 'opposes'} ${signal.action} signal`,
      details
    };
  }

  /**
   * 6. POSITION SIZE VALIDATOR
   * Ensures proper position sizing and portfolio allocation
   */
  private async validatePositionSize(
    signal: TradingSignal,
    portfolioValue: number,
    existingPositions: number,
    maxPositionPercent: number = 10,
    maxPositions: number = 10
  ): Promise<ValidationResult> {
    let score = 100; // Start with perfect score
    const details: any = {};

    // Check number of existing positions
    details.existingPositions = existingPositions;
    details.maxPositions = maxPositions;
    
    if (existingPositions >= maxPositions) {
      return {
        passed: false,
        score: 0,
        reason: `Maximum number of positions (${maxPositions}) reached`,
        details
      };
    }

    // Score based on position count
    const positionUtilization = existingPositions / maxPositions;
    if (positionUtilization > 0.8) {
      score -= 30;
      details.warning = 'Approaching maximum position limit';
    } else if (positionUtilization > 0.6) {
      score -= 15;
    }

    // Calculate recommended position size
    const basePositionSize = portfolioValue * (maxPositionPercent / 100);
    
    // Adjust based on signal strength
    const signalStrength = signal.confidence || 50;
    const adjustmentFactor = 0.5 + (signalStrength / 100);
    const recommendedSize = basePositionSize * adjustmentFactor;
    
    details.portfolioValue = portfolioValue;
    details.basePositionSize = basePositionSize;
    details.recommendedSize = recommendedSize;
    details.percentOfPortfolio = (recommendedSize / portfolioValue) * 100;

    // Ensure position size is reasonable
    if (recommendedSize < portfolioValue * 0.01) {
      score -= 30;
      details.issue = 'Position size too small';
    } else if (recommendedSize > portfolioValue * 0.15) {
      score -= 40;
      details.issue = 'Position size too large';
    }

    // Kelly Criterion calculation (simplified)
    const winProbability = signalStrength / 100;
    const winLossRatio = 2; // Assume 2:1 risk/reward
    const kellyPercent = ((winProbability * winLossRatio - (1 - winProbability)) / winLossRatio) * 100;
    const safeKelly = Math.max(0, Math.min(maxPositionPercent, kellyPercent * 0.25)); // Use 25% Kelly for safety
    
    details.kellyPercent = kellyPercent;
    details.safeKellyPercent = safeKelly;

    return {
      passed: score >= 60,
      score,
      reason: `Position sizing ${score >= 60 ? 'appropriate' : 'needs adjustment'}`,
      details
    };
  }

  /**
   * Helper function to analyze price trend
   */
  private analyzeTrend(highs: number[], lows: number[]): 'uptrend' | 'downtrend' | 'sideways' {
    let higherHighs = 0;
    let lowerLows = 0;
    
    for (let i = 1; i < highs.length; i++) {
      if (highs[i] > highs[i - 1]) higherHighs++;
      if (lows[i] < lows[i - 1]) lowerLows++;
    }
    
    if (higherHighs > lows.length * 0.6) return 'uptrend';
    if (lowerLows > highs.length * 0.6) return 'downtrend';
    return 'sideways';
  }

  /**
   * Main validation method - runs all validators and returns composite result
   */
  public async validate(
    signal: TradingSignal,
    priceData: PriceData[],
    cryptoData: CryptoData,
    context: {
      portfolioValue: number;
      existingPositions: number;
      fearGreedIndex?: number;
      settings?: any;
    }
  ): Promise<CompositeValidationResult> {
    const results: { [key: string]: ValidationResult } = {};
    const enabledValidators = this.validators.filter(v => v.enabled);
    
    // Run all validators in parallel
    const [trend, volume, volatility, riskReward, sentiment, positionSize] = await Promise.all([
      this.validateTrend(signal, priceData, cryptoData),
      this.validateVolume(signal, priceData, cryptoData),
      this.validateVolatility(signal, priceData, context.settings || {}),
      this.validateRiskReward(signal, priceData, context.settings?.minRiskReward || 2.0),
      this.validateSentiment(signal, cryptoData, context.fearGreedIndex),
      this.validatePositionSize(
        signal,
        context.portfolioValue,
        context.existingPositions,
        context.settings?.maxPositionSize || 10,
        context.settings?.maxPositions || 10
      )
    ]);

    // Store results
    results.trend = trend;
    results.volume = volume;
    results.volatility = volatility;
    results.riskReward = riskReward;
    results.sentiment = sentiment;
    results.positionSize = positionSize;

    // Calculate weighted final score
    let finalScore = 0;
    let totalWeight = 0;
    let passedValidators = 0;
    const reasons: string[] = [];

    for (const validator of enabledValidators) {
      const result = results[validator.name];
      if (result) {
        finalScore += result.score * validator.weight;
        totalWeight += validator.weight;
        
        if (result.passed) {
          passedValidators++;
          reasons.push(`✅ ${validator.name}: ${result.reason}`);
        } else {
          reasons.push(`❌ ${validator.name}: ${result.reason}`);
        }
      }
    }

    // Normalize final score
    finalScore = totalWeight > 0 ? finalScore / totalWeight : 0;
    
    // Determine if signal passes (at least 4/6 validators must pass)
    const passed = passedValidators >= 4 && finalScore >= 60;
    
    // Calculate confidence based on how many validators passed
    const confidence = (passedValidators / enabledValidators.length) * 100;
    
    // Determine recommendation
    let recommendation: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';
    if (signal.action === 'buy') {
      if (finalScore >= 80 && passedValidators >= 5) recommendation = 'strong_buy';
      else if (finalScore >= 60 && passedValidators >= 4) recommendation = 'buy';
      else recommendation = 'hold';
    } else {
      if (finalScore >= 80 && passedValidators >= 5) recommendation = 'strong_sell';
      else if (finalScore >= 60 && passedValidators >= 4) recommendation = 'sell';
      else recommendation = 'hold';
    }

    return {
      finalScore,
      passed,
      confidence,
      validators: results,
      recommendation,
      reasons
    };
  }
}

export default ValidatorSystem;