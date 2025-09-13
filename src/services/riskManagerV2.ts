/**
 * Enhanced Risk Manager Service V2
 * Advanced position sizing, risk controls, and portfolio protection
 */

import { Account, Position, CryptoData } from '../types/trading';
import { portfolioAnalytics } from './portfolioAnalytics';
import { fundingRatesService } from './fundingRatesService';
import { TechnicalIndicators, PriceData } from './technicalIndicators';

export interface RiskDecision {
  proceed: boolean;
  reason: string;
  positionSize: number; // USD amount
  leverage: number; // 1-10x
  stopLoss: number; // Price level
  takeProfit: number; // Price level
  stopLossPercent: number; // % from entry
  takeProfitPercent: number; // % from entry
  riskAmount: number; // USD at risk
  rewardAmount: number; // USD potential profit
  riskRewardRatio: number;
  kellyPercent: number; // Kelly Criterion suggestion
  confidenceAdjustedSize: number; // Final adjusted position size
}

export interface RiskParameters {
  maxPositionSizePercent: number; // Max % of portfolio per position
  maxTotalExposurePercent: number; // Max % of portfolio in all positions
  maxDrawdownPercent: number; // Max allowed drawdown
  minRiskRewardRatio: number; // Minimum R/R ratio
  maxLeverage: number; // Maximum leverage allowed
  volatilityMultiplier: number; // Position size adjustment for volatility
  useKellyCriterion: boolean; // Use Kelly for position sizing
  kellyFraction: number; // Fraction of Kelly to use (0.25 = 25% Kelly)
  trailingStopPercent?: number; // Trailing stop distance
  correlationLimit: number; // Max correlation between positions
}

export interface PortfolioRisk {
  totalExposure: number;
  totalExposurePercent: number;
  openPositions: number;
  averagePositionSize: number;
  largestPosition: number;
  portfolioVolatility: number;
  correlationRisk: number;
  marginUsed: number;
  marginAvailable: number;
  liquidationPrice?: number;
  riskScore: number; // 0-100, higher is riskier
}

class RiskManagerV2 {
  private peakEquity: number = 50000;
  private defaultParams: RiskParameters = {
    maxPositionSizePercent: 10,
    maxTotalExposurePercent: 60,
    maxDrawdownPercent: 15,
    minRiskRewardRatio: 2,
    maxLeverage: 3,
    volatilityMultiplier: 1,
    useKellyCriterion: true,
    kellyFraction: 0.25,
    trailingStopPercent: 5,
    correlationLimit: 0.7
  };
  
  private correlationMatrix: Map<string, Map<string, number>> = new Map();

  /**
   * Update peak equity for drawdown calculation
   */
  updatePeakEquity(account: Account | null) {
    const currentValue = account?.portfolio_value || 0;
    if (currentValue > this.peakEquity) {
      this.peakEquity = currentValue;
    }
  }

  /**
   * Main risk evaluation for new trades
   */
  async evaluateTrade(
    account: Account | null,
    positions: Position[],
    signal: {
      symbol: string;
      action: 'BUY' | 'SELL';
      price: number;
      confidence: number;
    },
    priceData: PriceData[],
    params?: Partial<RiskParameters>
  ): Promise<RiskDecision> {
    const riskParams = { ...this.defaultParams, ...params };
    const portfolioValue = account?.portfolio_value || 50000;
    
    // 1. Portfolio-level checks
    const portfolioRisk = await this.assessPortfolioRisk(account, positions);
    
    // Check drawdown
    const currentDrawdown = this.calculateDrawdown(portfolioValue);
    if (currentDrawdown > riskParams.maxDrawdownPercent) {
      return this.blockTrade(`Portfolio drawdown ${currentDrawdown.toFixed(1)}% exceeds limit ${riskParams.maxDrawdownPercent}%`);
    }
    
    // Check total exposure
    if (portfolioRisk.totalExposurePercent > riskParams.maxTotalExposurePercent) {
      return this.blockTrade(`Total exposure ${portfolioRisk.totalExposurePercent.toFixed(1)}% exceeds limit ${riskParams.maxTotalExposurePercent}%`);
    }
    
    // Check correlation risk
    if (portfolioRisk.correlationRisk > riskParams.correlationLimit) {
      return this.blockTrade(`Portfolio correlation risk ${portfolioRisk.correlationRisk.toFixed(2)} too high`);
    }
    
    // 2. Calculate volatility-adjusted position size
    const atr = TechnicalIndicators.ATR(priceData, 14);
    const atrPercent = (atr / signal.price) * 100;
    const volatilityAdjustment = this.calculateVolatilityAdjustment(atrPercent, riskParams.volatilityMultiplier);
    
    // 3. Calculate base position size
    const basePositionSize = portfolioValue * (riskParams.maxPositionSizePercent / 100);
    const volatilityAdjustedSize = basePositionSize * volatilityAdjustment;
    
    // 4. Apply Kelly Criterion if enabled
    let kellyPercent = 0;
    let kellySize = volatilityAdjustedSize;
    
    if (riskParams.useKellyCriterion) {
      kellyPercent = this.calculateKellyCriterion(signal.confidence, riskParams.minRiskRewardRatio);
      const fullKellySize = portfolioValue * (kellyPercent / 100);
      kellySize = fullKellySize * riskParams.kellyFraction; // Use fraction of Kelly
      kellySize = Math.min(kellySize, volatilityAdjustedSize); // Don't exceed volatility-adjusted size
    }
    
    // 5. Confidence adjustment
    const confidenceMultiplier = 0.5 + (signal.confidence / 100) * 0.5; // 50-100% of size based on confidence
    const confidenceAdjustedSize = kellySize * confidenceMultiplier;
    
    // 6. Calculate stop loss and take profit levels
    const { stopLoss, takeProfit, stopPercent, takeProfitPercent } = this.calculateExitLevels(
      signal.price,
      signal.action,
      priceData,
      riskParams.minRiskRewardRatio
    );
    
    // 7. Calculate risk and reward amounts
    const riskAmount = confidenceAdjustedSize * (stopPercent / 100);
    const rewardAmount = confidenceAdjustedSize * (takeProfitPercent / 100);
    const actualRiskRewardRatio = rewardAmount / riskAmount;
    
    // 8. Check if R/R ratio meets minimum
    if (actualRiskRewardRatio < riskParams.minRiskRewardRatio) {
      return this.blockTrade(`Risk/Reward ratio ${actualRiskRewardRatio.toFixed(2)} below minimum ${riskParams.minRiskRewardRatio}`);
    }
    
    // 9. Determine leverage (if allowed)
    let leverage = 1;
    if (riskParams.maxLeverage > 1) {
      // Use leverage based on confidence and volatility
      const maxSafeLeverage = Math.min(riskParams.maxLeverage, 10 / atrPercent); // Lower leverage for high volatility
      leverage = 1 + (signal.confidence / 100) * (maxSafeLeverage - 1);
      leverage = Math.min(leverage, riskParams.maxLeverage);
    }
    
    // 10. Check funding rates impact for leveraged positions
    if (leverage > 1) {
      const fundingAnalysis = await fundingRatesService.getFundingRates(signal.symbol);
      const fundingImpact = fundingRatesService.calculateFundingImpact(
        confidenceAdjustedSize * leverage,
        fundingAnalysis.averageRate,
        24 // Assume 24 hour hold
      );
      
      if (Math.abs(fundingImpact.costPercent) > 1) {
        // Reduce leverage if funding costs are too high
        leverage = Math.max(1, leverage * 0.5);
      }
    }
    
    // Final position size with leverage
    const finalPositionSize = Math.min(
      confidenceAdjustedSize,
      portfolioValue * 0.2, // Never more than 20% of portfolio
      account?.available_balance || portfolioValue
    );
    
    return {
      proceed: true,
      reason: 'Trade approved after risk assessment',
      positionSize: finalPositionSize,
      leverage,
      stopLoss,
      takeProfit,
      stopLossPercent: stopPercent,
      takeProfitPercent,
      riskAmount,
      rewardAmount,
      riskRewardRatio: actualRiskRewardRatio,
      kellyPercent,
      confidenceAdjustedSize: finalPositionSize
    };
  }

  /**
   * Calculate volatility adjustment factor
   */
  private calculateVolatilityAdjustment(atrPercent: number, multiplier: number): number {
    // Reduce position size for high volatility
    // ATR% of 2% = normal, 4% = half size, 1% = 1.5x size
    const baseVolatility = 2; // 2% ATR as baseline
    const adjustment = (baseVolatility / Math.max(0.5, atrPercent)) * multiplier;
    return Math.min(2, Math.max(0.25, adjustment)); // Clamp between 0.25x and 2x
  }

  /**
   * Calculate Kelly Criterion for position sizing
   */
  private calculateKellyCriterion(confidencePercent: number, riskRewardRatio: number): number {
    const winProbability = confidencePercent / 100;
    const lossProbability = 1 - winProbability;
    const b = riskRewardRatio; // Odds received on the wager
    
    // Kelly formula: f = (p*b - q) / b
    // where f = fraction of capital to wager, p = probability of win, q = probability of loss, b = odds
    const kelly = (winProbability * b - lossProbability) / b;
    
    // Convert to percentage and cap at reasonable levels
    const kellyPercent = Math.max(0, Math.min(25, kelly * 100)); // Cap at 25% max
    return kellyPercent;
  }

  /**
   * Calculate stop loss and take profit levels
   */
  private calculateExitLevels(
    entryPrice: number,
    action: 'BUY' | 'SELL',
    priceData: PriceData[],
    minRiskReward: number
  ): {
    stopLoss: number;
    takeProfit: number;
    stopPercent: number;
    takeProfitPercent: number;
  } {
    // Calculate ATR for dynamic stops
    const atr = TechnicalIndicators.ATR(priceData, 14);
    const supportResistance = TechnicalIndicators.SupportResistance(priceData, 20);
    
    let stopLoss: number;
    let takeProfit: number;
    
    if (action === 'BUY') {
      // For longs: stop below support or 1.5 ATR below entry
      const supportStop = supportResistance.support[0] || entryPrice * 0.97;
      const atrStop = entryPrice - (atr * 1.5);
      stopLoss = Math.max(supportStop, atrStop, entryPrice * 0.95); // At least 5% stop
      
      // Take profit at resistance or based on R/R ratio
      const resistanceTarget = supportResistance.resistance[0] || entryPrice * 1.1;
      const rrTarget = entryPrice + (entryPrice - stopLoss) * minRiskReward;
      takeProfit = Math.max(resistanceTarget, rrTarget);
    } else {
      // For shorts: stop above resistance or 1.5 ATR above entry
      const resistanceStop = supportResistance.resistance[0] || entryPrice * 1.03;
      const atrStop = entryPrice + (atr * 1.5);
      stopLoss = Math.min(resistanceStop, atrStop, entryPrice * 1.05);
      
      // Take profit at support or based on R/R ratio
      const supportTarget = supportResistance.support[0] || entryPrice * 0.9;
      const rrTarget = entryPrice - (stopLoss - entryPrice) * minRiskReward;
      takeProfit = Math.min(supportTarget, rrTarget);
    }
    
    const stopPercent = Math.abs((stopLoss - entryPrice) / entryPrice) * 100;
    const takeProfitPercent = Math.abs((takeProfit - entryPrice) / entryPrice) * 100;
    
    return {
      stopLoss,
      takeProfit,
      stopPercent,
      takeProfitPercent
    };
  }

  /**
   * Assess overall portfolio risk
   */
  async assessPortfolioRisk(account: Account | null, positions: Position[]): Promise<PortfolioRisk> {
    const portfolioValue = account?.portfolio_value || 50000;
    const totalExposure = positions.reduce((sum, p) => sum + p.market_value, 0);
    const totalExposurePercent = (totalExposure / portfolioValue) * 100;
    
    // Calculate position sizes
    const positionSizes = positions.map(p => p.market_value);
    const averagePositionSize = positionSizes.length > 0 
      ? positionSizes.reduce((a, b) => a + b, 0) / positionSizes.length 
      : 0;
    const largestPosition = positionSizes.length > 0 ? Math.max(...positionSizes) : 0;
    
    // Calculate portfolio volatility (simplified)
    const positionVolatilities = positions.map(p => Math.abs(p.unrealized_plpc || 0));
    const portfolioVolatility = positionVolatilities.length > 0
      ? Math.sqrt(positionVolatilities.reduce((sum, v) => sum + v * v, 0) / positionVolatilities.length)
      : 0;
    
    // Calculate correlation risk
    const correlationRisk = this.calculateCorrelationRisk(positions);
    
    // Margin calculations
    const marginUsed = totalExposure * 0.1; // Assume 10% margin requirement
    const marginAvailable = Math.max(0, portfolioValue - marginUsed);
    
    // Risk score (0-100)
    let riskScore = 0;
    riskScore += (totalExposurePercent / 100) * 30; // 30 points for exposure
    riskScore += (portfolioVolatility / 10) * 20; // 20 points for volatility
    riskScore += correlationRisk * 20; // 20 points for correlation
    riskScore += (positions.length / 20) * 15; // 15 points for position count
    riskScore += (largestPosition / portfolioValue) * 15; // 15 points for concentration
    
    return {
      totalExposure,
      totalExposurePercent,
      openPositions: positions.length,
      averagePositionSize,
      largestPosition,
      portfolioVolatility,
      correlationRisk,
      marginUsed,
      marginAvailable,
      riskScore: Math.min(100, riskScore)
    };
  }

  /**
   * Calculate correlation risk between positions
   */
  private calculateCorrelationRisk(positions: Position[]): number {
    if (positions.length < 2) return 0;
    
    // Simplified correlation based on symbol similarity
    // In production, would use actual price correlation data
    const symbols = positions.map(p => p.symbol);
    let correlationSum = 0;
    let pairCount = 0;
    
    for (let i = 0; i < symbols.length; i++) {
      for (let j = i + 1; j < symbols.length; j++) {
        const correlation = this.estimateCorrelation(symbols[i], symbols[j]);
        correlationSum += Math.abs(correlation);
        pairCount++;
      }
    }
    
    return pairCount > 0 ? correlationSum / pairCount : 0;
  }

  /**
   * Estimate correlation between two assets
   */
  private estimateCorrelation(symbol1: string, symbol2: string): number {
    // Simplified correlation estimates
    const highCorrelationPairs = [
      ['BTC', 'ETH'],
      ['SOL', 'AVAX'],
      ['MATIC', 'ARB']
    ];
    
    for (const pair of highCorrelationPairs) {
      if ((symbol1.includes(pair[0]) && symbol2.includes(pair[1])) ||
          (symbol1.includes(pair[1]) && symbol2.includes(pair[0]))) {
        return 0.7; // High correlation
      }
    }
    
    // Default moderate correlation for crypto assets
    return 0.4;
  }

  /**
   * Calculate current drawdown
   */
  private calculateDrawdown(currentValue: number): number {
    if (this.peakEquity === 0) return 0;
    const drawdown = ((this.peakEquity - currentValue) / this.peakEquity) * 100;
    return Math.max(0, drawdown);
  }

  /**
   * Block trade with reason
   */
  private blockTrade(reason: string): RiskDecision {
    return {
      proceed: false,
      reason,
      positionSize: 0,
      leverage: 1,
      stopLoss: 0,
      takeProfit: 0,
      stopLossPercent: 0,
      takeProfitPercent: 0,
      riskAmount: 0,
      rewardAmount: 0,
      riskRewardRatio: 0,
      kellyPercent: 0,
      confidenceAdjustedSize: 0
    };
  }

  /**
   * Update trailing stop for existing position
   */
  updateTrailingStop(
    position: Position,
    currentPrice: number,
    trailingPercent: number = 5
  ): number | null {
    if (position.side === 'long') {
      // For longs, trail below the highest price
      const trailPrice = currentPrice * (1 - trailingPercent / 100);
      const currentStop = position.cost_basis * 0.95; // Assume 5% initial stop
      return Math.max(currentStop, trailPrice);
    } else {
      // For shorts, trail above the lowest price
      const trailPrice = currentPrice * (1 + trailingPercent / 100);
      const currentStop = position.cost_basis * 1.05;
      return Math.min(currentStop, trailPrice);
    }
  }
}

// Export singleton instance
export const riskManagerV2 = new RiskManagerV2();
export default riskManagerV2;