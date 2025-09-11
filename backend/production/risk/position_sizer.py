"""
Advanced Position Sizing Algorithms
Implements Kelly Criterion, Fixed Fractional, and dynamic sizing methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    KELLY = "KELLY"
    OPTIMAL_F = "OPTIMAL_F"
    RISK_PARITY = "RISK_PARITY"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    MARTINGALE = "MARTINGALE"
    ANTI_MARTINGALE = "ANTI_MARTINGALE"
    DYNAMIC = "DYNAMIC"


@dataclass
class SizingConfig:
    """Position sizing configuration"""
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    base_risk_percent: float = 0.02  # 2% base risk per trade
    max_risk_percent: float = 0.05  # 5% max risk per trade
    kelly_fraction: float = 0.25  # Use 25% of Kelly suggestion
    confidence_threshold: float = 0.60  # 60% confidence required
    volatility_lookback: int = 20  # Days for volatility calculation
    use_leverage: bool = True
    max_leverage: float = 5.0
    scale_by_confidence: bool = True
    scale_by_volatility: bool = True
    scale_by_regime: bool = True
    min_position_size: float = 100  # Minimum position size in dollars
    round_to_lots: bool = True
    lot_size: int = 100  # Standard lot size


@dataclass
class MarketContext:
    """Current market context for position sizing"""
    volatility: float  # Current market volatility
    trend_strength: float  # Trend strength (0-1)
    regime: str  # Market regime (trending, ranging, volatile)
    correlation: float  # Correlation with market
    liquidity: float  # Liquidity score (0-1)
    confidence: float  # Signal confidence (0-1)
    win_rate: float  # Recent win rate
    expectancy: float  # Recent expectancy in R
    drawdown: float  # Current drawdown


class PositionSizer:
    """
    Advanced position sizing calculator with multiple algorithms
    """
    
    def __init__(self, 
                 account_balance: float,
                 config: Optional[SizingConfig] = None):
        """
        Initialize Position Sizer
        
        Args:
            account_balance: Current account balance
            config: Sizing configuration
        """
        self.account_balance = account_balance
        self.config = config or SizingConfig()
        
        # Performance tracking
        self.trade_results = deque(maxlen=100)  # Last 100 trades
        self.returns_history = deque(maxlen=252)  # 1 year of returns
        
        # Kelly calculation cache
        self.kelly_f = 0.02  # Default Kelly fraction
        self.optimal_f = 0.02  # Optimal f value
        
        # Volatility tracking
        self.volatility_history = deque(maxlen=self.config.volatility_lookback)
        self.current_volatility = 0.02  # Default 2% volatility
        
        logger.info(f"Position Sizer initialized - Method: {self.config.method.value}")
    
    def calculate_position_size(self,
                                signal_strength: float,
                                stop_distance: float,
                                entry_price: float,
                                market_context: Optional[MarketContext] = None) -> Dict[str, any]:
        """
        Calculate optimal position size based on configured method
        
        Args:
            signal_strength: Signal strength (0-1)
            stop_distance: Distance to stop loss in price units
            entry_price: Entry price
            market_context: Current market context
            
        Returns:
            Position sizing details
        """
        # Default context if not provided
        if market_context is None:
            market_context = MarketContext(
                volatility=self.current_volatility,
                trend_strength=0.5,
                regime="neutral",
                correlation=0.0,
                liquidity=1.0,
                confidence=signal_strength,
                win_rate=0.5,
                expectancy=0.0,
                drawdown=0.0
            )
        
        # Calculate base position size based on method
        if self.config.method == SizingMethod.FIXED_FRACTIONAL:
            base_size = self._fixed_fractional_size(stop_distance, entry_price)
        elif self.config.method == SizingMethod.KELLY:
            base_size = self._kelly_size(stop_distance, entry_price, market_context)
        elif self.config.method == SizingMethod.OPTIMAL_F:
            base_size = self._optimal_f_size(stop_distance, entry_price)
        elif self.config.method == SizingMethod.RISK_PARITY:
            base_size = self._risk_parity_size(stop_distance, entry_price, market_context)
        elif self.config.method == SizingMethod.VOLATILITY_BASED:
            base_size = self._volatility_based_size(stop_distance, entry_price, market_context)
        elif self.config.method == SizingMethod.MARTINGALE:
            base_size = self._martingale_size(stop_distance, entry_price, market_context)
        elif self.config.method == SizingMethod.ANTI_MARTINGALE:
            base_size = self._anti_martingale_size(stop_distance, entry_price, market_context)
        elif self.config.method == SizingMethod.DYNAMIC:
            base_size = self._dynamic_size(stop_distance, entry_price, market_context)
        else:
            base_size = self._fixed_fractional_size(stop_distance, entry_price)
        
        # Apply scaling factors
        scaled_size = self._apply_scaling(base_size, signal_strength, market_context)
        
        # Apply constraints
        final_size = self._apply_constraints(scaled_size, entry_price)
        
        # Calculate risk metrics
        position_value = final_size * entry_price
        risk_amount = final_size * stop_distance
        risk_percent = risk_amount / self.account_balance
        
        # Determine leverage if enabled
        leverage = 1.0
        if self.config.use_leverage and position_value > self.account_balance:
            leverage = min(position_value / self.account_balance, self.config.max_leverage)
            
        return {
            "shares": final_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percent": risk_percent,
            "leverage": leverage,
            "method": self.config.method.value,
            "confidence": market_context.confidence,
            "adjustments": {
                "base_size": base_size,
                "scaled_size": scaled_size,
                "final_size": final_size
            },
            "kelly_fraction": self.kelly_f if self.config.method == SizingMethod.KELLY else None,
            "volatility_adjustment": market_context.volatility / self.current_volatility if self.current_volatility > 0 else 1.0
        }
    
    def _fixed_fractional_size(self, stop_distance: float, entry_price: float) -> float:
        """Calculate position size using fixed fractional method"""
        risk_amount = self.account_balance * self.config.base_risk_percent
        shares = risk_amount / stop_distance
        return shares
    
    def _kelly_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Calculate position size using Kelly Criterion
        f* = (p*b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        # Update Kelly fraction based on recent performance
        if len(self.trade_results) >= 20:
            wins = [t for t in self.trade_results if t > 0]
            losses = [t for t in self.trade_results if t <= 0]
            
            if wins and losses:
                win_rate = len(wins) / len(self.trade_results)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                
                if avg_loss > 0:
                    b = avg_win / avg_loss  # Win/loss ratio
                    p = win_rate
                    q = 1 - p
                    
                    # Full Kelly
                    kelly_full = (p * b - q) / b if b > 0 else 0
                    
                    # Apply Kelly fraction (conservative)
                    self.kelly_f = max(0, min(kelly_full * self.config.kelly_fraction, self.config.max_risk_percent))
                else:
                    self.kelly_f = self.config.base_risk_percent
            else:
                self.kelly_f = self.config.base_risk_percent
        else:
            # Not enough data, use base risk
            self.kelly_f = self.config.base_risk_percent
        
        # Adjust for confidence
        adjusted_kelly = self.kelly_f * context.confidence
        
        risk_amount = self.account_balance * adjusted_kelly
        shares = risk_amount / stop_distance
        
        return shares
    
    def _optimal_f_size(self, stop_distance: float, entry_price: float) -> float:
        """
        Calculate position size using Optimal f method
        Based on Ralph Vince's optimal f calculation
        """
        if len(self.trade_results) < 30:
            # Not enough data, use fixed fractional
            return self._fixed_fractional_size(stop_distance, entry_price)
        
        # Find optimal f through iteration
        results = np.array(list(self.trade_results))
        
        best_f = 0.01
        best_twi = -np.inf  # Terminal Wealth Index
        
        for f in np.arange(0.01, 0.50, 0.01):
            twi = 1.0
            for result in results:
                hpr = 1.0 + (f * result)  # Holding Period Return
                if hpr <= 0:
                    twi = -np.inf
                    break
                twi *= hpr
            
            if twi > best_twi:
                best_twi = twi
                best_f = f
        
        self.optimal_f = min(best_f, self.config.max_risk_percent)
        
        risk_amount = self.account_balance * self.optimal_f
        shares = risk_amount / stop_distance
        
        return shares
    
    def _risk_parity_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Calculate position size using risk parity approach
        Size inversely proportional to volatility
        """
        # Target risk budget
        target_risk = self.account_balance * self.config.base_risk_percent
        
        # Adjust for asset volatility
        if context.volatility > 0:
            volatility_scalar = self.current_volatility / context.volatility
        else:
            volatility_scalar = 1.0
        
        # Adjust for correlation
        correlation_scalar = 1.0 - abs(context.correlation) * 0.5
        
        adjusted_risk = target_risk * volatility_scalar * correlation_scalar
        shares = adjusted_risk / stop_distance
        
        return shares
    
    def _volatility_based_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Calculate position size based on volatility targeting
        """
        # Target portfolio volatility (annualized)
        target_vol = 0.15  # 15% annual volatility target
        
        # Current position volatility (daily)
        position_vol = context.volatility
        
        # Calculate weight to achieve target volatility
        if position_vol > 0:
            weight = (target_vol / np.sqrt(252)) / position_vol
        else:
            weight = self.config.base_risk_percent
        
        # Convert weight to shares
        position_value = self.account_balance * min(weight, self.config.max_risk_percent)
        shares = position_value / entry_price
        
        # Ensure stop loss is respected
        max_shares_for_stop = (self.account_balance * self.config.max_risk_percent) / stop_distance
        shares = min(shares, max_shares_for_stop)
        
        return shares
    
    def _martingale_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Martingale sizing - increase size after losses
        WARNING: High risk strategy
        """
        base_size = self._fixed_fractional_size(stop_distance, entry_price)
        
        # Count consecutive losses
        consecutive_losses = 0
        for result in reversed(self.trade_results):
            if result < 0:
                consecutive_losses += 1
            else:
                break
        
        # Double size for each loss (with maximum)
        multiplier = min(2 ** consecutive_losses, 4)  # Max 4x
        
        return base_size * multiplier
    
    def _anti_martingale_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Anti-Martingale sizing - increase size after wins
        """
        base_size = self._fixed_fractional_size(stop_distance, entry_price)
        
        # Count consecutive wins
        consecutive_wins = 0
        for result in reversed(self.trade_results):
            if result > 0:
                consecutive_wins += 1
            else:
                break
        
        # Increase size gradually with wins
        multiplier = 1.0 + (consecutive_wins * 0.25)  # 25% increase per win
        multiplier = min(multiplier, 2.0)  # Max 2x
        
        # Reduce if in drawdown
        if context.drawdown > 0.05:
            multiplier *= (1.0 - context.drawdown)
        
        return base_size * multiplier
    
    def _dynamic_size(self, stop_distance: float, entry_price: float, context: MarketContext) -> float:
        """
        Dynamic position sizing based on multiple factors
        """
        # Start with base size
        base_size = self._fixed_fractional_size(stop_distance, entry_price)
        
        # Factor 1: Market regime
        regime_multiplier = {
            "trending": 1.2,
            "ranging": 0.8,
            "volatile": 0.6,
            "neutral": 1.0
        }.get(context.regime, 1.0)
        
        # Factor 2: Win rate adjustment
        if context.win_rate > 0.6:
            win_rate_multiplier = 1.2
        elif context.win_rate < 0.4:
            win_rate_multiplier = 0.8
        else:
            win_rate_multiplier = 1.0
        
        # Factor 3: Expectancy adjustment
        if context.expectancy > 0.5:
            expectancy_multiplier = 1.3
        elif context.expectancy < 0:
            expectancy_multiplier = 0.5
        else:
            expectancy_multiplier = 1.0
        
        # Factor 4: Drawdown adjustment
        drawdown_multiplier = max(0.5, 1.0 - context.drawdown * 2)
        
        # Factor 5: Confidence adjustment
        confidence_multiplier = 0.5 + (context.confidence * 0.5)
        
        # Combine all factors
        total_multiplier = (regime_multiplier * win_rate_multiplier * 
                          expectancy_multiplier * drawdown_multiplier * 
                          confidence_multiplier)
        
        # Apply bounds
        total_multiplier = max(0.25, min(2.0, total_multiplier))
        
        return base_size * total_multiplier
    
    def _apply_scaling(self, base_size: float, signal_strength: float, context: MarketContext) -> float:
        """Apply scaling factors to base position size"""
        scaled_size = base_size
        
        # Scale by confidence
        if self.config.scale_by_confidence:
            confidence_scalar = 0.5 + (context.confidence * 0.5)  # 50% to 100%
            scaled_size *= confidence_scalar
        
        # Scale by volatility
        if self.config.scale_by_volatility:
            if context.volatility > self.current_volatility * 1.5:
                # High volatility - reduce size
                scaled_size *= 0.7
            elif context.volatility < self.current_volatility * 0.5:
                # Low volatility - increase size
                scaled_size *= 1.3
        
        # Scale by regime
        if self.config.scale_by_regime:
            if context.regime == "volatile":
                scaled_size *= 0.6
            elif context.regime == "trending" and context.trend_strength > 0.7:
                scaled_size *= 1.2
        
        return scaled_size
    
    def _apply_constraints(self, size: float, entry_price: float) -> float:
        """Apply position size constraints"""
        # Minimum position size
        min_shares = self.config.min_position_size / entry_price
        size = max(size, min_shares)
        
        # Maximum position size (risk-based)
        max_risk_amount = self.account_balance * self.config.max_risk_percent
        max_shares = max_risk_amount / entry_price  # Simplified - should use stop distance
        size = min(size, max_shares)
        
        # Round to lot size if required
        if self.config.round_to_lots:
            size = round(size / self.config.lot_size) * self.config.lot_size
        
        return size
    
    def update_performance(self, trade_result: float):
        """
        Update performance metrics with trade result
        
        Args:
            trade_result: Trade result in R-multiples
        """
        self.trade_results.append(trade_result)
        
        # Update returns history
        if trade_result != 0:
            returns = trade_result * self.config.base_risk_percent
            self.returns_history.append(returns)
    
    def update_volatility(self, returns: List[float]):
        """
        Update volatility estimates
        
        Args:
            returns: List of recent returns
        """
        if len(returns) > 1:
            self.current_volatility = np.std(returns)
            self.volatility_history.append(self.current_volatility)
    
    def get_sizing_stats(self) -> Dict[str, any]:
        """Get position sizing statistics"""
        stats = {
            "method": self.config.method.value,
            "current_kelly_f": self.kelly_f,
            "optimal_f": self.optimal_f,
            "base_risk_percent": self.config.base_risk_percent,
            "max_risk_percent": self.config.max_risk_percent,
            "current_volatility": self.current_volatility,
            "avg_position_size": None,
            "performance": {
                "total_trades": len(self.trade_results),
                "avg_r_multiple": np.mean(self.trade_results) if self.trade_results else 0,
                "win_rate": len([r for r in self.trade_results if r > 0]) / len(self.trade_results) if self.trade_results else 0
            }
        }
        
        return stats
    
    def optimize_parameters(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize sizing parameters based on historical data
        
        Args:
            historical_data: Historical trade data
            
        Returns:
            Optimized parameters
        """
        # Monte Carlo simulation for parameter optimization
        best_params = {
            "base_risk": self.config.base_risk_percent,
            "kelly_fraction": self.config.kelly_fraction,
            "sharpe": 0
        }
        
        for _ in range(1000):
            # Random parameters
            test_risk = np.random.uniform(0.005, 0.05)
            test_kelly = np.random.uniform(0.1, 0.5)
            
            # Simulate with parameters
            returns = []
            for _, trade in historical_data.iterrows():
                # Simplified simulation
                size_multiplier = test_risk * test_kelly
                trade_return = trade.get('return', 0) * size_multiplier
                returns.append(trade_return)
            
            # Calculate Sharpe ratio
            if returns:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                
                if sharpe > best_params["sharpe"]:
                    best_params = {
                        "base_risk": test_risk,
                        "kelly_fraction": test_kelly,
                        "sharpe": sharpe
                    }
        
        return best_params