"""
Individual Reward Components for Multi-Objective Optimization

This module implements sophisticated reward components that align with SOW targets:
- 3-5% weekly returns
- Sharpe ratio > 1.5
- Maximum drawdown < 15%
- Win rate > 60%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RewardComponentConfig:
    """Configuration for individual reward components"""
    # Profit targets (SOW aligned)
    weekly_return_target: float = 0.04  # 4% weekly target (middle of 3-5% range)
    daily_return_target: float = 0.0057  # ~0.57% daily (4% weekly compounded)
    
    # Risk targets (SOW aligned)
    target_sharpe_ratio: float = 1.5
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    drawdown_warning_level: float = 0.10  # Start penalizing at 10%
    
    # Consistency targets (SOW aligned)
    target_win_rate: float = 0.60  # 60% win rate
    max_consecutive_losses: int = 5
    
    # Normalization parameters
    tanh_scale_profit: float = 100.0  # Scale for tanh normalization
    tanh_scale_sharpe: float = 3.0
    
    # Transaction costs
    base_transaction_cost: float = 0.001  # 0.1% base cost
    slippage_factor: float = 0.0005  # 0.05% slippage
    
    # Time decay
    urgency_decay_rate: float = 0.95  # Decay factor per time step
    
    # Exploration bonus
    exploration_bonus_base: float = 0.01
    exploration_decay_rate: float = 0.999


class ProfitComponent:
    """
    Profit-based reward component with tanh normalization to prevent greed
    
    Mathematical formulation:
    R_profit = tanh(returns / scale) * weight
    
    This creates a bounded reward that saturates at extreme values,
    preventing the agent from becoming too greedy.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.returns_history: List[float] = []
        
    def calculate(self, 
                  current_equity: float,
                  previous_equity: float,
                  time_steps: int = 1) -> float:
        """
        Calculate profit component with tanh normalization
        
        Args:
            current_equity: Current portfolio value
            previous_equity: Previous portfolio value
            time_steps: Number of time steps for return calculation
            
        Returns:
            Normalized profit reward component
        """
        try:
            # Calculate raw return
            raw_return = (current_equity - previous_equity) / previous_equity
            
            # Store in history
            self.returns_history.append(raw_return)
            if len(self.returns_history) > 100:
                self.returns_history.pop(0)
            
            # Apply tanh normalization to prevent greed
            # This creates diminishing returns for extreme profits
            normalized_return = np.tanh(raw_return * self.config.tanh_scale_profit)
            
            # Adjust for time steps (annualized comparison)
            if time_steps > 1:
                normalized_return = normalized_return / np.sqrt(time_steps)
            
            # Compare against target
            target_return = self.config.daily_return_target * time_steps
            target_bonus = 0.0
            
            if raw_return >= target_return:
                # Bonus for meeting target
                target_bonus = 0.1 * (1.0 - abs(raw_return - target_return) / target_return)
            
            return normalized_return + target_bonus
            
        except Exception as e:
            logger.error(f"Error calculating profit component: {e}")
            return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get profit component metrics"""
        if not self.returns_history:
            return {}
        
        returns = np.array(self.returns_history)
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'positive_returns': np.sum(returns > 0) / len(returns)
        }


class RiskAdjustedComponent:
    """
    Risk-adjusted returns component optimizing for Sharpe ratio
    
    Mathematical formulation:
    R_sharpe = (E[r] - r_f) / σ(r) * tanh(sharpe / target_sharpe)
    
    This rewards high Sharpe ratios while normalizing extreme values.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.returns_window: List[float] = []
        self.sharpe_history: List[float] = []
        
    def calculate(self,
                  returns_series: np.ndarray,
                  risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio component
        
        Args:
            returns_series: Array of recent returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio reward component
        """
        try:
            if len(returns_series) < 2:
                return 0.0
            
            # Calculate excess returns
            daily_rf = risk_free_rate / 252
            excess_returns = returns_series - daily_rf
            
            # Calculate Sharpe ratio
            mean_excess = np.mean(excess_returns)
            std_returns = np.std(returns_series)
            
            if std_returns == 0:
                sharpe = 0.0
            else:
                # Annualized Sharpe ratio
                sharpe = mean_excess / std_returns * np.sqrt(252)
            
            # Store history
            self.sharpe_history.append(sharpe)
            if len(self.sharpe_history) > 100:
                self.sharpe_history.pop(0)
            
            # Normalize with tanh to bound extreme values
            normalized_sharpe = np.tanh(sharpe / self.config.tanh_scale_sharpe)
            
            # Bonus for exceeding target Sharpe ratio
            target_bonus = 0.0
            if sharpe > self.config.target_sharpe_ratio:
                excess_sharpe = sharpe - self.config.target_sharpe_ratio
                target_bonus = 0.2 * np.tanh(excess_sharpe)
            
            return normalized_sharpe + target_bonus
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe component: {e}")
            return 0.0
    
    def calculate_sortino(self,
                         returns_series: np.ndarray,
                         target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted returns)
        
        Args:
            returns_series: Array of returns
            target_return: Minimum acceptable return
            
        Returns:
            Sortino ratio component
        """
        try:
            if len(returns_series) < 2:
                return 0.0
            
            excess_returns = returns_series - target_return
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return 1.0  # No downside risk
            
            downside_deviation = np.std(downside_returns)
            
            if downside_deviation == 0:
                return 1.0
            
            sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            
            # Normalize
            return np.tanh(sortino / 3.0)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0


class DrawdownComponent:
    """
    Drawdown penalty component with strict enforcement
    
    Mathematical formulation:
    R_dd = -exp(dd / dd_limit) * penalty_weight  if dd > warning_level
    
    This creates exponentially increasing penalties as drawdown approaches limits.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.peak_equity = 0.0
        self.drawdown_history: List[float] = []
        
    def calculate(self,
                  current_equity: float,
                  initial_equity: float) -> float:
        """
        Calculate drawdown penalty
        
        Args:
            current_equity: Current portfolio value
            initial_equity: Initial portfolio value
            
        Returns:
            Drawdown penalty (negative value)
        """
        try:
            # Update peak equity
            self.peak_equity = max(self.peak_equity, current_equity, initial_equity)
            
            # Calculate current drawdown
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            # Store history
            self.drawdown_history.append(drawdown)
            if len(self.drawdown_history) > 100:
                self.drawdown_history.pop(0)
            
            # No penalty if below warning level
            if drawdown < self.config.drawdown_warning_level:
                return 0.0
            
            # Exponential penalty between warning and limit
            if drawdown < self.config.max_drawdown_limit:
                penalty_factor = (drawdown - self.config.drawdown_warning_level) / \
                               (self.config.max_drawdown_limit - self.config.drawdown_warning_level)
                penalty = -np.exp(penalty_factor * 2) * 0.5
            else:
                # Severe penalty for exceeding limit
                excess = drawdown - self.config.max_drawdown_limit
                penalty = -np.exp(excess * 10) * 2.0
            
            return penalty
            
        except Exception as e:
            logger.error(f"Error calculating drawdown component: {e}")
            return 0.0
    
    def reset(self):
        """Reset drawdown tracking"""
        self.peak_equity = 0.0
        self.drawdown_history.clear()


class ConsistencyComponent:
    """
    Consistency bonus component rewarding win streaks and low variance
    
    Mathematical formulation:
    R_consistency = win_rate_bonus + streak_bonus - variance_penalty
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.trade_results: List[float] = []
        self.current_streak = 0
        self.max_streak = 0
        
    def calculate(self,
                  trade_result: Optional[float] = None,
                  returns_variance: Optional[float] = None) -> float:
        """
        Calculate consistency bonus
        
        Args:
            trade_result: Result of latest trade (profit/loss)
            returns_variance: Variance of recent returns
            
        Returns:
            Consistency reward component
        """
        try:
            reward = 0.0
            
            # Update trade results if provided
            if trade_result is not None:
                self.trade_results.append(trade_result)
                if len(self.trade_results) > 100:
                    self.trade_results.pop(0)
                
                # Update streak
                if trade_result > 0:
                    self.current_streak = max(0, self.current_streak) + 1
                    self.max_streak = max(self.max_streak, self.current_streak)
                else:
                    self.current_streak = min(0, self.current_streak) - 1
            
            # Win rate bonus
            if len(self.trade_results) >= 10:
                win_rate = sum(1 for r in self.trade_results if r > 0) / len(self.trade_results)
                
                if win_rate >= self.config.target_win_rate:
                    win_rate_bonus = 0.2 * (win_rate - self.config.target_win_rate) / \
                                   (1.0 - self.config.target_win_rate)
                    reward += win_rate_bonus
                else:
                    # Penalty for low win rate
                    win_rate_penalty = -0.1 * (self.config.target_win_rate - win_rate) / \
                                     self.config.target_win_rate
                    reward += win_rate_penalty
            
            # Streak bonus
            if self.current_streak > 0:
                streak_bonus = 0.1 * np.log1p(self.current_streak)
                reward += streak_bonus
            elif self.current_streak < -self.config.max_consecutive_losses:
                # Penalty for losing streak
                streak_penalty = -0.5 * np.exp(abs(self.current_streak) - \
                                             self.config.max_consecutive_losses)
                reward += streak_penalty
            
            # Variance penalty (prefer consistent returns)
            if returns_variance is not None:
                variance_penalty = -0.1 * np.tanh(returns_variance * 100)
                reward += variance_penalty
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating consistency component: {e}")
            return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get consistency metrics"""
        if not self.trade_results:
            return {}
        
        wins = sum(1 for r in self.trade_results if r > 0)
        losses = sum(1 for r in self.trade_results if r <= 0)
        
        return {
            'win_rate': wins / len(self.trade_results) if self.trade_results else 0,
            'total_trades': len(self.trade_results),
            'wins': wins,
            'losses': losses,
            'current_streak': self.current_streak,
            'max_streak': self.max_streak,
            'avg_win': np.mean([r for r in self.trade_results if r > 0]) if wins > 0 else 0,
            'avg_loss': np.mean([r for r in self.trade_results if r <= 0]) if losses > 0 else 0
        }


class TransactionCostComponent:
    """
    Transaction cost penalty component
    
    Penalizes excessive trading to encourage efficient position management.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.total_costs = 0.0
        self.trade_count = 0
        
    def calculate(self,
                  trade_volume: float,
                  is_trade_executed: bool,
                  market_impact: float = 0.0) -> float:
        """
        Calculate transaction cost penalty
        
        Args:
            trade_volume: Volume of trade in base currency
            is_trade_executed: Whether a trade was executed
            market_impact: Additional market impact cost
            
        Returns:
            Transaction cost penalty (negative value)
        """
        try:
            if not is_trade_executed:
                return 0.0
            
            # Calculate costs
            base_cost = trade_volume * self.config.base_transaction_cost
            slippage_cost = trade_volume * self.config.slippage_factor
            total_cost = base_cost + slippage_cost + market_impact
            
            # Update tracking
            self.total_costs += total_cost
            self.trade_count += 1
            
            # Normalize penalty
            penalty = -total_cost / trade_volume if trade_volume > 0 else 0
            
            # Additional penalty for overtrading
            if self.trade_count > 10:
                avg_trades_per_period = self.trade_count / 10
                if avg_trades_per_period > 2:  # More than 2 trades per period average
                    overtrading_penalty = -0.05 * np.log1p(avg_trades_per_period - 2)
                    penalty += overtrading_penalty
            
            return penalty
            
        except Exception as e:
            logger.error(f"Error calculating transaction cost: {e}")
            return 0.0
    
    def reset(self):
        """Reset cost tracking"""
        self.total_costs = 0.0
        self.trade_count = 0


class MarketRegimeComponent:
    """
    Market regime adjustment component
    
    Adjusts rewards based on current market conditions to encourage
    appropriate strategies in different regimes.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.regime_history: List[MarketRegime] = []
        
    def calculate(self,
                  current_regime: MarketRegime,
                  action_type: str,
                  position_size: float) -> float:
        """
        Calculate market regime adjustment
        
        Args:
            current_regime: Current market regime
            action_type: Type of action taken (buy/sell/hold)
            position_size: Size of position
            
        Returns:
            Regime-based reward adjustment
        """
        try:
            # Store regime history
            self.regime_history.append(current_regime)
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
            
            adjustment = 0.0
            
            # Regime-specific adjustments
            if current_regime == MarketRegime.BULL:
                if action_type == 'buy':
                    adjustment = 0.1 * position_size  # Reward buying in bull market
                elif action_type == 'sell':
                    adjustment = -0.05 * position_size  # Slight penalty for selling
                    
            elif current_regime == MarketRegime.BEAR:
                if action_type == 'sell':
                    adjustment = 0.1 * position_size  # Reward selling in bear market
                elif action_type == 'buy' and position_size > 0.5:
                    adjustment = -0.1 * position_size  # Penalty for large buys
                    
            elif current_regime == MarketRegime.HIGH_VOLATILITY:
                if action_type == 'hold':
                    adjustment = 0.05  # Reward caution in high volatility
                elif position_size > 0.3:
                    adjustment = -0.1 * position_size  # Penalty for large positions
                    
            elif current_regime == MarketRegime.SIDEWAYS:
                if action_type != 'hold' and position_size < 0.3:
                    adjustment = 0.05  # Small reward for small trades
                elif position_size > 0.5:
                    adjustment = -0.05 * position_size  # Penalty for large positions
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating regime adjustment: {e}")
            return 0.0
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of market regimes"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for regime in MarketRegime:
            count = sum(1 for r in self.regime_history if r == regime)
            regime_counts[regime.value] = count / len(self.regime_history)
        
        return regime_counts


class ExplorationComponent:
    """
    Exploration bonus component to prevent local optima
    
    Provides diminishing exploration bonuses over time to encourage
    initial exploration while converging to exploitation.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.exploration_count = 0
        self.action_history: List[int] = []
        self.state_visits: Dict[str, int] = {}
        
    def calculate(self,
                  action: int,
                  state_hash: str,
                  episode_step: int) -> float:
        """
        Calculate exploration bonus
        
        Args:
            action: Action taken
            state_hash: Hash of current state
            episode_step: Current step in episode
            
        Returns:
            Exploration bonus
        """
        try:
            # Update histories
            self.action_history.append(action)
            if len(self.action_history) > 100:
                self.action_history.pop(0)
            
            # Track state visits
            self.state_visits[state_hash] = self.state_visits.get(state_hash, 0) + 1
            
            # Calculate action entropy (diversity bonus)
            if len(self.action_history) >= 10:
                action_counts = {}
                for a in self.action_history[-10:]:
                    action_counts[a] = action_counts.get(a, 0) + 1
                
                # Calculate entropy
                probs = np.array(list(action_counts.values())) / 10
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                diversity_bonus = 0.01 * entropy
            else:
                diversity_bonus = 0.0
            
            # Novel state bonus
            visit_count = self.state_visits[state_hash]
            if visit_count == 1:
                novelty_bonus = self.config.exploration_bonus_base
            else:
                novelty_bonus = self.config.exploration_bonus_base / np.sqrt(visit_count)
            
            # Apply decay based on episode progress
            decay_factor = self.config.exploration_decay_rate ** episode_step
            
            total_bonus = (diversity_bonus + novelty_bonus) * decay_factor
            
            self.exploration_count += 1
            
            return total_bonus
            
        except Exception as e:
            logger.error(f"Error calculating exploration bonus: {e}")
            return 0.0
    
    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.action_history.clear()


class TimeDecayComponent:
    """
    Time-decay factor component for urgency
    
    Creates increasing urgency as episode progresses to encourage
    decisive action and prevent stalling.
    """
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        
    def calculate(self,
                  base_reward: float,
                  episode_step: int,
                  max_steps: int) -> float:
        """
        Apply time decay to base reward
        
        Args:
            base_reward: Base reward before decay
            episode_step: Current step in episode
            max_steps: Maximum steps in episode
            
        Returns:
            Time-adjusted reward
        """
        try:
            # Linear urgency factor
            urgency = episode_step / max_steps
            
            # Exponential decay for positive rewards
            if base_reward > 0:
                decay_factor = self.config.urgency_decay_rate ** (episode_step / 10)
                adjusted_reward = base_reward * decay_factor
                
                # Add urgency bonus for profitable actions near end
                if urgency > 0.8:
                    urgency_bonus = base_reward * 0.1 * (urgency - 0.8) * 5
                    adjusted_reward += urgency_bonus
            else:
                # Increase penalties over time for negative rewards
                adjusted_reward = base_reward * (1 + urgency * 0.5)
            
            return adjusted_reward
            
        except Exception as e:
            logger.error(f"Error calculating time decay: {e}")
            return base_reward


def create_reward_components(config: Optional[RewardComponentConfig] = None) -> Dict[str, Any]:
    """
    Factory function to create all reward components
    
    Args:
        config: Configuration for components
        
    Returns:
        Dictionary of initialized components
    """
    if config is None:
        config = RewardComponentConfig()
    
    return {
        'profit': ProfitComponent(config),
        'risk_adjusted': RiskAdjustedComponent(config),
        'drawdown': DrawdownComponent(config),
        'consistency': ConsistencyComponent(config),
        'transaction_cost': TransactionCostComponent(config),
        'market_regime': MarketRegimeComponent(config),
        'exploration': ExplorationComponent(config),
        'time_decay': TimeDecayComponent(config)
    }