"""
Multi-Objective Reward Optimization System

This module implements a sophisticated multi-objective optimization framework
that balances multiple competing objectives while maintaining SOW targets.
Uses Pareto optimization and dynamic weighting strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.stats import entropy
import json

from .reward_components import (
    RewardComponentConfig, MarketRegime,
    create_reward_components
)

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Multi-objective optimization strategies"""
    WEIGHTED_SUM = "weighted_sum"
    PARETO = "pareto"
    LEXICOGRAPHIC = "lexicographic"
    CONSTRAINT_BASED = "constraint_based"
    ADAPTIVE = "adaptive"


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective reward optimization"""
    
    # Optimization strategy
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    
    # Objective priorities (for lexicographic ordering)
    objective_priorities: List[str] = field(default_factory=lambda: [
        'risk_adjusted',  # Sharpe ratio first priority
        'drawdown',       # Risk management second
        'profit',         # Profitability third
        'consistency'     # Consistency fourth
    ])
    
    # Base weights for objectives
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        'profit': 0.25,
        'risk_adjusted': 0.30,
        'drawdown': 0.20,
        'consistency': 0.15,
        'transaction_cost': 0.05,
        'exploration': 0.05
    })
    
    # Constraints for objectives (min/max acceptable values)
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'drawdown': (-float('inf'), -0.15),  # Max 15% drawdown
        'risk_adjusted': (1.5, float('inf')),  # Min Sharpe 1.5
        'consistency': (0.6, float('inf')),  # Min 60% win rate
        'profit': (0.03, 0.05)  # 3-5% weekly target
    })
    
    # Adaptation parameters
    adaptation_rate: float = 0.01
    memory_size: int = 100
    update_frequency: int = 10
    
    # Pareto optimization parameters
    pareto_population_size: int = 50
    pareto_generations: int = 20
    pareto_mutation_rate: float = 0.1
    
    # Weight bounds
    weight_min: float = 0.0
    weight_max: float = 1.0
    
    # Normalization
    normalize_objectives: bool = True
    normalization_window: int = 100


class ParetoFrontier:
    """
    Pareto frontier calculation for multi-objective optimization
    
    Finds non-dominated solutions in the objective space.
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.frontier_points: List[Dict[str, float]] = []
        self.frontier_weights: List[Dict[str, float]] = []
        
    def update(self, objectives: Dict[str, float], weights: Dict[str, float]):
        """
        Update Pareto frontier with new solution
        
        Args:
            objectives: Objective values
            weights: Weights that produced these objectives
        """
        # Check if solution is dominated
        is_dominated = False
        to_remove = []
        
        for i, point in enumerate(self.frontier_points):
            if self._dominates(point, objectives):
                is_dominated = True
                break
            elif self._dominates(objectives, point):
                to_remove.append(i)
        
        # Remove dominated points
        for i in reversed(to_remove):
            self.frontier_points.pop(i)
            self.frontier_weights.pop(i)
        
        # Add if not dominated
        if not is_dominated:
            self.frontier_points.append(objectives.copy())
            self.frontier_weights.append(weights.copy())
            
            # Limit frontier size
            if len(self.frontier_points) > self.config.pareto_population_size:
                # Remove furthest from ideal point
                ideal_point = self._get_ideal_point()
                distances = [self._distance_to_ideal(p, ideal_point) 
                           for p in self.frontier_points]
                worst_idx = np.argmax(distances)
                self.frontier_points.pop(worst_idx)
                self.frontier_weights.pop(worst_idx)
    
    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if solution a dominates solution b"""
        better_in_at_least_one = False
        
        for key in a.keys():
            if key not in b:
                continue
                
            # Maximize all objectives (negative values for penalties)
            if a[key] < b[key]:
                return False
            elif a[key] > b[key]:
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _get_ideal_point(self) -> Dict[str, float]:
        """Get ideal point (best value for each objective)"""
        if not self.frontier_points:
            return {}
        
        ideal = {}
        for key in self.frontier_points[0].keys():
            ideal[key] = max(p.get(key, float('-inf')) for p in self.frontier_points)
        
        return ideal
    
    def _distance_to_ideal(self, point: Dict[str, float], ideal: Dict[str, float]) -> float:
        """Calculate distance from point to ideal point"""
        distance = 0.0
        for key in ideal.keys():
            if key in point:
                distance += (ideal[key] - point[key]) ** 2
        return np.sqrt(distance)
    
    def get_best_weights(self) -> Dict[str, float]:
        """Get weights for solution closest to ideal point"""
        if not self.frontier_points:
            return self.config.base_weights
        
        ideal = self._get_ideal_point()
        distances = [self._distance_to_ideal(p, ideal) for p in self.frontier_points]
        best_idx = np.argmin(distances)
        
        return self.frontier_weights[best_idx]


class AdaptiveWeightOptimizer:
    """
    Adaptive weight optimization using gradient-based methods
    
    Dynamically adjusts weights based on performance feedback.
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.current_weights = config.base_weights.copy()
        self.weight_history: List[Dict[str, float]] = []
        self.performance_history: List[float] = []
        self.gradient_estimates: Dict[str, float] = {k: 0.0 for k in config.base_weights}
        
    def update(self, objectives: Dict[str, float], overall_performance: float):
        """
        Update weights based on performance
        
        Args:
            objectives: Current objective values
            overall_performance: Overall performance metric
        """
        # Store history
        self.weight_history.append(self.current_weights.copy())
        self.performance_history.append(overall_performance)
        
        if len(self.weight_history) > self.config.memory_size:
            self.weight_history.pop(0)
            self.performance_history.pop(0)
        
        # Update gradient estimates
        if len(self.performance_history) >= 2:
            self._estimate_gradients()
        
        # Adapt weights
        if len(self.weight_history) % self.config.update_frequency == 0:
            self._adapt_weights(objectives)
    
    def _estimate_gradients(self):
        """Estimate gradients using finite differences"""
        if len(self.weight_history) < 2:
            return
        
        recent_weights = self.weight_history[-2:]
        recent_performance = self.performance_history[-2:]
        
        perf_diff = recent_performance[-1] - recent_performance[-2]
        
        for key in self.current_weights:
            weight_diff = recent_weights[-1].get(key, 0) - recent_weights[-2].get(key, 0)
            
            if abs(weight_diff) > 1e-6:
                gradient = perf_diff / weight_diff
                # Exponential moving average
                self.gradient_estimates[key] = 0.9 * self.gradient_estimates[key] + 0.1 * gradient
    
    def _adapt_weights(self, objectives: Dict[str, float]):
        """Adapt weights using gradient ascent with constraints"""
        new_weights = {}
        
        for key, current_weight in self.current_weights.items():
            # Gradient ascent step
            gradient = self.gradient_estimates.get(key, 0)
            new_weight = current_weight + self.config.adaptation_rate * gradient
            
            # Apply constraints
            if key in objectives and key in self.config.constraints:
                min_val, max_val = self.config.constraints[key]
                current_val = objectives[key]
                
                # Increase weight if constraint violated
                if current_val < min_val:
                    violation_factor = (min_val - current_val) / abs(min_val) if min_val != 0 else 1
                    new_weight *= (1 + violation_factor)
                elif current_val > max_val:
                    violation_factor = (current_val - max_val) / abs(max_val) if max_val != 0 else 1
                    new_weight *= (1 - violation_factor)
            
            # Bound weights
            new_weight = np.clip(new_weight, self.config.weight_min, self.config.weight_max)
            new_weights[key] = new_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for key in new_weights:
                new_weights[key] /= total_weight
        
        self.current_weights = new_weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current optimized weights"""
        return self.current_weights.copy()


class MultiObjectiveReward:
    """
    Main multi-objective reward system
    
    Combines multiple reward components using sophisticated optimization strategies
    to achieve balanced performance across all objectives.
    """
    
    def __init__(self, 
                 config: Optional[MultiObjectiveConfig] = None,
                 component_config: Optional[RewardComponentConfig] = None):
        self.config = config or MultiObjectiveConfig()
        self.component_config = component_config or RewardComponentConfig()
        
        # Initialize components
        self.components = create_reward_components(self.component_config)
        
        # Initialize optimization systems
        self.pareto_frontier = ParetoFrontier(self.config)
        self.weight_optimizer = AdaptiveWeightOptimizer(self.config)
        
        # Tracking
        self.objective_history: List[Dict[str, float]] = []
        self.reward_history: List[float] = []
        self.current_weights = self.config.base_weights.copy()
        
        # Normalization statistics
        self.objective_stats: Dict[str, Dict[str, float]] = {}
        
    def calculate_reward(self,
                        current_equity: float,
                        previous_equity: float,
                        portfolio_state: Dict[str, Any],
                        market_conditions: Dict[str, Any],
                        action_info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward
        
        Args:
            current_equity: Current portfolio value
            previous_equity: Previous portfolio value
            portfolio_state: Portfolio state information
            market_conditions: Market conditions
            action_info: Information about action taken
            
        Returns:
            Tuple of (total_reward, component_rewards)
        """
        try:
            # Calculate individual objectives
            objectives = self._calculate_objectives(
                current_equity, previous_equity,
                portfolio_state, market_conditions, action_info
            )
            
            # Normalize objectives if configured
            if self.config.normalize_objectives:
                objectives = self._normalize_objectives(objectives)
            
            # Update statistics
            self._update_statistics(objectives)
            
            # Calculate total reward based on strategy
            if self.config.strategy == OptimizationStrategy.WEIGHTED_SUM:
                total_reward = self._weighted_sum_reward(objectives)
                
            elif self.config.strategy == OptimizationStrategy.PARETO:
                total_reward = self._pareto_reward(objectives)
                
            elif self.config.strategy == OptimizationStrategy.LEXICOGRAPHIC:
                total_reward = self._lexicographic_reward(objectives)
                
            elif self.config.strategy == OptimizationStrategy.CONSTRAINT_BASED:
                total_reward = self._constraint_based_reward(objectives)
                
            else:  # ADAPTIVE
                total_reward = self._adaptive_reward(objectives)
            
            # Apply market regime adjustments
            regime = MarketRegime[market_conditions.get('regime', 'SIDEWAYS')]
            regime_adjustment = self.components['market_regime'].calculate(
                regime,
                action_info.get('type', 'hold'),
                action_info.get('size', 0.0)
            )
            total_reward += regime_adjustment
            
            # Apply time decay
            if 'episode_step' in action_info and 'max_steps' in action_info:
                total_reward = self.components['time_decay'].calculate(
                    total_reward,
                    action_info['episode_step'],
                    action_info['max_steps']
                )
            
            # Store history
            self.objective_history.append(objectives.copy())
            self.reward_history.append(total_reward)
            
            if len(self.objective_history) > self.config.memory_size:
                self.objective_history.pop(0)
                self.reward_history.pop(0)
            
            return total_reward, objectives
            
        except Exception as e:
            logger.error(f"Error calculating multi-objective reward: {e}")
            return 0.0, {}
    
    def _calculate_objectives(self,
                             current_equity: float,
                             previous_equity: float,
                             portfolio_state: Dict[str, Any],
                             market_conditions: Dict[str, Any],
                             action_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all objective values"""
        objectives = {}
        
        # Profit objective
        objectives['profit'] = self.components['profit'].calculate(
            current_equity, previous_equity
        )
        
        # Risk-adjusted objective (Sharpe)
        returns_history = portfolio_state.get('returns_history', [])
        if returns_history:
            objectives['risk_adjusted'] = self.components['risk_adjusted'].calculate(
                np.array(returns_history)
            )
        else:
            objectives['risk_adjusted'] = 0.0
        
        # Drawdown objective (negative value)
        initial_equity = portfolio_state.get('initial_equity', previous_equity)
        objectives['drawdown'] = self.components['drawdown'].calculate(
            current_equity, initial_equity
        )
        
        # Consistency objective
        trade_result = portfolio_state.get('last_trade_result')
        returns_variance = np.var(returns_history) if returns_history else None
        objectives['consistency'] = self.components['consistency'].calculate(
            trade_result, returns_variance
        )
        
        # Transaction cost objective (negative value)
        objectives['transaction_cost'] = self.components['transaction_cost'].calculate(
            action_info.get('trade_volume', 0),
            action_info.get('is_trade', False),
            market_conditions.get('market_impact', 0)
        )
        
        # Exploration objective
        if 'state_hash' in action_info:
            objectives['exploration'] = self.components['exploration'].calculate(
                action_info.get('action', 0),
                action_info['state_hash'],
                action_info.get('episode_step', 0)
            )
        else:
            objectives['exploration'] = 0.0
        
        return objectives
    
    def _normalize_objectives(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """Normalize objectives to [0, 1] range"""
        normalized = {}
        
        for key, value in objectives.items():
            if key not in self.objective_stats:
                self.objective_stats[key] = {
                    'min': value,
                    'max': value,
                    'mean': value,
                    'std': 0.0,
                    'count': 1
                }
            else:
                stats = self.objective_stats[key]
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
                stats['count'] += 1
                
                # Update mean incrementally
                delta = value - stats['mean']
                stats['mean'] += delta / stats['count']
                
                # Update std incrementally
                if stats['count'] > 1:
                    stats['std'] = np.sqrt(
                        ((stats['count'] - 2) * stats['std']**2 + delta**2) / 
                        (stats['count'] - 1)
                    )
            
            # Normalize using z-score
            stats = self.objective_stats[key]
            if stats['std'] > 0:
                normalized[key] = (value - stats['mean']) / stats['std']
                # Convert to [0, 1] using sigmoid
                normalized[key] = 1 / (1 + np.exp(-normalized[key]))
            else:
                normalized[key] = 0.5
        
        return normalized
    
    def _update_statistics(self, objectives: Dict[str, float]):
        """Update objective statistics for adaptation"""
        # Update Pareto frontier
        self.pareto_frontier.update(objectives, self.current_weights)
        
        # Update adaptive weights
        overall_performance = self._calculate_overall_performance(objectives)
        self.weight_optimizer.update(objectives, overall_performance)
    
    def _calculate_overall_performance(self, objectives: Dict[str, float]) -> float:
        """Calculate overall performance metric"""
        # Check constraint satisfaction
        constraint_violations = 0
        for key, (min_val, max_val) in self.config.constraints.items():
            if key in objectives:
                if objectives[key] < min_val or objectives[key] > max_val:
                    constraint_violations += 1
        
        # Base performance is weighted sum
        performance = sum(
            self.current_weights.get(k, 0) * v 
            for k, v in objectives.items()
        )
        
        # Penalize constraint violations
        performance -= constraint_violations * 0.5
        
        return performance
    
    def _weighted_sum_reward(self, objectives: Dict[str, float]) -> float:
        """Calculate weighted sum of objectives"""
        return sum(
            self.current_weights.get(k, 0) * v 
            for k, v in objectives.items()
        )
    
    def _pareto_reward(self, objectives: Dict[str, float]) -> float:
        """Calculate reward using Pareto optimization"""
        # Get best weights from Pareto frontier
        best_weights = self.pareto_frontier.get_best_weights()
        
        return sum(
            best_weights.get(k, 0) * v 
            for k, v in objectives.items()
        )
    
    def _lexicographic_reward(self, objectives: Dict[str, float]) -> float:
        """Calculate reward using lexicographic ordering"""
        reward = 0.0
        weight_multiplier = 1.0
        
        for priority_obj in self.config.objective_priorities:
            if priority_obj in objectives:
                reward += objectives[priority_obj] * weight_multiplier
                weight_multiplier *= 0.5  # Decay weight for lower priorities
        
        return reward
    
    def _constraint_based_reward(self, objectives: Dict[str, float]) -> float:
        """Calculate reward with hard constraints"""
        base_reward = self._weighted_sum_reward(objectives)
        
        # Apply constraint penalties
        for key, (min_val, max_val) in self.config.constraints.items():
            if key in objectives:
                if objectives[key] < min_val:
                    violation = (min_val - objectives[key]) / abs(min_val) if min_val != 0 else 1
                    base_reward -= violation * 2.0
                elif objectives[key] > max_val:
                    violation = (objectives[key] - max_val) / abs(max_val) if max_val != 0 else 1
                    base_reward -= violation * 2.0
        
        return base_reward
    
    def _adaptive_reward(self, objectives: Dict[str, float]) -> float:
        """Calculate reward using adaptive weights"""
        # Get adapted weights
        adapted_weights = self.weight_optimizer.get_weights()
        self.current_weights = adapted_weights
        
        # Calculate base reward
        base_reward = sum(
            adapted_weights.get(k, 0) * v 
            for k, v in objectives.items()
        )
        
        # Add constraint satisfaction bonus
        all_satisfied = True
        for key, (min_val, max_val) in self.config.constraints.items():
            if key in objectives:
                if objectives[key] < min_val or objectives[key] > max_val:
                    all_satisfied = False
                    break
        
        if all_satisfied:
            base_reward *= 1.2  # 20% bonus for meeting all constraints
        
        return base_reward
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            'current_weights': self.current_weights,
            'objective_stats': self.objective_stats,
            'pareto_frontier_size': len(self.pareto_frontier.frontier_points),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'std_reward': np.std(self.reward_history) if self.reward_history else 0
        }
        
        # Add component metrics
        for name, component in self.components.items():
            if hasattr(component, 'get_metrics'):
                metrics[f'{name}_metrics'] = component.get_metrics()
        
        return metrics
    
    def reset_episode(self):
        """Reset episode-specific tracking"""
        # Reset episode-specific components
        self.components['exploration'].reset_episode()
        
        # Clear short-term history
        if len(self.objective_history) > self.config.memory_size:
            self.objective_history = self.objective_history[-self.config.memory_size:]
            self.reward_history = self.reward_history[-self.config.memory_size:]
    
    def save_state(self, filepath: str):
        """Save optimizer state"""
        state = {
            'current_weights': self.current_weights,
            'objective_stats': self.objective_stats,
            'pareto_frontier': {
                'points': self.pareto_frontier.frontier_points,
                'weights': self.pareto_frontier.frontier_weights
            },
            'weight_optimizer': {
                'weights': self.weight_optimizer.current_weights,
                'gradients': self.weight_optimizer.gradient_estimates
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load optimizer state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_weights = state['current_weights']
        self.objective_stats = state['objective_stats']
        self.pareto_frontier.frontier_points = state['pareto_frontier']['points']
        self.pareto_frontier.frontier_weights = state['pareto_frontier']['weights']
        self.weight_optimizer.current_weights = state['weight_optimizer']['weights']
        self.weight_optimizer.gradient_estimates = state['weight_optimizer']['gradients']