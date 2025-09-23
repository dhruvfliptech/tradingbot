"""
Dynamic Weight Optimization for Reward Functions

This module implements sophisticated optimization algorithms for dynamically
adjusting reward component weights based on market conditions and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr, pearsonr
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .reward_components import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for reward weight optimizer"""
    
    # Optimization method
    method: str = 'bayesian'  # 'grid', 'random', 'bayesian', 'evolutionary', 'gradient'
    
    # Optimization objectives
    primary_metric: str = 'sharpe_ratio'
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'max_drawdown', 'win_rate'
    ])
    
    # Weight constraints
    weight_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'profit': (0.1, 0.4),
        'risk_adjusted': (0.2, 0.5),
        'drawdown': (0.1, 0.3),
        'consistency': (0.05, 0.25),
        'transaction_cost': (0.01, 0.1),
        'exploration': (0.0, 0.1)
    })
    
    # Sum constraint (weights should sum to approximately 1.0)
    weight_sum_target: float = 1.0
    weight_sum_tolerance: float = 0.05
    
    # Optimization parameters
    n_iterations: int = 100
    n_trials: int = 50  # For Optuna
    population_size: int = 20  # For evolutionary
    
    # Bayesian optimization
    acquisition_function: str = 'EI'  # Expected Improvement
    exploration_weight: float = 0.1
    
    # Performance evaluation
    evaluation_episodes: int = 10
    evaluation_window: int = 100  # Steps per evaluation
    
    # Adaptation
    adaptation_threshold: float = 0.1  # 10% performance change triggers adaptation
    min_samples_before_adapt: int = 50
    
    # Market regime specific weights
    regime_specific_weights: bool = True
    regime_transition_smoothing: float = 0.9  # Smooth weight transitions


class BayesianOptimizer:
    """
    Bayesian optimization for reward weights using Gaussian Processes
    
    This provides sample-efficient optimization by modeling the objective
    function and selecting promising weight configurations.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        
        # Gaussian Process model
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        # Observed data
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        
        # Best configuration
        self.best_weights: Optional[Dict[str, float]] = None
        self.best_performance: float = float('-inf')
        
    def optimize(self,
                evaluation_function: Callable,
                n_iterations: Optional[int] = None) -> Dict[str, float]:
        """
        Optimize weights using Bayesian optimization
        
        Args:
            evaluation_function: Function to evaluate weight configuration
            n_iterations: Number of optimization iterations
            
        Returns:
            Optimized weights
        """
        n_iterations = n_iterations or self.config.n_iterations
        
        # Initial random sampling
        n_initial = min(10, n_iterations // 5)
        for _ in range(n_initial):
            weights = self._sample_random_weights()
            performance = evaluation_function(weights)
            self._add_observation(weights, performance)
        
        # Bayesian optimization loop
        for iteration in range(n_iterations - n_initial):
            # Select next weights using acquisition function
            weights = self._select_next_weights()
            
            # Evaluate performance
            performance = evaluation_function(weights)
            
            # Update observations
            self._add_observation(weights, performance)
            
            # Update GP model
            if len(self.X_observed) > 2:
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            if iteration % 10 == 0:
                logger.info(f"Bayesian optimization iteration {iteration}: "
                          f"Best performance = {self.best_performance:.4f}")
        
        return self.best_weights
    
    def _add_observation(self, weights: Dict[str, float], performance: float):
        """Add observation to history"""
        # Convert weights to vector
        weight_vector = self._weights_to_vector(weights)
        self.X_observed.append(weight_vector)
        self.y_observed.append(performance)
        
        # Update best
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_weights = weights.copy()
    
    def _weights_to_vector(self, weights: Dict[str, float]) -> np.ndarray:
        """Convert weight dictionary to vector"""
        return np.array([weights.get(k, 0) for k in sorted(self.config.weight_bounds.keys())])
    
    def _vector_to_weights(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert vector to weight dictionary"""
        keys = sorted(self.config.weight_bounds.keys())
        weights = {k: float(v) for k, v in zip(keys, vector)}
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        
        return weights
    
    def _sample_random_weights(self) -> Dict[str, float]:
        """Sample random weight configuration"""
        weights = {}
        for key, (low, high) in self.config.weight_bounds.items():
            weights[key] = np.random.uniform(low, high)
        
        # Normalize
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        
        return weights
    
    def _select_next_weights(self) -> Dict[str, float]:
        """Select next weights using acquisition function"""
        if len(self.X_observed) < 3:
            return self._sample_random_weights()
        
        # Generate candidates
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            weights = self._sample_random_weights()
            candidates.append(self._weights_to_vector(weights))
        
        candidates = np.array(candidates)
        
        # Predict mean and std using GP
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        # Calculate acquisition function (Expected Improvement)
        if self.config.acquisition_function == 'EI':
            acquisition = self._expected_improvement(mu, sigma)
        elif self.config.acquisition_function == 'UCB':
            acquisition = self._upper_confidence_bound(mu, sigma)
        else:
            acquisition = mu  # Mean prediction
        
        # Select best candidate
        best_idx = np.argmax(acquisition)
        best_vector = candidates[best_idx]
        
        return self._vector_to_weights(best_vector)
    
    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        from scipy.stats import norm
        
        best_y = max(self.y_observed) if self.y_observed else 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mu - best_y - self.config.exploration_weight) / sigma
            ei = (mu - best_y - self.config.exploration_weight) * norm.cdf(Z) + \
                 sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Calculate Upper Confidence Bound acquisition function"""
        beta = 2 * np.log(len(self.X_observed) ** 2)
        return mu + np.sqrt(beta) * sigma


class EvolutionaryOptimizer:
    """
    Evolutionary algorithm for weight optimization
    
    Uses genetic algorithms to evolve optimal weight configurations.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.population: List[Dict[str, float]] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        
    def optimize(self,
                evaluation_function: Callable,
                n_generations: Optional[int] = None) -> Dict[str, float]:
        """
        Optimize weights using evolutionary algorithm
        
        Args:
            evaluation_function: Function to evaluate weight configuration
            n_generations: Number of generations
            
        Returns:
            Optimized weights
        """
        n_generations = n_generations or self.config.n_iterations
        
        # Initialize population
        self._initialize_population()
        
        best_weights = None
        best_fitness = float('-inf')
        
        for generation in range(n_generations):
            self.generation = generation
            
            # Evaluate fitness
            self.fitness_scores = [evaluation_function(w) for w in self.population]
            
            # Track best
            gen_best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = self.fitness_scores[gen_best_idx]
                best_weights = self.population[gen_best_idx].copy()
            
            # Selection
            selected = self._selection()
            
            # Crossover
            offspring = self._crossover(selected)
            
            # Mutation
            mutated = self._mutation(offspring)
            
            # Create new population
            self.population = mutated
            
            if generation % 10 == 0:
                logger.info(f"Evolution generation {generation}: "
                          f"Best fitness = {best_fitness:.4f}")
        
        return best_weights
    
    def _initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.config.population_size):
            weights = {}
            for key, (low, high) in self.config.weight_bounds.items():
                weights[key] = np.random.uniform(low, high)
            
            # Normalize
            total = sum(weights.values())
            for k in weights:
                weights[k] /= total
            
            self.population.append(weights)
    
    def _selection(self) -> List[Dict[str, float]]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.config.population_size):
            # Random tournament
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = [self.fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Uniform crossover"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Create two offspring
            child1 = {}
            child2 = {}
            
            for key in parent1.keys():
                if np.random.random() < 0.5:
                    child1[key] = parent1[key]
                    child2[key] = parent2[key]
                else:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
            
            # Normalize
            for child in [child1, child2]:
                total = sum(child.values())
                for k in child:
                    child[k] /= total
            
            offspring.extend([child1, child2])
        
        return offspring[:self.config.population_size]
    
    def _mutation(self, population: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Gaussian mutation"""
        mutation_rate = 0.1
        mutation_std = 0.05
        
        mutated = []
        for individual in population:
            if np.random.random() < mutation_rate:
                new_individual = {}
                for key, value in individual.items():
                    # Add Gaussian noise
                    noise = np.random.normal(0, mutation_std)
                    new_value = value + noise
                    
                    # Clip to bounds
                    low, high = self.config.weight_bounds[key]
                    new_value = np.clip(new_value, low, high)
                    new_individual[key] = new_value
                
                # Normalize
                total = sum(new_individual.values())
                for k in new_individual:
                    new_individual[k] /= total
                
                mutated.append(new_individual)
            else:
                mutated.append(individual.copy())
        
        return mutated


class MarketAdaptiveOptimizer:
    """
    Market regime adaptive weight optimizer
    
    Maintains separate weight configurations for different market regimes
    and smoothly transitions between them.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        
        # Regime-specific weights
        self.regime_weights: Dict[MarketRegime, Dict[str, float]] = {}
        
        # Initialize with base weights for each regime
        for regime in MarketRegime:
            self.regime_weights[regime] = self._get_initial_regime_weights(regime)
        
        # Current weights (smoothed)
        self.current_weights = config.weight_bounds.copy()
        self.previous_regime = MarketRegime.SIDEWAYS
        
        # Performance tracking per regime
        self.regime_performance: Dict[MarketRegime, List[float]] = {
            regime: [] for regime in MarketRegime
        }
        
    def _get_initial_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get initial weights for specific regime"""
        if regime == MarketRegime.BULL:
            return {
                'profit': 0.35,
                'risk_adjusted': 0.25,
                'drawdown': 0.15,
                'consistency': 0.15,
                'transaction_cost': 0.05,
                'exploration': 0.05
            }
        elif regime == MarketRegime.BEAR:
            return {
                'profit': 0.15,
                'risk_adjusted': 0.30,
                'drawdown': 0.30,
                'consistency': 0.15,
                'transaction_cost': 0.05,
                'exploration': 0.05
            }
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return {
                'profit': 0.20,
                'risk_adjusted': 0.35,
                'drawdown': 0.25,
                'consistency': 0.10,
                'transaction_cost': 0.08,
                'exploration': 0.02
            }
        else:  # SIDEWAYS or LOW_VOLATILITY
            return {
                'profit': 0.25,
                'risk_adjusted': 0.30,
                'drawdown': 0.20,
                'consistency': 0.15,
                'transaction_cost': 0.05,
                'exploration': 0.05
            }
    
    def get_weights(self, current_regime: MarketRegime) -> Dict[str, float]:
        """
        Get weights for current market regime with smooth transitions
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Smoothed weight configuration
        """
        if not self.config.regime_specific_weights:
            return self.current_weights
        
        target_weights = self.regime_weights[current_regime]
        
        # Smooth transition if regime changed
        if current_regime != self.previous_regime:
            alpha = self.config.regime_transition_smoothing
            
            smoothed_weights = {}
            for key in target_weights:
                smoothed_weights[key] = (
                    alpha * self.current_weights.get(key, 0) +
                    (1 - alpha) * target_weights[key]
                )
            
            self.current_weights = smoothed_weights
            self.previous_regime = current_regime
        else:
            self.current_weights = target_weights
        
        return self.current_weights
    
    def update_regime_weights(self,
                             regime: MarketRegime,
                             performance: float,
                             evaluation_function: Optional[Callable] = None):
        """
        Update weights for specific regime based on performance
        
        Args:
            regime: Market regime
            performance: Performance metric
            evaluation_function: Function to evaluate new weights
        """
        # Track performance
        self.regime_performance[regime].append(performance)
        
        if len(self.regime_performance[regime]) < self.config.min_samples_before_adapt:
            return
        
        # Check if adaptation needed
        recent_performance = np.mean(self.regime_performance[regime][-10:])
        historical_performance = np.mean(self.regime_performance[regime][:-10])
        
        if abs(recent_performance - historical_performance) > self.config.adaptation_threshold:
            if evaluation_function:
                # Optimize weights for this regime
                optimizer = BayesianOptimizer(self.config)
                
                # Create regime-specific evaluation
                def regime_eval(weights):
                    # Simulate performance in this regime
                    return evaluation_function(weights, regime)
                
                optimized_weights = optimizer.optimize(regime_eval, n_iterations=20)
                self.regime_weights[regime] = optimized_weights
                
                logger.info(f"Updated weights for {regime.value} regime")


class HybridOptimizer:
    """
    Hybrid optimizer combining multiple optimization strategies
    
    Uses ensemble of optimizers and selects best performing configuration.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        
        # Initialize component optimizers
        self.bayesian = BayesianOptimizer(config)
        self.evolutionary = EvolutionaryOptimizer(config)
        self.adaptive = MarketAdaptiveOptimizer(config)
        
        # Ensemble weights
        self.ensemble_weights: List[Dict[str, float]] = []
        self.ensemble_performance: List[float] = []
        
    def optimize(self,
                evaluation_function: Callable,
                current_regime: Optional[MarketRegime] = None) -> Dict[str, float]:
        """
        Optimize using ensemble of methods
        
        Args:
            evaluation_function: Function to evaluate weights
            current_regime: Current market regime
            
        Returns:
            Optimized weights
        """
        candidates = []
        
        # Bayesian optimization
        logger.info("Running Bayesian optimization...")
        bayesian_weights = self.bayesian.optimize(
            evaluation_function,
            n_iterations=self.config.n_iterations // 3
        )
        candidates.append(('bayesian', bayesian_weights))
        
        # Evolutionary optimization
        logger.info("Running evolutionary optimization...")
        evolutionary_weights = self.evolutionary.optimize(
            evaluation_function,
            n_generations=self.config.n_iterations // 3
        )
        candidates.append(('evolutionary', evolutionary_weights))
        
        # Market adaptive (if regime provided)
        if current_regime:
            adaptive_weights = self.adaptive.get_weights(current_regime)
            candidates.append(('adaptive', adaptive_weights))
        
        # Evaluate all candidates
        best_weights = None
        best_performance = float('-inf')
        best_method = None
        
        for method, weights in candidates:
            performance = evaluation_function(weights)
            logger.info(f"{method} optimization performance: {performance:.4f}")
            
            if performance > best_performance:
                best_performance = performance
                best_weights = weights
                best_method = method
        
        logger.info(f"Best optimization method: {best_method} with performance {best_performance:.4f}")
        
        # Store in ensemble
        self.ensemble_weights.append(best_weights)
        self.ensemble_performance.append(best_performance)
        
        return best_weights
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get weighted average of ensemble"""
        if not self.ensemble_weights:
            return self.config.weight_bounds
        
        # Weight by performance
        total_performance = sum(self.ensemble_performance)
        
        ensemble_avg = {}
        for key in self.ensemble_weights[0].keys():
            weighted_sum = sum(
                w[key] * p / total_performance
                for w, p in zip(self.ensemble_weights, self.ensemble_performance)
            )
            ensemble_avg[key] = weighted_sum
        
        return ensemble_avg


def create_optimizer(config: Optional[OptimizerConfig] = None,
                    method: Optional[str] = None) -> Any:
    """
    Factory function to create optimizer
    
    Args:
        config: Optimizer configuration
        method: Optimization method
        
    Returns:
        Optimizer instance
    """
    if config is None:
        config = OptimizerConfig()
    
    method = method or config.method
    
    if method == 'bayesian':
        return BayesianOptimizer(config)
    elif method == 'evolutionary':
        return EvolutionaryOptimizer(config)
    elif method == 'adaptive':
        return MarketAdaptiveOptimizer(config)
    elif method == 'hybrid':
        return HybridOptimizer(config)
    else:
        raise ValueError(f"Unknown optimization method: {method}")