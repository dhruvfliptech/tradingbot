"""
Multi-Armed Bandit Strategy Selector

This module implements sophisticated multi-armed bandit algorithms to dynamically
select the best trading agents and strategies based on their performance.

Features:
- Multiple bandit algorithms (UCB, Thompson Sampling, Epsilon-Greedy, etc.)
- Contextual bandits for regime-aware selection
- Non-stationary reward handling for changing market conditions
- Bayesian updates and confidence intervals
- Performance tracking and regret minimization

The bandit selector continuously learns which agents perform best in different
contexts and automatically adapts the selection strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
import math
from collections import defaultdict, deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import from other ensemble modules
from ..agents.specialized_agents import AgentType
from ..regime.market_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class BanditAlgorithm(Enum):
    """Types of bandit algorithms"""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"
    EXP3 = "exp3"  # Exponential-weight algorithm for Exploration and Exploitation
    CONTEXTUAL_UCB = "contextual_ucb"
    BAYESIAN_UCB = "bayesian_ucb"
    SLIDING_WINDOW_UCB = "sliding_window_ucb"


@dataclass
class BanditConfig:
    """Configuration for bandit strategy selector"""
    
    # Algorithm selection
    algorithm: BanditAlgorithm = BanditAlgorithm.CONTEXTUAL_UCB
    fallback_algorithm: BanditAlgorithm = BanditAlgorithm.UCB
    
    # Epsilon-greedy parameters
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    
    # UCB parameters
    ucb_confidence: float = 1.0  # Exploration parameter
    ucb_decay: float = 1.0  # Decay factor for non-stationary environments
    
    # Thompson Sampling parameters
    alpha_prior: float = 1.0  # Prior for Beta distribution
    beta_prior: float = 1.0
    
    # EXP3 parameters
    exp3_gamma: float = 0.1  # Exploration parameter
    
    # Contextual bandit parameters
    context_window: int = 20  # Number of features for context
    context_learning_rate: float = 0.01
    context_regularization: float = 0.1
    
    # Performance tracking
    reward_window: int = 100  # Window for reward tracking
    performance_threshold: float = 0.05  # Minimum performance difference
    regret_calculation_window: int = 500
    
    # Non-stationary handling
    forgetting_factor: float = 0.95  # For exponential forgetting
    change_detection_threshold: float = 0.1
    adaptation_rate: float = 0.1
    
    # Minimum pulls before selection
    min_pulls_per_arm: int = 5
    warm_up_period: int = 50
    
    # Logging and debugging
    enable_logging: bool = True
    log_frequency: int = 50
    save_history: bool = True


@dataclass
class BanditArm:
    """Represents a bandit arm (agent/strategy)"""
    
    agent_type: AgentType
    agent_name: str
    
    # Reward statistics
    num_pulls: int = 0
    total_reward: float = 0.0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # UCB statistics
    mean_reward: float = 0.0
    confidence_bound: float = float('inf')
    last_pull_time: int = 0
    
    # Thompson Sampling parameters
    alpha: float = 1.0  # Successes
    beta: float = 1.0   # Failures
    
    # Contextual bandit features
    context_weights: np.ndarray = field(default_factory=lambda: np.zeros(20))
    context_covariance: np.ndarray = field(default_factory=lambda: np.eye(20))
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    
    def update_reward(self, reward: float, timestamp: int):
        """Update arm statistics with new reward"""
        self.num_pulls += 1
        self.total_reward += reward
        self.reward_history.append((reward, timestamp))
        self.last_pull_time = timestamp
        
        # Update mean
        self.mean_reward = self.total_reward / self.num_pulls
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update derived performance metrics"""
        if len(self.reward_history) < 5:
            return
        
        rewards = [r for r, _ in self.reward_history[-50:]]  # Last 50 rewards
        
        # Sharpe ratio
        if len(rewards) > 1:
            mean_return = np.mean(rewards)
            std_return = np.std(rewards)
            self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Win rate
        positive_rewards = [r for r in rewards if r > 0]
        self.win_rate = len(positive_rewards) / len(rewards) if rewards else 0
        
        # Max drawdown (simplified)
        cumulative = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / np.maximum(running_max, 1)
        self.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Volatility
        self.volatility = float(np.std(rewards)) if len(rewards) > 1 else 0
    
    def get_thompson_sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling"""
        return np.random.beta(self.alpha, self.beta)
    
    def update_thompson_params(self, reward: float):
        """Update Thompson Sampling parameters"""
        if reward > 0:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get arm performance metrics"""
        return {
            'num_pulls': self.num_pulls,
            'mean_reward': self.mean_reward,
            'total_reward': self.total_reward,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'confidence_bound': self.confidence_bound,
            'alpha': self.alpha,
            'beta': self.beta
        }


class BanditAlgorithmBase(ABC):
    """Abstract base class for bandit algorithms"""
    
    def __init__(self, config: BanditConfig, arms: Dict[AgentType, BanditArm]):
        self.config = config
        self.arms = arms
        self.total_pulls = 0
        self.algorithm_history = []
    
    @abstractmethod
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select an arm based on the algorithm"""
        pass
    
    @abstractmethod
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update algorithm state after receiving reward"""
        pass
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get current scores for all arms"""
        return {arm_type: 0.0 for arm_type in self.arms.keys()}


class EpsilonGreedyBandit(BanditAlgorithmBase):
    """Epsilon-greedy bandit algorithm"""
    
    def __init__(self, config: BanditConfig, arms: Dict[AgentType, BanditArm]):
        super().__init__(config, arms)
        self.current_epsilon = config.epsilon
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select arm using epsilon-greedy strategy"""
        
        # Exploration vs exploitation
        if np.random.random() < self.current_epsilon:
            # Explore: random selection
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: select best arm
            best_arm = max(self.arms.keys(), key=lambda a: self.arms[a].mean_reward)
            return best_arm
    
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update epsilon and arm statistics"""
        
        self.total_pulls += 1
        self.arms[arm].update_reward(reward, self.total_pulls)
        
        # Decay epsilon
        self.current_epsilon = max(
            self.config.min_epsilon,
            self.current_epsilon * self.config.epsilon_decay
        )
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get epsilon-greedy scores"""
        return {arm_type: arm.mean_reward for arm_type, arm in self.arms.items()}


class UCBBandit(BanditAlgorithmBase):
    """Upper Confidence Bound bandit algorithm"""
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select arm using UCB strategy"""
        
        # Calculate UCB for each arm
        ucb_scores = {}
        
        for arm_type, arm in self.arms.items():
            if arm.num_pulls == 0:
                # Unplayed arms get infinite confidence
                ucb_scores[arm_type] = float('inf')
            else:
                # UCB formula: mean + confidence_bound * sqrt(ln(total_pulls) / arm_pulls)
                confidence_term = self.config.ucb_confidence * math.sqrt(
                    math.log(max(self.total_pulls, 1)) / arm.num_pulls
                )
                ucb_scores[arm_type] = arm.mean_reward + confidence_term
                arm.confidence_bound = confidence_term
        
        # Select arm with highest UCB
        return max(ucb_scores.keys(), key=ucb_scores.get)
    
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update UCB statistics"""
        
        self.total_pulls += 1
        self.arms[arm].update_reward(reward, self.total_pulls)
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get UCB scores"""
        
        scores = {}
        for arm_type, arm in self.arms.items():
            if arm.num_pulls == 0:
                scores[arm_type] = float('inf')
            else:
                confidence_term = self.config.ucb_confidence * math.sqrt(
                    math.log(max(self.total_pulls, 1)) / arm.num_pulls
                )
                scores[arm_type] = arm.mean_reward + confidence_term
        
        return scores


class ThompsonSamplingBandit(BanditAlgorithmBase):
    """Thompson Sampling bandit algorithm"""
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select arm using Thompson Sampling"""
        
        # Sample from each arm's posterior distribution
        samples = {}
        for arm_type, arm in self.arms.items():
            samples[arm_type] = arm.get_thompson_sample()
        
        # Select arm with highest sample
        return max(samples.keys(), key=samples.get)
    
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update Thompson Sampling parameters"""
        
        self.total_pulls += 1
        self.arms[arm].update_reward(reward, self.total_pulls)
        self.arms[arm].update_thompson_params(reward)
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get Thompson Sampling scores (expected values)"""
        
        scores = {}
        for arm_type, arm in self.arms.items():
            # Expected value of Beta distribution
            scores[arm_type] = arm.alpha / (arm.alpha + arm.beta)
        
        return scores


class ContextualUCBBandit(BanditAlgorithmBase):
    """Contextual UCB bandit with linear rewards"""
    
    def __init__(self, config: BanditConfig, arms: Dict[AgentType, BanditArm]):
        super().__init__(config, arms)
        self.context_dim = config.context_window
        
        # Initialize linear model parameters for each arm
        for arm in self.arms.values():
            arm.context_weights = np.zeros(self.context_dim)
            arm.context_covariance = np.eye(self.context_dim) * config.context_regularization
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select arm using contextual UCB"""
        
        if context is None:
            # Fall back to standard UCB
            return UCBBandit(self.config, self.arms).select_arm()
        
        context = context[:self.context_dim]  # Ensure correct dimension
        if len(context) < self.context_dim:
            context = np.pad(context, (0, self.context_dim - len(context)))
        
        ucb_scores = {}
        
        for arm_type, arm in self.arms.items():
            # Linear prediction: theta^T * context
            predicted_reward = np.dot(arm.context_weights, context)
            
            # Confidence bound: sqrt(context^T * A^-1 * context)
            try:
                confidence_bound = math.sqrt(
                    np.dot(context, np.dot(np.linalg.inv(arm.context_covariance), context))
                ) * self.config.ucb_confidence
            except np.linalg.LinAlgError:
                confidence_bound = self.config.ucb_confidence
            
            ucb_scores[arm_type] = predicted_reward + confidence_bound
        
        return max(ucb_scores.keys(), key=ucb_scores.get)
    
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update contextual model"""
        
        self.total_pulls += 1
        self.arms[arm].update_reward(reward, self.total_pulls)
        
        if context is not None:
            context = context[:self.context_dim]
            if len(context) < self.context_dim:
                context = np.pad(context, (0, self.context_dim - len(context)))
            
            arm_obj = self.arms[arm]
            
            # Update covariance matrix: A = A + x * x^T
            arm_obj.context_covariance += np.outer(context, context)
            
            # Update weights using Sherman-Morrison formula for efficiency
            try:
                A_inv = np.linalg.inv(arm_obj.context_covariance)
                arm_obj.context_weights = np.dot(A_inv, 
                                               np.dot(arm_obj.context_covariance, arm_obj.context_weights) + 
                                               context * reward)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                arm_obj.context_weights = np.dot(
                    np.linalg.pinv(arm_obj.context_covariance),
                    context * reward
                )
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get contextual UCB scores"""
        
        if context is None:
            return {arm_type: arm.mean_reward for arm_type, arm in self.arms.items()}
        
        context = context[:self.context_dim]
        if len(context) < self.context_dim:
            context = np.pad(context, (0, self.context_dim - len(context)))
        
        scores = {}
        for arm_type, arm in self.arms.items():
            predicted_reward = np.dot(arm.context_weights, context)
            
            try:
                confidence_bound = math.sqrt(
                    np.dot(context, np.dot(np.linalg.inv(arm.context_covariance), context))
                ) * self.config.ucb_confidence
            except np.linalg.LinAlgError:
                confidence_bound = self.config.ucb_confidence
            
            scores[arm_type] = predicted_reward + confidence_bound
        
        return scores


class SlidingWindowUCBBandit(BanditAlgorithmBase):
    """UCB with sliding window for non-stationary environments"""
    
    def __init__(self, config: BanditConfig, arms: Dict[AgentType, BanditArm]):
        super().__init__(config, arms)
        self.window_size = config.reward_window
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> AgentType:
        """Select arm using sliding window UCB"""
        
        ucb_scores = {}
        
        for arm_type, arm in self.arms.items():
            # Use only recent rewards
            recent_rewards = [r for r, t in arm.reward_history 
                            if t > self.total_pulls - self.window_size]
            
            if len(recent_rewards) == 0:
                ucb_scores[arm_type] = float('inf')
            else:
                mean_reward = np.mean(recent_rewards)
                confidence_term = self.config.ucb_confidence * math.sqrt(
                    math.log(max(self.total_pulls, 1)) / len(recent_rewards)
                )
                ucb_scores[arm_type] = mean_reward + confidence_term
        
        return max(ucb_scores.keys(), key=ucb_scores.get)
    
    def update(self, arm: AgentType, reward: float, context: Optional[np.ndarray] = None):
        """Update sliding window statistics"""
        
        self.total_pulls += 1
        self.arms[arm].update_reward(reward, self.total_pulls)
    
    def get_arm_scores(self, context: Optional[np.ndarray] = None) -> Dict[AgentType, float]:
        """Get sliding window UCB scores"""
        
        scores = {}
        for arm_type, arm in self.arms.items():
            recent_rewards = [r for r, t in arm.reward_history 
                            if t > self.total_pulls - self.window_size]
            
            if len(recent_rewards) == 0:
                scores[arm_type] = 0.0
            else:
                mean_reward = np.mean(recent_rewards)
                confidence_term = self.config.ucb_confidence * math.sqrt(
                    math.log(max(self.total_pulls, 1)) / len(recent_rewards)
                )
                scores[arm_type] = mean_reward + confidence_term
        
        return scores


class MultiArmedBanditSelector:
    """
    Main multi-armed bandit strategy selector
    
    Manages multiple bandit algorithms and provides intelligent agent selection
    based on contextual information and performance tracking.
    """
    
    def __init__(self, agent_types: List[AgentType], config: Optional[BanditConfig] = None):
        self.config = config or BanditConfig()
        self.agent_types = agent_types
        
        # Initialize bandit arms
        self.arms = {}
        for agent_type in agent_types:
            arm = BanditArm(
                agent_type=agent_type,
                agent_name=agent_type.value.replace('_', ' ').title()
            )
            # Initialize Thompson Sampling parameters
            arm.alpha = self.config.alpha_prior
            arm.beta = self.config.beta_prior
            self.arms[agent_type] = arm
        
        # Initialize algorithms
        self.algorithms = {
            BanditAlgorithm.EPSILON_GREEDY: EpsilonGreedyBandit(self.config, self.arms),
            BanditAlgorithm.UCB: UCBBandit(self.config, self.arms),
            BanditAlgorithm.THOMPSON_SAMPLING: ThompsonSamplingBandit(self.config, self.arms),
            BanditAlgorithm.CONTEXTUAL_UCB: ContextualUCBBandit(self.config, self.arms),
            BanditAlgorithm.SLIDING_WINDOW_UCB: SlidingWindowUCBBandit(self.config, self.arms)
        }
        
        self.current_algorithm = self.algorithms[self.config.algorithm]
        
        # Selection tracking
        self.selection_history = []
        self.performance_history = []
        self.regret_history = []
        
        # Context management
        self.context_buffer = deque(maxlen=self.config.context_window)
        
        logger.info(f"Bandit selector initialized with {len(self.arms)} arms using {self.config.algorithm.value}")
    
    def select_agent(self, context: Optional[Dict[str, Any]] = None) -> Tuple[AgentType, Dict[str, Any]]:
        """
        Select the best agent based on current context
        
        Args:
            context: Context information (market regime, features, etc.)
            
        Returns:
            Tuple of (selected_agent_type, selection_info)
        """
        
        # Convert context to feature vector
        context_vector = self._extract_context_features(context)
        
        # Warm-up period: ensure each arm is pulled minimum times
        if self._needs_warmup():
            selected_agent = self._warmup_selection()
        else:
            # Use bandit algorithm for selection
            selected_agent = self.current_algorithm.select_arm(context_vector)
        
        # Get selection information
        arm_scores = self.current_algorithm.get_arm_scores(context_vector)
        
        selection_info = {
            'algorithm': self.config.algorithm.value,
            'selected_agent': selected_agent.value,
            'arm_scores': {agent.value: score for agent, score in arm_scores.items()},
            'context_features': context_vector.tolist() if context_vector is not None else None,
            'selection_reason': self._get_selection_reason(selected_agent, arm_scores),
            'timestamp': datetime.now().isoformat()
        }
        
        # Record selection
        self.selection_history.append({
            'agent': selected_agent,
            'context': context,
            'context_vector': context_vector,
            'arm_scores': arm_scores,
            'timestamp': datetime.now()
        })
        
        # Keep history manageable
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-500:]
        
        if self.config.enable_logging and len(self.selection_history) % self.config.log_frequency == 0:
            self._log_selection_stats()
        
        return selected_agent, selection_info
    
    def update_reward(self, agent_type: AgentType, reward: float, 
                     context: Optional[Dict[str, Any]] = None):
        """
        Update bandit with reward for selected agent
        
        Args:
            agent_type: Agent that was selected
            reward: Reward received
            context: Context when agent was selected
        """
        
        # Convert context to feature vector
        context_vector = self._extract_context_features(context)
        
        # Update algorithm
        self.current_algorithm.update(agent_type, reward, context_vector)
        
        # Record performance
        self.performance_history.append({
            'agent': agent_type,
            'reward': reward,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Calculate and record regret
        self._update_regret(reward)
        
        # Check for algorithm adaptation
        if len(self.performance_history) % 100 == 0:
            self._check_algorithm_adaptation()
    
    def _extract_context_features(self, context: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract feature vector from context dictionary"""
        
        if context is None:
            return None
        
        features = []
        
        # Market regime features
        regime = context.get('market_regime', MarketRegime.UNKNOWN)
        regime_confidence = context.get('regime_confidence', 0.0)
        
        # One-hot encode regime
        regime_features = [0.0] * len(MarketRegime)
        try:
            regime_idx = list(MarketRegime).index(regime)
            regime_features[regime_idx] = regime_confidence
        except (ValueError, IndexError):
            pass
        
        features.extend(regime_features)
        
        # Market condition features
        features.extend([
            context.get('volatility', 0.0),
            context.get('trend_strength', 0.0),
            context.get('momentum', 0.0),
            context.get('mean_reversion_signal', 0.0),
            context.get('volume_ratio', 1.0)
        ])
        
        # Portfolio features
        features.extend([
            context.get('current_drawdown', 0.0),
            context.get('portfolio_return', 0.0),
            context.get('sharpe_ratio', 0.0),
            context.get('win_rate', 0.0)
        ])
        
        # Pad or truncate to required size
        feature_vector = np.array(features[:self.config.context_window])
        if len(feature_vector) < self.config.context_window:
            feature_vector = np.pad(feature_vector, 
                                  (0, self.config.context_window - len(feature_vector)))
        
        # Add to context buffer
        self.context_buffer.append(feature_vector)
        
        return feature_vector
    
    def _needs_warmup(self) -> bool:
        """Check if we're still in the warm-up period"""
        
        # Check if any arm has been pulled less than minimum
        min_pulls = min(arm.num_pulls for arm in self.arms.values())
        return min_pulls < self.config.min_pulls_per_arm
    
    def _warmup_selection(self) -> AgentType:
        """Select agent during warm-up period"""
        
        # Find arms with minimum pulls
        min_pulls = min(arm.num_pulls for arm in self.arms.values())
        candidate_arms = [agent_type for agent_type, arm in self.arms.items() 
                         if arm.num_pulls == min_pulls]
        
        # Random selection among candidates
        return np.random.choice(candidate_arms)
    
    def _get_selection_reason(self, selected_agent: AgentType, 
                            arm_scores: Dict[AgentType, float]) -> str:
        """Get human-readable reason for selection"""
        
        selected_score = arm_scores.get(selected_agent, 0.0)
        arm = self.arms[selected_agent]
        
        if arm.num_pulls < self.config.min_pulls_per_arm:
            return f"Warm-up: exploring {selected_agent.value} (pulls: {arm.num_pulls})"
        
        if selected_score == float('inf'):
            return f"Exploration: {selected_agent.value} has infinite confidence"
        
        if self.config.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            if np.random.random() < self.current_algorithm.current_epsilon:
                return f"Exploration: random selection ({self.current_algorithm.current_epsilon:.3f})"
            else:
                return f"Exploitation: best mean reward ({arm.mean_reward:.3f})"
        
        elif self.config.algorithm in [BanditAlgorithm.UCB, BanditAlgorithm.CONTEXTUAL_UCB]:
            return f"UCB: highest confidence bound ({selected_score:.3f})"
        
        elif self.config.algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            return f"Thompson: sampled value ({selected_score:.3f})"
        
        else:
            return f"Algorithm: {self.config.algorithm.value} selected {selected_agent.value}"
    
    def _update_regret(self, received_reward: float):
        """Update regret calculation"""
        
        # Simple regret: difference from best possible reward
        best_possible = max(arm.mean_reward for arm in self.arms.values() if arm.num_pulls > 0)
        if best_possible > 0:
            regret = best_possible - received_reward
            self.regret_history.append(regret)
            
            # Keep regret history manageable
            if len(self.regret_history) > self.config.regret_calculation_window:
                self.regret_history = self.regret_history[-self.config.regret_calculation_window//2:]
    
    def _check_algorithm_adaptation(self):
        """Check if algorithm should be adapted based on performance"""
        
        if len(self.performance_history) < 100:
            return
        
        # Calculate recent performance
        recent_rewards = [p['reward'] for p in self.performance_history[-50:]]
        recent_performance = np.mean(recent_rewards)
        
        # Calculate regret
        if len(self.regret_history) > 10:
            recent_regret = np.mean(self.regret_history[-10:])
            
            # If regret is too high, consider switching algorithm
            if recent_regret > self.config.change_detection_threshold:
                logger.info(f"High regret detected ({recent_regret:.3f}), considering algorithm switch")
                # Could implement algorithm switching logic here
    
    def _log_selection_stats(self):
        """Log selection statistics"""
        
        if not self.config.enable_logging:
            return
        
        # Selection counts
        agent_counts = {}
        for selection in self.selection_history[-100:]:  # Last 100 selections
            agent = selection['agent']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Performance summary
        arm_metrics = {agent_type: arm.get_metrics() for agent_type, arm in self.arms.items()}
        
        logger.info("Bandit Selection Statistics:")
        logger.info(f"  Total selections: {len(self.selection_history)}")
        logger.info(f"  Algorithm: {self.config.algorithm.value}")
        
        for agent_type, count in agent_counts.items():
            metrics = arm_metrics[agent_type]
            logger.info(f"  {agent_type.value}: {count} selections, "
                       f"mean reward: {metrics['mean_reward']:.3f}, "
                       f"pulls: {metrics['num_pulls']}")
        
        if self.regret_history:
            avg_regret = np.mean(self.regret_history[-20:])
            logger.info(f"  Average regret (last 20): {avg_regret:.3f}")
    
    def get_agent_rankings(self) -> List[Tuple[AgentType, Dict[str, float]]]:
        """Get agents ranked by performance"""
        
        rankings = []
        for agent_type, arm in self.arms.items():
            metrics = arm.get_metrics()
            
            # Composite score
            score = (
                metrics['mean_reward'] * 0.4 +
                metrics['sharpe_ratio'] * 0.3 +
                metrics['win_rate'] * 0.2 +
                (1 + metrics['max_drawdown']) * 0.1  # Convert drawdown to positive
            )
            
            metrics['composite_score'] = score
            rankings.append((agent_type, metrics))
        
        return sorted(rankings, key=lambda x: x[1]['composite_score'], reverse=True)
    
    def get_bandit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bandit statistics"""
        
        # Agent metrics
        agent_metrics = {agent_type.value: arm.get_metrics() 
                        for agent_type, arm in self.arms.items()}
        
        # Selection statistics
        total_selections = len(self.selection_history)
        if total_selections > 0:
            recent_selections = self.selection_history[-100:]
            agent_selection_counts = {}
            for selection in recent_selections:
                agent = selection['agent'].value
                agent_selection_counts[agent] = agent_selection_counts.get(agent, 0) + 1
        else:
            agent_selection_counts = {}
        
        # Performance statistics
        performance_stats = {}
        if self.performance_history:
            recent_rewards = [p['reward'] for p in self.performance_history[-100:]]
            performance_stats = {
                'mean_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'min_reward': np.min(recent_rewards),
                'max_reward': np.max(recent_rewards),
                'total_episodes': len(self.performance_history)
            }
        
        # Regret statistics
        regret_stats = {}
        if self.regret_history:
            regret_stats = {
                'mean_regret': np.mean(self.regret_history),
                'cumulative_regret': np.sum(self.regret_history),
                'recent_regret': np.mean(self.regret_history[-20:]) if len(self.regret_history) >= 20 else 0
            }
        
        return {
            'algorithm': self.config.algorithm.value,
            'total_selections': total_selections,
            'agent_metrics': agent_metrics,
            'agent_selection_counts': agent_selection_counts,
            'performance_stats': performance_stats,
            'regret_stats': regret_stats,
            'warmup_complete': not self._needs_warmup(),
            'context_buffer_size': len(self.context_buffer)
        }
    
    def save_state(self, filepath: str):
        """Save bandit state"""
        
        # Prepare arm data for serialization
        arms_data = {}
        for agent_type, arm in self.arms.items():
            arms_data[agent_type.value] = {
                'num_pulls': arm.num_pulls,
                'total_reward': arm.total_reward,
                'mean_reward': arm.mean_reward,
                'alpha': arm.alpha,
                'beta': arm.beta,
                'reward_history': list(arm.reward_history)[-100:],  # Save last 100
                'metrics': arm.get_metrics()
            }
        
        state = {
            'config': {
                'algorithm': self.config.algorithm.value,
                'epsilon': self.config.epsilon,
                'ucb_confidence': self.config.ucb_confidence,
                'context_window': self.config.context_window
            },
            'arms': arms_data,
            'selection_history': [
                {
                    'agent': s['agent'].value,
                    'timestamp': s['timestamp'].isoformat(),
                    'context': s['context']
                }
                for s in self.selection_history[-100:]  # Save last 100
            ],
            'performance_summary': self.get_bandit_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Bandit state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load bandit state"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore arm data
            for agent_str, arm_data in state['arms'].items():
                try:
                    agent_type = AgentType(agent_str)
                    if agent_type in self.arms:
                        arm = self.arms[agent_type]
                        arm.num_pulls = arm_data['num_pulls']
                        arm.total_reward = arm_data['total_reward']
                        arm.mean_reward = arm_data['mean_reward']
                        arm.alpha = arm_data.get('alpha', self.config.alpha_prior)
                        arm.beta = arm_data.get('beta', self.config.beta_prior)
                        
                        # Restore reward history
                        arm.reward_history.clear()
                        for reward_data in arm_data['reward_history']:
                            if isinstance(reward_data, list) and len(reward_data) == 2:
                                arm.reward_history.append((reward_data[0], reward_data[1]))
                
                except ValueError:
                    continue
            
            logger.info(f"Bandit state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading bandit state: {e}")


# Factory function for creating bandit selectors
def create_bandit_selector(agent_types: List[AgentType], 
                          algorithm: BanditAlgorithm = BanditAlgorithm.CONTEXTUAL_UCB,
                          **config_kwargs) -> MultiArmedBanditSelector:
    """
    Factory function to create bandit selector with specified configuration
    
    Args:
        agent_types: List of agent types to include
        algorithm: Bandit algorithm to use
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured MultiArmedBanditSelector
    """
    
    config = BanditConfig(algorithm=algorithm, **config_kwargs)
    return MultiArmedBanditSelector(agent_types, config)


if __name__ == "__main__":
    # Example usage and testing
    
    # Create bandit selector
    agent_types = [AgentType.CONSERVATIVE, AgentType.AGGRESSIVE, AgentType.BALANCED, AgentType.CONTRARIAN]
    
    bandit_selector = create_bandit_selector(
        agent_types=agent_types,
        algorithm=BanditAlgorithm.CONTEXTUAL_UCB,
        epsilon=0.1,
        ucb_confidence=1.0,
        enable_logging=True
    )
    
    print(f"Created bandit selector with {len(agent_types)} agents")
    print(f"Using algorithm: {bandit_selector.config.algorithm.value}")
    
    # Simulate some selections and updates
    for i in range(50):
        # Create mock context
        context = {
            'market_regime': np.random.choice(list(MarketRegime)),
            'regime_confidence': np.random.random(),
            'volatility': np.random.random() * 0.3,
            'trend_strength': np.random.random() - 0.5,
            'current_drawdown': np.random.random() * 0.1
        }
        
        # Select agent
        selected_agent, info = bandit_selector.select_agent(context)
        
        # Simulate reward (higher for some agents in certain conditions)
        if context['market_regime'] == MarketRegime.BULL and selected_agent == AgentType.AGGRESSIVE:
            reward = np.random.normal(0.02, 0.01)  # Higher reward
        elif context['market_regime'] == MarketRegime.BEAR and selected_agent == AgentType.CONSERVATIVE:
            reward = np.random.normal(0.01, 0.005)  # Moderate reward
        else:
            reward = np.random.normal(0.0, 0.01)  # Random reward
        
        # Update bandit
        bandit_selector.update_reward(selected_agent, reward, context)
        
        if i % 10 == 0:
            print(f"Step {i}: Selected {selected_agent.value}, reward: {reward:.3f}")
    
    # Get final statistics
    stats = bandit_selector.get_bandit_statistics()
    print(f"\nFinal Statistics:")
    print(f"Algorithm: {stats['algorithm']}")
    print(f"Total selections: {stats['total_selections']}")
    print(f"Performance: {stats['performance_stats']}")
    
    # Get rankings
    rankings = bandit_selector.get_agent_rankings()
    print(f"\nAgent Rankings:")
    for i, (agent_type, metrics) in enumerate(rankings):
        print(f"{i+1}. {agent_type.value}: score={metrics['composite_score']:.3f}, "
              f"pulls={metrics['num_pulls']}, mean_reward={metrics['mean_reward']:.3f}")