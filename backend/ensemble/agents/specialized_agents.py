"""
Specialized Trading Agents for Multi-Agent Ensemble System

This module implements specialized agents with different objectives and reward functions:
- ConservativeAgent: Minimize drawdown, steady returns
- AggressiveAgent: Maximize returns, accept higher risk  
- BalancedAgent: Optimize Sharpe ratio
- ContrarianAgent: Mean reversion focus

Each agent is optimized for different market conditions and risk profiles.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from enum import Enum
import json

# Import base classes from existing RL service
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../rl-service'))

from agents.ppo_agent import PPOAgent, PPOConfig
from rewards.reward_calculator import RewardCalculator, RewardCalculatorConfig
from rewards.reward_components import RewardComponentConfig, MarketRegime

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"


@dataclass
class SpecializedAgentConfig:
    """Base configuration for specialized agents"""
    agent_type: AgentType
    name: str
    description: str
    
    # Risk parameters
    max_position_size: float = 1.0
    max_drawdown_threshold: float = 0.15
    risk_free_rate: float = 0.02
    
    # Performance targets
    target_return: float = 0.15
    target_sharpe: float = 1.0
    target_win_rate: float = 0.55
    
    # Agent-specific parameters
    reward_weights: Dict[str, float] = field(default_factory=dict)
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training_timesteps: int = 100000
    eval_frequency: int = 5000
    learning_rate: float = 3e-4
    
    def __post_init__(self):
        if not self.reward_weights:
            self.reward_weights = self._get_default_weights()
        if not self.strategy_params:
            self.strategy_params = self._get_default_params()
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default reward weights for agent type"""
        return {
            'return': 1.0,
            'risk': 0.5,
            'drawdown': 0.5,
            'volatility': 0.3,
            'transaction_cost': 0.1
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default strategy parameters"""
        return {
            'lookback_window': 20,
            'signal_threshold': 0.1,
            'position_sizing': 'fixed'
        }


class SpecializedRewardCalculator(RewardCalculator):
    """Specialized reward calculator for different agent types"""
    
    def __init__(self, agent_config: SpecializedAgentConfig):
        self.agent_config = agent_config
        
        # Create base reward config
        reward_config = RewardCalculatorConfig()
        super().__init__(reward_config)
        
        # Customize for agent type
        self._customize_rewards()
    
    def _customize_rewards(self):
        """Customize reward function based on agent type"""
        weights = self.agent_config.reward_weights
        agent_type = self.agent_config.agent_type
        
        if agent_type == AgentType.CONSERVATIVE:
            # Conservative: Heavy penalty for drawdown, reward stability
            weights.update({
                'return': 0.6,
                'risk': 1.5,
                'drawdown': 2.0,
                'volatility': 1.2,
                'consistency': 1.0,
                'transaction_cost': 0.8
            })
            
        elif agent_type == AgentType.AGGRESSIVE:
            # Aggressive: Maximize returns, accept higher risk
            weights.update({
                'return': 2.0,
                'risk': 0.3,
                'drawdown': 0.5,
                'volatility': 0.2,
                'momentum': 1.5,
                'transaction_cost': 0.3
            })
            
        elif agent_type == AgentType.BALANCED:
            # Balanced: Optimize risk-adjusted returns (Sharpe ratio)
            weights.update({
                'return': 1.0,
                'risk': 1.0,
                'drawdown': 1.0,
                'volatility': 0.8,
                'sharpe_ratio': 1.5,
                'transaction_cost': 0.6
            })
            
        elif agent_type == AgentType.CONTRARIAN:
            # Contrarian: Mean reversion focus
            weights.update({
                'return': 1.0,
                'risk': 0.8,
                'mean_reversion': 1.8,
                'oversold_bought': 1.5,
                'volatility': 0.6,
                'transaction_cost': 0.4
            })
    
    def calculate_specialized_reward(self, 
                                   current_state: Dict[str, Any],
                                   previous_state: Optional[Dict[str, Any]] = None,
                                   action_info: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """Calculate reward with agent-specific modifications"""
        
        # Get base reward
        base_reward, base_info = self.calculate(current_state, previous_state, action_info)
        
        # Add agent-specific reward components
        specialized_reward = self._add_specialized_components(
            base_reward, current_state, previous_state, action_info
        )
        
        # Combine rewards
        final_reward = base_reward + specialized_reward
        
        # Update info
        info = base_info.copy()
        info.update({
            'base_reward': base_reward,
            'specialized_reward': specialized_reward,
            'final_reward': final_reward,
            'agent_type': self.agent_config.agent_type.value
        })
        
        return final_reward, info
    
    def _add_specialized_components(self,
                                  base_reward: float,
                                  current_state: Dict[str, Any],
                                  previous_state: Optional[Dict[str, Any]],
                                  action_info: Optional[Dict[str, Any]]) -> float:
        """Add agent-specific reward components"""
        
        specialized_reward = 0.0
        weights = self.agent_config.reward_weights
        
        # Current portfolio metrics
        portfolio_value = current_state.get('portfolio_value', 0)
        previous_value = previous_state.get('portfolio_value', portfolio_value) if previous_state else portfolio_value
        
        # Calculate return
        if previous_value > 0:
            period_return = (portfolio_value - previous_value) / previous_value
        else:
            period_return = 0
        
        if self.agent_config.agent_type == AgentType.CONSERVATIVE:
            specialized_reward += self._conservative_rewards(current_state, period_return)
            
        elif self.agent_config.agent_type == AgentType.AGGRESSIVE:
            specialized_reward += self._aggressive_rewards(current_state, period_return)
            
        elif self.agent_config.agent_type == AgentType.BALANCED:
            specialized_reward += self._balanced_rewards(current_state, period_return)
            
        elif self.agent_config.agent_type == AgentType.CONTRARIAN:
            specialized_reward += self._contrarian_rewards(current_state, period_return, action_info)
        
        return specialized_reward
    
    def _conservative_rewards(self, state: Dict[str, Any], period_return: float) -> float:
        """Conservative agent specific rewards"""
        reward = 0.0
        weights = self.agent_config.reward_weights
        
        # Reward steady positive returns
        if 0 < period_return < 0.02:  # 0-2% return
            reward += weights.get('consistency', 1.0) * 0.5
        
        # Heavy penalty for large losses
        if period_return < -0.05:  # >5% loss
            reward -= weights.get('drawdown', 2.0) * abs(period_return) * 10
        
        # Reward low volatility
        volatility = state.get('portfolio_volatility', 0)
        if volatility < 0.15:  # Low volatility
            reward += weights.get('volatility', 1.2) * 0.3
        
        # Reward high win rate
        win_rate = state.get('win_rate', 0)
        if win_rate > 0.6:
            reward += weights.get('consistency', 1.0) * (win_rate - 0.6)
        
        # Penalty for large positions
        position_size = abs(state.get('position_size', 0))
        if position_size > 0.5:
            reward -= weights.get('risk', 1.5) * (position_size - 0.5)
        
        return reward
    
    def _aggressive_rewards(self, state: Dict[str, Any], period_return: float) -> float:
        """Aggressive agent specific rewards"""
        reward = 0.0
        weights = self.agent_config.reward_weights
        
        # Reward large positive returns
        if period_return > 0.03:  # >3% return
            reward += weights.get('return', 2.0) * period_return * 5
        
        # Reward momentum following
        market_trend = state.get('market_trend', 0)
        position_size = state.get('position_size', 0)
        
        if market_trend * position_size > 0:  # Following trend
            reward += weights.get('momentum', 1.5) * abs(market_trend) * 0.5
        
        # Reward high volatility environments (more opportunities)
        volatility = state.get('market_volatility', 0)
        if volatility > 0.2:
            reward += weights.get('volatility', 0.2) * volatility
        
        # Less penalty for moderate losses (accept risk)
        if -0.1 < period_return < 0:
            reward -= weights.get('risk', 0.3) * abs(period_return) * 2
        
        return reward
    
    def _balanced_rewards(self, state: Dict[str, Any], period_return: float) -> float:
        """Balanced agent specific rewards"""
        reward = 0.0
        weights = self.agent_config.reward_weights
        
        # Calculate risk-adjusted return (simplified Sharpe)
        returns_std = state.get('returns_std', 0.01)
        if returns_std > 0:
            risk_adjusted_return = period_return / returns_std
            reward += weights.get('sharpe_ratio', 1.5) * risk_adjusted_return
        
        # Reward optimal position sizing
        position_size = abs(state.get('position_size', 0))
        optimal_size = 0.3  # 30% position
        size_deviation = abs(position_size - optimal_size)
        if size_deviation < 0.1:
            reward += weights.get('risk', 1.0) * 0.2
        
        # Balance between return and risk
        if period_return > 0 and state.get('portfolio_volatility', 0) < 0.2:
            reward += weights.get('return', 1.0) * period_return * 3
        
        return reward
    
    def _contrarian_rewards(self, state: Dict[str, Any], period_return: float, action_info: Optional[Dict[str, Any]]) -> float:
        """Contrarian agent specific rewards"""
        reward = 0.0
        weights = self.agent_config.reward_weights
        
        # Reward mean reversion trades
        market_trend = state.get('market_trend', 0)
        position_size = state.get('position_size', 0)
        
        # Reward going against trend (contrarian)
        if market_trend * position_size < 0:  # Against trend
            reward += weights.get('mean_reversion', 1.8) * abs(market_trend) * 0.8
        
        # Reward buying oversold/selling overbought
        rsi = state.get('rsi', 50)
        if action_info:
            action = action_info.get('action', 0)
            if action > 0 and rsi < 30:  # Buy oversold
                reward += weights.get('oversold_bought', 1.5) * (30 - rsi) / 30
            elif action < 0 and rsi > 70:  # Sell overbought
                reward += weights.get('oversold_bought', 1.5) * (rsi - 70) / 30
        
        # Reward profit from mean reversion
        if period_return > 0.02:  # Good profit from contrarian trade
            reward += weights.get('return', 1.0) * period_return * 4
        
        return reward


class SpecializedAgent(ABC):
    """Abstract base class for specialized agents"""
    
    def __init__(self, agent_config: SpecializedAgentConfig):
        self.config = agent_config
        self.agent_type = agent_config.agent_type
        self.name = agent_config.name
        
        # Initialize reward calculator
        self.reward_calculator = SpecializedRewardCalculator(agent_config)
        
        # Initialize PPO agent (will be set up in subclasses)
        self.ppo_agent = None
        
        # Performance tracking
        self.performance_history = []
        self.training_history = []
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.episode_count = 0
        
        logger.info(f"Initialized {self.name} ({self.agent_type.value})")
    
    @abstractmethod
    def setup_agent(self, env):
        """Setup the PPO agent with environment"""
        pass
    
    @abstractmethod
    def get_action_preprocessing(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess observation for agent-specific features"""
        pass
    
    @abstractmethod
    def get_action_postprocessing(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Postprocess action based on agent strategy"""
        pass
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict action with agent-specific processing"""
        if not self.ppo_agent:
            raise ValueError(f"Agent {self.name} not initialized")
        
        # Preprocess observation
        processed_obs = self.get_action_preprocessing(observation)
        
        # Get base action
        action, info = self.ppo_agent.predict(processed_obs, deterministic)
        
        # Postprocess action
        final_action = self.get_action_postprocessing(action, observation)
        
        # Add agent info
        info = info or {}
        info.update({
            'agent_type': self.agent_type.value,
            'agent_name': self.name,
            'processed_observation': processed_obs.tolist() if hasattr(processed_obs, 'tolist') else processed_obs,
            'original_action': action.tolist() if hasattr(action, 'tolist') else action,
            'final_action': final_action.tolist() if hasattr(final_action, 'tolist') else final_action
        })
        
        self.last_action = final_action
        return final_action, info
    
    def train(self, env, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Train the specialized agent"""
        if not self.ppo_agent:
            self.setup_agent(env)
        
        timesteps = total_timesteps or self.config.training_timesteps
        
        # Train with specialized reward function
        training_summary = self.ppo_agent.train(total_timesteps=timesteps, eval_env=env)
        
        # Record training
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'timesteps': timesteps,
            'summary': training_summary
        })
        
        logger.info(f"Training completed for {self.name}: {training_summary.get('best_mean_reward', 'N/A')}")
        return training_summary
    
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        if not self.ppo_agent:
            raise ValueError(f"Agent {self.name} not trained")
        
        metrics = self.ppo_agent.evaluate(env, n_episodes=n_episodes)
        
        # Add agent-specific metrics
        metrics.update({
            'agent_type': self.agent_type.value,
            'agent_name': self.name,
            'evaluation_episodes': n_episodes
        })
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def save_agent(self, path: str):
        """Save agent state and model"""
        # Save PPO model
        if self.ppo_agent and self.ppo_agent.model:
            model_path = f"{path}_{self.agent_type.value}_model.zip"
            self.ppo_agent.model.save(model_path)
        
        # Save agent state
        state_path = f"{path}_{self.agent_type.value}_state.json"
        state = {
            'config': {
                'agent_type': self.agent_type.value,
                'name': self.name,
                'description': self.config.description,
                'reward_weights': self.config.reward_weights,
                'strategy_params': self.config.strategy_params
            },
            'performance_history': self.performance_history,
            'training_history': self.training_history,
            'episode_count': self.episode_count
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Agent {self.name} saved to {path}")
    
    def load_agent(self, path: str, env):
        """Load agent state and model"""
        # Load PPO model
        model_path = f"{path}_{self.agent_type.value}_model.zip"
        if os.path.exists(model_path):
            self.setup_agent(env)
            self.ppo_agent.load_model(model_path)
        
        # Load agent state
        state_path = f"{path}_{self.agent_type.value}_state.json"
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.performance_history = state.get('performance_history', [])
            self.training_history = state.get('training_history', [])
            self.episode_count = state.get('episode_count', 0)
        
        logger.info(f"Agent {self.name} loaded from {path}")


class ConservativeAgent(SpecializedAgent):
    """Conservative trading agent focused on capital preservation"""
    
    def __init__(self):
        config = SpecializedAgentConfig(
            agent_type=AgentType.CONSERVATIVE,
            name="Conservative Protector",
            description="Capital preservation focused agent with strict risk controls",
            max_position_size=0.5,
            max_drawdown_threshold=0.10,
            target_return=0.08,
            target_sharpe=1.2,
            target_win_rate=0.65,
            reward_weights={
                'return': 0.6,
                'risk': 1.5,
                'drawdown': 2.0,
                'volatility': 1.2,
                'consistency': 1.0,
                'transaction_cost': 0.8
            },
            strategy_params={
                'position_sizing': 'kelly_conservative',
                'stop_loss': 0.03,
                'take_profit': 0.06,
                'lookback_window': 30,
                'volatility_threshold': 0.15
            }
        )
        super().__init__(config)
    
    def setup_agent(self, env):
        """Setup PPO agent for conservative trading"""
        ppo_config = PPOConfig(
            policy_type='risk_aware',
            learning_rate=self.config.learning_rate * 0.8,  # Conservative learning
            n_steps=1024,  # Smaller steps for stability
            batch_size=32,
            clip_range=0.15,  # Smaller clip range
            total_timesteps=self.config.training_timesteps,
            eval_freq=self.config.eval_frequency,
            enable_risk_constraints=True,
            max_drawdown_threshold=self.config.max_drawdown_threshold
        )
        
        self.ppo_agent = PPOAgent(env, ppo_config)
    
    def get_action_preprocessing(self, observation: np.ndarray) -> np.ndarray:
        """Add risk-focused features"""
        # Add volatility and drawdown indicators
        processed = observation.copy()
        
        # Calculate recent volatility
        if len(processed) > 20:
            recent_returns = processed[-20:]
            volatility = np.std(recent_returns)
            
            # Add volatility scaling
            processed = np.append(processed, [volatility, volatility**2])
        
        return processed
    
    def get_action_postprocessing(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Apply conservative position sizing and risk controls"""
        processed_action = action.copy()
        
        # Limit position size
        max_position = self.config.max_position_size
        processed_action = np.clip(processed_action, -max_position, max_position)
        
        # Reduce position size during high volatility
        if len(observation) > 10:
            recent_volatility = np.std(observation[-10:])
            if recent_volatility > self.config.strategy_params['volatility_threshold']:
                processed_action *= 0.5  # Halve position during high volatility
        
        return processed_action


class AggressiveAgent(SpecializedAgent):
    """Aggressive trading agent focused on maximum returns"""
    
    def __init__(self):
        config = SpecializedAgentConfig(
            agent_type=AgentType.AGGRESSIVE,
            name="Alpha Hunter",
            description="High-return focused agent with aggressive position sizing",
            max_position_size=2.0,  # Allow leverage
            max_drawdown_threshold=0.25,
            target_return=0.30,
            target_sharpe=0.8,
            target_win_rate=0.45,
            reward_weights={
                'return': 2.0,
                'risk': 0.3,
                'drawdown': 0.5,
                'volatility': 0.2,
                'momentum': 1.5,
                'transaction_cost': 0.3
            },
            strategy_params={
                'position_sizing': 'kelly_aggressive',
                'stop_loss': 0.08,
                'take_profit': 0.15,
                'lookback_window': 10,
                'momentum_threshold': 0.05
            }
        )
        super().__init__(config)
    
    def setup_agent(self, env):
        """Setup PPO agent for aggressive trading"""
        ppo_config = PPOConfig(
            policy_type='attention',
            learning_rate=self.config.learning_rate * 1.2,  # Higher learning rate
            n_steps=2048,  # Larger steps
            batch_size=128,
            clip_range=0.25,  # Larger clip range
            ent_coef=0.01,  # More exploration
            total_timesteps=self.config.training_timesteps,
            eval_freq=self.config.eval_frequency,
            enable_risk_constraints=False,  # Disable constraints
            max_drawdown_threshold=self.config.max_drawdown_threshold
        )
        
        self.ppo_agent = PPOAgent(env, ppo_config)
    
    def get_action_preprocessing(self, observation: np.ndarray) -> np.ndarray:
        """Add momentum and trend features"""
        processed = observation.copy()
        
        # Add momentum indicators
        if len(processed) > 10:
            short_ma = np.mean(processed[-5:])
            long_ma = np.mean(processed[-10:])
            momentum = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            # Add momentum features
            processed = np.append(processed, [momentum, momentum**2, np.sign(momentum)])
        
        return processed
    
    def get_action_postprocessing(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Apply aggressive position sizing and momentum following"""
        processed_action = action.copy()
        
        # Allow larger positions
        max_position = self.config.max_position_size
        processed_action = np.clip(processed_action, -max_position, max_position)
        
        # Amplify positions during strong momentum
        if len(observation) > 5:
            short_momentum = np.mean(observation[-5:]) - np.mean(observation[-10:]) if len(observation) > 10 else 0
            if abs(short_momentum) > self.config.strategy_params['momentum_threshold']:
                processed_action *= 1.5  # Amplify during strong momentum
        
        return processed_action


class BalancedAgent(SpecializedAgent):
    """Balanced trading agent optimizing risk-adjusted returns"""
    
    def __init__(self):
        config = SpecializedAgentConfig(
            agent_type=AgentType.BALANCED,
            name="Sharpe Optimizer",
            description="Risk-adjusted return focused agent optimizing Sharpe ratio",
            max_position_size=1.0,
            max_drawdown_threshold=0.15,
            target_return=0.18,
            target_sharpe=1.5,
            target_win_rate=0.58,
            reward_weights={
                'return': 1.0,
                'risk': 1.0,
                'drawdown': 1.0,
                'volatility': 0.8,
                'sharpe_ratio': 1.5,
                'transaction_cost': 0.6
            },
            strategy_params={
                'position_sizing': 'kelly_balanced',
                'stop_loss': 0.05,
                'take_profit': 0.10,
                'lookback_window': 20,
                'rebalance_frequency': 5
            }
        )
        super().__init__(config)
    
    def setup_agent(self, env):
        """Setup PPO agent for balanced trading"""
        ppo_config = PPOConfig(
            policy_type='attention',
            learning_rate=self.config.learning_rate,
            n_steps=1536,
            batch_size=64,
            clip_range=0.2,
            total_timesteps=self.config.training_timesteps,
            eval_freq=self.config.eval_frequency,
            enable_risk_constraints=True,
            max_drawdown_threshold=self.config.max_drawdown_threshold
        )
        
        self.ppo_agent = PPOAgent(env, ppo_config)
    
    def get_action_preprocessing(self, observation: np.ndarray) -> np.ndarray:
        """Add risk-return balance features"""
        processed = observation.copy()
        
        # Add Sharpe ratio components
        if len(processed) > 20:
            returns = np.diff(processed[-20:]) / processed[-20:-1]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
            
            # Add balance features
            processed = np.append(processed, [mean_return, std_return, sharpe])
        
        return processed
    
    def get_action_postprocessing(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Apply balanced position sizing"""
        processed_action = action.copy()
        
        # Standard position limits
        max_position = self.config.max_position_size
        processed_action = np.clip(processed_action, -max_position, max_position)
        
        # Adjust based on current risk-return profile
        if len(observation) > 20:
            recent_returns = np.diff(observation[-20:]) / observation[-20:-1]
            current_vol = np.std(recent_returns)
            target_vol = 0.15  # 15% target volatility
            
            # Scale position based on volatility
            vol_adjustment = target_vol / max(current_vol, 0.01)
            processed_action *= min(vol_adjustment, 2.0)  # Cap adjustment
        
        return processed_action


class ContrarianAgent(SpecializedAgent):
    """Contrarian trading agent focused on mean reversion"""
    
    def __init__(self):
        config = SpecializedAgentConfig(
            agent_type=AgentType.CONTRARIAN,
            name="Mean Reverter",
            description="Mean reversion focused agent trading against momentum",
            max_position_size=1.2,
            max_drawdown_threshold=0.18,
            target_return=0.20,
            target_sharpe=1.1,
            target_win_rate=0.52,
            reward_weights={
                'return': 1.0,
                'risk': 0.8,
                'mean_reversion': 1.8,
                'oversold_bought': 1.5,
                'volatility': 0.6,
                'transaction_cost': 0.4
            },
            strategy_params={
                'position_sizing': 'contrarian',
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'lookback_window': 15,
                'mean_reversion_period': 10
            }
        )
        super().__init__(config)
    
    def setup_agent(self, env):
        """Setup PPO agent for contrarian trading"""
        ppo_config = PPOConfig(
            policy_type='attention',
            learning_rate=self.config.learning_rate * 0.9,
            n_steps=1024,
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.005,  # Some exploration for contrarian signals
            total_timesteps=self.config.training_timesteps,
            eval_freq=self.config.eval_frequency,
            enable_risk_constraints=True,
            max_drawdown_threshold=self.config.max_drawdown_threshold
        )
        
        self.ppo_agent = PPOAgent(env, ppo_config)
    
    def get_action_preprocessing(self, observation: np.ndarray) -> np.ndarray:
        """Add mean reversion and momentum features"""
        processed = observation.copy()
        
        # Add mean reversion indicators
        if len(processed) > 15:
            # Calculate RSI
            prices = processed[-15:]
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # Distance from mean
            mean_price = np.mean(prices)
            current_price = prices[-1]
            distance_from_mean = (current_price - mean_price) / mean_price if mean_price > 0 else 0
            
            # Add contrarian features
            processed = np.append(processed, [rsi, distance_from_mean, -distance_from_mean])
        
        return processed
    
    def get_action_postprocessing(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Apply contrarian position sizing"""
        processed_action = action.copy()
        
        # Standard position limits
        max_position = self.config.max_position_size
        processed_action = np.clip(processed_action, -max_position, max_position)
        
        # Contrarian adjustments
        if len(observation) > 15:
            # Calculate simple momentum
            short_ma = np.mean(observation[-5:])
            long_ma = np.mean(observation[-15:])
            momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            
            # Reverse the action if strong momentum (contrarian)
            if abs(momentum) > 0.05:  # Strong momentum
                processed_action *= -1  # Go against momentum
                processed_action *= min(abs(momentum) * 10, 1.5)  # Scale by momentum strength
        
        return processed_action


def create_specialized_agent(agent_type: AgentType) -> SpecializedAgent:
    """Factory function to create specialized agents"""
    
    agent_map = {
        AgentType.CONSERVATIVE: ConservativeAgent,
        AgentType.AGGRESSIVE: AggressiveAgent,
        AgentType.BALANCED: BalancedAgent,
        AgentType.CONTRARIAN: ContrarianAgent
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_map[agent_type]()


def create_agent_ensemble() -> Dict[AgentType, SpecializedAgent]:
    """Create a complete ensemble of all specialized agents"""
    
    ensemble = {}
    for agent_type in AgentType:
        ensemble[agent_type] = create_specialized_agent(agent_type)
    
    logger.info("Created complete agent ensemble with all specialized agents")
    return ensemble


if __name__ == "__main__":
    # Example usage and testing
    
    # Create individual agents
    conservative = create_specialized_agent(AgentType.CONSERVATIVE)
    aggressive = create_specialized_agent(AgentType.AGGRESSIVE)
    balanced = create_specialized_agent(AgentType.BALANCED)
    contrarian = create_specialized_agent(AgentType.CONTRARIAN)
    
    print(f"Conservative Agent: {conservative.name}")
    print(f"  Reward weights: {conservative.config.reward_weights}")
    print(f"  Strategy params: {conservative.config.strategy_params}")
    
    print(f"\nAggressive Agent: {aggressive.name}")
    print(f"  Reward weights: {aggressive.config.reward_weights}")
    print(f"  Strategy params: {aggressive.config.strategy_params}")
    
    print(f"\nBalanced Agent: {balanced.name}")
    print(f"  Reward weights: {balanced.config.reward_weights}")
    print(f"  Strategy params: {balanced.config.strategy_params}")
    
    print(f"\nContrarian Agent: {contrarian.name}")
    print(f"  Reward weights: {contrarian.config.reward_weights}")
    print(f"  Strategy params: {contrarian.config.strategy_params}")
    
    # Create full ensemble
    ensemble = create_agent_ensemble()
    print(f"\nCreated ensemble with {len(ensemble)} specialized agents")