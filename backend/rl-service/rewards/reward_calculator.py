"""
Main Reward Calculation System

This module provides the primary interface for calculating rewards in the
RL trading environment, integrating all reward components and optimization strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib

from .reward_components import (
    RewardComponentConfig, MarketRegime,
    create_reward_components
)
from .multi_objective_reward import (
    MultiObjectiveConfig, OptimizationStrategy,
    MultiObjectiveReward
)

logger = logging.getLogger(__name__)


@dataclass
class RewardCalculatorConfig:
    """Main configuration for reward calculator"""
    
    # Component configuration
    component_config: RewardComponentConfig = field(default_factory=RewardComponentConfig)
    
    # Multi-objective configuration
    multi_objective_config: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    
    # Reward shaping parameters
    enable_reward_shaping: bool = True
    shaping_decay_rate: float = 0.99
    shaping_magnitude: float = 0.1
    
    # Sparse reward handling
    sparse_reward_threshold: int = 10  # Steps before sparse reward
    sparse_reward_bonus: float = 0.5
    
    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'stage': 1, 'difficulty': 0.3, 'min_episodes': 100},
        {'stage': 2, 'difficulty': 0.6, 'min_episodes': 500},
        {'stage': 3, 'difficulty': 1.0, 'min_episodes': 1000}
    ])
    
    # Performance targets (SOW aligned)
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        'weekly_return': 0.04,  # 4% weekly
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.15,
        'win_rate': 0.60,
        'profit_factor': 1.5  # Ratio of gross profit to gross loss
    })
    
    # Logging and debugging
    log_frequency: int = 100
    debug_mode: bool = False


class RewardShaper:
    """
    Reward shaping for sparse reward environments
    
    Provides intermediate rewards to guide learning in sparse reward scenarios.
    """
    
    def __init__(self, config: RewardCalculatorConfig):
        self.config = config
        self.potential_history: List[float] = []
        self.shaping_rewards: List[float] = []
        
    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """
        Calculate potential function for current state
        
        The potential function estimates future reward potential.
        """
        potential = 0.0
        
        # Portfolio value component
        portfolio_value = state.get('portfolio_value', 0)
        initial_value = state.get('initial_value', portfolio_value)
        if initial_value > 0:
            value_ratio = portfolio_value / initial_value
            potential += np.tanh(value_ratio - 1.0) * 0.3
        
        # Position quality component
        position_info = state.get('position_info', {})
        if position_info:
            unrealized_pnl = position_info.get('unrealized_pnl', 0)
            position_size = position_info.get('size', 0)
            
            if position_size > 0:
                # Reward profitable positions
                potential += np.tanh(unrealized_pnl / position_size) * 0.2
        
        # Market alignment component
        market_trend = state.get('market_trend', 0)
        position_direction = 1 if position_size > 0 else -1 if position_size < 0 else 0
        
        if position_direction != 0:
            alignment = market_trend * position_direction
            potential += alignment * 0.1
        
        # Risk management component
        current_drawdown = state.get('current_drawdown', 0)
        if current_drawdown < self.config.component_config.drawdown_warning_level:
            potential += 0.1
        else:
            potential -= (current_drawdown - self.config.component_config.drawdown_warning_level) * 2
        
        return potential
    
    def shape_reward(self,
                     base_reward: float,
                     current_state: Dict[str, Any],
                     previous_state: Optional[Dict[str, Any]] = None,
                     episode_step: int = 0) -> float:
        """
        Apply reward shaping to base reward
        
        Args:
            base_reward: Original reward
            current_state: Current state
            previous_state: Previous state
            episode_step: Current step in episode
            
        Returns:
            Shaped reward
        """
        if not self.config.enable_reward_shaping:
            return base_reward
        
        # Calculate potential-based shaping
        current_potential = self.calculate_potential(current_state)
        
        if previous_state is not None:
            previous_potential = self.calculate_potential(previous_state)
            shaping_reward = self.config.shaping_magnitude * (
                current_potential - previous_potential
            )
        else:
            shaping_reward = 0.0
        
        # Apply decay
        decay_factor = self.config.shaping_decay_rate ** episode_step
        shaping_reward *= decay_factor
        
        # Store for analysis
        self.potential_history.append(current_potential)
        self.shaping_rewards.append(shaping_reward)
        
        if len(self.potential_history) > 1000:
            self.potential_history.pop(0)
            self.shaping_rewards.pop(0)
        
        return base_reward + shaping_reward


class SparseRewardHandler:
    """
    Handles sparse reward scenarios
    
    Provides intermediate rewards and bonuses for achieving sub-goals.
    """
    
    def __init__(self, config: RewardCalculatorConfig):
        self.config = config
        self.steps_since_reward = 0
        self.sub_goals_achieved: List[str] = []
        
    def check_sparse_reward(self,
                           base_reward: float,
                           state: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Check and handle sparse reward conditions
        
        Args:
            base_reward: Base reward value
            state: Current state
            
        Returns:
            Tuple of (adjusted_reward, is_sparse)
        """
        is_sparse = abs(base_reward) < 0.001  # Near-zero reward
        
        if is_sparse:
            self.steps_since_reward += 1
        else:
            self.steps_since_reward = 0
        
        adjusted_reward = base_reward
        
        # Check sub-goals
        sub_goal_reward = self._check_sub_goals(state)
        adjusted_reward += sub_goal_reward
        
        # Add exploration bonus in sparse regions
        if self.steps_since_reward > self.config.sparse_reward_threshold:
            exploration_bonus = self.config.sparse_reward_bonus * \
                              (1.0 - np.exp(-self.steps_since_reward / 20))
            adjusted_reward += exploration_bonus
        
        return adjusted_reward, is_sparse
    
    def _check_sub_goals(self, state: Dict[str, Any]) -> float:
        """Check and reward achievement of sub-goals"""
        reward = 0.0
        
        # Sub-goal: First profitable trade
        if 'first_profit' not in self.sub_goals_achieved:
            if state.get('realized_pnl', 0) > 0:
                self.sub_goals_achieved.append('first_profit')
                reward += 0.2
        
        # Sub-goal: Risk management
        if 'risk_managed' not in self.sub_goals_achieved:
            if state.get('current_drawdown', 1.0) < 0.05:  # Less than 5% drawdown
                if state.get('total_trades', 0) > 10:
                    self.sub_goals_achieved.append('risk_managed')
                    reward += 0.3
        
        # Sub-goal: Consistency
        if 'consistent_trading' not in self.sub_goals_achieved:
            win_rate = state.get('win_rate', 0)
            if win_rate > 0.6 and state.get('total_trades', 0) > 20:
                self.sub_goals_achieved.append('consistent_trading')
                reward += 0.4
        
        # Sub-goal: Target achievement
        if 'target_achieved' not in self.sub_goals_achieved:
            weekly_return = state.get('weekly_return', 0)
            if weekly_return >= self.config.performance_targets['weekly_return']:
                self.sub_goals_achieved.append('target_achieved')
                reward += 0.5
        
        return reward
    
    def reset(self):
        """Reset sparse reward tracking"""
        self.steps_since_reward = 0
        self.sub_goals_achieved.clear()


class CurriculumManager:
    """
    Manages curriculum learning for progressive difficulty
    
    Adjusts reward complexity and targets based on agent progress.
    """
    
    def __init__(self, config: RewardCalculatorConfig):
        self.config = config
        self.current_stage = 0
        self.episodes_completed = 0
        self.stage_performance: List[float] = []
        
    def update(self, episode_performance: float):
        """Update curriculum based on performance"""
        self.episodes_completed += 1
        self.stage_performance.append(episode_performance)
        
        if not self.config.enable_curriculum:
            return
        
        # Check for stage progression
        if self.current_stage < len(self.config.curriculum_stages) - 1:
            current_config = self.config.curriculum_stages[self.current_stage]
            
            if self.episodes_completed >= current_config['min_episodes']:
                # Check performance threshold
                if len(self.stage_performance) >= 10:
                    avg_performance = np.mean(self.stage_performance[-10:])
                    
                    if avg_performance > 0.6:  # 60% success rate
                        self.current_stage += 1
                        self.stage_performance.clear()
                        logger.info(f"Progressed to curriculum stage {self.current_stage + 1}")
    
    def get_difficulty_multiplier(self) -> float:
        """Get current difficulty multiplier"""
        if not self.config.enable_curriculum:
            return 1.0
        
        if self.current_stage < len(self.config.curriculum_stages):
            return self.config.curriculum_stages[self.current_stage]['difficulty']
        
        return 1.0
    
    def adjust_targets(self, base_targets: Dict[str, float]) -> Dict[str, float]:
        """Adjust performance targets based on curriculum stage"""
        multiplier = self.get_difficulty_multiplier()
        
        adjusted_targets = {}
        for key, value in base_targets.items():
            if key == 'max_drawdown':
                # Relax drawdown constraint in early stages
                adjusted_targets[key] = value * (2 - multiplier)
            elif key == 'win_rate':
                # Lower win rate requirement in early stages
                adjusted_targets[key] = value * multiplier
            else:
                # Scale other targets
                adjusted_targets[key] = value * multiplier
        
        return adjusted_targets


class RewardCalculator:
    """
    Main reward calculator integrating all components
    
    This is the primary interface for calculating rewards in the RL environment.
    """
    
    def __init__(self, config: Optional[RewardCalculatorConfig] = None):
        self.config = config or RewardCalculatorConfig()
        
        # Initialize multi-objective reward system
        self.multi_objective = MultiObjectiveReward(
            self.config.multi_objective_config,
            self.config.component_config
        )
        
        # Initialize auxiliary systems
        self.reward_shaper = RewardShaper(self.config)
        self.sparse_handler = SparseRewardHandler(self.config)
        self.curriculum_manager = CurriculumManager(self.config)
        
        # Tracking
        self.episode_rewards: List[float] = []
        self.total_rewards: List[float] = []
        self.previous_state: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.episode_count = 0
        
        logger.info("RewardCalculator initialized with multi-objective optimization")
    
    def calculate(self,
                 current_state: Dict[str, Any],
                 previous_state: Optional[Dict[str, Any]] = None,
                 action_info: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward for current state transition
        
        Args:
            current_state: Current environment state
            previous_state: Previous environment state
            action_info: Information about action taken
            
        Returns:
            Tuple of (reward, info_dict)
        """
        try:
            self.step_count += 1
            
            # Prepare state information
            if previous_state is None:
                previous_state = self.previous_state or current_state
            
            # Extract key values
            current_equity = current_state.get('portfolio_value', 0)
            previous_equity = previous_state.get('portfolio_value', current_equity)
            
            # Prepare portfolio state
            portfolio_state = {
                'initial_equity': current_state.get('initial_equity', previous_equity),
                'returns_history': current_state.get('returns_history', []),
                'last_trade_result': current_state.get('last_trade_result'),
                'current_drawdown': current_state.get('current_drawdown', 0),
                'win_rate': current_state.get('win_rate', 0),
                'total_trades': current_state.get('total_trades', 0)
            }
            
            # Prepare market conditions
            market_conditions = {
                'regime': current_state.get('market_regime', 'SIDEWAYS'),
                'volatility': current_state.get('volatility', 0.02),
                'trend': current_state.get('trend', 0),
                'market_impact': current_state.get('market_impact', 0)
            }
            
            # Prepare action info
            if action_info is None:
                action_info = {}
            
            action_info.update({
                'episode_step': self.step_count,
                'max_steps': current_state.get('max_steps', 1000),
                'state_hash': self._hash_state(current_state)
            })
            
            # Calculate multi-objective reward
            base_reward, objectives = self.multi_objective.calculate_reward(
                current_equity,
                previous_equity,
                portfolio_state,
                market_conditions,
                action_info
            )
            
            # Apply curriculum learning adjustments
            difficulty = self.curriculum_manager.get_difficulty_multiplier()
            base_reward *= difficulty
            
            # Handle sparse rewards
            sparse_adjusted, is_sparse = self.sparse_handler.check_sparse_reward(
                base_reward, current_state
            )
            
            # Apply reward shaping
            shaped_reward = self.reward_shaper.shape_reward(
                sparse_adjusted,
                current_state,
                previous_state,
                self.step_count
            )
            
            # Final reward
            final_reward = shaped_reward
            
            # Track rewards
            self.episode_rewards.append(final_reward)
            self.total_rewards.append(final_reward)
            
            # Prepare info dictionary
            info = {
                'base_reward': base_reward,
                'shaped_reward': shaped_reward,
                'final_reward': final_reward,
                'objectives': objectives,
                'is_sparse': is_sparse,
                'curriculum_stage': self.curriculum_manager.current_stage,
                'difficulty': difficulty,
                'step': self.step_count,
                'episode': self.episode_count
            }
            
            # Add performance metrics
            if len(self.episode_rewards) > 0:
                info['episode_return'] = sum(self.episode_rewards)
                info['avg_reward'] = np.mean(self.episode_rewards)
                info['reward_std'] = np.std(self.episode_rewards)
            
            # Log if needed
            if self.step_count % self.config.log_frequency == 0:
                self._log_metrics(info)
            
            # Store state for next calculation
            self.previous_state = current_state.copy()
            
            return final_reward, info
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0, {'error': str(e)}
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1
        
        # Update curriculum
        if len(self.episode_rewards) > 0:
            episode_return = sum(self.episode_rewards)
            self.curriculum_manager.update(episode_return)
        
        # Reset tracking
        self.episode_rewards.clear()
        self.step_count = 0
        self.previous_state = None
        
        # Reset components
        self.multi_objective.reset_episode()
        self.sparse_handler.reset()
        
        logger.debug(f"Episode {self.episode_count} reset")
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash of state for tracking"""
        # Select key features for hashing
        key_features = [
            state.get('portfolio_value', 0),
            state.get('position_size', 0),
            state.get('market_regime', ''),
            state.get('current_price', 0)
        ]
        
        state_str = '_'.join(str(f) for f in key_features)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]
    
    def _log_metrics(self, info: Dict[str, Any]):
        """Log metrics for debugging"""
        if self.config.debug_mode:
            logger.info(f"Step {self.step_count}: Reward={info['final_reward']:.4f}")
            logger.info(f"  Objectives: {info['objectives']}")
            logger.info(f"  Curriculum: Stage {info['curriculum_stage']}, Difficulty {info['difficulty']:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'current_episode_return': sum(self.episode_rewards),
            'avg_reward_all_time': np.mean(self.total_rewards) if self.total_rewards else 0,
            'curriculum_stage': self.curriculum_manager.current_stage,
            'sparse_steps': self.sparse_handler.steps_since_reward,
            'sub_goals_achieved': len(self.sparse_handler.sub_goals_achieved)
        }
        
        # Add multi-objective metrics
        metrics.update(self.multi_objective.get_metrics())
        
        # Add shaping metrics
        if self.reward_shaper.potential_history:
            metrics['avg_potential'] = np.mean(self.reward_shaper.potential_history)
            metrics['avg_shaping_reward'] = np.mean(self.reward_shaper.shaping_rewards)
        
        return metrics
    
    def save_state(self, filepath: str):
        """Save calculator state"""
        import json
        
        state = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'curriculum_stage': self.curriculum_manager.current_stage,
            'episodes_completed': self.curriculum_manager.episodes_completed,
            'sub_goals': self.sparse_handler.sub_goals_achieved
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save multi-objective state
        mo_filepath = filepath.replace('.json', '_mo.json')
        self.multi_objective.save_state(mo_filepath)
    
    def load_state(self, filepath: str):
        """Load calculator state"""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.step_count = state['step_count']
        self.episode_count = state['episode_count']
        self.curriculum_manager.current_stage = state['curriculum_stage']
        self.curriculum_manager.episodes_completed = state['episodes_completed']
        self.sparse_handler.sub_goals_achieved = state['sub_goals']
        
        # Load multi-objective state
        mo_filepath = filepath.replace('.json', '_mo.json')
        self.multi_objective.load_state(mo_filepath)