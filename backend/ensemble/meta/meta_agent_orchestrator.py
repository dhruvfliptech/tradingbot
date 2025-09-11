"""
Meta-Agent Orchestrator for Multi-Agent Ensemble System

This module implements the central orchestrator that coordinates multiple specialized
trading agents based on market conditions, performance, and strategic objectives.

The orchestrator:
- Manages the ensemble of specialized agents
- Selects appropriate agents based on market regime
- Coordinates agent decisions and weights
- Tracks ensemble performance
- Provides explainable decisions
- Handles agent lifecycle and updates

Key features:
- Dynamic agent selection based on market conditions
- Performance-weighted ensemble decisions
- Risk management and position sizing
- Real-time regime adaptation
- Explainability and decision transparency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from abc import ABC, abstractmethod
import threading
import time
from collections import defaultdict, deque

# Import from other ensemble modules
from ..agents.specialized_agents import (
    SpecializedAgent, AgentType, create_specialized_agent, create_agent_ensemble
)
from ..regime.market_regime_detector import (
    MarketRegimeDetector, MarketRegime, RegimeDetectionConfig, get_optimal_agents_for_regime
)

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    """Strategies for combining agent decisions"""
    SINGLE_BEST = "single_best"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    MAJORITY_VOTE = "majority_vote"
    REGIME_BASED = "regime_based"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    ADAPTIVE_WEIGHT = "adaptive_weight"


class RiskLevel(Enum):
    """Risk levels for position management"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class OrchestratorConfig:
    """Configuration for meta-agent orchestrator"""
    
    # Agent management
    enabled_agents: List[AgentType] = field(default_factory=lambda: list(AgentType))
    max_active_agents: int = 3
    min_agent_confidence: float = 0.3
    
    # Decision strategy
    decision_strategy: DecisionStrategy = DecisionStrategy.ADAPTIVE_WEIGHT
    fallback_strategy: DecisionStrategy = DecisionStrategy.REGIME_BASED
    
    # Performance tracking
    performance_window: int = 100  # Steps to track performance
    performance_update_frequency: int = 10
    agent_ranking_frequency: int = 50
    
    # Risk management
    risk_level: RiskLevel = RiskLevel.ADAPTIVE
    max_total_position: float = 1.0
    position_scaling_factor: float = 0.8
    emergency_stop_loss: float = 0.10  # 10% stop loss
    
    # Regime adaptation
    regime_update_frequency: int = 5
    regime_confidence_threshold: float = 0.6
    regime_persistence_required: int = 3
    
    # Ensemble weights
    initial_agent_weights: Dict[str, float] = field(default_factory=dict)
    weight_decay_rate: float = 0.95
    weight_learning_rate: float = 0.1
    min_agent_weight: float = 0.05
    
    # Explainability
    enable_decision_logging: bool = True
    log_frequency: int = 20
    save_decision_history: bool = True
    
    # Performance thresholds
    agent_replacement_threshold: float = -0.15  # Replace if 15% underperformance
    ensemble_rebalance_threshold: float = 0.05
    
    def __post_init__(self):
        if not self.enabled_agents:
            self.enabled_agents = list(AgentType)
        
        if not self.initial_agent_weights:
            # Equal weights initially
            weight = 1.0 / len(self.enabled_agents)
            self.initial_agent_weights = {agent.value: weight for agent in self.enabled_agents}


@dataclass
class AgentDecision:
    """Individual agent decision"""
    agent_type: AgentType
    agent_name: str
    action: np.ndarray
    confidence: float
    reasoning: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_type': self.agent_type.value,
            'agent_name': self.agent_name,
            'action': self.action.tolist() if hasattr(self.action, 'tolist') else self.action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class EnsembleDecision:
    """Final ensemble decision"""
    final_action: np.ndarray
    agent_decisions: List[AgentDecision]
    agent_weights: Dict[str, float]
    regime: MarketRegime
    regime_confidence: float
    strategy_used: DecisionStrategy
    risk_level: RiskLevel
    explanation: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_action': self.final_action.tolist() if hasattr(self.final_action, 'tolist') else self.final_action,
            'agent_decisions': [d.to_dict() for d in self.agent_decisions],
            'agent_weights': self.agent_weights,
            'regime': self.regime.value,
            'regime_confidence': self.regime_confidence,
            'strategy_used': self.strategy_used.value,
            'risk_level': self.risk_level.value,
            'explanation': self.explanation,
            'timestamp': self.timestamp.isoformat()
        }


class AgentPerformanceTracker:
    """Track individual agent performance within ensemble"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_data = defaultdict(lambda: {
            'returns': deque(maxlen=window_size),
            'decisions': deque(maxlen=window_size),
            'accuracy': deque(maxlen=window_size),
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'last_update': datetime.now()
        })
    
    def update_performance(self, agent_type: AgentType, decision: AgentDecision, 
                          actual_return: float, is_correct: bool):
        """Update agent performance metrics"""
        
        data = self.performance_data[agent_type.value]
        
        # Update metrics
        data['returns'].append(actual_return)
        data['decisions'].append(decision)
        data['accuracy'].append(is_correct)
        data['last_update'] = datetime.now()
        
        # Calculate derived metrics
        if len(data['returns']) > 10:
            returns = list(data['returns'])
            data['total_return'] = sum(returns)
            data['sharpe_ratio'] = self._calculate_sharpe(returns)
            data['win_rate'] = sum(data['accuracy']) / len(data['accuracy'])
            data['max_drawdown'] = self._calculate_max_drawdown(returns)
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / np.maximum(running_max, 1)
        return float(np.min(drawdowns))
    
    def get_agent_ranking(self) -> List[Tuple[AgentType, float]]:
        """Get agents ranked by performance score"""
        
        rankings = []
        
        for agent_str, data in self.performance_data.items():
            try:
                agent_type = AgentType(agent_str)
                
                # Calculate composite score
                score = (
                    data['total_return'] * 0.3 +
                    data['sharpe_ratio'] * 0.3 +
                    data['win_rate'] * 0.2 +
                    (1 + data['max_drawdown']) * 0.2  # Invert drawdown (less negative is better)
                )
                
                rankings.append((agent_type, score))
                
            except ValueError:
                continue
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_agent_metrics(self, agent_type: AgentType) -> Dict[str, float]:
        """Get performance metrics for specific agent"""
        
        data = self.performance_data.get(agent_type.value, {})
        
        return {
            'total_return': data.get('total_return', 0.0),
            'sharpe_ratio': data.get('sharpe_ratio', 0.0),
            'win_rate': data.get('win_rate', 0.0),
            'max_drawdown': data.get('max_drawdown', 0.0),
            'num_decisions': len(data.get('returns', [])),
            'last_update': data.get('last_update', datetime.now()).isoformat()
        }


class WeightManager:
    """Manage dynamic agent weights based on performance and regime"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.weights = config.initial_agent_weights.copy()
        self.weight_history = []
        self.last_update = datetime.now()
    
    def update_weights(self, performance_tracker: AgentPerformanceTracker,
                      current_regime: MarketRegime, regime_confidence: float):
        """Update agent weights based on performance and regime"""
        
        # Get performance rankings
        rankings = performance_tracker.get_agent_ranking()
        
        # Base weights on performance
        performance_weights = {}
        total_score = sum(max(score, 0) for _, score in rankings)
        
        if total_score > 0:
            for agent_type, score in rankings:
                normalized_score = max(score, 0) / total_score
                performance_weights[agent_type.value] = normalized_score
        else:
            # Equal weights if no positive performance
            for agent_type, _ in rankings:
                performance_weights[agent_type.value] = 1.0 / len(rankings)
        
        # Adjust weights based on regime
        regime_weights = self._get_regime_weights(current_regime, regime_confidence)
        
        # Combine performance and regime weights
        new_weights = {}
        for agent_str in self.weights.keys():
            perf_weight = performance_weights.get(agent_str, 0.0)
            regime_weight = regime_weights.get(agent_str, 0.0)
            
            # Weighted combination
            combined_weight = (
                perf_weight * (1 - regime_confidence) + 
                regime_weight * regime_confidence
            )
            
            new_weights[agent_str] = combined_weight
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for agent_str in new_weights:
                new_weights[agent_str] /= total_weight
                # Apply minimum weight
                new_weights[agent_str] = max(new_weights[agent_str], self.config.min_agent_weight)
        
        # Apply learning rate for smooth updates
        for agent_str in self.weights:
            if agent_str in new_weights:
                self.weights[agent_str] = (
                    self.weights[agent_str] * (1 - self.config.weight_learning_rate) +
                    new_weights[agent_str] * self.config.weight_learning_rate
                )
        
        # Record weight history
        self.weight_history.append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights.copy(),
            'regime': current_regime.value,
            'regime_confidence': regime_confidence
        })
        
        # Keep history manageable
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-500:]
        
        self.last_update = datetime.now()
    
    def _get_regime_weights(self, regime: MarketRegime, confidence: float) -> Dict[str, float]:
        """Get regime-based weights for agents"""
        
        # Get optimal agents for regime
        optimal_agents = get_optimal_agents_for_regime(regime)
        
        # Create weights
        regime_weights = {}
        
        # Higher weight for optimal agents
        optimal_weight = 0.8 / len(optimal_agents) if optimal_agents else 0.0
        other_weight = 0.2 / max(len(AgentType) - len(optimal_agents), 1)
        
        for agent_type in AgentType:
            if agent_type.value in optimal_agents:
                regime_weights[agent_type.value] = optimal_weight
            else:
                regime_weights[agent_type.value] = other_weight
        
        return regime_weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current agent weights"""
        return self.weights.copy()
    
    def get_weight_history(self) -> List[Dict[str, Any]]:
        """Get weight update history"""
        return self.weight_history.copy()


class DecisionCombiner:
    """Combine multiple agent decisions into final ensemble decision"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
    
    def combine_decisions(self, agent_decisions: List[AgentDecision],
                         agent_weights: Dict[str, float],
                         strategy: DecisionStrategy) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combine agent decisions using specified strategy
        
        Returns:
            Tuple of (final_action, explanation)
        """
        
        if not agent_decisions:
            return np.array([0.0]), {'error': 'No agent decisions available'}
        
        if strategy == DecisionStrategy.SINGLE_BEST:
            return self._single_best_decision(agent_decisions)
        
        elif strategy == DecisionStrategy.WEIGHTED_ENSEMBLE:
            return self._weighted_ensemble_decision(agent_decisions, agent_weights)
        
        elif strategy == DecisionStrategy.MAJORITY_VOTE:
            return self._majority_vote_decision(agent_decisions)
        
        elif strategy == DecisionStrategy.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_decision(agent_decisions)
        
        elif strategy == DecisionStrategy.ADAPTIVE_WEIGHT:
            return self._adaptive_weight_decision(agent_decisions, agent_weights)
        
        else:
            # Default to weighted ensemble
            return self._weighted_ensemble_decision(agent_decisions, agent_weights)
    
    def _single_best_decision(self, decisions: List[AgentDecision]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Use decision from highest confidence agent"""
        
        best_decision = max(decisions, key=lambda d: d.confidence)
        
        explanation = {
            'strategy': 'single_best',
            'selected_agent': best_decision.agent_type.value,
            'confidence': best_decision.confidence,
            'reasoning': 'Selected agent with highest confidence'
        }
        
        return best_decision.action, explanation
    
    def _weighted_ensemble_decision(self, decisions: List[AgentDecision],
                                  weights: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Weight decisions by agent weights"""
        
        if not decisions:
            return np.array([0.0]), {'error': 'No decisions to combine'}
        
        # Initialize action with zeros
        action_shape = decisions[0].action.shape if hasattr(decisions[0].action, 'shape') else (1,)
        final_action = np.zeros(action_shape)
        total_weight = 0.0
        used_weights = {}
        
        for decision in decisions:
            agent_weight = weights.get(decision.agent_type.value, 0.0)
            if agent_weight > 0:
                final_action += decision.action * agent_weight
                total_weight += agent_weight
                used_weights[decision.agent_type.value] = agent_weight
        
        if total_weight > 0:
            final_action /= total_weight
        
        explanation = {
            'strategy': 'weighted_ensemble',
            'used_weights': used_weights,
            'total_weight': total_weight,
            'num_agents': len(decisions)
        }
        
        return final_action, explanation
    
    def _majority_vote_decision(self, decisions: List[AgentDecision]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Use majority vote for discrete actions"""
        
        # Convert actions to discrete votes (buy/sell/hold)
        votes = []
        for decision in decisions:
            action_value = float(decision.action[0]) if hasattr(decision.action, '__getitem__') else float(decision.action)
            
            if action_value > 0.1:
                votes.append(1)  # Buy
            elif action_value < -0.1:
                votes.append(-1)  # Sell
            else:
                votes.append(0)  # Hold
        
        # Count votes
        vote_counts = {-1: 0, 0: 0, 1: 0}
        for vote in votes:
            vote_counts[vote] += 1
        
        # Get majority vote
        majority_vote = max(vote_counts, key=vote_counts.get)
        final_action = np.array([float(majority_vote)])
        
        explanation = {
            'strategy': 'majority_vote',
            'vote_counts': vote_counts,
            'majority_vote': majority_vote,
            'total_votes': len(votes)
        }
        
        return final_action, explanation
    
    def _performance_weighted_decision(self, decisions: List[AgentDecision]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Weight by recent agent performance (confidence as proxy)"""
        
        if not decisions:
            return np.array([0.0]), {'error': 'No decisions to combine'}
        
        # Use confidence as performance proxy
        total_confidence = sum(d.confidence for d in decisions)
        
        if total_confidence == 0:
            # Equal weights if no confidence
            weight = 1.0 / len(decisions)
            final_action = sum(d.action for d in decisions) * weight
        else:
            final_action = sum(d.action * (d.confidence / total_confidence) for d in decisions)
        
        explanation = {
            'strategy': 'performance_weighted',
            'total_confidence': total_confidence,
            'agent_confidences': {d.agent_type.value: d.confidence for d in decisions}
        }
        
        return final_action, explanation
    
    def _adaptive_weight_decision(self, decisions: List[AgentDecision],
                                weights: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adaptive weighting combining confidence and performance"""
        
        if not decisions:
            return np.array([0.0]), {'error': 'No decisions to combine'}
        
        action_shape = decisions[0].action.shape if hasattr(decisions[0].action, 'shape') else (1,)
        final_action = np.zeros(action_shape)
        total_adaptive_weight = 0.0
        adaptive_weights = {}
        
        for decision in decisions:
            base_weight = weights.get(decision.agent_type.value, 0.0)
            confidence_boost = decision.confidence
            
            # Adaptive weight combines base weight with confidence
            adaptive_weight = base_weight * (0.7 + 0.3 * confidence_boost)
            
            final_action += decision.action * adaptive_weight
            total_adaptive_weight += adaptive_weight
            adaptive_weights[decision.agent_type.value] = adaptive_weight
        
        if total_adaptive_weight > 0:
            final_action /= total_adaptive_weight
        
        explanation = {
            'strategy': 'adaptive_weight',
            'adaptive_weights': adaptive_weights,
            'total_weight': total_adaptive_weight
        }
        
        return final_action, explanation


class RiskManager:
    """Manage ensemble risk and position sizing"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.current_positions = {}
        self.risk_metrics = {}
        self.emergency_stop_triggered = False
    
    def apply_risk_management(self, action: np.ndarray, current_state: Dict[str, Any],
                            risk_level: RiskLevel) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply risk management to ensemble action"""
        
        # Get current portfolio state
        portfolio_value = current_state.get('portfolio_value', 0)
        current_drawdown = current_state.get('current_drawdown', 0)
        position_size = current_state.get('position_size', 0)
        
        # Check emergency stop
        if current_drawdown > self.config.emergency_stop_loss:
            if not self.emergency_stop_triggered:
                logger.warning(f"Emergency stop triggered: drawdown {current_drawdown:.2%}")
                self.emergency_stop_triggered = True
            
            # Force close positions
            risk_adjusted_action = np.array([0.0]) if position_size != 0 else action * 0.1
            
            explanation = {
                'emergency_stop': True,
                'drawdown': current_drawdown,
                'original_action': action.tolist() if hasattr(action, 'tolist') else action,
                'adjusted_action': risk_adjusted_action.tolist() if hasattr(risk_adjusted_action, 'tolist') else risk_adjusted_action
            }
            
            return risk_adjusted_action, explanation
        
        # Reset emergency stop if drawdown recovers
        if current_drawdown < self.config.emergency_stop_loss * 0.5:
            self.emergency_stop_triggered = False
        
        # Apply position sizing based on risk level
        position_scalar = self._get_position_scalar(risk_level, current_state)
        risk_adjusted_action = action * position_scalar * self.config.position_scaling_factor
        
        # Ensure total position doesn't exceed limit
        projected_position = abs(position_size + float(risk_adjusted_action[0])) if len(risk_adjusted_action) > 0 else abs(position_size)
        
        if projected_position > self.config.max_total_position:
            reduction_factor = self.config.max_total_position / max(projected_position, 0.001)
            risk_adjusted_action *= reduction_factor
        
        explanation = {
            'risk_level': risk_level.value,
            'position_scalar': position_scalar,
            'drawdown': current_drawdown,
            'emergency_stop': False,
            'position_limit_applied': projected_position > self.config.max_total_position
        }
        
        return risk_adjusted_action, explanation
    
    def _get_position_scalar(self, risk_level: RiskLevel, state: Dict[str, Any]) -> float:
        """Get position sizing scalar based on risk level"""
        
        volatility = state.get('volatility', 0.02)
        
        if risk_level == RiskLevel.CONSERVATIVE:
            # Reduce size with volatility
            return max(0.3, 1.0 - volatility * 3)
        
        elif risk_level == RiskLevel.MODERATE:
            return max(0.5, 1.0 - volatility * 2)
        
        elif risk_level == RiskLevel.AGGRESSIVE:
            # Increase size with volatility (up to a point)
            return min(1.5, 1.0 + volatility)
        
        elif risk_level == RiskLevel.ADAPTIVE:
            # Adaptive based on market conditions
            drawdown = state.get('current_drawdown', 0)
            sharpe = state.get('sharpe_ratio', 0)
            
            # Reduce size if drawdown is high or Sharpe is poor
            risk_scalar = 1.0
            if drawdown > 0.05:
                risk_scalar *= (1 - drawdown * 2)
            if sharpe < 0.5:
                risk_scalar *= 0.7
            
            return max(0.2, min(1.2, risk_scalar))
        
        return 1.0


class MetaAgentOrchestrator:
    """
    Main orchestrator class that coordinates the ensemble of specialized agents
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.performance_tracker = AgentPerformanceTracker(self.config.performance_window)
        self.weight_manager = WeightManager(self.config)
        self.decision_combiner = DecisionCombiner(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Initialize agent ensemble
        self.agents: Dict[AgentType, SpecializedAgent] = {}
        self.active_agents: List[AgentType] = []
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.decision_history = []
        self.step_count = 0
        self.last_regime_update = 0
        self.last_weight_update = 0
        
        # Performance metrics
        self.ensemble_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }
        
        logger.info("Meta-agent orchestrator initialized")
    
    def initialize_agents(self, env):
        """Initialize all specialized agents"""
        
        logger.info("Initializing specialized agents...")
        
        for agent_type in self.config.enabled_agents:
            try:
                agent = create_specialized_agent(agent_type)
                agent.setup_agent(env)
                self.agents[agent_type] = agent
                
                logger.info(f"Initialized {agent_type.value} agent: {agent.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {agent_type.value} agent: {e}")
        
        # Set initial active agents
        self.active_agents = list(self.agents.keys())[:self.config.max_active_agents]
        
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents, {len(self.active_agents)} active")
    
    def train_agents(self, env, timesteps_per_agent: int = 50000):
        """Train all agents in the ensemble"""
        
        logger.info("Training ensemble agents...")
        
        training_results = {}
        
        for agent_type, agent in self.agents.items():
            try:
                logger.info(f"Training {agent_type.value} agent...")
                
                # Train agent
                result = agent.train(env, total_timesteps=timesteps_per_agent)
                training_results[agent_type.value] = result
                
                logger.info(f"Completed training {agent_type.value}: best reward {result.get('best_mean_reward', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Failed to train {agent_type.value} agent: {e}")
                training_results[agent_type.value] = {'error': str(e)}
        
        logger.info("Ensemble training completed")
        return training_results
    
    def predict(self, observation: np.ndarray, market_data: Dict[str, Any],
               deterministic: bool = True) -> Tuple[np.ndarray, EnsembleDecision]:
        """
        Generate ensemble prediction
        
        Args:
            observation: Environment observation
            market_data: Additional market data for regime detection
            deterministic: Whether to use deterministic policies
            
        Returns:
            Tuple of (final_action, ensemble_decision)
        """
        
        self.step_count += 1
        
        try:
            # Update market regime
            if self.step_count - self.last_regime_update >= self.config.regime_update_frequency:
                self._update_market_regime(market_data)
                self.last_regime_update = self.step_count
            
            # Update agent weights
            if self.step_count - self.last_weight_update >= self.config.performance_update_frequency:
                self._update_agent_weights()
                self.last_weight_update = self.step_count
            
            # Get active agents for current regime
            active_agents = self._select_active_agents()
            
            # Get decisions from active agents
            agent_decisions = []
            for agent_type in active_agents:
                if agent_type in self.agents:
                    decision = self._get_agent_decision(agent_type, observation, deterministic)
                    if decision:
                        agent_decisions.append(decision)
            
            # Combine decisions
            strategy = self._select_decision_strategy()
            current_weights = self.weight_manager.get_current_weights()
            
            combined_action, combination_explanation = self.decision_combiner.combine_decisions(
                agent_decisions, current_weights, strategy
            )
            
            # Apply risk management
            risk_level = self._determine_risk_level()
            final_action, risk_explanation = self.risk_manager.apply_risk_management(
                combined_action, market_data, risk_level
            )
            
            # Create ensemble decision
            ensemble_decision = EnsembleDecision(
                final_action=final_action,
                agent_decisions=agent_decisions,
                agent_weights=current_weights,
                regime=self.current_regime,
                regime_confidence=getattr(self, 'regime_confidence', 0.5),
                strategy_used=strategy,
                risk_level=risk_level,
                explanation={
                    'combination': combination_explanation,
                    'risk_management': risk_explanation,
                    'active_agents': [a.value for a in active_agents],
                    'step': self.step_count
                },
                timestamp=datetime.now()
            )
            
            # Record decision
            if self.config.save_decision_history:
                self.decision_history.append(ensemble_decision)
                
                # Keep history manageable
                if len(self.decision_history) > 1000:
                    self.decision_history = self.decision_history[-500:]
            
            # Log decision
            if self.config.enable_decision_logging and self.step_count % self.config.log_frequency == 0:
                self._log_decision(ensemble_decision)
            
            return final_action, ensemble_decision
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            
            # Return safe default action
            default_action = np.array([0.0])
            default_decision = EnsembleDecision(
                final_action=default_action,
                agent_decisions=[],
                agent_weights={},
                regime=MarketRegime.UNKNOWN,
                regime_confidence=0.0,
                strategy_used=DecisionStrategy.SINGLE_BEST,
                risk_level=RiskLevel.CONSERVATIVE,
                explanation={'error': str(e)},
                timestamp=datetime.now()
            )
            
            return default_action, default_decision
    
    def _update_market_regime(self, market_data: Dict[str, Any]):
        """Update market regime detection"""
        
        # Extract price data for regime detection
        prices = market_data.get('price_history', np.array([]))
        volumes = market_data.get('volume_history', None)
        
        if len(prices) > 20:  # Need sufficient data
            regime, confidence, info = self.regime_detector.detect_regime(prices, volumes)
            
            if confidence >= self.config.regime_confidence_threshold:
                self.current_regime = regime
                self.regime_confidence = confidence
                
                if self.config.enable_decision_logging:
                    logger.info(f"Regime updated: {regime.value} (confidence: {confidence:.2f})")
    
    def _update_agent_weights(self):
        """Update agent weights based on performance"""
        
        self.weight_manager.update_weights(
            self.performance_tracker,
            self.current_regime,
            getattr(self, 'regime_confidence', 0.5)
        )
    
    def _select_active_agents(self) -> List[AgentType]:
        """Select active agents based on regime and performance"""
        
        # Get optimal agents for current regime
        optimal_agents = get_optimal_agents_for_regime(self.current_regime)
        
        # Convert to AgentType
        regime_agents = []
        for agent_str in optimal_agents:
            try:
                agent_type = AgentType(agent_str)
                if agent_type in self.agents:
                    regime_agents.append(agent_type)
            except ValueError:
                continue
        
        # Get top performing agents
        rankings = self.performance_tracker.get_agent_ranking()
        top_performers = [agent_type for agent_type, _ in rankings[:self.config.max_active_agents]]
        
        # Combine regime-optimal and top-performing agents
        active_agents = list(set(regime_agents + top_performers))
        
        # Limit to max active agents
        active_agents = active_agents[:self.config.max_active_agents]
        
        # Ensure at least one agent is active
        if not active_agents and self.agents:
            active_agents = [list(self.agents.keys())[0]]
        
        return active_agents
    
    def _get_agent_decision(self, agent_type: AgentType, observation: np.ndarray,
                          deterministic: bool) -> Optional[AgentDecision]:
        """Get decision from specific agent"""
        
        try:
            agent = self.agents[agent_type]
            action, info = agent.predict(observation, deterministic)
            
            # Extract confidence from info
            confidence = info.get('confidence', 0.5)
            if isinstance(confidence, (list, np.ndarray)):
                confidence = float(confidence[0]) if len(confidence) > 0 else 0.5
            
            decision = AgentDecision(
                agent_type=agent_type,
                agent_name=agent.name,
                action=action,
                confidence=float(confidence),
                reasoning=info,
                timestamp=datetime.now()
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting decision from {agent_type.value}: {e}")
            return None
    
    def _select_decision_strategy(self) -> DecisionStrategy:
        """Select appropriate decision strategy"""
        
        # Use adaptive strategy based on regime confidence
        regime_confidence = getattr(self, 'regime_confidence', 0.5)
        
        if regime_confidence > 0.8:
            return DecisionStrategy.REGIME_BASED
        elif regime_confidence > 0.6:
            return DecisionStrategy.ADAPTIVE_WEIGHT
        else:
            return self.config.decision_strategy
    
    def _determine_risk_level(self) -> RiskLevel:
        """Determine appropriate risk level"""
        
        if self.config.risk_level != RiskLevel.ADAPTIVE:
            return self.config.risk_level
        
        # Adaptive risk based on regime and performance
        regime_confidence = getattr(self, 'regime_confidence', 0.5)
        
        if self.current_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BEAR]:
            return RiskLevel.CONSERVATIVE
        elif self.current_regime in [MarketRegime.BULL, MarketRegime.MOMENTUM]:
            return RiskLevel.AGGRESSIVE if regime_confidence > 0.7 else RiskLevel.MODERATE
        else:
            return RiskLevel.MODERATE
    
    def _log_decision(self, decision: EnsembleDecision):
        """Log ensemble decision for debugging"""
        
        logger.info(f"Ensemble Decision at step {self.step_count}:")
        logger.info(f"  Final action: {decision.final_action}")
        logger.info(f"  Regime: {decision.regime.value} (confidence: {decision.regime_confidence:.2f})")
        logger.info(f"  Strategy: {decision.strategy_used.value}")
        logger.info(f"  Risk level: {decision.risk_level.value}")
        logger.info(f"  Active agents: {decision.explanation.get('active_agents', [])}")
        logger.info(f"  Agent weights: {decision.agent_weights}")
    
    def update_performance(self, agent_type: AgentType, decision: AgentDecision,
                          actual_return: float, is_correct: bool):
        """Update agent performance after trade execution"""
        
        self.performance_tracker.update_performance(agent_type, decision, actual_return, is_correct)
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble metrics"""
        
        # Agent rankings
        rankings = self.performance_tracker.get_agent_ranking()
        
        # Current weights
        current_weights = self.weight_manager.get_current_weights()
        
        # Regime statistics
        regime_stats = self.regime_detector.get_regime_statistics()
        
        # Decision history statistics
        decision_stats = self._get_decision_statistics()
        
        return {
            'ensemble_metrics': self.ensemble_metrics,
            'agent_rankings': [(agent.value, score) for agent, score in rankings],
            'current_weights': current_weights,
            'current_regime': self.current_regime.value,
            'regime_confidence': getattr(self, 'regime_confidence', 0.0),
            'regime_statistics': regime_stats,
            'decision_statistics': decision_stats,
            'active_agents': [a.value for a in self.active_agents],
            'total_agents': len(self.agents),
            'step_count': self.step_count
        }
    
    def _get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics from decision history"""
        
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-100:]
        
        # Strategy usage
        strategy_counts = {}
        for decision in recent_decisions:
            strategy = decision.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Risk level usage
        risk_counts = {}
        for decision in recent_decisions:
            risk_level = decision.risk_level.value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'strategy_usage': strategy_counts,
            'risk_level_usage': risk_counts
        }
    
    def save_state(self, filepath: str):
        """Save orchestrator state"""
        
        state = {
            'config': {
                'enabled_agents': [a.value for a in self.config.enabled_agents],
                'max_active_agents': self.config.max_active_agents,
                'decision_strategy': self.config.decision_strategy.value,
                'risk_level': self.config.risk_level.value
            },
            'current_regime': self.current_regime.value,
            'regime_confidence': getattr(self, 'regime_confidence', 0.0),
            'step_count': self.step_count,
            'ensemble_metrics': self.ensemble_metrics,
            'agent_weights': self.weight_manager.get_current_weights(),
            'weight_history': self.weight_manager.get_weight_history()[-50:],  # Last 50 weight updates
            'decision_history': [d.to_dict() for d in self.decision_history[-100:]]  # Last 100 decisions
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save regime detector state
        regime_filepath = filepath.replace('.json', '_regime.json')
        self.regime_detector.save_state(regime_filepath)
        
        # Save individual agents
        for agent_type, agent in self.agents.items():
            agent_filepath = filepath.replace('.json', f'_{agent_type.value}')
            agent.save_agent(agent_filepath)
        
        logger.info(f"Orchestrator state saved to {filepath}")
    
    def load_state(self, filepath: str, env):
        """Load orchestrator state"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_regime = MarketRegime(state['current_regime'])
            self.regime_confidence = state['regime_confidence']
            self.step_count = state['step_count']
            self.ensemble_metrics = state['ensemble_metrics']
            
            # Load regime detector state
            regime_filepath = filepath.replace('.json', '_regime.json')
            if os.path.exists(regime_filepath):
                self.regime_detector.load_state(regime_filepath)
            
            # Load individual agents
            for agent_type in self.config.enabled_agents:
                agent_filepath = filepath.replace('.json', f'_{agent_type.value}')
                if agent_type in self.agents:
                    self.agents[agent_type].load_agent(agent_filepath, env)
            
            logger.info(f"Orchestrator state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading orchestrator state: {e}")


if __name__ == "__main__":
    # Example usage and testing
    
    from rl_config import get_rl_config
    from environment.trading_env import TradingEnvironment
    
    # Create configuration
    config = OrchestratorConfig(
        enabled_agents=[AgentType.CONSERVATIVE, AgentType.AGGRESSIVE, AgentType.BALANCED],
        max_active_agents=2,
        decision_strategy=DecisionStrategy.ADAPTIVE_WEIGHT,
        enable_decision_logging=True
    )
    
    # Create orchestrator
    orchestrator = MetaAgentOrchestrator(config)
    
    # Create mock environment
    rl_config = get_rl_config()
    env = TradingEnvironment(config=rl_config, mode='train')
    
    # Initialize agents
    orchestrator.initialize_agents(env)
    
    # Get ensemble metrics
    metrics = orchestrator.get_ensemble_metrics()
    print(f"Ensemble initialized with {metrics['total_agents']} agents")
    print(f"Active agents: {metrics['active_agents']}")
    print(f"Current regime: {metrics['current_regime']}")
    
    # Simulate prediction
    observation = np.random.random(50)
    market_data = {
        'portfolio_value': 10000,
        'price_history': np.random.random(100) * 100 + 1000,
        'volume_history': np.random.random(100) * 1000
    }
    
    action, decision = orchestrator.predict(observation, market_data)
    print(f"\nEnsemble prediction:")
    print(f"Action: {action}")
    print(f"Strategy used: {decision.strategy_used.value}")
    print(f"Risk level: {decision.risk_level.value}")
    print(f"Agent decisions: {len(decision.agent_decisions)}")