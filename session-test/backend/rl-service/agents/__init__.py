"""
RL Agents Package for Crypto Trading Bot

This package contains:
- PPO Agent: Main reinforcement learning agent using Proximal Policy Optimization
- Policy Networks: Custom neural network architectures for trading
- Training Pipeline: Comprehensive training loop with monitoring and optimization
- Ensemble Agent: Multi-agent system for different market conditions
- Explainer: Decision explainability and feature importance analysis
"""

from .ppo_agent import PPOAgent, PPOConfig
from .policy_network import TradingPolicyNetwork, AttentionPolicy, RiskAwarePolicy
from .trainer import PPOTrainer, TrainingConfig, TrainingMetrics
from .ensemble_agent import EnsembleAgent, MarketRegimeDetector
from .explainer import AgentExplainer, FeatureImportance, DecisionExplanation

__all__ = [
    'PPOAgent',
    'PPOConfig', 
    'TradingPolicyNetwork',
    'AttentionPolicy',
    'RiskAwarePolicy',
    'PPOTrainer',
    'TrainingConfig',
    'TrainingMetrics',
    'EnsembleAgent',
    'MarketRegimeDetector',
    'AgentExplainer',
    'FeatureImportance',
    'DecisionExplanation'
]

__version__ = "1.0.0"