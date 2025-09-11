"""
Multi-Objective Reward System for Crypto Trading RL Agent

This package implements a sophisticated reward function that balances:
- Profit maximization (with tanh normalization to prevent greed)
- Risk-adjusted returns (Sharpe ratio optimization)
- Drawdown management (strict enforcement at 10-15%)
- Trading consistency (win streaks, low variance)
- Transaction costs and market impact
- Exploration bonuses and time decay factors

Aligned with SOW targets:
- 3-5% weekly returns
- Sharpe ratio > 1.5
- Maximum drawdown < 15%
- Win rate > 60%
"""

from .reward_components import (
    RewardComponentConfig,
    MarketRegime,
    ProfitComponent,
    RiskAdjustedComponent,
    DrawdownComponent,
    ConsistencyComponent,
    TransactionCostComponent,
    MarketRegimeComponent,
    ExplorationComponent,
    TimeDecayComponent,
    create_reward_components
)

from .multi_objective_reward import (
    MultiObjectiveConfig,
    OptimizationStrategy,
    MultiObjectiveReward,
    ParetoFrontier,
    AdaptiveWeightOptimizer
)

from .reward_calculator import (
    RewardCalculatorConfig,
    RewardCalculator,
    RewardShaper,
    SparseRewardHandler,
    CurriculumManager
)

from .reward_optimizer import (
    OptimizerConfig,
    BayesianOptimizer,
    EvolutionaryOptimizer,
    MarketAdaptiveOptimizer,
    HybridOptimizer,
    create_optimizer
)

from .reward_analysis import (
    AnalysisConfig,
    RewardAnalyzer,
    create_analyzer
)

__all__ = [
    # Components
    'RewardComponentConfig',
    'MarketRegime',
    'ProfitComponent',
    'RiskAdjustedComponent',
    'DrawdownComponent',
    'ConsistencyComponent',
    'TransactionCostComponent',
    'MarketRegimeComponent',
    'ExplorationComponent',
    'TimeDecayComponent',
    'create_reward_components',
    
    # Multi-objective
    'MultiObjectiveConfig',
    'OptimizationStrategy',
    'MultiObjectiveReward',
    'ParetoFrontier',
    'AdaptiveWeightOptimizer',
    
    # Calculator
    'RewardCalculatorConfig',
    'RewardCalculator',
    'RewardShaper',
    'SparseRewardHandler',
    'CurriculumManager',
    
    # Optimizer
    'OptimizerConfig',
    'BayesianOptimizer',
    'EvolutionaryOptimizer',
    'MarketAdaptiveOptimizer',
    'HybridOptimizer',
    'create_optimizer',
    
    # Analysis
    'AnalysisConfig',
    'RewardAnalyzer',
    'create_analyzer'
]

# Version
__version__ = '1.0.0'