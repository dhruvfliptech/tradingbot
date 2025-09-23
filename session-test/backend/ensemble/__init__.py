"""
Multi-Agent Ensemble Trading System

This module provides a comprehensive multi-agent ensemble system for cryptocurrency trading
with specialized agents, market regime detection, and intelligent orchestration.

The ensemble system consists of:

1. Specialized Agents (agents/specialized_agents.py):
   - ConservativeAgent: Capital preservation focused
   - AggressiveAgent: Maximum returns focused  
   - BalancedAgent: Risk-adjusted returns optimized
   - ContrarianAgent: Mean reversion focused

2. Market Regime Detection (regime/market_regime_detector.py):
   - Real-time market condition classification
   - Multiple regime types (Bull, Bear, Sideways, High Volatility, etc.)
   - ML-enhanced detection with rule-based fallback

3. Meta-Agent Orchestrator (meta/meta_agent_orchestrator.py):
   - Coordinates agent selection and decision combination
   - Dynamic weight adjustment based on performance
   - Risk management and position sizing
   - Explainable decision making

4. Multi-Armed Bandit Selector (bandit/strategy_selector.py):
   - Intelligent agent selection using bandit algorithms
   - Contextual selection based on market conditions
   - Performance-based learning and adaptation

5. Performance Tracker (performance/ensemble_tracker.py):
   - Comprehensive performance monitoring
   - Individual agent and ensemble analytics
   - Risk-adjusted metrics and reporting

Key Features:
- Prevents single point of failure through diversification
- Adapts to changing market conditions automatically
- Provides explainable trading decisions
- Tracks and optimizes performance continuously
- Robust risk management across all market regimes

Usage:
    from ensemble import EnsembleSystem
    
    # Initialize ensemble
    ensemble = EnsembleSystem()
    ensemble.initialize(env)
    
    # Train agents
    ensemble.train_all_agents(total_timesteps=100000)
    
    # Make predictions
    action, decision = ensemble.predict(observation, market_data)
    
    # Update performance
    ensemble.update_performance(actual_return)
"""

from .agents.specialized_agents import (
    SpecializedAgent, AgentType, ConservativeAgent, AggressiveAgent, 
    BalancedAgent, ContrarianAgent, create_specialized_agent, create_agent_ensemble
)

from .regime.market_regime_detector import (
    MarketRegime, MarketRegimeDetector, RegimeDetectionConfig, get_optimal_agents_for_regime
)

from .meta.meta_agent_orchestrator import (
    MetaAgentOrchestrator, OrchestratorConfig, EnsembleDecision, DecisionStrategy, RiskLevel
)

from .bandit.strategy_selector import (
    MultiArmedBanditSelector, BanditConfig, BanditAlgorithm, create_bandit_selector
)

from .performance.ensemble_tracker import (
    EnsemblePerformanceTracker, PerformanceConfig, PerformanceMetric, PerformanceSnapshot
)

__version__ = "1.0.0"
__author__ = "Trading Bot Development Team"

__all__ = [
    # Main system classes
    'EnsembleSystem',
    
    # Specialized agents
    'SpecializedAgent', 'AgentType', 'ConservativeAgent', 'AggressiveAgent',
    'BalancedAgent', 'ContrarianAgent', 'create_specialized_agent', 'create_agent_ensemble',
    
    # Market regime detection
    'MarketRegime', 'MarketRegimeDetector', 'RegimeDetectionConfig', 'get_optimal_agents_for_regime',
    
    # Meta-agent orchestration
    'MetaAgentOrchestrator', 'OrchestratorConfig', 'EnsembleDecision', 'DecisionStrategy', 'RiskLevel',
    
    # Bandit selection
    'MultiArmedBanditSelector', 'BanditConfig', 'BanditAlgorithm', 'create_bandit_selector',
    
    # Performance tracking
    'EnsemblePerformanceTracker', 'PerformanceConfig', 'PerformanceMetric', 'PerformanceSnapshot'
]


class EnsembleSystem:
    """
    Main ensemble system that integrates all components
    
    This is the primary interface for using the multi-agent ensemble system.
    It coordinates all components and provides a simple API for training,
    prediction, and performance tracking.
    """
    
    def __init__(self, 
                 orchestrator_config: OrchestratorConfig = None,
                 regime_config: RegimeDetectionConfig = None,
                 bandit_config: BanditConfig = None,
                 performance_config: PerformanceConfig = None):
        """
        Initialize the ensemble system
        
        Args:
            orchestrator_config: Configuration for meta-agent orchestrator
            regime_config: Configuration for regime detection
            bandit_config: Configuration for bandit selection
            performance_config: Configuration for performance tracking
        """
        
        # Initialize configurations
        self.orchestrator_config = orchestrator_config or OrchestratorConfig()
        self.regime_config = regime_config or RegimeDetectionConfig()
        self.bandit_config = bandit_config or BanditConfig()
        self.performance_config = performance_config or PerformanceConfig()
        
        # Initialize components
        self.orchestrator = MetaAgentOrchestrator(self.orchestrator_config)
        self.performance_tracker = EnsemblePerformanceTracker(self.performance_config)
        
        # Bandit selector (initialized after agents are set up)
        self.bandit_selector = None
        
        # State tracking
        self.is_initialized = False
        self.is_trained = False
        self.env = None
        
        print("Ensemble system initialized with multi-agent orchestration")
    
    def initialize(self, env):
        """
        Initialize the ensemble system with trading environment
        
        Args:
            env: Trading environment for agent training and evaluation
        """
        
        self.env = env
        
        # Initialize orchestrator agents
        self.orchestrator.initialize_agents(env)
        
        # Initialize bandit selector with available agents
        agent_types = list(self.orchestrator.agents.keys())
        self.bandit_selector = create_bandit_selector(
            agent_types=agent_types,
            algorithm=self.bandit_config.algorithm,
            **{
                'epsilon': self.bandit_config.epsilon,
                'ucb_confidence': self.bandit_config.ucb_confidence,
                'context_window': self.bandit_config.context_window,
                'enable_logging': self.bandit_config.enable_logging
            }
        )
        
        # Initialize performance trackers for each agent
        for agent_type in agent_types:
            self.performance_tracker.initialize_agent_tracker(agent_type)
        
        self.is_initialized = True
        print(f"Ensemble system initialized with {len(agent_types)} specialized agents")
    
    def train_all_agents(self, total_timesteps: int = 100000, timesteps_per_agent: int = None):
        """
        Train all agents in the ensemble
        
        Args:
            total_timesteps: Total timesteps for training
            timesteps_per_agent: Timesteps per individual agent (defaults to total_timesteps)
        """
        
        if not self.is_initialized:
            raise ValueError("Ensemble system must be initialized before training")
        
        timesteps_per_agent = timesteps_per_agent or total_timesteps
        
        print(f"Training ensemble agents with {timesteps_per_agent:,} timesteps each...")
        
        # Train all agents
        training_results = self.orchestrator.train_agents(self.env, timesteps_per_agent)
        
        self.is_trained = True
        
        print("Ensemble training completed")
        print("Training Results:")
        for agent_type, result in training_results.items():
            if 'error' not in result:
                best_reward = result.get('best_mean_reward', 'N/A')
                print(f"  {agent_type}: Best Mean Reward = {best_reward}")
            else:
                print(f"  {agent_type}: Training failed - {result['error']}")
        
        return training_results
    
    def predict(self, observation, market_data: dict, deterministic: bool = True):
        """
        Generate ensemble prediction
        
        Args:
            observation: Environment observation
            market_data: Market data for regime detection and context
            deterministic: Whether to use deterministic policies
            
        Returns:
            Tuple of (final_action, ensemble_decision_info)
        """
        
        if not self.is_trained:
            print("Warning: Agents not trained yet. Using untrained models.")
        
        # Get orchestrator prediction
        action, ensemble_decision = self.orchestrator.predict(observation, market_data, deterministic)
        
        # Use bandit selector for additional agent selection insights
        if self.bandit_selector:
            # Create context for bandit
            bandit_context = {
                'market_regime': ensemble_decision.regime,
                'regime_confidence': ensemble_decision.regime_confidence,
                'volatility': market_data.get('volatility', 0.02),
                'trend_strength': market_data.get('trend_strength', 0.0),
                'current_drawdown': market_data.get('current_drawdown', 0.0),
                'portfolio_return': market_data.get('portfolio_return', 0.0)
            }
            
            # Get bandit recommendation
            bandit_agent, bandit_info = self.bandit_selector.select_agent(bandit_context)
            
            # Add bandit insights to decision
            ensemble_decision.explanation['bandit_recommendation'] = {
                'selected_agent': bandit_agent.value,
                'selection_info': bandit_info
            }
        
        return action, ensemble_decision
    
    def update_performance(self, actual_return: float, ensemble_decision: EnsembleDecision = None,
                          individual_returns: dict = None):
        """
        Update performance tracking after trade execution
        
        Args:
            actual_return: Actual return achieved
            ensemble_decision: Previous ensemble decision (optional)
            individual_returns: Individual agent returns (optional)
        """
        
        # Update ensemble performance tracker
        if ensemble_decision:
            self.performance_tracker.update_ensemble_performance(
                actual_return, ensemble_decision, individual_returns
            )
        
        # Update bandit selector if available
        if self.bandit_selector and ensemble_decision:
            # Extract context used for selection
            context = {
                'market_regime': ensemble_decision.regime,
                'regime_confidence': ensemble_decision.regime_confidence
            }
            
            # Update bandit for each agent that contributed
            for agent_decision in ensemble_decision.agent_decisions:
                agent_type = agent_decision.agent_type
                
                # Calculate agent-specific reward (simplified)
                agent_return = individual_returns.get(agent_type, actual_return) if individual_returns else actual_return
                
                self.bandit_selector.update_reward(agent_type, agent_return, context)
    
    def get_comprehensive_metrics(self):
        """Get comprehensive performance and system metrics"""
        
        metrics = {
            'ensemble_performance': self.performance_tracker.get_ensemble_metrics(),
            'agent_comparison': self.performance_tracker.get_agent_comparison(),
            'regime_analysis': self.performance_tracker.get_regime_analysis(),
            'attribution_analysis': self.performance_tracker.get_attribution_analysis(),
            'orchestrator_metrics': self.orchestrator.get_ensemble_metrics() if hasattr(self.orchestrator, 'get_ensemble_metrics') else {},
            'bandit_statistics': self.bandit_selector.get_bandit_statistics() if self.bandit_selector else {},
            'system_status': {
                'is_initialized': self.is_initialized,
                'is_trained': self.is_trained,
                'num_agents': len(self.orchestrator.agents) if self.is_initialized else 0,
                'current_regime': self.orchestrator.current_regime.value if self.is_initialized else 'unknown'
            }
        }
        
        return metrics
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        return self.performance_tracker.generate_performance_report()
    
    def save_system_state(self, base_filepath: str):
        """
        Save complete system state
        
        Args:
            base_filepath: Base path for saving (without extension)
        """
        
        # Save orchestrator state
        orchestrator_path = f"{base_filepath}_orchestrator.json"
        self.orchestrator.save_state(orchestrator_path)
        
        # Save performance tracker state
        performance_path = f"{base_filepath}_performance.json"
        self.performance_tracker.save_state(performance_path)
        
        # Save bandit selector state
        if self.bandit_selector:
            bandit_path = f"{base_filepath}_bandit.json"
            self.bandit_selector.save_state(bandit_path)
        
        print(f"Ensemble system state saved to {base_filepath}_*")
    
    def load_system_state(self, base_filepath: str):
        """
        Load complete system state
        
        Args:
            base_filepath: Base path for loading (without extension)
        """
        
        if not self.is_initialized:
            raise ValueError("Ensemble system must be initialized before loading state")
        
        # Load orchestrator state
        orchestrator_path = f"{base_filepath}_orchestrator.json"
        self.orchestrator.load_state(orchestrator_path, self.env)
        
        # Load performance tracker state
        performance_path = f"{base_filepath}_performance.json"
        self.performance_tracker.load_state(performance_path)
        
        # Load bandit selector state
        if self.bandit_selector:
            bandit_path = f"{base_filepath}_bandit.json"
            self.bandit_selector.load_state(bandit_path)
        
        self.is_trained = True  # Assume trained if loading state
        print(f"Ensemble system state loaded from {base_filepath}_*")
    
    def get_system_summary(self):
        """Get a summary of the ensemble system status"""
        
        summary = {
            'system_info': {
                'version': __version__,
                'initialized': self.is_initialized,
                'trained': self.is_trained,
                'num_agents': len(self.orchestrator.agents) if self.is_initialized else 0
            }
        }
        
        if self.is_initialized:
            # Agent information
            summary['agents'] = {}
            for agent_type, agent in self.orchestrator.agents.items():
                summary['agents'][agent_type.value] = {
                    'name': agent.name,
                    'description': agent.config.description,
                    'trained': hasattr(agent, 'ppo_agent') and agent.ppo_agent is not None
                }
            
            # Current regime
            summary['current_regime'] = {
                'regime': self.orchestrator.current_regime.value,
                'confidence': getattr(self.orchestrator, 'regime_confidence', 0.0)
            }
            
            # Performance summary
            if hasattr(self.performance_tracker, 'get_ensemble_metrics'):
                ensemble_metrics = self.performance_tracker.get_ensemble_metrics()
                if ensemble_metrics:
                    summary['performance'] = {
                        'total_return': ensemble_metrics.get('total_return', 0.0),
                        'sharpe_ratio': ensemble_metrics.get('sharpe_ratio', 0.0),
                        'max_drawdown': ensemble_metrics.get('max_drawdown', 0.0),
                        'num_trades': ensemble_metrics.get('num_trades', 0)
                    }
        
        return summary


# Convenience functions for quick setup
def create_default_ensemble_system():
    """Create ensemble system with default configurations"""
    
    return EnsembleSystem(
        orchestrator_config=OrchestratorConfig(
            enabled_agents=[AgentType.CONSERVATIVE, AgentType.AGGRESSIVE, AgentType.BALANCED, AgentType.CONTRARIAN],
            decision_strategy=DecisionStrategy.ADAPTIVE_WEIGHT,
            enable_decision_logging=True
        ),
        regime_config=RegimeDetectionConfig(
            enable_ml_detection=True,
            debug_mode=False
        ),
        bandit_config=BanditConfig(
            algorithm=BanditAlgorithm.CONTEXTUAL_UCB,
            enable_logging=True
        ),
        performance_config=PerformanceConfig(
            enable_detailed_logging=True,
            enable_regime_analysis=True,
            enable_attribution_analysis=True
        )
    )


def create_conservative_ensemble_system():
    """Create ensemble system optimized for conservative trading"""
    
    return EnsembleSystem(
        orchestrator_config=OrchestratorConfig(
            enabled_agents=[AgentType.CONSERVATIVE, AgentType.BALANCED],
            decision_strategy=DecisionStrategy.PERFORMANCE_WEIGHTED,
            risk_level=RiskLevel.CONSERVATIVE,
            max_total_position=0.5
        ),
        performance_config=PerformanceConfig(
            enable_detailed_logging=True,
            risk_free_rate=0.02
        )
    )


def create_aggressive_ensemble_system():
    """Create ensemble system optimized for aggressive trading"""
    
    return EnsembleSystem(
        orchestrator_config=OrchestratorConfig(
            enabled_agents=[AgentType.AGGRESSIVE, AgentType.BALANCED, AgentType.CONTRARIAN],
            decision_strategy=DecisionStrategy.REGIME_BASED,
            risk_level=RiskLevel.AGGRESSIVE,
            max_total_position=1.5
        ),
        bandit_config=BanditConfig(
            algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
            epsilon=0.15  # More exploration
        )
    )