"""
Strategy Integration Layer
==========================

Advanced integration layer connecting institutional trading strategies with the RL system.
Provides seamless feature extraction, signal processing, and performance monitoring.

Main Components:
- StrategyIntegrationManager: Main orchestrator
- FeatureAggregator: Feature processing and aggregation
- SignalProcessor: Signal processing and conflict resolution
- RLConnector: Direct connection to RL environment
- PerformanceTracker: Performance monitoring and analytics

Usage:
    from backend.strategies.integration import StrategyIntegrationManager
    
    manager = StrategyIntegrationManager()
    await manager.start()
    
    # The manager will automatically coordinate all components
"""

from .strategy_manager import StrategyIntegrationManager, create_strategy_manager
from .feature_aggregator import FeatureAggregator
from .signal_processor import SignalProcessor
from .rl_connector import RLConnector
from .performance_tracker import PerformanceTracker

__version__ = "1.0.0"
__author__ = "Trading Bot Team"

__all__ = [
    "StrategyIntegrationManager",
    "create_strategy_manager",
    "FeatureAggregator", 
    "SignalProcessor",
    "RLConnector",
    "PerformanceTracker"
]

# Quick setup function
async def setup_integration_layer(config=None):
    """
    Quick setup function for the integration layer.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized and started StrategyIntegrationManager
    """
    manager = create_strategy_manager(config)
    await manager.start()
    return manager