"""
Strategy Integration Manager
===========================

Main orchestrator for connecting all institutional strategies with the RL system.
Manages strategy lifecycle, feature extraction, and signal coordination.

Key Features:
- Dynamic strategy loading and management
- Real-time feature aggregation
- Signal prioritization and routing
- A/B testing framework
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from threading import Lock, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import inspect
import traceback

from .feature_aggregator import FeatureAggregator
from .signal_processor import SignalProcessor
from .rl_connector import RLConnector
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    TESTING = "testing"


class StrategyPriority(Enum):
    """Strategy priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StrategyConfig:
    """Configuration for individual strategies"""
    name: str
    module_path: str
    class_name: str
    enabled: bool = True
    priority: StrategyPriority = StrategyPriority.MEDIUM
    weight: float = 1.0
    timeout: float = 5.0  # seconds
    feature_names: List[str] = field(default_factory=list)
    signal_names: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    ab_test_group: Optional[str] = None
    last_update: Optional[datetime] = None


@dataclass
class StrategyExecution:
    """Strategy execution state and results"""
    strategy_name: str
    status: StrategyStatus
    features: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_execution: Optional[datetime] = None
    success_rate: float = 1.0
    feature_latency: float = 0.0
    signal_latency: float = 0.0


class StrategyIntegrationManager:
    """
    Main integration manager for institutional strategies and RL system.
    
    Orchestrates the flow of data between strategies and RL environment,
    ensuring optimal performance and reliability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy integration manager"""
        self.config = config or self._default_config()
        
        # Core components
        self.feature_aggregator = FeatureAggregator(self.config.get('feature_config', {}))
        self.signal_processor = SignalProcessor(self.config.get('signal_config', {}))
        self.rl_connector = RLConnector(self.config.get('rl_config', {}))
        self.performance_tracker = PerformanceTracker(self.config.get('performance_config', {}))
        
        # Strategy management
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_instances: Dict[str, Any] = {}
        self.strategy_executions: Dict[str, StrategyExecution] = {}
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        self.execution_lock = Lock()
        self.shutdown_event = Event()
        self.running = False
        
        # Feature and signal state
        self.latest_features: Dict[str, float] = {}
        self.latest_signals: Dict[str, Any] = {}
        self.feature_history: Dict[str, List[float]] = {}
        self.signal_history: Dict[str, List[Any]] = {}
        
        # Performance metrics
        self.metrics = {
            'total_strategies': 0,
            'active_strategies': 0,
            'total_features': 0,
            'total_signals': 0,
            'avg_execution_time': 0.0,
            'avg_feature_latency': 0.0,
            'total_executions': 0,
            'error_rate': 0.0,
            'uptime': 0.0
        }
        
        # Load and initialize strategies
        self._load_strategies()
        
        logger.info("Strategy Integration Manager initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_workers': 10,
            'execution_interval': 1.0,  # seconds
            'feature_cache_size': 1000,
            'signal_cache_size': 100,
            'max_feature_latency': 0.01,  # 10ms
            'max_signal_latency': 0.05,  # 50ms
            'error_threshold': 0.1,  # 10% error rate
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'strategy_timeout': 5.0,
            'enable_ab_testing': True,
            'enable_monitoring': True,
            'metrics_update_interval': 60.0,  # seconds
            'strategies': {
                'whale_tracker': {
                    'module_path': 'backend.strategies.institutional.whale_tracker',
                    'class_name': 'WhaleTracker',
                    'priority': 'HIGH',
                    'weight': 2.0,
                    'enabled': True,
                    'feature_names': ['whale_sentiment', 'large_transfers', 'accumulation_score'],
                    'signal_names': ['whale_buy_signal', 'whale_sell_signal']
                },
                'volume_profile': {
                    'module_path': 'backend.strategies.institutional.volume_profile',
                    'class_name': 'VolumeProfile',
                    'priority': 'HIGH',
                    'weight': 1.8,
                    'enabled': True,
                    'feature_names': ['poc_distance', 'volume_imbalance', 'value_area'],
                    'signal_names': ['volume_breakout', 'support_resistance']
                },
                'order_book_analyzer': {
                    'module_path': 'backend.strategies.institutional.order_book_analyzer',
                    'class_name': 'OrderBookAnalyzer',
                    'priority': 'HIGH',
                    'weight': 1.5,
                    'enabled': True,
                    'feature_names': ['bid_ask_spread', 'order_flow', 'liquidity_depth'],
                    'signal_names': ['liquidity_signal', 'order_flow_signal']
                },
                'regime_detection': {
                    'module_path': 'backend.strategies.institutional.regime_detection',
                    'class_name': 'RegimeDetection',
                    'priority': 'MEDIUM',
                    'weight': 1.2,
                    'enabled': True,
                    'feature_names': ['regime_state', 'regime_confidence', 'volatility_regime'],
                    'signal_names': ['regime_change_signal']
                },
                'smart_money_divergence': {
                    'module_path': 'backend.strategies.institutional.smart_money_divergence',
                    'class_name': 'SmartMoneyDivergence',
                    'priority': 'MEDIUM',
                    'weight': 1.0,
                    'enabled': True,
                    'feature_names': ['smart_money_flow', 'divergence_strength', 'institutional_sentiment'],
                    'signal_names': ['smart_money_signal']
                }
            }
        }
    
    def _load_strategies(self):
        """Load and initialize strategy configurations"""
        strategy_configs = self.config.get('strategies', {})
        
        for name, config in strategy_configs.items():
            try:
                strategy_config = StrategyConfig(
                    name=name,
                    module_path=config['module_path'],
                    class_name=config['class_name'],
                    enabled=config.get('enabled', True),
                    priority=StrategyPriority[config.get('priority', 'MEDIUM')],
                    weight=config.get('weight', 1.0),
                    timeout=config.get('timeout', self.config['strategy_timeout']),
                    feature_names=config.get('feature_names', []),
                    signal_names=config.get('signal_names', []),
                    dependencies=config.get('dependencies', []),
                    config_params=config.get('config_params', {}),
                    ab_test_group=config.get('ab_test_group')
                )
                
                self.strategies[name] = strategy_config
                
                # Initialize strategy execution state
                self.strategy_executions[name] = StrategyExecution(
                    strategy_name=name,
                    status=StrategyStatus.INACTIVE
                )
                
                logger.info(f"Loaded strategy configuration: {name}")
                
            except Exception as e:
                logger.error(f"Error loading strategy {name}: {e}")
        
        self.metrics['total_strategies'] = len(self.strategies)
        logger.info(f"Loaded {len(self.strategies)} strategy configurations")
    
    async def initialize_strategies(self):
        """Initialize strategy instances"""
        for name, config in self.strategies.items():
            if not config.enabled:
                continue
            
            try:
                # Import strategy module
                module = importlib.import_module(config.module_path)
                strategy_class = getattr(module, config.class_name)
                
                # Initialize strategy instance
                strategy_instance = strategy_class(config.config_params)
                self.strategy_instances[name] = strategy_instance
                
                # Update execution status
                self.strategy_executions[name].status = StrategyStatus.ACTIVE
                
                logger.info(f"Initialized strategy: {name}")
                
            except Exception as e:
                logger.error(f"Error initializing strategy {name}: {e}")
                self.strategy_executions[name].status = StrategyStatus.ERROR
                self.strategy_executions[name].last_error = str(e)
        
        active_count = sum(1 for exec_state in self.strategy_executions.values() 
                          if exec_state.status == StrategyStatus.ACTIVE)
        self.metrics['active_strategies'] = active_count
        
        logger.info(f"Initialized {active_count} strategies successfully")
    
    async def start(self):
        """Start the strategy integration manager"""
        if self.running:
            logger.warning("Strategy manager already running")
            return
        
        logger.info("Starting Strategy Integration Manager...")
        
        # Initialize strategies
        await self.initialize_strategies()
        
        # Start core components
        await self.feature_aggregator.start()
        await self.signal_processor.start()
        await self.rl_connector.start()
        await self.performance_tracker.start()
        
        self.running = True
        
        # Start main execution loop
        asyncio.create_task(self._execution_loop())
        
        # Start monitoring
        if self.config.get('enable_monitoring', True):
            asyncio.create_task(self._monitoring_loop())
        
        logger.info("Strategy Integration Manager started successfully")
    
    async def stop(self):
        """Stop the strategy integration manager"""
        logger.info("Stopping Strategy Integration Manager...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop core components
        await self.feature_aggregator.stop()
        await self.signal_processor.stop()
        await self.rl_connector.stop()
        await self.performance_tracker.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Strategy Integration Manager stopped")
    
    async def _execution_loop(self):
        """Main execution loop for strategy processing"""
        while self.running and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Execute strategies in parallel
                execution_tasks = []
                for name, config in self.strategies.items():
                    if (config.enabled and 
                        name in self.strategy_instances and 
                        self.strategy_executions[name].status == StrategyStatus.ACTIVE):
                        
                        task = asyncio.create_task(self._execute_strategy(name))
                        execution_tasks.append(task)
                
                # Wait for all strategies to complete
                if execution_tasks:
                    results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                    
                    # Process results
                    await self._process_execution_results(results)
                
                # Update metrics
                execution_time = time.time() - start_time
                self.metrics['avg_execution_time'] = (
                    self.metrics['avg_execution_time'] * 0.9 + execution_time * 0.1
                )
                
                # Wait for next interval
                await asyncio.sleep(max(0, self.config['execution_interval'] - execution_time))
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Execute individual strategy"""
        start_time = time.time()
        execution = self.strategy_executions[strategy_name]
        strategy_instance = self.strategy_instances[strategy_name]
        config = self.strategies[strategy_name]
        
        try:
            # Get market data for strategy
            market_data = await self._get_market_data()
            
            # Execute strategy methods
            features = {}
            signals = {}
            
            # Extract features
            if hasattr(strategy_instance, 'extract_features'):
                feature_start = time.time()
                features = await self._call_strategy_method(
                    strategy_instance.extract_features, 
                    market_data,
                    timeout=config.timeout
                )
                execution.feature_latency = time.time() - feature_start
            
            # Generate signals
            if hasattr(strategy_instance, 'generate_signals'):
                signal_start = time.time()
                signals = await self._call_strategy_method(
                    strategy_instance.generate_signals,
                    market_data,
                    timeout=config.timeout
                )
                execution.signal_latency = time.time() - signal_start
            
            # Update execution state
            execution.features = features or {}
            execution.signals = signals or {}
            execution.execution_time = time.time() - start_time
            execution.last_execution = datetime.now()
            execution.status = StrategyStatus.ACTIVE
            
            # Update success rate
            execution.success_rate = execution.success_rate * 0.95 + 0.05
            
            return {
                'strategy_name': strategy_name,
                'features': features,
                'signals': signals,
                'execution_time': execution.execution_time,
                'success': True
            }
            
        except Exception as e:
            # Handle execution error
            execution.error_count += 1
            execution.last_error = str(e)
            execution.status = StrategyStatus.ERROR
            execution.success_rate = execution.success_rate * 0.95
            
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            
            return {
                'strategy_name': strategy_name,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'success': False
            }
    
    async def _call_strategy_method(self, method: Callable, *args, timeout: float) -> Any:
        """Call strategy method with timeout"""
        if asyncio.iscoroutinefunction(method):
            return await asyncio.wait_for(method(*args), timeout=timeout)
        else:
            # Run synchronous method in executor
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, method, *args),
                timeout=timeout
            )
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for strategies"""
        # This would typically fetch from data sources
        # For now, return placeholder data
        return {
            'timestamp': datetime.now(),
            'price_data': {
                'symbol': 'BTC/USD',
                'price': 30000.0,
                'volume': 1000000,
                'high': 31000.0,
                'low': 29000.0,
                'open': 30500.0
            },
            'order_book': {
                'bids': [[29990, 10], [29980, 20]],
                'asks': [[30010, 15], [30020, 25]]
            },
            'trades': [],
            'indicators': {}
        }
    
    async def _process_execution_results(self, results: List[Any]):
        """Process strategy execution results"""
        all_features = {}
        all_signals = {}
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Strategy execution exception: {result}")
                continue
            
            if not result.get('success', False):
                continue
            
            strategy_name = result['strategy_name']
            features = result.get('features', {})
            signals = result.get('signals', {})
            
            # Aggregate features
            for name, value in features.items():
                feature_key = f"{strategy_name}_{name}"
                all_features[feature_key] = value
                
                # Update feature history
                if feature_key not in self.feature_history:
                    self.feature_history[feature_key] = []
                self.feature_history[feature_key].append(value)
                
                # Keep only recent history
                if len(self.feature_history[feature_key]) > self.config['feature_cache_size']:
                    self.feature_history[feature_key] = self.feature_history[feature_key][-self.config['feature_cache_size']:]
            
            # Aggregate signals
            for name, value in signals.items():
                signal_key = f"{strategy_name}_{name}"
                all_signals[signal_key] = value
                
                # Update signal history
                if signal_key not in self.signal_history:
                    self.signal_history[signal_key] = []
                self.signal_history[signal_key].append(value)
                
                # Keep only recent history
                if len(self.signal_history[signal_key]) > self.config['signal_cache_size']:
                    self.signal_history[signal_key] = self.signal_history[signal_key][-self.config['signal_cache_size']:]
        
        # Update latest state
        self.latest_features.update(all_features)
        self.latest_signals.update(all_signals)
        
        # Send to feature aggregator
        if all_features:
            await self.feature_aggregator.add_features(all_features)
        
        # Send to signal processor
        if all_signals:
            await self.signal_processor.add_signals(all_signals)
        
        # Update metrics
        self.metrics['total_features'] = len(self.latest_features)
        self.metrics['total_signals'] = len(self.latest_signals)
        self.metrics['total_executions'] += len([r for r in results if not isinstance(r, Exception)])
        
        # Calculate average feature latency
        feature_latencies = [exec_state.feature_latency for exec_state in self.strategy_executions.values() 
                           if exec_state.feature_latency > 0]
        if feature_latencies:
            self.metrics['avg_feature_latency'] = np.mean(feature_latencies)
    
    async def _monitoring_loop(self):
        """Monitoring loop for health checks and alerts"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check strategy health
                await self._check_strategy_health()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Sleep until next check
                await asyncio.sleep(self.config['metrics_update_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _check_strategy_health(self):
        """Check health of all strategies"""
        for name, execution in self.strategy_executions.items():
            config = self.strategies[name]
            
            if not config.enabled:
                continue
            
            # Check execution frequency
            if execution.last_execution:
                time_since_last = datetime.now() - execution.last_execution
                if time_since_last > timedelta(seconds=self.config['execution_interval'] * 5):
                    logger.warning(f"Strategy {name} hasn't executed recently")
                    execution.status = StrategyStatus.ERROR
            
            # Check error rate
            error_rate = execution.error_count / max(1, execution.error_count + 1)
            if error_rate > self.config['error_threshold']:
                logger.warning(f"Strategy {name} has high error rate: {error_rate:.2%}")
                execution.status = StrategyStatus.PAUSED
            
            # Check feature latency
            if execution.feature_latency > self.config['max_feature_latency']:
                logger.warning(f"Strategy {name} feature latency too high: {execution.feature_latency:.3f}s")
            
            # Check signal latency
            if execution.signal_latency > self.config['max_signal_latency']:
                logger.warning(f"Strategy {name} signal latency too high: {execution.signal_latency:.3f}s")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        # Calculate uptime
        # This would track actual uptime in production
        
        # Calculate error rate
        total_errors = sum(exec_state.error_count for exec_state in self.strategy_executions.values())
        total_executions = max(1, self.metrics['total_executions'])
        self.metrics['error_rate'] = total_errors / total_executions
        
        # Send metrics to performance tracker
        await self.performance_tracker.update_metrics(self.metrics)
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        # Check overall system health
        active_strategies = sum(1 for exec_state in self.strategy_executions.values() 
                              if exec_state.status == StrategyStatus.ACTIVE)
        
        if active_strategies < len(self.strategies) * 0.5:
            logger.critical(f"Only {active_strategies} strategies active out of {len(self.strategies)}")
        
        # Check feature latency
        if self.metrics['avg_feature_latency'] > self.config['max_feature_latency']:
            logger.warning(f"High average feature latency: {self.metrics['avg_feature_latency']:.3f}s")
        
        # Check error rate
        if self.metrics['error_rate'] > self.config['error_threshold']:
            logger.warning(f"High error rate: {self.metrics['error_rate']:.2%}")
    
    def get_strategy_status(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific strategy"""
        if strategy_name not in self.strategy_executions:
            return None
        
        execution = self.strategy_executions[strategy_name]
        config = self.strategies[strategy_name]
        
        return {
            'name': strategy_name,
            'status': execution.status.value,
            'enabled': config.enabled,
            'priority': config.priority.value,
            'weight': config.weight,
            'features': execution.features,
            'signals': execution.signals,
            'execution_time': execution.execution_time,
            'feature_latency': execution.feature_latency,
            'signal_latency': execution.signal_latency,
            'error_count': execution.error_count,
            'success_rate': execution.success_rate,
            'last_execution': execution.last_execution.isoformat() if execution.last_execution else None,
            'last_error': execution.last_error
        }
    
    def get_all_features(self) -> Dict[str, float]:
        """Get all current features"""
        return self.latest_features.copy()
    
    def get_all_signals(self) -> Dict[str, Any]:
        """Get all current signals"""
        return self.latest_signals.copy()
    
    def get_aggregated_features(self) -> Dict[str, float]:
        """Get aggregated features for RL system"""
        return self.feature_aggregator.get_aggregated_features()
    
    def get_processed_signals(self) -> Dict[str, Any]:
        """Get processed signals for RL system"""
        return self.signal_processor.get_processed_signals()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            **self.metrics,
            'strategies': {
                name: self.get_strategy_status(name) 
                for name in self.strategies.keys()
            },
            'feature_aggregator_metrics': self.feature_aggregator.get_metrics(),
            'signal_processor_metrics': self.signal_processor.get_metrics(),
            'rl_connector_metrics': self.rl_connector.get_metrics(),
            'performance_tracker_metrics': self.performance_tracker.get_metrics()
        }
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy"""
        if strategy_name not in self.strategies:
            return False
        
        self.strategies[strategy_name].enabled = True
        if strategy_name in self.strategy_executions:
            self.strategy_executions[strategy_name].status = StrategyStatus.ACTIVE
        
        logger.info(f"Enabled strategy: {strategy_name}")
        return True
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy"""
        if strategy_name not in self.strategies:
            return False
        
        self.strategies[strategy_name].enabled = False
        if strategy_name in self.strategy_executions:
            self.strategy_executions[strategy_name].status = StrategyStatus.INACTIVE
        
        logger.info(f"Disabled strategy: {strategy_name}")
        return True
    
    def update_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """Update strategy weight"""
        if strategy_name not in self.strategies:
            return False
        
        self.strategies[strategy_name].weight = weight
        
        # Update in feature aggregator and signal processor
        self.feature_aggregator.update_strategy_weight(strategy_name, weight)
        self.signal_processor.update_strategy_weight(strategy_name, weight)
        
        logger.info(f"Updated strategy {strategy_name} weight to {weight}")
        return True
    
    async def reload_strategy(self, strategy_name: str) -> bool:
        """Reload a strategy instance"""
        if strategy_name not in self.strategies:
            return False
        
        try:
            config = self.strategies[strategy_name]
            
            # Import strategy module
            module = importlib.reload(importlib.import_module(config.module_path))
            strategy_class = getattr(module, config.class_name)
            
            # Create new instance
            strategy_instance = strategy_class(config.config_params)
            self.strategy_instances[strategy_name] = strategy_instance
            
            # Reset execution state
            self.strategy_executions[strategy_name] = StrategyExecution(
                strategy_name=strategy_name,
                status=StrategyStatus.ACTIVE if config.enabled else StrategyStatus.INACTIVE
            )
            
            logger.info(f"Reloaded strategy: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading strategy {strategy_name}: {e}")
            return False


# Factory function for easy initialization
def create_strategy_manager(config: Optional[Dict] = None) -> StrategyIntegrationManager:
    """Create and return a strategy integration manager instance"""
    return StrategyIntegrationManager(config)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create strategy manager
        manager = create_strategy_manager()
        
        try:
            # Start the manager
            await manager.start()
            
            # Run for a while
            await asyncio.sleep(30)
            
            # Get system status
            metrics = manager.get_system_metrics()
            print(f"System Metrics: {metrics}")
            
        finally:
            # Stop the manager
            await manager.stop()
    
    asyncio.run(main())