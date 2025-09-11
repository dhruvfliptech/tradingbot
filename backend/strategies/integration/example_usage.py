"""
Example Usage of Strategy Integration Layer
==========================================

Demonstrates how to use the strategy integration layer to connect
institutional strategies with the RL system for enhanced trading decisions.

This example shows:
1. Setting up the integration manager
2. Adding custom strategies
3. Processing features and signals
4. Monitoring performance
5. A/B testing strategies
"""

import asyncio
import logging
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any

# Import integration components
from strategy_manager import StrategyIntegrationManager
from feature_aggregator import FeatureAggregator
from signal_processor import SignalProcessor, SignalType, SignalStrength
from rl_connector import RLConnector
from performance_tracker import PerformanceTracker, PerformanceMetric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleStrategy:
    """Example strategy for demonstration"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.call_count = 0
    
    async def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from market data"""
        self.call_count += 1
        
        # Simulate feature extraction
        features = {
            f'{self.name}_trend_strength': np.random.uniform(0, 1),
            f'{self.name}_volume_ratio': np.random.uniform(0.5, 2.0),
            f'{self.name}_price_momentum': np.random.uniform(-1, 1),
            f'{self.name}_volatility': np.random.uniform(0.1, 0.5),
            f'{self.name}_support_resistance': np.random.uniform(0, 1)
        }
        
        # Add some correlation with time for realistic behavior
        time_factor = np.sin(time.time() / 100) * 0.1
        for key in features:
            features[key] += time_factor
        
        return features
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals"""
        
        # Simulate signal generation
        trend_strength = np.random.uniform(0, 1)
        confidence = np.random.uniform(0.3, 0.9)
        
        if trend_strength > 0.7:
            signal_type = 'buy' if np.random.random() > 0.5 else 'strong_buy'
            strength = 3 if signal_type == 'strong_buy' else 2
        elif trend_strength < 0.3:
            signal_type = 'sell' if np.random.random() > 0.5 else 'strong_sell'
            strength = 3 if signal_type == 'strong_sell' else 2
        else:
            signal_type = 'hold'
            strength = 1
        
        signals = {
            f'{self.name}_main_signal': {
                'type': signal_type,
                'strength': strength,
                'confidence': confidence,
                'reasoning': f'{self.name} analysis: trend_strength={trend_strength:.2f}'
            },
            f'{self.name}_risk_signal': {
                'type': 'hold' if confidence < 0.5 else signal_type,
                'strength': max(1, strength - 1),
                'confidence': confidence * 0.8,
                'reasoning': f'Risk-adjusted signal from {self.name}'
            }
        }
        
        return signals


async def demo_basic_integration():
    """Demonstrate basic integration functionality"""
    print("\n=== Basic Integration Demo ===")
    
    # Create integration manager with custom config
    config = {
        'max_workers': 5,
        'execution_interval': 0.5,
        'strategies': {
            'momentum_strategy': {
                'module_path': '__main__',  # Use current module
                'class_name': 'ExampleStrategy',
                'priority': 'HIGH',
                'weight': 1.5,
                'enabled': True,
                'feature_names': ['trend_strength', 'volume_ratio', 'price_momentum'],
                'signal_names': ['main_signal', 'risk_signal'],
                'config_params': {'lookback': 20}
            },
            'volume_strategy': {
                'module_path': '__main__',
                'class_name': 'ExampleStrategy', 
                'priority': 'MEDIUM',
                'weight': 1.2,
                'enabled': True,
                'feature_names': ['volume_ratio', 'volatility', 'support_resistance'],
                'signal_names': ['main_signal'],
                'config_params': {'threshold': 0.8}
            }
        }
    }
    
    manager = StrategyIntegrationManager(config)
    
    try:
        # Manually add strategy instances for demo
        manager.strategy_instances['momentum_strategy'] = ExampleStrategy('momentum_strategy')
        manager.strategy_instances['volume_strategy'] = ExampleStrategy('volume_strategy')
        
        await manager.start()
        
        print("Integration manager started successfully")
        
        # Let it run for a while
        await asyncio.sleep(5)
        
        # Get current features and signals
        features = manager.get_aggregated_features()
        signals = manager.get_processed_signals()
        
        print(f"Aggregated features ({len(features)}): {list(features.keys())[:5]}...")
        print(f"Processed signals ({len(signals)}): {list(signals.keys())[:3]}...")
        
        # Get system metrics
        metrics = manager.get_system_metrics()
        print(f"Total executions: {metrics['total_executions']}")
        print(f"Active strategies: {metrics['active_strategies']}")
        print(f"Avg execution time: {metrics['avg_execution_time']:.3f}s")
        
    finally:
        await manager.stop()


async def demo_feature_aggregation():
    """Demonstrate feature aggregation capabilities"""
    print("\n=== Feature Aggregation Demo ===")
    
    aggregator = FeatureAggregator({
        'max_features': 20,
        'normalization_method': 'robust',
        'enable_feature_selection': True,
        'correlation_threshold': 0.9
    })
    
    try:
        await aggregator.start()
        
        # Simulate multiple feature updates
        for i in range(10):
            features = {
                f'strategy_a_feature_{j}': np.random.normal(0, 1) for j in range(5)
            }
            features.update({
                f'strategy_b_feature_{j}': np.random.normal(0.5, 0.8) for j in range(3)
            })
            
            # Add target for supervised learning
            target = sum(features.values()) * 0.1 + np.random.normal(0, 0.1)
            
            await aggregator.add_features(features, target)
            await asyncio.sleep(0.1)
        
        # Get aggregated results
        final_features = aggregator.get_aggregated_features()
        feature_vector = aggregator.get_feature_vector()
        importance_ranking = aggregator.get_feature_importance_ranking()
        
        print(f"Final aggregated features: {len(final_features)}")
        print(f"Feature vector shape: {feature_vector.shape}")
        print(f"Top 3 important features: {importance_ranking[:3]}")
        
        # Get metrics
        metrics = aggregator.get_metrics()
        print(f"Total updates: {metrics['stats']['update_count']}")
        print(f"Avg processing time: {metrics['stats']['aggregation_time']:.4f}s")
        
    finally:
        await aggregator.stop()


async def demo_signal_processing():
    """Demonstrate signal processing and conflict resolution"""
    print("\n=== Signal Processing Demo ===")
    
    processor = SignalProcessor({
        'conflict_resolution_method': 'weighted_average',
        'consensus_threshold': 0.6,
        'enable_conflict_detection': True
    })
    
    try:
        await processor.start()
        
        # Add conflicting signals
        conflicting_signals = {
            'strategy_a_buy_signal': {
                'type': 'buy',
                'strength': 3,
                'confidence': 0.8,
                'reasoning': 'Strong upward momentum detected'
            },
            'strategy_b_sell_signal': {
                'type': 'sell',
                'strength': 2,
                'confidence': 0.6,
                'reasoning': 'Resistance level reached'
            },
            'strategy_c_buy_signal': {
                'type': 'buy',
                'strength': 2,
                'confidence': 0.7,
                'reasoning': 'Volume breakout confirmation'
            }
        }
        
        await processor.add_signals(conflicting_signals)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Get processed results
        latest_signal = processor.get_latest_signal()
        if latest_signal:
            print(f"Aggregated signal: {latest_signal.signal_type.value}")
            print(f"Confidence: {latest_signal.confidence:.2f}")
            print(f"Consensus score: {latest_signal.consensus_score:.2f}")
            print(f"Contributing strategies: {latest_signal.contributing_strategies}")
            print(f"Reasoning: {latest_signal.reasoning}")
        
        # Get metrics
        metrics = processor.get_metrics()
        print(f"Signals processed: {metrics['stats']['signals_processed']}")
        print(f"Conflicts detected: {metrics['stats']['conflicts_detected']}")
        
    finally:
        await processor.stop()


async def demo_rl_connection():
    """Demonstrate RL environment connection"""
    print("\n=== RL Connection Demo ===")
    
    connector = RLConnector({
        'enable_websocket_server': False,  # Disable for demo
        'action_mode': 'HYBRID'
    })
    
    try:
        await connector.start()
        
        # Update features
        features = {
            'whale_sentiment': 0.75,
            'volume_poc': 30000,
            'order_spread': 0.001,
            'market_regime': 2.0,
            'institutional_flow': 0.3
        }
        
        await connector.update_features(features)
        
        # Request action from RL
        action_response = await connector.request_action()
        if action_response:
            print(f"RL Action: {action_response.action_type}")
            print(f"Confidence: {action_response.confidence:.2f}")
            print(f"Reasoning: {action_response.reasoning}")
        
        # Send reward feedback
        await connector.send_reward(0.02, step_id=1, components={
            'return': 0.015,
            'risk_penalty': -0.005,
            'transaction_cost': -0.002
        })
        
        # Get status and metrics
        status = connector.get_status()
        metrics = connector.get_metrics()
        
        print(f"Connection status: {status['status']}")
        print(f"Features sent: {metrics['connection_metrics']['feature_updates_sent']}")
        print(f"Actions received: {metrics['connection_metrics']['actions_received']}")
        
    finally:
        await connector.stop()


async def demo_performance_tracking():
    """Demonstrate performance tracking and A/B testing"""
    print("\n=== Performance Tracking Demo ===")
    
    tracker = PerformanceTracker({
        'enable_alerting': True,
        'enable_ab_testing': True,
        'performance_thresholds': {
            'strategy_accuracy_min': 0.6,
            'strategy_latency_max': 0.5
        }
    })
    
    try:
        await tracker.start()
        
        # Simulate strategy performance updates
        strategies = ['momentum_strategy', 'volume_strategy', 'whale_strategy']
        
        for i in range(5):
            for strategy in strategies:
                # Simulate performance metrics
                accuracy = np.random.uniform(0.5, 0.8)
                latency = np.random.uniform(0.01, 0.3)
                
                await tracker.update_strategy_performance(strategy, {
                    'total_signals': 20 + i * 5,
                    'successful_signals': int((20 + i * 5) * accuracy),
                    'avg_latency': latency,
                    'uptime_ratio': np.random.uniform(0.95, 1.0),
                    'returns': [np.random.normal(0.001, 0.02) for _ in range(5)]
                })
            
            await asyncio.sleep(0.5)
        
        # Start A/B test
        test_id = await tracker.start_ab_test(
            'momentum_strategy', 
            'volume_strategy', 
            PerformanceMetric.ACCURACY
        )
        print(f"Started A/B test: {test_id}")
        
        # Wait and end test
        await asyncio.sleep(1)
        result = await tracker.end_ab_test(test_id)
        
        if result:
            print(f"A/B test result - Winner: {result.winner}")
            print(f"Statistical significance: {result.statistical_significance:.3f}")
        
        # Get performance summary
        all_performances = tracker.get_all_strategy_performances()
        for name, perf in all_performances.items():
            print(f"{name}: accuracy={perf.signal_accuracy:.2f}, "
                  f"latency={perf.avg_latency:.3f}s, "
                  f"sharpe={perf.sharpe_ratio:.2f}")
        
        # Check for alerts
        alerts = tracker.get_active_alerts()
        if alerts:
            print(f"Active alerts: {len(alerts)}")
            for alert in alerts[:2]:  # Show first 2
                print(f"  - {alert.message} (severity: {alert.severity.value})")
        
    finally:
        await tracker.stop()


async def demo_full_integration():
    """Demonstrate full end-to-end integration"""
    print("\n=== Full Integration Demo ===")
    
    # Custom config for full integration
    config = {
        'max_workers': 8,
        'execution_interval': 1.0,
        'enable_monitoring': True,
        'strategies': {
            'momentum_strategy': {
                'module_path': '__main__',
                'class_name': 'ExampleStrategy',
                'priority': 'HIGH',
                'weight': 1.5,
                'enabled': True,
                'feature_names': ['trend_strength', 'volume_ratio'],
                'signal_names': ['main_signal']
            },
            'volume_strategy': {
                'module_path': '__main__',
                'class_name': 'ExampleStrategy',
                'priority': 'MEDIUM', 
                'weight': 1.2,
                'enabled': True,
                'feature_names': ['volume_ratio', 'volatility'],
                'signal_names': ['main_signal']
            },
            'whale_strategy': {
                'module_path': '__main__',
                'class_name': 'ExampleStrategy',
                'priority': 'HIGH',
                'weight': 1.8,
                'enabled': True,
                'feature_names': ['whale_sentiment', 'large_transfers'],
                'signal_names': ['whale_signal']
            }
        },
        'feature_config': {
            'max_features': 30,
            'normalization_method': 'robust',
            'enable_feature_selection': True
        },
        'signal_config': {
            'conflict_resolution_method': 'weighted_average',
            'consensus_threshold': 0.7
        },
        'rl_config': {
            'enable_websocket_server': False
        },
        'performance_config': {
            'enable_ab_testing': True,
            'monitoring_interval': 5.0
        }
    }
    
    manager = StrategyIntegrationManager(config)
    
    # Add strategy instances
    manager.strategy_instances['momentum_strategy'] = ExampleStrategy('momentum_strategy')
    manager.strategy_instances['volume_strategy'] = ExampleStrategy('volume_strategy') 
    manager.strategy_instances['whale_strategy'] = ExampleStrategy('whale_strategy')
    
    try:
        await manager.start()
        print("Full integration system started")
        
        # Run for a period to collect data
        print("Running integration for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get comprehensive metrics
        system_metrics = manager.get_system_metrics()
        
        print(f"\n=== Final System Status ===")
        print(f"Total strategies: {system_metrics['total_strategies']}")
        print(f"Active strategies: {system_metrics['active_strategies']}")
        print(f"Total features: {system_metrics['total_features']}")
        print(f"Total signals: {system_metrics['total_signals']}")
        print(f"Avg execution time: {system_metrics['avg_execution_time']:.3f}s")
        print(f"Error rate: {system_metrics['error_rate']:.3%}")
        
        # Show strategy details
        print(f"\n=== Strategy Performance ===")
        for name, strategy_info in system_metrics['strategies'].items():
            if strategy_info:
                print(f"{name}:")
                print(f"  Features: {len(strategy_info.get('features', {}))}")
                print(f"  Signals: {len(strategy_info.get('signals', {}))}")
                print(f"  Execution time: {strategy_info.get('execution_time', 0):.3f}s")
                print(f"  Success rate: {strategy_info.get('success_rate', 0):.2%}")
        
        # Test dynamic strategy management
        print(f"\n=== Testing Dynamic Management ===")
        
        # Disable a strategy
        success = manager.disable_strategy('volume_strategy')
        print(f"Disabled volume_strategy: {success}")
        
        await asyncio.sleep(2)
        
        # Update strategy weight
        success = manager.update_strategy_weight('whale_strategy', 2.5)
        print(f"Updated whale_strategy weight: {success}")
        
        await asyncio.sleep(2)
        
        # Re-enable strategy
        success = manager.enable_strategy('volume_strategy')
        print(f"Re-enabled volume_strategy: {success}")
        
        await asyncio.sleep(2)
        
        # Final status
        final_metrics = manager.get_system_metrics()
        print(f"Final active strategies: {final_metrics['active_strategies']}")
        
    finally:
        await manager.stop()
        print("Full integration system stopped")


async def main():
    """Run all demonstrations"""
    print("Strategy Integration Layer Demonstrations")
    print("=" * 50)
    
    try:
        # Run individual component demos
        await demo_basic_integration()
        await demo_feature_aggregation()
        await demo_signal_processing()
        await demo_rl_connection()
        await demo_performance_tracking()
        
        # Run full integration demo
        await demo_full_integration()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    # Set up event loop and run demonstrations
    asyncio.run(main())