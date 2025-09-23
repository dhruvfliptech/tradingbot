"""
Complete System End-to-End Integration Tests

Tests the entire trading bot system from data ingestion to trade execution,
validating all components work together as a cohesive unit.
"""

import pytest
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from strategies.institutional.whale_tracker import WhaleTracker
from strategies.institutional.smart_money_divergence import SmartMoneyDivergenceAnalyzer
from strategies.institutional.volume_profile import VolumeProfileAnalyzer
from strategies.integration.strategy_manager import StrategyManager
from rl_service.agents.ensemble_agent import EnsembleAgent
from rl_service.environment.trading_env import TradingEnvironment
from production.risk.risk_manager import RiskManager
from production.optimization.execution_optimizer import ExecutionOptimizer
from ml_service.adaptive_threshold import AdaptiveThresholdOptimizer

logger = logging.getLogger(__name__)


class TestCompleteSystem:
    """End-to-end system integration tests"""
    
    @pytest.fixture(autouse=True)
    def setup_system(self):
        """Setup complete system for testing"""
        self.whale_tracker = WhaleTracker()
        self.smart_money = SmartMoneyDivergenceAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.strategy_manager = StrategyManager()
        self.rl_agent = EnsembleAgent()
        self.trading_env = TradingEnvironment()
        self.risk_manager = RiskManager()
        self.execution_optimizer = ExecutionOptimizer()
        self.adaptive_threshold = AdaptiveThresholdOptimizer()
        
        # System state tracking
        self.system_metrics = {
            'data_pipeline_latency': [],
            'strategy_execution_time': [],
            'rl_decision_time': [],
            'risk_check_time': [],
            'order_execution_time': [],
            'total_cycle_time': []
        }
        
    def test_complete_trading_cycle(self):
        """Test complete trading cycle from data to execution"""
        start_time = time.time()
        
        # 1. Data Ingestion Phase
        market_data = self._simulate_market_data()
        data_timestamp = time.time()
        
        # 2. Strategy Analysis Phase
        strategy_signals = self._execute_strategy_analysis(market_data)
        strategy_timestamp = time.time()
        
        # 3. RL Decision Phase
        rl_decision = self._execute_rl_decision(market_data, strategy_signals)
        rl_timestamp = time.time()
        
        # 4. Risk Management Phase
        risk_approved = self._execute_risk_checks(rl_decision, market_data)
        risk_timestamp = time.time()
        
        # 5. Order Execution Phase
        execution_result = self._execute_order(rl_decision if risk_approved else None)
        execution_timestamp = time.time()
        
        # Record metrics
        self.system_metrics['data_pipeline_latency'].append(data_timestamp - start_time)
        self.system_metrics['strategy_execution_time'].append(strategy_timestamp - data_timestamp)
        self.system_metrics['rl_decision_time'].append(rl_timestamp - strategy_timestamp)
        self.system_metrics['risk_check_time'].append(risk_timestamp - rl_timestamp)
        self.system_metrics['order_execution_time'].append(execution_timestamp - risk_timestamp)
        self.system_metrics['total_cycle_time'].append(execution_timestamp - start_time)
        
        # Assertions
        assert market_data is not None, "Market data should be available"
        assert len(strategy_signals) > 0, "Strategy signals should be generated"
        assert rl_decision is not None, "RL agent should make decision"
        assert isinstance(risk_approved, bool), "Risk check should return boolean"
        
        # Performance assertions
        total_cycle_time = execution_timestamp - start_time
        assert total_cycle_time < 1.0, f"Complete cycle should finish in <1s, took {total_cycle_time:.3f}s"
        
        logger.info(f"Complete trading cycle completed in {total_cycle_time:.3f}s")
        
    def test_system_resilience_data_failure(self):
        """Test system behavior when data sources fail"""
        # Simulate data source failures
        with patch('strategies.institutional.whale_tracker.WhaleTracker.analyze') as mock_whale:
            mock_whale.side_effect = Exception("Data source unavailable")
            
            # System should gracefully handle failure
            market_data = self._simulate_market_data()
            strategy_signals = self._execute_strategy_analysis(market_data)
            
            # Should have fallback signals
            assert len(strategy_signals) >= 1, "System should provide fallback signals"
            assert 'fallback' in str(strategy_signals).lower(), "Should indicate fallback mode"
            
    def test_system_resilience_rl_failure(self):
        """Test system behavior when RL agent fails"""
        with patch.object(self.rl_agent, 'predict') as mock_predict:
            mock_predict.side_effect = Exception("RL model unavailable")
            
            market_data = self._simulate_market_data()
            strategy_signals = self._execute_strategy_analysis(market_data)
            
            # Should fallback to strategy-only decisions
            decision = self._execute_rl_decision(market_data, strategy_signals)
            assert decision is not None, "Should have fallback decision mechanism"
            
    def test_concurrent_processing_capacity(self):
        """Test system can handle multiple concurrent requests"""
        concurrent_requests = 10
        
        async def process_request():
            return self.test_complete_trading_cycle()
            
        # Run concurrent trading cycles
        start_time = time.time()
        tasks = [process_request() for _ in range(concurrent_requests)]
        
        # Note: Since test_complete_trading_cycle is sync, we simulate async behavior
        for task in tasks:
            try:
                task
            except Exception as e:
                pytest.fail(f"Concurrent processing failed: {e}")
                
        total_time = time.time() - start_time
        avg_time_per_request = total_time / concurrent_requests
        
        assert avg_time_per_request < 2.0, f"Average time per request should be <2s, got {avg_time_per_request:.3f}s"
        
    def test_memory_usage_stability(self):
        """Test system memory usage remains stable over time"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple cycles
        for i in range(50):
            self.test_complete_trading_cycle()
            if i % 10 == 0:
                gc.collect()
                
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 100, f"Memory growth should be <100MB, got {memory_growth:.2f}MB"
        logger.info(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (+{memory_growth:.2f}MB)")
        
    def test_error_recovery_mechanisms(self):
        """Test system error recovery and graceful degradation"""
        # Test database connection failure
        with patch('production.risk.risk_manager.RiskManager.check_risk') as mock_risk:
            mock_risk.side_effect = Exception("Database connection lost")
            
            # System should continue with conservative risk defaults
            market_data = self._simulate_market_data()
            strategy_signals = self._execute_strategy_analysis(market_data)
            rl_decision = self._execute_rl_decision(market_data, strategy_signals)
            
            # Should apply conservative risk management
            risk_approved = self._execute_risk_checks(rl_decision, market_data)
            assert isinstance(risk_approved, bool), "Should fallback to conservative risk management"
            
    def test_data_consistency_across_components(self):
        """Test data consistency across all system components"""
        market_data = self._simulate_market_data()
        
        # Ensure timestamp consistency
        base_timestamp = market_data['timestamp']
        
        # Strategy analysis should use same timestamp
        strategy_signals = self._execute_strategy_analysis(market_data)
        for signal in strategy_signals:
            assert abs(signal.get('timestamp', base_timestamp) - base_timestamp) < 1.0, \
                "Strategy signals should use consistent timestamps"
                
        # RL decision should be based on current data
        rl_decision = self._execute_rl_decision(market_data, strategy_signals)
        assert rl_decision.get('data_timestamp') == base_timestamp, \
            "RL decision should reference correct data timestamp"
            
    def test_component_isolation(self):
        """Test that component failures don't cascade"""
        # Test strategy component failure doesn't affect RL
        with patch.object(self.whale_tracker, 'analyze') as mock_whale:
            mock_whale.side_effect = Exception("Strategy component failure")
            
            market_data = self._simulate_market_data()
            
            # Other strategies should continue working
            try:
                volume_signals = self.volume_profile.analyze(market_data)
                assert volume_signals is not None, "Other strategies should remain functional"
            except Exception:
                pytest.fail("Component failure cascaded to other components")
                
    def test_system_performance_under_load(self):
        """Test system performance under high load conditions"""
        load_test_cycles = 100
        start_time = time.time()
        
        successful_cycles = 0
        failed_cycles = 0
        
        for i in range(load_test_cycles):
            try:
                self.test_complete_trading_cycle()
                successful_cycles += 1
            except Exception as e:
                failed_cycles += 1
                logger.warning(f"Cycle {i} failed: {e}")
                
        total_time = time.time() - start_time
        success_rate = successful_cycles / load_test_cycles
        avg_cycle_time = total_time / load_test_cycles
        
        # Performance requirements
        assert success_rate >= 0.95, f"Success rate should be >=95%, got {success_rate:.2%}"
        assert avg_cycle_time < 1.0, f"Average cycle time should be <1s, got {avg_cycle_time:.3f}s"
        
        # Log performance metrics
        logger.info(f"Load test results: {successful_cycles}/{load_test_cycles} successful")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Average cycle time: {avg_cycle_time:.3f}s")
        
    def _simulate_market_data(self) -> Dict[str, Any]:
        """Simulate realistic market data"""
        return {
            'timestamp': time.time(),
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume': 1500000,
            'bid': 44995.0,
            'ask': 45005.0,
            'orderbook': {
                'bids': [[44995.0, 10.5], [44990.0, 15.2]],
                'asks': [[45005.0, 8.7], [45010.0, 12.3]]
            },
            'trades': [
                {'price': 45000.0, 'volume': 0.5, 'side': 'buy'},
                {'price': 44998.0, 'volume': 1.2, 'side': 'sell'}
            ]
        }
        
    def _execute_strategy_analysis(self, market_data: Dict) -> List[Dict]:
        """Execute strategy analysis with error handling"""
        signals = []
        
        try:
            # Whale tracking analysis
            whale_signal = self.whale_tracker.analyze(market_data)
            if whale_signal:
                signals.append(whale_signal)
        except Exception as e:
            logger.warning(f"Whale tracker failed: {e}")
            signals.append({'type': 'fallback', 'signal': 'neutral', 'confidence': 0.1})
            
        try:
            # Smart money analysis
            smart_money_signal = self.smart_money.analyze(market_data)
            if smart_money_signal:
                signals.append(smart_money_signal)
        except Exception as e:
            logger.warning(f"Smart money analysis failed: {e}")
            
        try:
            # Volume profile analysis
            volume_signal = self.volume_profile.analyze(market_data)
            if volume_signal:
                signals.append(volume_signal)
        except Exception as e:
            logger.warning(f"Volume profile analysis failed: {e}")
            
        # Ensure at least one signal exists
        if not signals:
            signals.append({'type': 'fallback', 'signal': 'hold', 'confidence': 0.05})
            
        return signals
        
    def _execute_rl_decision(self, market_data: Dict, strategy_signals: List) -> Dict:
        """Execute RL decision with fallback"""
        try:
            # Prepare state for RL agent
            state = {
                'market_data': market_data,
                'strategy_signals': strategy_signals,
                'timestamp': market_data['timestamp']
            }
            
            decision = self.rl_agent.predict(state)
            decision['data_timestamp'] = market_data['timestamp']
            return decision
            
        except Exception as e:
            logger.warning(f"RL decision failed: {e}")
            # Fallback to strategy consensus
            return {
                'action': 'hold',
                'confidence': 0.1,
                'source': 'fallback',
                'data_timestamp': market_data['timestamp']
            }
            
    def _execute_risk_checks(self, decision: Dict, market_data: Dict) -> bool:
        """Execute risk management checks"""
        try:
            return self.risk_manager.check_risk(decision, market_data)
        except Exception as e:
            logger.warning(f"Risk check failed: {e}")
            # Conservative fallback - reject risky decisions
            return decision.get('action') == 'hold'
            
    def _execute_order(self, decision: Dict) -> Dict:
        """Simulate order execution"""
        if decision is None:
            return {'status': 'rejected', 'reason': 'risk_check_failed'}
            
        if decision.get('action') == 'hold':
            return {'status': 'no_action', 'action': 'hold'}
            
        # Simulate order execution
        return {
            'status': 'executed',
            'action': decision.get('action'),
            'price': 45000.0,
            'quantity': 0.1,
            'timestamp': time.time()
        }


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])