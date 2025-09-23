"""
Comprehensive Test Suite for AdaptiveThreshold System
Tests all aspects of the adaptive threshold functionality including performance tracking,
parameter adaptation, database operations, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import asdict
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

from adaptive_threshold import (
    AdaptiveThreshold,
    AdaptiveThresholdManager,
    PerformanceMetrics,
    ThresholdUpdate,
    threshold_manager
)


@pytest.fixture
def mock_db_engine():
    """Mock SQLAlchemy engine for database operations"""
    engine = Mock()
    connection = Mock()
    result = Mock()
    
    # Mock successful database operations
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=None)
    connection.execute.return_value = result
    connection.commit.return_value = None
    
    engine.connect.return_value = connection
    return engine


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing"""
    return [
        (10.5, 0.05, 1.5, datetime.now()),  # pnl, pnl_percent, hold_duration, closed_at
        (-5.2, -0.02, 2.0, datetime.now() - timedelta(days=1)),
        (15.8, 0.08, 0.5, datetime.now() - timedelta(days=2)),
        (-8.3, -0.03, 1.2, datetime.now() - timedelta(days=3)),
        (22.1, 0.11, 3.0, datetime.now() - timedelta(days=4)),
    ]


@pytest.fixture
def adaptive_threshold():
    """Create AdaptiveThreshold instance with mocked database"""
    with patch('adaptive_threshold.create_engine') as mock_create_engine:
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock successful database operations
        mock_connection = Mock()
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        mock_connection.execute.return_value = Mock(fetchall=Mock(return_value=[]))
        mock_connection.commit.return_value = None
        mock_engine.connect.return_value = mock_connection
        
        threshold = AdaptiveThreshold(user_id="test_user_123", symbol="BTCUSD")
        return threshold


class TestAdaptiveThreshold:
    """Test cases for AdaptiveThreshold class"""
    
    def test_initialization(self, adaptive_threshold):
        """Test proper initialization of AdaptiveThreshold"""
        assert adaptive_threshold.user_id == "test_user_123"
        assert adaptive_threshold.symbol == "BTCUSD"
        assert adaptive_threshold.learning_rate == 0.01
        assert adaptive_threshold.performance_window == 100
        assert adaptive_threshold.min_trades_for_adaptation == 10
        
        # Check default parameters
        assert "rsi_threshold" in adaptive_threshold.parameters
        assert "confidence_threshold" in adaptive_threshold.parameters
        assert adaptive_threshold.parameters["rsi_threshold"] == 70.0
        assert adaptive_threshold.parameters["confidence_threshold"] == 0.75
    
    def test_parameter_bounds(self, adaptive_threshold):
        """Test parameter bounds are properly defined"""
        for param_name in adaptive_threshold.parameters.keys():
            assert param_name in adaptive_threshold.adaptation_bounds
            min_val, max_val = adaptive_threshold.adaptation_bounds[param_name]
            assert min_val < max_val
            assert min_val <= adaptive_threshold.parameters[param_name] <= max_val
    
    def test_performance_metrics_calculation(self, adaptive_threshold, sample_performance_data):
        """Test performance metrics calculation"""
        with patch.object(adaptive_threshold.db_engine, 'connect') as mock_connect:
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            
            # Mock database result
            mock_result = Mock()
            mock_result.fetchall.return_value = [
                Mock(pnl_percent=0.05, pnl=10.5),
                Mock(pnl_percent=-0.02, pnl=-5.2),
                Mock(pnl_percent=0.08, pnl=15.8),
                Mock(pnl_percent=-0.03, pnl=-8.3),
                Mock(pnl_percent=0.11, pnl=22.1),
            ]
            mock_connection.execute.return_value = mock_result
            mock_connect.return_value = mock_connection
            
            metrics = adaptive_threshold.get_performance_metrics(days_back=30)
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.total_return > 0  # Should be positive overall
            assert 0 <= metrics.win_rate <= 1
            assert metrics.volatility >= 0
            assert metrics.max_drawdown >= 0
    
    def test_performance_score_calculation(self, adaptive_threshold):
        """Test performance score calculation"""
        # Test with good performance metrics
        good_metrics = PerformanceMetrics(
            total_return=15.5,
            sharpe_ratio=1.2,
            win_rate=0.65,
            avg_trade_return=3.1,
            max_drawdown=5.2,
            volatility=2.1
        )
        score = adaptive_threshold.calculate_performance_score(good_metrics)
        assert 0.4 <= score <= 1.0  # Should be reasonably high
        
        # Test with poor performance metrics
        poor_metrics = PerformanceMetrics(
            total_return=-10.2,
            sharpe_ratio=-0.5,
            win_rate=0.35,
            avg_trade_return=-2.0,
            max_drawdown=15.0,
            volatility=8.5
        )
        score = adaptive_threshold.calculate_performance_score(poor_metrics)
        assert 0.0 <= score <= 0.6  # Should be lower
    
    def test_parameter_adaptation(self, adaptive_threshold):
        """Test parameter adaptation logic"""
        # Test RSI threshold adaptation
        good_metrics = PerformanceMetrics(
            total_return=10.0, sharpe_ratio=1.0, win_rate=0.7,
            avg_trade_return=2.0, max_drawdown=3.0, volatility=2.0
        )
        
        update = adaptive_threshold._adapt_parameter(
            "rsi_threshold", 70.0, 0.75, 0.1, good_metrics
        )
        
        if update:  # May not update if change is too small
            assert isinstance(update, ThresholdUpdate)
            assert update.parameter_name == "rsi_threshold"
            assert 50.0 <= update.new_value <= 90.0  # Within bounds
            assert update.confidence > 0
    
    def test_bounds_enforcement(self, adaptive_threshold):
        """Test that parameter updates respect bounds"""
        # Test with extreme performance to trigger large adaptations
        extreme_metrics = PerformanceMetrics(
            total_return=100.0, sharpe_ratio=5.0, win_rate=0.95,
            avg_trade_return=20.0, max_drawdown=1.0, volatility=1.0
        )
        
        # Try to adapt RSI threshold
        update = adaptive_threshold._adapt_parameter(
            "rsi_threshold", 70.0, 0.95, 0.5, extreme_metrics
        )
        
        if update:
            min_val, max_val = adaptive_threshold.adaptation_bounds["rsi_threshold"]
            assert min_val <= update.new_value <= max_val
    
    def test_signal_evaluation(self, adaptive_threshold):
        """Test signal evaluation against thresholds"""
        # Test signal that should pass all checks
        good_signal = {
            'confidence': 0.80,
            'rsi': 65,
            'change_percent': 3.5,
            'volume': 1500000000,
            'action': 'BUY'
        }
        assert adaptive_threshold.should_trade(good_signal) == True
        
        # Test signal with low confidence
        low_confidence_signal = {
            'confidence': 0.60,
            'rsi': 65,
            'change_percent': 3.5,
            'volume': 1500000000,
            'action': 'BUY'
        }
        assert adaptive_threshold.should_trade(low_confidence_signal) == False
        
        # Test signal with high RSI for BUY action
        high_rsi_signal = {
            'confidence': 0.80,
            'rsi': 85,
            'change_percent': 3.5,
            'volume': 1500000000,
            'action': 'BUY'
        }
        assert adaptive_threshold.should_trade(high_rsi_signal) == False
    
    def test_threshold_reset(self, adaptive_threshold):
        """Test threshold reset functionality"""
        # Modify some thresholds
        adaptive_threshold.parameters['rsi_threshold'] = 85.0
        adaptive_threshold.parameters['confidence_threshold'] = 0.90
        
        # Reset thresholds
        with patch.object(adaptive_threshold, '_save_threshold'):
            adaptive_threshold.reset_thresholds()
        
        # Check if reset to defaults
        assert adaptive_threshold.parameters['rsi_threshold'] == 70.0
        assert adaptive_threshold.parameters['confidence_threshold'] == 0.75
    
    @patch('adaptive_threshold.logger')
    def test_error_handling(self, mock_logger, adaptive_threshold):
        """Test error handling in database operations"""
        with patch.object(adaptive_threshold.db_engine, 'connect') as mock_connect:
            # Simulate database error
            mock_connect.side_effect = Exception("Database connection failed")
            
            metrics = adaptive_threshold.get_performance_metrics()
            
            # Should return default metrics on error
            assert metrics.total_return == 0
            assert metrics.win_rate == 0
            mock_logger.error.assert_called()


class TestAdaptiveThresholdManager:
    """Test cases for AdaptiveThresholdManager"""
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = AdaptiveThresholdManager()
        assert len(manager.instances) == 0
    
    def test_instance_creation_and_retrieval(self):
        """Test instance creation and retrieval"""
        manager = AdaptiveThresholdManager()
        
        with patch('adaptive_threshold.create_engine'):
            instance1 = manager.get_instance("user1", "BTCUSD")
            instance2 = manager.get_instance("user1", "BTCUSD")
            
            # Should return the same instance
            assert instance1 is instance2
            assert len(manager.instances) == 1
            
            # Different user should create new instance
            instance3 = manager.get_instance("user2", "BTCUSD")
            assert instance3 is not instance1
            assert len(manager.instances) == 2
    
    def test_adapt_all_users(self):
        """Test adapting thresholds for all users"""
        manager = AdaptiveThresholdManager()
        
        with patch('adaptive_threshold.create_engine'):
            # Create some instances
            manager.get_instance("user1", "BTCUSD")
            manager.get_instance("user2", "ETHUSD")
            
            # Mock adaptation methods
            for instance in manager.instances.values():
                instance.adapt_thresholds = Mock(return_value=[])
            
            results = manager.adapt_all_users()
            
            assert len(results) == 2
            assert all(isinstance(updates, list) for updates in results.values())


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation"""
        metrics = PerformanceMetrics(
            total_return=15.5,
            sharpe_ratio=1.2,
            win_rate=0.65,
            avg_trade_return=3.1,
            max_drawdown=5.2,
            volatility=2.1
        )
        
        assert metrics.total_return == 15.5
        assert metrics.sharpe_ratio == 1.2
        assert metrics.win_rate == 0.65
        assert metrics.avg_trade_return == 3.1
        assert metrics.max_drawdown == 5.2
        assert metrics.volatility == 2.1
    
    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics can be serialized"""
        metrics = PerformanceMetrics(
            total_return=15.5,
            sharpe_ratio=1.2,
            win_rate=0.65,
            avg_trade_return=3.1,
            max_drawdown=5.2,
            volatility=2.1
        )
        
        # Should be able to convert to dict
        metrics_dict = asdict(metrics)
        assert isinstance(metrics_dict, dict)
        assert len(metrics_dict) == 6


class TestThresholdUpdate:
    """Test cases for ThresholdUpdate dataclass"""
    
    def test_threshold_update_creation(self):
        """Test ThresholdUpdate creation"""
        update = ThresholdUpdate(
            parameter_name="rsi_threshold",
            old_value=70.0,
            new_value=72.5,
            reason="Performance improved",
            confidence=0.85
        )
        
        assert update.parameter_name == "rsi_threshold"
        assert update.old_value == 70.0
        assert update.new_value == 72.5
        assert update.reason == "Performance improved"
        assert update.confidence == 0.85


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    def test_complete_adaptation_cycle(self, adaptive_threshold):
        """Test a complete adaptation cycle"""
        with patch.object(adaptive_threshold.db_engine, 'connect') as mock_connect:
            # Mock database operations
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_connection.execute.return_value = Mock(fetchall=Mock(return_value=[
                Mock(pnl_percent=0.05, pnl=10.5),
                Mock(pnl_percent=0.08, pnl=15.8),
                Mock(pnl_percent=0.03, pnl=7.2),
                Mock(pnl_percent=0.06, pnl=12.1),
                Mock(pnl_percent=0.04, pnl=8.9),
                Mock(pnl_percent=0.09, pnl=18.3),
                Mock(pnl_percent=0.02, pnl=5.4),
                Mock(pnl_percent=0.07, pnl=14.6),
                Mock(pnl_percent=0.05, pnl=11.2),
                Mock(pnl_percent=0.08, pnl=16.7),
                Mock(pnl_percent=0.06, pnl=13.8),
            ]))
            mock_connection.commit.return_value = None
            mock_connect.return_value = mock_connection
            
            # Run adaptation
            updates = adaptive_threshold.adapt_thresholds()
            
            # Should have some updates for good performance
            assert isinstance(updates, list)
            # Performance history should be updated
            assert len(adaptive_threshold.performance_history) > 0
    
    def test_poor_performance_adaptation(self, adaptive_threshold):
        """Test adaptation with poor performance data"""
        with patch.object(adaptive_threshold.db_engine, 'connect') as mock_connect:
            # Mock poor performance data
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_connection.execute.return_value = Mock(fetchall=Mock(return_value=[
                Mock(pnl_percent=-0.05, pnl=-10.5),
                Mock(pnl_percent=-0.08, pnl=-15.8),
                Mock(pnl_percent=-0.03, pnl=-7.2),
                Mock(pnl_percent=-0.06, pnl=-12.1),
                Mock(pnl_percent=-0.04, pnl=-8.9),
                Mock(pnl_percent=-0.09, pnl=-18.3),
                Mock(pnl_percent=-0.02, pnl=-5.4),
                Mock(pnl_percent=-0.07, pnl=-14.6),
                Mock(pnl_percent=-0.05, pnl=-11.2),
                Mock(pnl_percent=-0.08, pnl=-16.7),
                Mock(pnl_percent=-0.06, pnl=-13.8),
            ]))
            mock_connection.commit.return_value = None
            mock_connect.return_value = mock_connection
            
            # Store initial thresholds
            initial_rsi = adaptive_threshold.parameters['rsi_threshold']
            
            # Run adaptation
            updates = adaptive_threshold.adapt_thresholds()
            
            # Thresholds should adapt to poor performance
            # (e.g., RSI threshold might increase to be more conservative)
            if updates:
                rsi_update = next((u for u in updates if u.parameter_name == 'rsi_threshold'), None)
                if rsi_update:
                    # For poor performance, RSI threshold should typically increase (more conservative)
                    # But this depends on the specific adaptation logic
                    assert rsi_update.new_value != initial_rsi


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_insufficient_data(self, adaptive_threshold):
        """Test behavior with insufficient trading data"""
        with patch.object(adaptive_threshold.db_engine, 'connect') as mock_connect:
            # Mock insufficient data (less than min_trades_for_adaptation)
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_connection.execute.return_value = Mock(fetchall=Mock(return_value=[
                Mock(pnl_percent=0.05, pnl=10.5),
                Mock(pnl_percent=-0.02, pnl=-5.2),
            ]))  # Only 2 trades, less than min_trades_for_adaptation (10)
            mock_connect.return_value = mock_connection
            
            metrics = adaptive_threshold.get_performance_metrics()
            
            # Should return zero metrics
            assert metrics.total_return == 0
            assert metrics.win_rate == 0
    
    def test_extreme_parameter_values(self, adaptive_threshold):
        """Test adaptation with parameters at extreme bounds"""
        # Set RSI threshold to minimum bound
        min_rsi, max_rsi = adaptive_threshold.adaptation_bounds['rsi_threshold']
        adaptive_threshold.parameters['rsi_threshold'] = min_rsi
        
        # Try to adapt with negative performance (should try to increase threshold)
        poor_metrics = PerformanceMetrics(
            total_return=-20.0, sharpe_ratio=-1.5, win_rate=0.2,
            avg_trade_return=-4.0, max_drawdown=25.0, volatility=10.0
        )
        
        update = adaptive_threshold._adapt_parameter(
            'rsi_threshold', min_rsi, 0.1, -0.2, poor_metrics
        )
        
        if update:
            # Should respect bounds
            assert min_rsi <= update.new_value <= max_rsi
    
    def test_null_symbol_handling(self):
        """Test handling of null symbol (global thresholds)"""
        with patch('adaptive_threshold.create_engine'):
            threshold = AdaptiveThreshold(user_id="test_user", symbol=None)
            assert threshold.symbol is None
            assert threshold.user_id == "test_user"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])