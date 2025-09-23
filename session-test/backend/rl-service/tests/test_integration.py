"""
Integration Tests for RL Service Connections
Tests integration between RL service and external systems:
- Data aggregation service
- ML service (AdaptiveThreshold)
- Trading API
- Database connections
- Monitoring systems
"""

import pytest
import asyncio
import requests
import json
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from sqlalchemy import create_engine, text
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.rl_service import RLService
from integration.data_connector import DataConnector
from integration.trading_bridge import TradingBridge
from integration.api_routes import create_app
from integration.monitoring import MonitoringService
from rl_config import RLConfig

logger = logging.getLogger(__name__)


class TestDataIntegration:
    """Test data pipeline integrations"""

    @pytest.fixture
    def data_config(self):
        """Configuration for data integration tests"""
        config = RLConfig()
        config.data.price_data_source = 'alpaca'
        config.data.sentiment_data_source = 'groq'
        config.data.alternative_data_sources = ['coinglass', 'bitquery']
        return config

    @pytest.fixture
    def mock_data_responses(self):
        """Mock responses from external data sources"""
        return {
            'alpaca': {
                'bars': [{
                    'c': 50000, 'h': 51000, 'l': 49000, 'o': 49500,
                    'v': 1000000, 't': '2024-08-15T12:00:00Z'
                }]
            },
            'groq': {
                'sentiment_score': 0.2,
                'fear_greed_index': 55,
                'news_sentiment': 0.1
            },
            'coinglass': {
                'funding_rate': 0.0001,
                'open_interest': 25000000000,
                'long_short_ratio': 1.2
            },
            'bitquery': {
                'whale_activity': 0.8,
                'exchange_inflows': 500000000,
                'network_activity': 150000
            }
        }

    def test_data_connector_initialization(self, data_config):
        """Test DataConnector initialization and configuration"""
        
        connector = DataConnector(config=data_config)
        
        # Test basic initialization
        assert connector.config == data_config
        assert hasattr(connector, 'price_client')
        assert hasattr(connector, 'sentiment_client')
        assert hasattr(connector, 'alternative_clients')
        
        # Test client configuration
        assert connector.price_data_source == 'alpaca'
        assert connector.sentiment_data_source == 'groq'
        assert 'coinglass' in connector.alternative_data_sources
        
        logger.info("DataConnector initialization test passed")

    @patch('requests.get')
    def test_price_data_fetching(self, mock_get, data_config, mock_data_responses):
        """Test price data fetching from external APIs"""
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = mock_data_responses['alpaca']
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        connector = DataConnector(config=data_config)
        
        # Test data fetching
        symbol = 'BTC/USD'
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        price_data = connector.fetch_price_data(symbol, start_time, end_time)
        
        # Validate response
        assert price_data is not None
        assert isinstance(price_data, dict)
        assert 'timestamp' in price_data
        assert 'ohlcv' in price_data
        
        # Test API call was made correctly
        mock_get.assert_called_once()
        
        logger.info("Price data fetching test passed")

    @patch('requests.post')
    def test_sentiment_data_fetching(self, mock_post, data_config, mock_data_responses):
        """Test sentiment data fetching from Groq/ML services"""
        
        # Mock Groq API response
        mock_response = Mock()
        mock_response.json.return_value = mock_data_responses['groq']
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        connector = DataConnector(config=data_config)
        
        # Test sentiment data fetching
        symbol = 'BTC/USD'
        sentiment_data = connector.fetch_sentiment_data(symbol)
        
        # Validate response
        assert sentiment_data is not None
        assert 'sentiment_score' in sentiment_data
        assert 'fear_greed_index' in sentiment_data
        assert -1 <= sentiment_data['sentiment_score'] <= 1
        assert 0 <= sentiment_data['fear_greed_index'] <= 100
        
        logger.info("Sentiment data fetching test passed")

    @patch('requests.get')
    def test_alternative_data_integration(self, mock_get, data_config, mock_data_responses):
        """Test integration with alternative data sources"""
        
        def mock_response_factory(url):
            """Create appropriate mock response based on URL"""
            mock_response = Mock()
            mock_response.status_code = 200
            
            if 'coinglass' in url:
                mock_response.json.return_value = mock_data_responses['coinglass']
            elif 'bitquery' in url:
                mock_response.json.return_value = mock_data_responses['bitquery']
            else:
                mock_response.json.return_value = {}
            
            return mock_response
        
        mock_get.side_effect = lambda url, **kwargs: mock_response_factory(url)
        
        connector = DataConnector(config=data_config)
        
        # Test alternative data fetching
        symbol = 'BTC/USD'
        alt_data = connector.fetch_alternative_data(symbol)
        
        # Validate response structure
        assert alt_data is not None
        assert isinstance(alt_data, dict)
        
        # Check for expected data sources
        expected_sources = ['coinglass', 'bitquery']
        for source in expected_sources:
            assert source in alt_data or any(source in str(v) for v in alt_data.values())
        
        logger.info("Alternative data integration test passed")

    def test_data_aggregation_workflow(self, data_config):
        """Test complete data aggregation workflow"""
        
        with patch.multiple(
            'integration.data_connector.DataConnector',
            fetch_price_data=Mock(return_value={'timestamp': datetime.now(), 'ohlcv': {'close': 50000}}),
            fetch_sentiment_data=Mock(return_value={'sentiment_score': 0.1, 'fear_greed_index': 55}),
            fetch_alternative_data=Mock(return_value={'funding_rate': 0.0001, 'open_interest': 25000000000})
        ):
            connector = DataConnector(config=data_config)
            
            # Test complete data fetch
            symbol = 'BTC/USD'
            complete_data = connector.fetch_complete_dataset(symbol)
            
            # Validate complete dataset structure
            assert 'price_data' in complete_data
            assert 'sentiment_data' in complete_data
            assert 'alternative_data' in complete_data
            assert 'timestamp' in complete_data
            
            # Validate data consistency
            assert complete_data['timestamp'] is not None
            assert isinstance(complete_data['price_data'], dict)
            assert isinstance(complete_data['sentiment_data'], dict)
            assert isinstance(complete_data['alternative_data'], dict)
        
        logger.info("Data aggregation workflow test passed")

    def test_data_caching_mechanism(self, data_config):
        """Test data caching for performance optimization"""
        
        connector = DataConnector(config=data_config)
        
        with patch.object(connector, '_fetch_from_api') as mock_fetch:
            mock_fetch.return_value = {'test': 'data', 'timestamp': datetime.now()}
            
            # First call should hit the API
            data1 = connector.fetch_cached_data('BTC/USD', 'price')
            assert mock_fetch.call_count == 1
            
            # Second call within cache window should use cache
            data2 = connector.fetch_cached_data('BTC/USD', 'price')
            assert mock_fetch.call_count == 1  # No additional call
            assert data1 == data2
            
            # Test cache expiration
            connector.cache_expiry = timedelta(seconds=1)
            time.sleep(1.1)
            
            data3 = connector.fetch_cached_data('BTC/USD', 'price')
            assert mock_fetch.call_count == 2  # New API call after expiration
        
        logger.info("Data caching mechanism test passed")


class TestMLServiceIntegration:
    """Test integration with ML service (AdaptiveThreshold)"""

    @pytest.fixture
    def ml_service_config(self):
        """Configuration for ML service integration"""
        config = RLConfig()
        config.model.algorithm = 'PPO'
        return config

    def test_adaptive_threshold_connection(self, ml_service_config):
        """Test connection to AdaptiveThreshold ML service"""
        
        # Mock the ML service API
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                'thresholds': {
                    'rsi_threshold': 70.0,
                    'confidence_threshold': 0.75,
                    'momentum_threshold': 2.0
                },
                'status': 'success'
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            rl_service = RLService(config=ml_service_config)
            
            # Test threshold retrieval
            thresholds = rl_service.get_adaptive_thresholds('user_123', 'BTC/USD')
            
            assert thresholds is not None
            assert 'rsi_threshold' in thresholds
            assert 'confidence_threshold' in thresholds
            assert isinstance(thresholds['rsi_threshold'], (int, float))
        
        logger.info("AdaptiveThreshold connection test passed")

    def test_threshold_adaptation_integration(self, ml_service_config):
        """Test threshold adaptation integration with RL system"""
        
        # Mock threshold updates
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                'updates': [
                    {
                        'parameter': 'rsi_threshold',
                        'old_value': 70.0,
                        'new_value': 68.0,
                        'reason': 'performance_improvement',
                        'confidence': 0.85
                    }
                ],
                'status': 'success'
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            rl_service = RLService(config=ml_service_config)
            
            # Test threshold adaptation
            performance_data = {
                'total_return': 0.05,
                'sharpe_ratio': 1.8,
                'win_rate': 0.65,
                'max_drawdown': 0.08
            }
            
            updates = rl_service.trigger_threshold_adaptation('user_123', performance_data)
            
            assert updates is not None
            assert len(updates) > 0
            assert 'parameter' in updates[0]
            assert 'new_value' in updates[0]
        
        logger.info("Threshold adaptation integration test passed")

    def test_ml_service_fallback_mechanism(self, ml_service_config):
        """Test fallback when ML service is unavailable"""
        
        # Simulate ML service failure
        with patch('requests.post', side_effect=requests.exceptions.ConnectionError()):
            rl_service = RLService(config=ml_service_config)
            
            # Should use default thresholds when service fails
            thresholds = rl_service.get_adaptive_thresholds('user_123', 'BTC/USD')
            
            assert thresholds is not None
            assert isinstance(thresholds, dict)
            
            # Should contain default values
            expected_defaults = ['rsi_threshold', 'confidence_threshold', 'momentum_threshold']
            for key in expected_defaults:
                assert key in thresholds
                assert isinstance(thresholds[key], (int, float))
        
        logger.info("ML service fallback mechanism test passed")


class TestTradingAPIIntegration:
    """Test integration with external trading APIs"""

    @pytest.fixture
    def trading_config(self):
        """Configuration for trading API integration"""
        config = RLConfig()
        config.env.enable_live_trading = False  # Paper trading for tests
        config.env.paper_trading = True
        return config

    def test_trading_bridge_initialization(self, trading_config):
        """Test TradingBridge initialization with mock API"""
        
        mock_api = Mock()
        mock_api.get_account.return_value = {'buying_power': 10000.0, 'equity': 10000.0}
        
        bridge = TradingBridge(config=trading_config, trading_api=mock_api)
        
        assert bridge.config == trading_config
        assert bridge.trading_api == mock_api
        assert bridge.paper_trading == True
        
        # Test account validation
        account_valid = bridge.validate_account()
        assert account_valid == True
        
        logger.info("TradingBridge initialization test passed")

    def test_order_placement_integration(self, trading_config):
        """Test order placement through trading API"""
        
        mock_api = Mock()
        mock_api.place_order.return_value = {
            'order_id': 'test_order_123',
            'status': 'pending',
            'symbol': 'BTCUSD',
            'qty': 0.1,
            'side': 'buy'
        }
        
        bridge = TradingBridge(config=trading_config, trading_api=mock_api)
        
        # Test signal execution
        signal = {
            'action': 'BUY_20',
            'symbol': 'BTC/USD',
            'confidence': 0.8,
            'reasoning': 'Strong bullish signal'
        }
        
        order_result = bridge.execute_signal(signal)
        
        assert order_result is not None
        assert 'order_id' in order_result
        assert order_result['status'] in ['pending', 'filled']
        
        # Verify API was called correctly
        mock_api.place_order.assert_called_once()
        
        logger.info("Order placement integration test passed")

    def test_order_monitoring_workflow(self, trading_config):
        """Test order monitoring and status updates"""
        
        mock_api = Mock()
        
        # Mock order lifecycle
        order_id = 'test_order_123'
        mock_api.get_order.side_effect = [
            {'order_id': order_id, 'status': 'pending', 'filled_qty': 0},
            {'order_id': order_id, 'status': 'partially_filled', 'filled_qty': 0.05},
            {'order_id': order_id, 'status': 'filled', 'filled_qty': 0.1}
        ]
        
        bridge = TradingBridge(config=trading_config, trading_api=mock_api)
        
        # Test order monitoring
        order_updates = []
        for _ in range(3):
            status = bridge.monitor_order(order_id)
            order_updates.append(status)
        
        # Verify order progression
        assert order_updates[0]['status'] == 'pending'
        assert order_updates[1]['status'] == 'partially_filled'
        assert order_updates[2]['status'] == 'filled'
        
        # Verify monitoring calls
        assert mock_api.get_order.call_count == 3
        
        logger.info("Order monitoring workflow test passed")

    def test_risk_management_integration(self, trading_config):
        """Test risk management checks during order placement"""
        
        mock_api = Mock()
        mock_api.get_account.return_value = {
            'buying_power': 1000.0,  # Limited buying power
            'equity': 1000.0
        }
        
        bridge = TradingBridge(config=trading_config, trading_api=mock_api)
        
        # Test position sizing limits
        large_signal = {
            'action': 'BUY_100',  # 100% position
            'symbol': 'BTC/USD',
            'confidence': 0.9
        }
        
        # Should adjust position size based on available capital
        risk_check = bridge.validate_signal(large_signal)
        assert isinstance(risk_check, bool)
        
        # Test maximum position constraints
        bridge.max_position_size = 0.5  # 50% max position
        adjusted_signal = bridge.apply_risk_management(large_signal)
        
        # Position should be capped at maximum
        assert 'adjusted_position_size' in adjusted_signal
        assert adjusted_signal['adjusted_position_size'] <= 0.5
        
        logger.info("Risk management integration test passed")


class TestDatabaseIntegration:
    """Test database connections and operations"""

    @pytest.fixture
    def db_config(self):
        """Database configuration for testing"""
        config = RLConfig()
        # Use test database URL
        config.database_url = "sqlite:///:memory:"  # In-memory SQLite for testing
        return config

    def test_database_connection(self, db_config):
        """Test database connection and basic operations"""
        
        from sqlalchemy import create_engine, text
        
        # Test connection
        engine = create_engine(db_config.database_url)
        
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT 1 as test_value"))
            row = result.fetchone()
            assert row[0] == 1
        
        logger.info("Database connection test passed")

    def test_trade_logging_integration(self, db_config):
        """Test trade logging to database"""
        
        rl_service = RLService(config=db_config)
        
        # Mock database operations
        with patch('sqlalchemy.engine.Engine.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Test trade logging
            trade_data = {
                'symbol': 'BTC/USD',
                'action': 'BUY',
                'quantity': 0.1,
                'price': 50000.0,
                'timestamp': datetime.now(),
                'confidence': 0.8
            }
            
            success = rl_service.log_trade(trade_data)
            assert success == True
            
            # Verify database interaction
            mock_conn.execute.assert_called()
        
        logger.info("Trade logging integration test passed")

    def test_performance_metrics_storage(self, db_config):
        """Test storage of performance metrics in database"""
        
        rl_service = RLService(config=db_config)
        
        with patch('sqlalchemy.engine.Engine.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            # Test metrics storage
            metrics = {
                'timestamp': datetime.now(),
                'total_return': 0.05,
                'sharpe_ratio': 1.6,
                'max_drawdown': 0.08,
                'win_rate': 0.62,
                'total_trades': 150
            }
            
            success = rl_service.store_performance_metrics(metrics)
            assert success == True
            
            # Verify storage operation
            mock_conn.execute.assert_called()
        
        logger.info("Performance metrics storage test passed")


class TestMonitoringIntegration:
    """Test monitoring and alerting integrations"""

    @pytest.fixture
    def monitoring_config(self):
        """Configuration for monitoring integration"""
        config = RLConfig()
        config.monitoring = {
            'enable_alerts': True,
            'alert_thresholds': {
                'max_drawdown': 0.10,
                'min_sharpe': 1.0,
                'min_win_rate': 0.55
            }
        }
        return config

    def test_monitoring_service_initialization(self, monitoring_config):
        """Test MonitoringService initialization"""
        
        monitoring = MonitoringService(config=monitoring_config)
        
        assert monitoring.config == monitoring_config
        assert hasattr(monitoring, 'alert_thresholds')
        assert monitoring.alerts_enabled == True
        
        logger.info("MonitoringService initialization test passed")

    def test_real_time_metrics_ingestion(self, monitoring_config):
        """Test real-time metrics ingestion and processing"""
        
        monitoring = MonitoringService(config=monitoring_config)
        
        # Test metrics ingestion
        test_metrics = {
            'timestamp': datetime.now(),
            'portfolio_value': 10500.0,
            'total_return': 0.05,
            'sharpe_ratio': 1.8,
            'drawdown': 0.03,
            'win_rate': 0.65
        }
        
        result = monitoring.ingest_metrics(test_metrics)
        assert result['status'] == 'success'
        assert 'metrics_id' in result
        
        # Test metrics retrieval
        retrieved_metrics = monitoring.get_recent_metrics(hours=1)
        assert len(retrieved_metrics) > 0
        assert retrieved_metrics[0]['portfolio_value'] == 10500.0
        
        logger.info("Real-time metrics ingestion test passed")

    def test_alert_triggering_mechanism(self, monitoring_config):
        """Test alert triggering for performance degradation"""
        
        monitoring = MonitoringService(config=monitoring_config)
        
        # Test normal performance (no alerts)
        normal_metrics = {
            'timestamp': datetime.now(),
            'sharpe_ratio': 1.8,
            'drawdown': 0.05,
            'win_rate': 0.65
        }
        
        alerts = monitoring.check_alerts(normal_metrics)
        assert len(alerts) == 0
        
        # Test degraded performance (should trigger alerts)
        degraded_metrics = {
            'timestamp': datetime.now(),
            'sharpe_ratio': 0.8,  # Below threshold
            'drawdown': 0.12,     # Above threshold
            'win_rate': 0.50      # Below threshold
        }
        
        alerts = monitoring.check_alerts(degraded_metrics)
        assert len(alerts) > 0
        
        # Validate alert structure
        for alert in alerts:
            assert 'severity' in alert
            assert 'message' in alert
            assert 'metric' in alert
            assert 'threshold' in alert
        
        logger.info("Alert triggering mechanism test passed")

    def test_notification_system_integration(self, monitoring_config):
        """Test integration with notification systems"""
        
        monitoring = MonitoringService(config=monitoring_config)
        
        # Mock notification services
        with patch.multiple(
            monitoring,
            send_email_alert=Mock(return_value=True),
            send_slack_alert=Mock(return_value=True),
            send_webhook_alert=Mock(return_value=True)
        ):
            # Test alert notification
            alert = {
                'severity': 'high',
                'message': 'Maximum drawdown exceeded',
                'metric': 'drawdown',
                'value': 0.12,
                'threshold': 0.10
            }
            
            notification_result = monitoring.send_notifications([alert])
            
            assert notification_result['email_sent'] == True
            assert notification_result['slack_sent'] == True
            assert notification_result['webhook_sent'] == True
            
            # Verify notification calls
            monitoring.send_email_alert.assert_called_once()
            monitoring.send_slack_alert.assert_called_once()
            monitoring.send_webhook_alert.assert_called_once()
        
        logger.info("Notification system integration test passed")


class TestAPIEndpointIntegration:
    """Test RL service API endpoints integration"""

    @pytest.fixture
    def api_config(self):
        """Configuration for API testing"""
        config = RLConfig()
        config.api = {
            'host': '127.0.0.1',
            'port': 8080,
            'debug': True
        }
        return config

    @pytest.fixture
    def api_client(self, api_config):
        """Create test client for API testing"""
        app = create_app(config=api_config)
        app.config['TESTING'] = True
        return app.test_client()

    def test_health_check_endpoint(self, api_client):
        """Test health check API endpoint"""
        
        response = api_client.get('/health')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
        
        logger.info("Health check endpoint test passed")

    def test_prediction_endpoint(self, api_client):
        """Test prediction API endpoint"""
        
        # Test prediction request
        request_data = {
            'symbol': 'BTC/USD',
            'market_data': {
                'price': 50000.0,
                'volume': 1000000,
                'rsi': 65.0,
                'macd': 100.0
            },
            'sentiment_data': {
                'fear_greed_index': 55,
                'sentiment_score': 0.1
            }
        }
        
        response = api_client.post(
            '/predict',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'action' in data
        assert 'confidence' in data
        assert 'reasoning' in data
        assert data['confidence'] >= 0 and data['confidence'] <= 1
        
        logger.info("Prediction endpoint test passed")

    def test_training_endpoint(self, api_client):
        """Test model training API endpoint"""
        
        request_data = {
            'user_id': 'test_user_123',
            'training_params': {
                'timesteps': 1000,
                'learning_rate': 0.0003,
                'batch_size': 64
            }
        }
        
        response = api_client.post(
            '/train',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'training_id' in data
        assert 'status' in data
        assert data['status'] in ['started', 'queued']
        
        logger.info("Training endpoint test passed")

    def test_metrics_endpoint(self, api_client):
        """Test metrics retrieval API endpoint"""
        
        response = api_client.get('/metrics/test_user_123')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'performance_metrics' in data
        assert 'portfolio_metrics' in data
        assert 'system_metrics' in data
        
        logger.info("Metrics endpoint test passed")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])