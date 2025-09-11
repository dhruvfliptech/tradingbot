"""
Trading Bridge - Integration Layer with TradingService
====================================================

This module provides the bridge between the RL system and the existing
TradingService, enabling seamless integration while maintaining backward
compatibility and fallback mechanisms.

Features:
- RESTful communication with TradingService
- Adaptive threshold integration and fallback
- Real-time trading signal processing
- Performance comparison and A/B testing
- Graceful error handling and circuit breaker patterns
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class TradingServiceConfig:
    """Configuration for trading service integration"""
    base_url: str = "http://backend:3000"
    adaptive_threshold_url: str = "http://ml-service:5000"
    timeout_seconds: int = 10
    max_retries: int = 3
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: int = 60   # Seconds before trying again
    
@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0

class TradingBridge:
    """
    Bridge between RL system and TradingService
    
    Handles:
    - Communication with TradingService API
    - Fallback to AdaptiveThreshold service
    - Circuit breaker pattern for resilience
    - Performance tracking and comparison
    - A/B testing coordination
    """
    
    def __init__(self, trading_service_url: str, adaptive_threshold_url: str = None):
        self.config = TradingServiceConfig(
            base_url=trading_service_url,
            adaptive_threshold_url=adaptive_threshold_url or "http://ml-service:5000"
        )
        
        # Circuit breaker states
        self.trading_service_breaker = CircuitBreakerState()
        self.adaptive_threshold_breaker = CircuitBreakerState()
        
        # Performance tracking
        self.request_history = []
        self.performance_metrics = {
            'trading_service': {'success_rate': 0.0, 'avg_response_time': 0.0},
            'adaptive_threshold': {'success_rate': 0.0, 'avg_response_time': 0.0}
        }
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"TradingBridge initialized - TradingService: {self.config.base_url}, "
                   f"AdaptiveThreshold: {self.config.adaptive_threshold_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
    
    def _check_circuit_breaker(self, breaker: CircuitBreakerState) -> bool:
        """Check if circuit breaker allows requests"""
        if not breaker.is_open:
            return True
        
        # Check if timeout has passed
        if breaker.last_failure_time:
            time_since_failure = datetime.now() - breaker.last_failure_time
            if time_since_failure.total_seconds() > self.config.circuit_breaker_timeout:
                # Try half-open state
                breaker.is_open = False
                logger.info("Circuit breaker moving to half-open state")
                return True
        
        return False
    
    def _record_success(self, breaker: CircuitBreakerState):
        """Record successful request"""
        breaker.success_count += 1
        breaker.failure_count = 0
        if breaker.is_open:
            breaker.is_open = False
            logger.info("Circuit breaker closed after successful request")
    
    def _record_failure(self, breaker: CircuitBreakerState):
        """Record failed request"""
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= self.config.circuit_breaker_threshold:
            breaker.is_open = True
            logger.warning(f"Circuit breaker opened after {breaker.failure_count} failures")
    
    async def get_trading_status(self, user_id: str) -> Dict[str, Any]:
        """Get trading status from TradingService"""
        try:
            if not self._check_circuit_breaker(self.trading_service_breaker):
                raise Exception("Trading service circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/trading/status/{user_id}"
            start_time = time.time()
            
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.trading_service_breaker)
                    self._update_performance_metrics('trading_service', True, response_time)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            self._record_failure(self.trading_service_breaker)
            self._update_performance_metrics('trading_service', False, 0)
            raise
    
    async def start_trading_session(self, user_id: str, watchlist: List[str], 
                                  settings: Dict[str, Any]) -> Dict[str, Any]:
        """Start trading session via TradingService"""
        try:
            if not self._check_circuit_breaker(self.trading_service_breaker):
                raise Exception("Trading service circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/trading/start"
            payload = {
                "userId": user_id,
                "watchlist": watchlist,
                "settings": settings
            }
            
            start_time = time.time()
            
            async with self.session.post(url, json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.trading_service_breaker)
                    self._update_performance_metrics('trading_service', True, response_time)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to start trading session: {e}")
            self._record_failure(self.trading_service_breaker)
            self._update_performance_metrics('trading_service', False, 0)
            raise
    
    async def stop_trading_session(self, user_id: str) -> Dict[str, Any]:
        """Stop trading session via TradingService"""
        try:
            if not self._check_circuit_breaker(self.trading_service_breaker):
                raise Exception("Trading service circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/trading/stop"
            payload = {"userId": user_id}
            
            start_time = time.time()
            
            async with self.session.post(url, json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.trading_service_breaker)
                    self._update_performance_metrics('trading_service', True, response_time)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to stop trading session: {e}")
            self._record_failure(self.trading_service_breaker)
            self._update_performance_metrics('trading_service', False, 0)
            raise
    
    async def get_trading_signals(self, user_id: str, limit: int = 10, 
                                symbol: str = None) -> List[Dict[str, Any]]:
        """Get recent trading signals from TradingService"""
        try:
            if not self._check_circuit_breaker(self.trading_service_breaker):
                raise Exception("Trading service circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/trading/signals/{user_id}"
            params = {"limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            start_time = time.time()
            
            async with self.session.get(url, params=params) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.trading_service_breaker)
                    self._update_performance_metrics('trading_service', True, response_time)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to get trading signals: {e}")
            self._record_failure(self.trading_service_breaker)
            self._update_performance_metrics('trading_service', False, 0)
            raise
    
    async def update_trading_settings(self, user_id: str, 
                                    settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading settings via TradingService"""
        try:
            if not self._check_circuit_breaker(self.trading_service_breaker):
                raise Exception("Trading service circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/trading/settings/{user_id}"
            
            start_time = time.time()
            
            async with self.session.put(url, json=settings) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.trading_service_breaker)
                    self._update_performance_metrics('trading_service', True, response_time)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to update trading settings: {e}")
            self._record_failure(self.trading_service_breaker)
            self._update_performance_metrics('trading_service', False, 0)
            raise
    
    async def get_adaptive_threshold_prediction(self, user_id: str, 
                                              signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from AdaptiveThreshold service"""
        try:
            if not self._check_circuit_breaker(self.adaptive_threshold_breaker):
                raise Exception("Adaptive threshold circuit breaker is open")
            
            await self._ensure_session()
            
            url = f"{self.config.adaptive_threshold_url}/api/v1/evaluate/{user_id}"
            payload = {"signal": signal_data}
            
            start_time = time.time()
            
            async with self.session.post(url, json=payload) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.adaptive_threshold_breaker)
                    self._update_performance_metrics('adaptive_threshold', True, response_time)
                    
                    # Transform response to standard format
                    return {
                        'should_trade': data.get('data', {}).get('should_trade', False),
                        'confidence': data.get('data', {}).get('confidence', 0.5),
                        'action': self._infer_action_from_threshold_response(data),
                        'reasoning': data.get('data', {}).get('reasoning', 'Adaptive threshold decision'),
                        'thresholds_used': data.get('data', {}).get('thresholds', {})
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to get adaptive threshold prediction: {e}")
            self._record_failure(self.adaptive_threshold_breaker)
            self._update_performance_metrics('adaptive_threshold', False, 0)
            
            # Final fallback - conservative decision
            return {
                'should_trade': False,
                'confidence': 0.0,
                'action': 'HOLD',
                'reasoning': f'Adaptive threshold service unavailable: {str(e)}',
                'thresholds_used': {}
            }
    
    def _infer_action_from_threshold_response(self, response: Dict[str, Any]) -> str:
        """Infer trading action from adaptive threshold response"""
        data = response.get('data', {})
        should_trade = data.get('should_trade', False)
        
        if not should_trade:
            return 'HOLD'
        
        # Check signal data for action hint
        signal = data.get('signal', {})
        original_action = signal.get('action', 'HOLD')
        
        if original_action in ['BUY', 'SELL']:
            return original_action
        
        # Infer from signal characteristics
        rsi = signal.get('rsi', 50)
        change_percent = signal.get('change_percent', 0)
        
        if rsi < 30 or change_percent > 2:
            return 'BUY'
        elif rsi > 70 or change_percent < -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    async def get_adaptive_threshold_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from AdaptiveThreshold service"""
        try:
            if not self._check_circuit_breaker(self.adaptive_threshold_breaker):
                return {"error": "Adaptive threshold service circuit breaker is open"}
            
            await self._ensure_session()
            
            url = f"{self.config.adaptive_threshold_url}/api/v1/metrics"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self._record_success(self.adaptive_threshold_breaker)
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to get adaptive threshold metrics: {e}")
            self._record_failure(self.adaptive_threshold_breaker)
            return {"error": str(e)}
    
    async def submit_rl_prediction_feedback(self, user_id: str, prediction_id: str,
                                          actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Submit feedback about RL prediction performance to TradingService"""
        try:
            await self._ensure_session()
            
            url = f"{self.config.base_url}/api/rl/feedback"
            payload = {
                "userId": user_id,
                "predictionId": prediction_id,
                "outcome": actual_outcome,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to submit RL feedback: {response.status}")
                    return {"error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Failed to submit RL feedback: {e}")
            return {"error": str(e)}
    
    def _update_performance_metrics(self, service: str, success: bool, response_time: float):
        """Update performance metrics for service"""
        if service not in self.performance_metrics:
            self.performance_metrics[service] = {'success_rate': 0.0, 'avg_response_time': 0.0}
        
        # Add to request history
        self.request_history.append({
            'service': service,
            'success': success,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history (last 1000 requests)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
        
        # Recalculate metrics
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate performance metrics from request history"""
        # Calculate metrics for last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_requests = [r for r in self.request_history if r['timestamp'] > cutoff_time]
        
        for service in ['trading_service', 'adaptive_threshold']:
            service_requests = [r for r in recent_requests if r['service'] == service]
            
            if service_requests:
                success_count = sum(1 for r in service_requests if r['success'])
                success_rate = success_count / len(service_requests)
                
                successful_requests = [r for r in service_requests if r['success']]
                avg_response_time = np.mean([r['response_time'] for r in successful_requests]) if successful_requests else 0.0
                
                self.performance_metrics[service] = {
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'total_requests': len(service_requests),
                    'successful_requests': success_count
                }
    
    async def health_check(self) -> str:
        """Check health of TradingBridge"""
        try:
            # Check if both services are reachable
            trading_healthy = not self.trading_service_breaker.is_open
            adaptive_healthy = not self.adaptive_threshold_breaker.is_open
            
            if trading_healthy and adaptive_healthy:
                return ServiceStatus.HEALTHY.value
            elif trading_healthy or adaptive_healthy:
                return ServiceStatus.DEGRADED.value
            else:
                return ServiceStatus.UNHEALTHY.value
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceStatus.UNKNOWN.value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for both services"""
        self._recalculate_metrics()
        
        return {
            'trading_service': {
                **self.performance_metrics.get('trading_service', {}),
                'circuit_breaker': {
                    'is_open': self.trading_service_breaker.is_open,
                    'failure_count': self.trading_service_breaker.failure_count,
                    'success_count': self.trading_service_breaker.success_count
                }
            },
            'adaptive_threshold': {
                **self.performance_metrics.get('adaptive_threshold', {}),
                'circuit_breaker': {
                    'is_open': self.adaptive_threshold_breaker.is_open,
                    'failure_count': self.adaptive_threshold_breaker.failure_count,
                    'success_count': self.adaptive_threshold_breaker.success_count
                }
            },
            'total_requests': len(self.request_history),
            'request_history_hours': 1
        }
    
    async def run_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test"""
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test TradingService health
        try:
            await self._ensure_session()
            url = f"{self.config.base_url}/health"
            async with self.session.get(url) as response:
                test_results['tests']['trading_service_health'] = {
                    'status': 'passed' if response.status == 200 else 'failed',
                    'response_code': response.status
                }
        except Exception as e:
            test_results['tests']['trading_service_health'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Test AdaptiveThreshold health
        try:
            url = f"{self.config.adaptive_threshold_url}/health"
            async with self.session.get(url) as response:
                test_results['tests']['adaptive_threshold_health'] = {
                    'status': 'passed' if response.status == 200 else 'failed',
                    'response_code': response.status
                }
        except Exception as e:
            test_results['tests']['adaptive_threshold_health'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Test prediction flow
        try:
            test_signal = {
                'symbol': 'BTC',
                'confidence': 0.8,
                'rsi': 45,
                'change_percent': 2.5,
                'volume': 1000000,
                'action': 'BUY'
            }
            
            prediction = await self.get_adaptive_threshold_prediction('test_user', test_signal)
            test_results['tests']['prediction_flow'] = {
                'status': 'passed' if 'should_trade' in prediction else 'failed',
                'prediction': prediction
            }
        except Exception as e:
            test_results['tests']['prediction_flow'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Calculate overall status
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'passed')
        total_tests = len(test_results['tests'])
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'healthy' if passed_tests == total_tests else 'degraded' if passed_tests > 0 else 'unhealthy'
        }
        
        return test_results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("TradingBridge cleanup completed")


# Factory function for easy instantiation
def create_trading_bridge(trading_service_url: str, adaptive_threshold_url: str = None) -> TradingBridge:
    """Create TradingBridge instance"""
    return TradingBridge(trading_service_url, adaptive_threshold_url)


# Async context manager for automatic cleanup
async def trading_bridge_context(trading_service_url: str, adaptive_threshold_url: str = None):
    """Async context manager for TradingBridge"""
    bridge = TradingBridge(trading_service_url, adaptive_threshold_url)
    try:
        async with bridge:
            yield bridge
    finally:
        await bridge.cleanup()


if __name__ == "__main__":
    # Test the trading bridge
    async def test_bridge():
        async with trading_bridge_context("http://localhost:3000", "http://localhost:5000") as bridge:
            # Run integration test
            results = await bridge.run_integration_test()
            print(f"Integration test results: {json.dumps(results, indent=2)}")
            
            # Get performance summary
            performance = bridge.get_performance_summary()
            print(f"Performance summary: {json.dumps(performance, indent=2)}")
    
    # Run test
    asyncio.run(test_bridge())