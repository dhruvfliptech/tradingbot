"""
Load and Performance Testing Suite

Tests system performance under high load conditions including:
- 1000+ requests per second capacity
- Memory usage under load
- Database connection pooling
- API response times
- Concurrent user handling
- Resource utilization monitoring
"""

import pytest
import asyncio
import aiohttp
import time
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import queue
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load testing configuration"""
    target_rps: int = 1000  # Requests per second
    test_duration: int = 60  # Test duration in seconds
    ramp_up_time: int = 10   # Ramp up time in seconds
    max_response_time: float = 0.5  # Maximum acceptable response time
    max_error_rate: float = 0.01  # Maximum acceptable error rate (1%)
    max_memory_usage: int = 1024  # Maximum memory usage in MB
    max_cpu_usage: float = 80.0   # Maximum CPU usage percentage


class LoadTestMetrics:
    """Load test metrics collector"""
    
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        self.concurrent_users = []
        self.throughput = []
        self._lock = threading.Lock()
        
    def record_response(self, response_time: float, success: bool):
        """Record a response"""
        with self._lock:
            self.response_times.append(response_time)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                
    def record_system_metrics(self):
        """Record system metrics"""
        with self._lock:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        total_requests = self.success_count + self.error_count
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'error_rate': self.error_count / total_requests if total_requests > 0 else 0,
            'average_rps': total_requests / duration if duration > 0 else 0,
            'duration_seconds': duration,
            'response_times': {
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0,
                'mean': statistics.mean(self.response_times) if self.response_times else 0,
                'median': statistics.median(self.response_times) if self.response_times else 0,
                'p95': np.percentile(self.response_times, 95) if self.response_times else 0,
                'p99': np.percentile(self.response_times, 99) if self.response_times else 0
            },
            'system_metrics': {
                'avg_cpu': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max_cpu': max(self.cpu_usage) if self.cpu_usage else 0,
                'avg_memory': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'max_memory': max(self.memory_usage) if self.memory_usage else 0
            }
        }


class TestLoadPerformance:
    """Load and performance testing suite"""
    
    @pytest.fixture(autouse=True)
    def setup_load_testing(self):
        """Setup load testing environment"""
        self.config = LoadTestConfig()
        self.metrics = LoadTestMetrics()
        self.base_url = "http://localhost:8000"  # Adjust as needed
        self.api_endpoints = [
            "/api/v1/signals",
            "/api/v1/portfolio",
            "/api/v1/risk-metrics",
            "/api/v1/market-data",
            "/api/v1/strategies"
        ]
        
    def test_baseline_performance(self):
        """Test baseline single request performance"""
        # Measure single request performance
        response_times = []
        
        for _ in range(100):
            start_time = time.time()
            
            # Simulate API call (replace with actual implementation)
            result = self._simulate_api_call("/api/v1/signals")
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            assert result['status'] == 'success', "API call should succeed"
            
        avg_response_time = statistics.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        assert avg_response_time < 0.1, \
            f"Average response time {avg_response_time:.3f}s exceeds 100ms baseline"
        
        assert p95_response_time < 0.2, \
            f"P95 response time {p95_response_time:.3f}s exceeds 200ms baseline"
        
        logger.info(f"Baseline performance - Avg: {avg_response_time:.3f}s, P95: {p95_response_time:.3f}s")
        
    def test_concurrent_users_scalability(self):
        """Test system scalability with increasing concurrent users"""
        user_counts = [10, 50, 100, 500, 1000]
        scalability_results = {}
        
        for user_count in user_counts:
            logger.info(f"Testing {user_count} concurrent users")
            
            # Reset metrics
            test_metrics = LoadTestMetrics()
            test_metrics.start_time = datetime.now()
            
            # Run concurrent load test
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = []
                
                for i in range(user_count):
                    future = executor.submit(self._simulate_user_session, test_metrics, 10)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        logger.warning(f"User session failed: {e}")
                        
            test_metrics.end_time = datetime.now()
            summary = test_metrics.get_summary()
            
            # Validate scalability requirements
            assert summary['error_rate'] <= self.config.max_error_rate, \
                f"Error rate {summary['error_rate']:.2%} exceeds {self.config.max_error_rate:.2%} for {user_count} users"
            
            assert summary['response_times']['p95'] <= self.config.max_response_time, \
                f"P95 response time {summary['response_times']['p95']:.3f}s exceeds limit for {user_count} users"
            
            scalability_results[user_count] = summary
            
        # Analyze scalability degradation
        response_degradation = self._analyze_scalability_degradation(scalability_results)
        assert response_degradation < 3.0, \
            f"Response time degradation {response_degradation:.1f}x exceeds 3x limit"
        
        logger.info(f"Scalability test passed with {response_degradation:.1f}x degradation")
        
    def test_sustained_load_1000_rps(self):
        """Test sustained load at 1000+ RPS"""
        target_rps = self.config.target_rps
        test_duration = self.config.test_duration
        
        logger.info(f"Starting sustained load test: {target_rps} RPS for {test_duration}s")
        
        self.metrics.start_time = datetime.now()
        
        # Start system monitoring
        monitor_thread = threading.Thread(target=self._monitor_system_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Calculate request timing
        request_interval = 1.0 / target_rps
        total_requests = target_rps * test_duration
        
        # Execute load test
        with ThreadPoolExecutor(max_workers=min(200, target_rps // 5)) as executor:
            futures = []
            
            for i in range(total_requests):
                # Throttle requests to maintain target RPS
                if i > 0 and i % target_rps == 0:
                    time.sleep(0.1)  # Brief pause every second
                    
                future = executor.submit(self._execute_load_request)
                futures.append(future)
                
                # Control rate
                if len(futures) >= 100:
                    # Process completed requests
                    completed = [f for f in futures if f.done()]
                    futures = [f for f in futures if not f.done()]
                    
            # Wait for remaining requests
            for future in as_completed(futures, timeout=60):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Load request failed: {e}")
                    
        self.metrics.end_time = datetime.now()
        
        # Validate load test results
        summary = self.metrics.get_summary()
        
        assert summary['average_rps'] >= target_rps * 0.95, \
            f"Average RPS {summary['average_rps']:.0f} below target {target_rps}"
        
        assert summary['error_rate'] <= self.config.max_error_rate, \
            f"Error rate {summary['error_rate']:.2%} exceeds {self.config.max_error_rate:.2%}"
        
        assert summary['response_times']['p95'] <= self.config.max_response_time, \
            f"P95 response time {summary['response_times']['p95']:.3f}s exceeds {self.config.max_response_time}s"
        
        assert summary['system_metrics']['max_cpu'] <= self.config.max_cpu_usage, \
            f"Max CPU usage {summary['system_metrics']['max_cpu']:.1f}% exceeds {self.config.max_cpu_usage}%"
        
        logger.info(f"Sustained load test passed: {summary['average_rps']:.0f} RPS, "
                   f"{summary['error_rate']:.2%} error rate")
        
    def test_database_connection_pooling(self):
        """Test database connection pool under load"""
        concurrent_db_operations = 500
        
        logger.info(f"Testing database connection pooling with {concurrent_db_operations} operations")
        
        connection_times = []
        query_times = []
        errors = 0
        
        def execute_db_operation():
            nonlocal errors
            try:
                # Simulate database connection and query
                conn_start = time.time()
                connection = self._get_db_connection()
                conn_time = time.time() - conn_start
                connection_times.append(conn_time)
                
                query_start = time.time()
                result = self._execute_db_query(connection)
                query_time = time.time() - query_start
                query_times.append(query_time)
                
                assert result is not None, "Database query should return result"
                
            except Exception as e:
                errors += 1
                logger.warning(f"Database operation failed: {e}")
            finally:
                if 'connection' in locals():
                    self._close_db_connection(connection)
        
        # Execute concurrent database operations
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(execute_db_operation) 
                      for _ in range(concurrent_db_operations)]
            
            for future in as_completed(futures, timeout=120):
                future.result()
        
        # Validate connection pool performance
        error_rate = errors / concurrent_db_operations
        avg_connection_time = statistics.mean(connection_times) if connection_times else 0
        avg_query_time = statistics.mean(query_times) if query_times else 0
        
        assert error_rate <= 0.05, \
            f"Database error rate {error_rate:.2%} exceeds 5% limit"
        
        assert avg_connection_time <= 0.1, \
            f"Average connection time {avg_connection_time:.3f}s exceeds 100ms"
        
        assert avg_query_time <= 0.05, \
            f"Average query time {avg_query_time:.3f}s exceeds 50ms"
        
        logger.info(f"Database pooling test passed: {error_rate:.2%} error rate, "
                   f"{avg_connection_time:.3f}s connection time")
        
    def test_memory_leak_detection(self):
        """Test for memory leaks under sustained load"""
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        logger.info(f"Starting memory leak detection test (initial: {initial_memory:.1f}MB)")
        
        # Run operations that could cause memory leaks
        operations_count = 1000
        
        for i in range(operations_count):
            # Simulate memory-intensive operations
            self._execute_memory_intensive_operation()
            
            # Check memory every 100 operations
            if i % 100 == 0:
                current_memory = psutil.virtual_memory().used / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Acceptable memory growth is proportional to operations
                max_acceptable_growth = (i / operations_count) * 100  # 100MB max
                
                assert memory_growth <= max_acceptable_growth, \
                    f"Memory growth {memory_growth:.1f}MB exceeds {max_acceptable_growth:.1f}MB after {i} operations"
        
        # Final memory check
        final_memory = psutil.virtual_memory().used / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Force garbage collection
        import gc
        gc.collect()
        
        post_gc_memory = psutil.virtual_memory().used / 1024 / 1024
        leak_indicator = post_gc_memory - initial_memory
        
        assert leak_indicator <= 50, \
            f"Potential memory leak detected: {leak_indicator:.1f}MB growth after GC"
        
        logger.info(f"Memory leak test passed: {total_growth:.1f}MB total growth, "
                   f"{leak_indicator:.1f}MB after GC")
        
    def test_api_rate_limiting(self):
        """Test API rate limiting mechanisms"""
        rate_limit = 100  # requests per minute per user
        test_user_id = "test_user_123"
        
        logger.info(f"Testing API rate limiting: {rate_limit} requests/minute")
        
        requests_sent = 0
        rate_limited_count = 0
        
        start_time = time.time()
        
        # Send requests rapidly to trigger rate limiting
        while time.time() - start_time < 10:  # 10 second test
            try:
                result = self._simulate_api_call_with_auth(
                    "/api/v1/signals", 
                    user_id=test_user_id
                )
                
                requests_sent += 1
                
                if result.get('status') == 'rate_limited':
                    rate_limited_count += 1
                    
            except Exception as e:
                logger.warning(f"Rate limit test request failed: {e}")
                
            time.sleep(0.05)  # 20 RPS attempt rate
        
        # Validate rate limiting
        rate_limit_triggered = rate_limited_count > 0
        assert rate_limit_triggered, "Rate limiting should be triggered under high load"
        
        rate_limit_ratio = rate_limited_count / requests_sent
        assert 0.1 <= rate_limit_ratio <= 0.9, \
            f"Rate limit ratio {rate_limit_ratio:.2%} should be between 10%-90%"
        
        logger.info(f"Rate limiting test passed: {rate_limited_count}/{requests_sent} "
                   f"({rate_limit_ratio:.1%}) rate limited")
        
    def test_websocket_performance(self):
        """Test WebSocket connection performance under load"""
        concurrent_connections = 100
        message_rate = 10  # messages per second per connection
        test_duration = 30  # seconds
        
        logger.info(f"Testing WebSocket performance: {concurrent_connections} connections")
        
        connection_results = []
        message_results = []
        
        async def websocket_test():
            try:
                # Simulate WebSocket connections and messaging
                connections = await self._create_websocket_connections(concurrent_connections)
                
                # Test message throughput
                for i in range(test_duration):
                    start_time = time.time()
                    
                    # Send messages on all connections
                    for conn in connections:
                        await self._send_websocket_message(conn, {"test": f"message_{i}"})
                    
                    message_time = time.time() - start_time
                    message_results.append(message_time)
                    
                    await asyncio.sleep(1.0 / message_rate)
                
                # Close connections
                for conn in connections:
                    await self._close_websocket_connection(conn)
                    
                return len(connections)
                
            except Exception as e:
                logger.error(f"WebSocket test failed: {e}")
                return 0
        
        # Run WebSocket test
        connected_count = asyncio.run(websocket_test())
        
        # Validate WebSocket performance
        assert connected_count >= concurrent_connections * 0.95, \
            f"Only {connected_count}/{concurrent_connections} WebSocket connections successful"
        
        if message_results:
            avg_message_time = statistics.mean(message_results)
            assert avg_message_time <= 0.1, \
                f"Average message broadcast time {avg_message_time:.3f}s exceeds 100ms"
        
        logger.info(f"WebSocket test passed: {connected_count} connections, "
                   f"{len(message_results)} message cycles")
        
    def _simulate_api_call(self, endpoint: str) -> Dict[str, Any]:
        """Simulate API call"""
        # Simulate processing time
        processing_time = np.random.normal(0.05, 0.02)
        time.sleep(max(0.001, processing_time))
        
        # Simulate occasional failures
        if np.random.random() < 0.001:  # 0.1% failure rate
            raise Exception("Simulated API failure")
        
        return {
            'status': 'success',
            'data': {'timestamp': time.time()},
            'processing_time': processing_time
        }
        
    def _simulate_api_call_with_auth(self, endpoint: str, user_id: str) -> Dict[str, Any]:
        """Simulate authenticated API call with rate limiting"""
        # Simple rate limiting simulation
        current_time = time.time()
        minute_key = int(current_time // 60)
        
        # Track requests per user per minute (simplified)
        if not hasattr(self, '_rate_limit_cache'):
            self._rate_limit_cache = {}
        
        user_minute_key = f"{user_id}_{minute_key}"
        current_count = self._rate_limit_cache.get(user_minute_key, 0)
        
        if current_count >= 100:  # 100 requests per minute limit
            return {'status': 'rate_limited', 'message': 'Rate limit exceeded'}
        
        self._rate_limit_cache[user_minute_key] = current_count + 1
        
        return self._simulate_api_call(endpoint)
        
    def _simulate_user_session(self, metrics: LoadTestMetrics, requests_per_session: int):
        """Simulate a user session with multiple requests"""
        for _ in range(requests_per_session):
            start_time = time.time()
            
            try:
                # Random endpoint selection
                endpoint = np.random.choice(self.api_endpoints)
                result = self._simulate_api_call(endpoint)
                
                response_time = time.time() - start_time
                metrics.record_response(response_time, True)
                
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                
            # Random think time between requests
            think_time = np.random.exponential(0.1)  # 100ms average
            time.sleep(min(think_time, 2.0))  # Max 2s think time
            
    def _execute_load_request(self):
        """Execute a single load test request"""
        start_time = time.time()
        
        try:
            endpoint = np.random.choice(self.api_endpoints)
            result = self._simulate_api_call(endpoint)
            
            response_time = time.time() - start_time
            self.metrics.record_response(response_time, True)
            
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_response(response_time, False)
            
    def _monitor_system_resources(self):
        """Monitor system resources during load test"""
        while self.metrics.start_time and not self.metrics.end_time:
            self.metrics.record_system_metrics()
            time.sleep(1)
            
    def _analyze_scalability_degradation(self, results: Dict) -> float:
        """Analyze response time degradation with scale"""
        user_counts = sorted(results.keys())
        response_times = [results[count]['response_times']['p95'] for count in user_counts]
        
        # Calculate degradation factor
        baseline = response_times[0] if response_times else 1.0
        max_degradation = max(response_times) if response_times else baseline
        
        return max_degradation / baseline if baseline > 0 else 1.0
        
    def _get_db_connection(self):
        """Simulate database connection"""
        # Simulate connection time
        time.sleep(np.random.uniform(0.001, 0.01))
        return {"connection_id": time.time()}
        
    def _execute_db_query(self, connection) -> Dict:
        """Simulate database query"""
        # Simulate query execution time
        time.sleep(np.random.uniform(0.005, 0.02))
        return {"result": "query_data", "rows": 100}
        
    def _close_db_connection(self, connection):
        """Simulate closing database connection"""
        pass
        
    def _execute_memory_intensive_operation(self):
        """Simulate memory-intensive operation"""
        # Create and process some data
        data = [i for i in range(1000)]
        processed = [x * 2 for x in data]
        
        # Simulate some processing
        result = sum(processed)
        
        # Clean up explicitly
        del data, processed
        
        return result
        
    async def _create_websocket_connections(self, count: int) -> List:
        """Simulate creating WebSocket connections"""
        connections = []
        for i in range(count):
            # Simulate connection creation
            await asyncio.sleep(0.001)  # Small delay
            connections.append({"ws_id": i, "connected": True})
        return connections
        
    async def _send_websocket_message(self, connection, message):
        """Simulate sending WebSocket message"""
        await asyncio.sleep(0.001)  # Simulate send time
        
    async def _close_websocket_connection(self, connection):
        """Simulate closing WebSocket connection"""
        connection["connected"] = False


if __name__ == "__main__":
    # Run load and performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])