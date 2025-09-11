"""
Example Usage of Trading Optimization Components
================================================

Demonstrates how to integrate and use the optimization components
to achieve sub-100ms trading execution latency.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any
import logging

# Import optimization components
from execution_optimizer import create_execution_optimizer, ExecutionRequest
from feature_cache import create_feature_cache
from connection_pool import create_connection_pool
from vectorized_calculations import get_calculator
from performance_profiler import get_profiler, measure
from async_engine import create_async_engine, TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedTradingSystem:
    """
    Example of an optimized trading system using all components.
    Demonstrates sub-100ms end-to-end latency.
    """
    
    def __init__(self):
        # Initialize components
        self.execution_optimizer = create_execution_optimizer({
            'max_workers': 8,
            'batch_size': 100,
            'prefetch_size': 10,
            'trading_api_url': 'http://localhost:8000'
        })
        
        self.feature_cache = create_feature_cache({
            'l1_size': 10000,
            'l2_size': 100000,
            'redis_host': 'localhost',
            'redis_port': 6379
        })
        
        self.connection_pool = create_connection_pool({
            'pool_size': 100,
            'connection_timeout': 0.5,
            'enable_tcp_nodelay': True
        })
        
        self.calculator = get_calculator()
        self.profiler = get_profiler()
        
        self.async_engine = create_async_engine({
            'num_workers': 10,
            'critical_concurrency': 100,
            'high_concurrency': 50
        })
        
        logger.info("Optimized Trading System initialized")
    
    async def initialize(self):
        """Initialize all async components"""
        # Start profiler
        await self.profiler.start()
        
        # Initialize cache
        await self.feature_cache.initialize()
        
        # Initialize connection pools
        endpoints = [
            'http://api.alpaca.markets',
            'wss://stream.alpaca.markets',
            'http://localhost:8000'
        ]
        await self.connection_pool.initialize(endpoints)
        
        # Initialize execution optimizer
        await self.execution_optimizer.initialize()
        
        # Start async engine
        await self.async_engine.start()
        
        logger.info("All components initialized")
    
    @measure('process_market_data')
    async def process_market_data(self, symbol: str) -> Dict[str, np.ndarray]:
        """
        Process market data with caching and vectorization.
        Target: < 10ms
        """
        # Check cache first
        cache_key = f"features_{symbol}_{int(time.time())}"
        cached_features = await self.feature_cache.get(cache_key)
        
        if cached_features is not None:
            logger.info(f"Cache hit for {symbol}")
            return cached_features
        
        # Fetch market data (simulated)
        with self.profiler.measure('fetch_market_data'):
            market_data = await self._fetch_market_data(symbol)
        
        # Calculate features using vectorized operations
        with self.profiler.measure('calculate_features'):
            features = self.calculator.calculate_features(market_data)
        
        # Cache features
        await self.feature_cache.set(cache_key, features, ttl=60)
        
        return features
    
    @measure('make_trading_decision')
    async def make_trading_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make trading decision using optimized execution.
        Target: < 40ms
        """
        # Create execution request
        request = ExecutionRequest(
            request_id=f"req_{time.time()}",
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            features=features,
            priority=1
        )
        
        # Execute with optimizer
        result = await self.execution_optimizer.execute_trade(request)
        
        return result
    
    async def execute_trade_pipeline(self, symbol: str) -> Dict[str, Any]:
        """
        Complete trade execution pipeline.
        Target: < 100ms total
        """
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Market data processing (< 10ms)
            features = await self.process_market_data(symbol)
            
            # Stage 2: Decision making (< 40ms)
            decision = await self.make_trading_decision(features[-1])
            
            # Stage 3: Order execution (< 50ms)
            # Already included in decision making
            
            total_latency = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'symbol': symbol,
                'decision': decision,
                'total_latency_ms': total_latency,
                'target_met': total_latency < 100
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_latency_ms': (time.perf_counter() - start_time) * 1000
            }
    
    async def parallel_execution_demo(self):
        """Demonstrate parallel execution capabilities"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Submit tasks with priority
        task_ids = []
        for symbol in symbols:
            task_id = await self.async_engine.submit(
                self.execute_trade_pipeline(symbol),
                priority=TaskPriority.HIGH,
                timeout=0.2  # 200ms timeout
            )
            task_ids.append(task_id)
        
        # Wait for results
        results = await self.async_engine.wait_for_all(task_ids, timeout=1.0)
        
        # Analyze results
        successful = sum(1 for r in results if r.success)
        avg_latency = np.mean([r.result['total_latency_ms'] 
                               for r in results if r.success])
        
        logger.info(f"Parallel execution: {successful}/{len(symbols)} successful")
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        
        return results
    
    async def stress_test(self, duration_seconds: int = 10):
        """
        Stress test the system to measure sustained performance.
        """
        logger.info(f"Starting {duration_seconds}s stress test...")
        
        start_time = time.time()
        total_requests = 0
        successful_requests = 0
        latencies = []
        
        while time.time() - start_time < duration_seconds:
            # Execute trade
            result = await self.execute_trade_pipeline('AAPL')
            
            total_requests += 1
            if result['success']:
                successful_requests += 1
                latencies.append(result['total_latency_ms'])
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)  # 1ms between requests
        
        # Calculate statistics
        success_rate = successful_requests / total_requests
        avg_latency = np.mean(latencies) if latencies else 0
        p50_latency = np.percentile(latencies, 50) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        throughput = total_requests / duration_seconds
        
        logger.info(f"\nStress Test Results:")
        logger.info(f"Total Requests: {total_requests}")
        logger.info(f"Success Rate: {success_rate:.2%}")
        logger.info(f"Throughput: {throughput:.1f} req/s")
        logger.info(f"Average Latency: {avg_latency:.2f}ms")
        logger.info(f"P50 Latency: {p50_latency:.2f}ms")
        logger.info(f"P95 Latency: {p95_latency:.2f}ms")
        logger.info(f"P99 Latency: {p99_latency:.2f}ms")
        logger.info(f"Target Met (<100ms): {sum(1 for l in latencies if l < 100)/len(latencies):.2%}")
        
        return {
            'total_requests': total_requests,
            'success_rate': success_rate,
            'throughput': throughput,
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency
        }
    
    async def _fetch_market_data(self, symbol: str) -> Dict[str, np.ndarray]:
        """Simulate fetching market data"""
        # In production, this would use the connection pool
        # to fetch real market data
        
        # Simulate with random data
        n = 1000
        return {
            'open': np.random.randn(n) * 10 + 150,
            'high': np.random.randn(n) * 10 + 155,
            'low': np.random.randn(n) * 10 + 145,
            'close': np.random.randn(n) * 10 + 150,
            'volume': np.abs(np.random.randn(n) * 1000000)
        }
    
    async def shutdown(self):
        """Clean shutdown of all components"""
        await self.async_engine.stop()
        await self.execution_optimizer.shutdown()
        await self.feature_cache.shutdown()
        await self.connection_pool.shutdown()
        await self.profiler.stop()
        
        logger.info("System shutdown complete")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'execution_optimizer': self.execution_optimizer.get_metrics(),
            'feature_cache': self.feature_cache.get_stats(),
            'connection_pool': self.connection_pool.get_stats(),
            'async_engine': self.async_engine.get_stats(),
            'profiler': self.profiler.get_metrics()
        }


async def main():
    """Main demonstration function"""
    system = OptimizedTradingSystem()
    
    try:
        # Initialize system
        await system.initialize()
        
        logger.info("\n" + "="*50)
        logger.info("TRADING OPTIMIZATION DEMONSTRATION")
        logger.info("="*50)
        
        # 1. Single trade execution
        logger.info("\n1. Single Trade Execution:")
        result = await system.execute_trade_pipeline('AAPL')
        logger.info(f"Result: {result}")
        
        # 2. Parallel execution
        logger.info("\n2. Parallel Execution Demo:")
        await system.parallel_execution_demo()
        
        # 3. Stress test
        logger.info("\n3. Stress Test (10 seconds):")
        stress_results = await system.stress_test(duration_seconds=10)
        
        # 4. Performance report
        logger.info("\n4. Performance Report:")
        report = system.get_performance_report()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*50)
        
        exec_metrics = report['execution_optimizer']
        logger.info(f"Decision Latency: {exec_metrics['decision_latency_ms']:.2f}ms")
        logger.info(f"Execution Latency: {exec_metrics['execution_latency_ms']:.2f}ms")
        logger.info(f"Total Latency: {exec_metrics['total_latency_ms']:.2f}ms")
        logger.info(f"P95 Latency: {exec_metrics['p95_latency_ms']:.2f}ms")
        logger.info(f"P99 Latency: {exec_metrics['p99_latency_ms']:.2f}ms")
        
        cache_stats = report['feature_cache']
        logger.info(f"\nCache Hit Rate: {cache_stats['overall_hit_rate']:.2%}")
        logger.info(f"Cache Latency: {cache_stats['avg_latency_us']:.2f}μs")
        
        engine_stats = report['async_engine']
        logger.info(f"\nAsync Engine Success Rate: {engine_stats['success_rate']:.2%}")
        logger.info(f"Avg Task Execution: {engine_stats['avg_execution_time_ms']:.2f}ms")
        
        # Check if we met the target
        logger.info("\n" + "="*50)
        if exec_metrics['total_latency_ms'] < 100:
            logger.info("SUCCESS: Achieved sub-100ms latency target!")
            logger.info(f"Total Latency: {exec_metrics['total_latency_ms']:.2f}ms < 100ms")
        else:
            logger.info("Target not met. Further optimization needed.")
            logger.info(f"Total Latency: {exec_metrics['total_latency_ms']:.2f}ms")
        
    finally:
        # Cleanup
        await system.shutdown()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())