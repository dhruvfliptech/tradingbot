"""
Trading Execution Optimization Package
======================================

High-performance optimization components for sub-100ms trading execution.
"""

from .execution_optimizer import ExecutionOptimizer, ExecutionRequest, create_execution_optimizer
from .feature_cache import FeatureCache, create_feature_cache
from .connection_pool import ConnectionPool, create_connection_pool
from .vectorized_calculations import VectorizedCalculator, get_calculator
from .performance_profiler import PerformanceProfiler, get_profiler, measure
from .async_engine import AsyncExecutor, AsyncPipeline, create_async_engine, TaskPriority

__all__ = [
    # Execution Optimizer
    'ExecutionOptimizer',
    'ExecutionRequest',
    'create_execution_optimizer',
    
    # Feature Cache
    'FeatureCache',
    'create_feature_cache',
    
    # Connection Pool
    'ConnectionPool',
    'create_connection_pool',
    
    # Vectorized Calculations
    'VectorizedCalculator',
    'get_calculator',
    
    # Performance Profiler
    'PerformanceProfiler',
    'get_profiler',
    'measure',
    
    # Async Engine
    'AsyncExecutor',
    'AsyncPipeline',
    'create_async_engine',
    'TaskPriority'
]

__version__ = '1.0.0'