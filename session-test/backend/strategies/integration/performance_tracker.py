"""
Performance Tracker
===================

Comprehensive performance monitoring and analytics system for the strategy
integration layer. Tracks strategy performance, system metrics, and provides
detailed analytics and reporting capabilities.

Key Features:
- Real-time strategy performance monitoring
- System health and reliability tracking
- A/B testing framework integration
- Performance analytics and reporting
- Alert system for performance issues
- Historical performance analysis
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import sqlite3
from threading import Lock
import statistics
import traceback

# Statistical and ML libraries
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    LATENCY = "latency"
    UPTIME = "uptime"
    ERROR_RATE = "error_rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class StrategyPerformance:
    """Performance metrics for individual strategy"""
    strategy_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_confidence: float = 0.0
    avg_latency: float = 0.0
    uptime_ratio: float = 1.0
    error_count: int = 0
    feature_count: int = 0
    signal_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    returns: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemPerformance:
    """Overall system performance metrics"""
    timestamp: datetime
    total_strategies: int = 0
    active_strategies: int = 0
    total_features: int = 0
    total_signals: int = 0
    avg_processing_time: float = 0.0
    system_uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # Messages per second
    latency_p95: float = 0.0
    latency_p99: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric: PerformanceMetric
    strategy_name: Optional[str]
    message: str
    current_value: float
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    strategy_a: str
    strategy_b: str
    start_time: datetime
    end_time: Optional[datetime]
    metric: PerformanceMetric
    a_performance: float
    b_performance: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    winner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Advanced performance tracking and analytics system.
    
    Monitors strategy and system performance, provides alerts,
    and supports A/B testing for strategy optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance tracker"""
        self.config = config or self._default_config()
        
        # Performance data storage
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.system_performance_history: deque = deque(maxlen=self.config['history_length'])
        self.performance_snapshots: deque = deque(maxlen=self.config['snapshot_history'])
        
        # Alert system
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=self.config['alert_history_length'])
        self.alert_callbacks: List[Callable] = []
        
        # A/B testing
        self.active_ab_tests: Dict[str, ABTestResult] = {}
        self.completed_ab_tests: List[ABTestResult] = []
        
        # Database for persistence
        self.db_path = self.config.get('db_path', 'performance_tracker.db')
        self.db_lock = Lock()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.last_snapshot_time: Optional[datetime] = None
        
        # Performance thresholds
        self.thresholds = self.config.get('performance_thresholds', {})
        
        # Statistics tracking
        self.stats = {
            'alerts_generated': 0,
            'alerts_resolved': 0,
            'ab_tests_started': 0,
            'ab_tests_completed': 0,
            'uptime_start': datetime.now()
        }
        
        logger.info("Performance Tracker initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'history_length': 10000,
            'snapshot_history': 1000,
            'alert_history_length': 1000,
            'monitoring_interval': 10.0,  # seconds
            'snapshot_interval': 60.0,  # seconds
            'db_path': 'performance_tracker.db',
            'enable_persistence': True,
            'enable_real_time_monitoring': True,
            'enable_alerting': True,
            'enable_ab_testing': True,
            'performance_thresholds': {
                'strategy_accuracy_min': 0.55,
                'strategy_latency_max': 1.0,  # seconds
                'strategy_uptime_min': 0.95,
                'strategy_error_rate_max': 0.05,
                'system_latency_p95_max': 0.1,  # seconds
                'system_error_rate_max': 0.01,
                'system_uptime_min': 0.99,
                'memory_usage_max': 0.8,  # 80% of available
                'cpu_usage_max': 0.8  # 80% of available
            },
            'alert_cooldown': 300,  # seconds between similar alerts
            'ab_test_min_samples': 100,
            'ab_test_significance_level': 0.05,
            'retention_days': 30,  # Data retention period
            'aggregation_intervals': [
                '1m', '5m', '15m', '1h', '1d'
            ]
        }
    
    async def start(self):
        """Start the performance tracker"""
        logger.info("Starting Performance Tracker...")
        
        # Initialize database
        if self.config.get('enable_persistence', True):
            await self._initialize_database()
        
        # Start monitoring tasks
        if self.config.get('enable_real_time_monitoring', True):
            self.monitoring_active = True
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._snapshot_loop())
        
        # Load historical data
        await self._load_historical_data()
        
        logger.info("Performance Tracker started")
    
    async def stop(self):
        """Stop the performance tracker"""
        logger.info("Stopping Performance Tracker...")
        
        self.monitoring_active = False
        
        # Save current state
        if self.config.get('enable_persistence', True):
            await self._save_current_state()
        
        logger.info("Performance Tracker stopped")
    
    async def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        timestamp DATETIME NOT NULL,
                        severity TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        strategy_name TEXT,
                        message TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE,
                        metadata TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_id TEXT UNIQUE NOT NULL,
                        strategy_a TEXT NOT NULL,
                        strategy_b TEXT NOT NULL,
                        start_time DATETIME NOT NULL,
                        end_time DATETIME,
                        metric TEXT NOT NULL,
                        a_performance REAL,
                        b_performance REAL,
                        statistical_significance REAL,
                        winner TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_perf_timestamp ON strategy_performance(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_perf_timestamp ON system_performance(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON performance_alerts(timestamp)')
                
                conn.commit()
                conn.close()
                
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    async def update_strategy_performance(self, 
                                        strategy_name: str, 
                                        metrics: Dict[str, Any]):
        """Update performance metrics for a strategy"""
        try:
            if strategy_name not in self.strategy_performances:
                self.strategy_performances[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    start_time=datetime.now()
                )
            
            performance = self.strategy_performances[strategy_name]
            
            # Update metrics
            for metric_name, value in metrics.items():
                if hasattr(performance, metric_name):
                    setattr(performance, metric_name, value)
                else:
                    performance.metadata[metric_name] = value
            
            # Calculate derived metrics
            await self._calculate_derived_metrics(performance)
            
            # Check thresholds and generate alerts
            if self.config.get('enable_alerting', True):
                await self._check_strategy_thresholds(strategy_name, performance)
            
            # Save to database
            if self.config.get('enable_persistence', True):
                await self._save_strategy_performance(strategy_name, metrics)
            
            logger.debug(f"Updated performance for strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def _calculate_derived_metrics(self, performance: StrategyPerformance):
        """Calculate derived performance metrics"""
        try:
            # Calculate signal accuracy
            total_signals = performance.total_signals
            if total_signals > 0:
                performance.signal_accuracy = performance.successful_signals / total_signals
            
            # Calculate win rate
            if performance.returns:
                winning_trades = sum(1 for r in performance.returns if r > 0)
                performance.win_rate = winning_trades / len(performance.returns)
                
                # Calculate Sharpe ratio
                if len(performance.returns) > 1:
                    mean_return = np.mean(performance.returns)
                    std_return = np.std(performance.returns)
                    if std_return > 0:
                        performance.sharpe_ratio = mean_return / std_return
                
                # Calculate max drawdown
                cumulative_returns = np.cumsum(performance.returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                performance.max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
                
                # Calculate profit factor
                positive_returns = [r for r in performance.returns if r > 0]
                negative_returns = [abs(r) for r in performance.returns if r < 0]
                
                if positive_returns and negative_returns:
                    total_profit = sum(positive_returns)
                    total_loss = sum(negative_returns)
                    performance.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
    
    async def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system-wide performance metrics"""
        try:
            current_time = datetime.now()
            
            # Create system performance snapshot
            system_perf = SystemPerformance(
                timestamp=current_time,
                **metrics
            )
            
            # Add to history
            self.system_performance_history.append(system_perf)
            
            # Check system thresholds
            if self.config.get('enable_alerting', True):
                await self._check_system_thresholds(system_perf)
            
            # Save to database
            if self.config.get('enable_persistence', True):
                await self._save_system_performance(metrics)
            
            logger.debug("Updated system performance metrics")
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def _check_strategy_thresholds(self, 
                                       strategy_name: str, 
                                       performance: StrategyPerformance):
        """Check strategy performance against thresholds"""
        thresholds = self.thresholds
        
        checks = [
            ('accuracy', performance.signal_accuracy, thresholds.get('strategy_accuracy_min')),
            ('latency', performance.avg_latency, thresholds.get('strategy_latency_max')),
            ('uptime', performance.uptime_ratio, thresholds.get('strategy_uptime_min')),
            ('error_rate', performance.error_count / max(performance.total_signals, 1), 
             thresholds.get('strategy_error_rate_max'))
        ]
        
        for metric_name, current_value, threshold in checks:
            if threshold is None:
                continue
            
            violation = False
            if metric_name in ['accuracy', 'uptime']:
                violation = current_value < threshold
            else:
                violation = current_value > threshold
            
            if violation:
                await self._generate_alert(
                    metric=PerformanceMetric(metric_name.upper()),
                    strategy_name=strategy_name,
                    current_value=current_value,
                    threshold_value=threshold,
                    message=f"Strategy {strategy_name} {metric_name} threshold violated"
                )
    
    async def _check_system_thresholds(self, system_perf: SystemPerformance):
        """Check system performance against thresholds"""
        thresholds = self.thresholds
        
        checks = [
            ('latency_p95', system_perf.latency_p95, thresholds.get('system_latency_p95_max')),
            ('error_rate', system_perf.error_rate, thresholds.get('system_error_rate_max')),
            ('uptime', system_perf.system_uptime, thresholds.get('system_uptime_min')),
            ('memory_usage', system_perf.memory_usage, thresholds.get('memory_usage_max')),
            ('cpu_usage', system_perf.cpu_usage, thresholds.get('cpu_usage_max'))
        ]
        
        for metric_name, current_value, threshold in checks:
            if threshold is None:
                continue
            
            violation = False
            if metric_name == 'uptime':
                violation = current_value < threshold
            else:
                violation = current_value > threshold
            
            if violation:
                await self._generate_alert(
                    metric=PerformanceMetric(metric_name.upper()),
                    strategy_name=None,
                    current_value=current_value,
                    threshold_value=threshold,
                    message=f"System {metric_name} threshold violated"
                )
    
    async def _generate_alert(self, 
                            metric: PerformanceMetric,
                            current_value: float,
                            threshold_value: float,
                            message: str,
                            strategy_name: Optional[str] = None,
                            severity: AlertSeverity = AlertSeverity.WARNING):
        """Generate performance alert"""
        try:
            # Check alert cooldown
            alert_key = f"{metric.value}_{strategy_name or 'system'}"
            if await self._is_alert_in_cooldown(alert_key):
                return
            
            # Create alert
            alert_id = f"{alert_key}_{int(time.time())}"
            alert = PerformanceAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                metric=metric,
                strategy_name=strategy_name,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update stats
            self.stats['alerts_generated'] += 1
            
            # Save to database
            if self.config.get('enable_persistence', True):
                await self._save_alert(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.warning(f"Error in alert callback: {e}")
            
            logger.warning(f"Generated alert: {message}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    async def _is_alert_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert type is in cooldown period"""
        cooldown_period = self.config.get('alert_cooldown', 300)
        current_time = datetime.now()
        
        # Check recent alerts for this key
        for alert in reversed(list(self.alert_history)):
            if (alert.metric.value in alert_key and 
                alert.strategy_name in alert_key and
                (current_time - alert.timestamp).total_seconds() < cooldown_period):
                return True
        
        return False
    
    async def start_ab_test(self, 
                          strategy_a: str, 
                          strategy_b: str, 
                          metric: PerformanceMetric,
                          test_id: Optional[str] = None) -> str:
        """Start A/B test between two strategies"""
        if not self.config.get('enable_ab_testing', True):
            raise ValueError("A/B testing is disabled")
        
        test_id = test_id or f"ab_test_{strategy_a}_vs_{strategy_b}_{int(time.time())}"
        
        try:
            ab_test = ABTestResult(
                test_id=test_id,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                start_time=datetime.now(),
                end_time=None,
                metric=metric,
                a_performance=0.0,
                b_performance=0.0,
                statistical_significance=0.0,
                confidence_interval=(0.0, 0.0)
            )
            
            self.active_ab_tests[test_id] = ab_test
            self.stats['ab_tests_started'] += 1
            
            logger.info(f"Started A/B test: {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            raise
    
    async def end_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """End A/B test and calculate results"""
        if test_id not in self.active_ab_tests:
            return None
        
        try:
            ab_test = self.active_ab_tests[test_id]
            ab_test.end_time = datetime.now()
            
            # Get performance data for both strategies
            strategy_a_perf = self.strategy_performances.get(ab_test.strategy_a)
            strategy_b_perf = self.strategy_performances.get(ab_test.strategy_b)
            
            if not strategy_a_perf or not strategy_b_perf:
                logger.warning(f"Missing performance data for A/B test {test_id}")
                return None
            
            # Extract metric values
            metric_name = ab_test.metric.value.lower()
            ab_test.a_performance = getattr(strategy_a_perf, metric_name, 0.0)
            ab_test.b_performance = getattr(strategy_b_perf, metric_name, 0.0)
            
            # Calculate statistical significance
            await self._calculate_ab_test_significance(ab_test)
            
            # Move to completed tests
            del self.active_ab_tests[test_id]
            self.completed_ab_tests.append(ab_test)
            self.stats['ab_tests_completed'] += 1
            
            # Save to database
            if self.config.get('enable_persistence', True):
                await self._save_ab_test_result(ab_test)
            
            logger.info(f"Completed A/B test: {test_id}, Winner: {ab_test.winner}")
            return ab_test
            
        except Exception as e:
            logger.error(f"Error ending A/B test: {e}")
            return None
    
    async def _calculate_ab_test_significance(self, ab_test: ABTestResult):
        """Calculate statistical significance of A/B test"""
        try:
            # Get sample data for both strategies
            strategy_a_perf = self.strategy_performances.get(ab_test.strategy_a)
            strategy_b_perf = self.strategy_performances.get(ab_test.strategy_b)
            
            # Use returns as sample data (simplified)
            a_samples = strategy_a_perf.returns if strategy_a_perf else []
            b_samples = strategy_b_perf.returns if strategy_b_perf else []
            
            min_samples = self.config.get('ab_test_min_samples', 100)
            
            if len(a_samples) < min_samples or len(b_samples) < min_samples:
                ab_test.statistical_significance = 0.0
                ab_test.winner = None
                return
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(a_samples, b_samples)
            ab_test.statistical_significance = 1 - p_value
            
            # Calculate confidence interval
            significance_level = self.config.get('ab_test_significance_level', 0.05)
            
            if p_value < significance_level:
                # Statistically significant
                if ab_test.a_performance > ab_test.b_performance:
                    ab_test.winner = ab_test.strategy_a
                else:
                    ab_test.winner = ab_test.strategy_b
            
            # Calculate confidence interval for difference
            a_mean, a_std = np.mean(a_samples), np.std(a_samples)
            b_mean, b_std = np.mean(b_samples), np.std(b_samples)
            
            pooled_std = np.sqrt((a_std**2 + b_std**2) / 2)
            margin_of_error = stats.t.ppf(1 - significance_level/2, len(a_samples) + len(b_samples) - 2) * pooled_std
            
            diff = a_mean - b_mean
            ab_test.confidence_interval = (diff - margin_of_error, diff + margin_of_error)
            
        except Exception as e:
            logger.error(f"Error calculating A/B test significance: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config['monitoring_interval'])
                
                # Update system metrics
                await self._collect_system_metrics()
                
                # Check for stale data
                await self._check_stale_data()
                
                # Clean up old data
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _snapshot_loop(self):
        """Background snapshot loop"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config['snapshot_interval'])
                
                # Take performance snapshot
                await self._take_performance_snapshot()
                
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Calculate system uptime
            uptime_seconds = (datetime.now() - self.stats['uptime_start']).total_seconds()
            
            # Calculate error rate
            total_errors = sum(perf.error_count for perf in self.strategy_performances.values())
            total_signals = sum(perf.total_signals for perf in self.strategy_performances.values())
            error_rate = total_errors / max(total_signals, 1)
            
            # Calculate latency percentiles
            all_latencies = []
            for perf in self.strategy_performances.values():
                if perf.avg_latency > 0:
                    all_latencies.append(perf.avg_latency)
            
            latency_p95 = np.percentile(all_latencies, 95) if all_latencies else 0
            latency_p99 = np.percentile(all_latencies, 99) if all_latencies else 0
            
            system_metrics = {
                'total_strategies': len(self.strategy_performances),
                'active_strategies': sum(1 for p in self.strategy_performances.values() 
                                       if p.uptime_ratio > 0.5),
                'system_uptime': uptime_seconds,
                'memory_usage': memory.percent / 100.0,
                'cpu_usage': cpu_percent / 100.0,
                'error_rate': error_rate,
                'latency_p95': latency_p95,
                'latency_p99': latency_p99,
                'total_features': sum(p.feature_count for p in self.strategy_performances.values()),
                'total_signals': total_signals
            }
            
            await self.update_system_metrics(system_metrics)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _take_performance_snapshot(self):
        """Take comprehensive performance snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'strategies': {},
                'system': {},
                'alerts': len(self.active_alerts),
                'ab_tests': len(self.active_ab_tests)
            }
            
            # Strategy snapshots
            for name, perf in self.strategy_performances.items():
                snapshot['strategies'][name] = {
                    'signal_accuracy': perf.signal_accuracy,
                    'avg_latency': perf.avg_latency,
                    'uptime_ratio': perf.uptime_ratio,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'max_drawdown': perf.max_drawdown,
                    'win_rate': perf.win_rate,
                    'total_signals': perf.total_signals,
                    'error_count': perf.error_count
                }
            
            # System snapshot
            if self.system_performance_history:
                latest_system = self.system_performance_history[-1]
                snapshot['system'] = {
                    'cpu_usage': latest_system.cpu_usage,
                    'memory_usage': latest_system.memory_usage,
                    'error_rate': latest_system.error_rate,
                    'latency_p95': latest_system.latency_p95,
                    'throughput': latest_system.throughput
                }
            
            self.performance_snapshots.append(snapshot)
            self.last_snapshot_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error taking performance snapshot: {e}")
    
    async def _check_stale_data(self):
        """Check for stale performance data"""
        current_time = datetime.now()
        stale_threshold = timedelta(minutes=5)
        
        for name, perf in self.strategy_performances.items():
            if perf.end_time and current_time - perf.end_time > stale_threshold:
                logger.warning(f"Stale data detected for strategy: {name}")
    
    async def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            current_time = datetime.now()
            retention_period = timedelta(days=self.config.get('retention_days', 30))
            
            # Clean up old alerts
            expired_alerts = []
            for alert_id, alert in self.active_alerts.items():
                if current_time - alert.timestamp > retention_period:
                    expired_alerts.append(alert_id)
            
            for alert_id in expired_alerts:
                del self.active_alerts[alert_id]
            
            # Clean up old A/B tests
            self.completed_ab_tests = [
                test for test in self.completed_ab_tests
                if test.end_time and current_time - test.end_time < retention_period
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def _load_historical_data(self):
        """Load historical performance data from database"""
        if not self.config.get('enable_persistence', True):
            return
        
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Load recent strategy performance
                cursor.execute('''
                    SELECT strategy_name, metric_name, metric_value, timestamp
                    FROM strategy_performance
                    WHERE timestamp > datetime('now', '-1 day')
                    ORDER BY timestamp DESC
                ''')
                
                # Process results
                # (Implementation would aggregate and restore strategy performance objects)
                
                conn.close()
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _save_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]):
        """Save strategy performance to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                timestamp = datetime.now()
                
                for metric_name, value in metrics.items():
                    cursor.execute('''
                        INSERT INTO strategy_performance 
                        (strategy_name, timestamp, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (strategy_name, timestamp, metric_name, float(value), '{}'))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving strategy performance: {e}")
    
    async def _save_system_performance(self, metrics: Dict[str, Any]):
        """Save system performance to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                timestamp = datetime.now()
                
                for metric_name, value in metrics.items():
                    cursor.execute('''
                        INSERT INTO system_performance 
                        (timestamp, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, metric_name, float(value), '{}'))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving system performance: {e}")
    
    async def _save_alert(self, alert: PerformanceAlert):
        """Save alert to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_alerts 
                    (alert_id, timestamp, severity, metric, strategy_name, message,
                     current_value, threshold_value, acknowledged, resolved, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.timestamp, alert.severity.value,
                    alert.metric.value, alert.strategy_name, alert.message,
                    alert.current_value, alert.threshold_value,
                    alert.acknowledged, alert.resolved, json.dumps(alert.metadata)
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    async def _save_ab_test_result(self, ab_test: ABTestResult):
        """Save A/B test result to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO ab_test_results 
                    (test_id, strategy_a, strategy_b, start_time, end_time, metric,
                     a_performance, b_performance, statistical_significance, winner, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ab_test.test_id, ab_test.strategy_a, ab_test.strategy_b,
                    ab_test.start_time, ab_test.end_time, ab_test.metric.value,
                    ab_test.a_performance, ab_test.b_performance,
                    ab_test.statistical_significance, ab_test.winner,
                    json.dumps(ab_test.metadata)
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving A/B test result: {e}")
    
    async def _save_current_state(self):
        """Save current state to database"""
        try:
            # Save all current performance data
            for name, perf in self.strategy_performances.items():
                metrics = {
                    'signal_accuracy': perf.signal_accuracy,
                    'avg_latency': perf.avg_latency,
                    'uptime_ratio': perf.uptime_ratio,
                    'total_signals': perf.total_signals,
                    'error_count': perf.error_count
                }
                await self._save_strategy_performance(name, metrics)
            
            logger.info("Saved current state to database")
            
        except Exception as e:
            logger.error(f"Error saving current state: {e}")
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for specific strategy"""
        return self.strategy_performances.get(strategy_name)
    
    def get_all_strategy_performances(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies"""
        return self.strategy_performances.copy()
    
    def get_system_performance(self) -> Optional[SystemPerformance]:
        """Get latest system performance metrics"""
        return self.system_performance_history[-1] if self.system_performance_history else None
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_ab_test_results(self) -> List[ABTestResult]:
        """Get completed A/B test results"""
        return self.completed_ab_tests.copy()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            self.stats['alerts_resolved'] += 1
            # Move to history only (keep in active for a while)
            return True
        return False
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'tracker_stats': self.stats,
            'strategies': {
                name: {
                    'signal_accuracy': perf.signal_accuracy,
                    'avg_latency': perf.avg_latency,
                    'uptime_ratio': perf.uptime_ratio,
                    'total_signals': perf.total_signals,
                    'error_count': perf.error_count,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'max_drawdown': perf.max_drawdown,
                    'win_rate': perf.win_rate
                }
                for name, perf in self.strategy_performances.items()
            },
            'system': self.get_system_performance().__dict__ if self.get_system_performance() else {},
            'alerts': {
                'active_count': len(self.active_alerts),
                'total_generated': self.stats['alerts_generated'],
                'total_resolved': self.stats['alerts_resolved']
            },
            'ab_tests': {
                'active_count': len(self.active_ab_tests),
                'completed_count': len(self.completed_ab_tests),
                'total_started': self.stats['ab_tests_started']
            },
            'config': self.config
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create performance tracker
        tracker = PerformanceTracker()
        
        try:
            await tracker.start()
            
            # Update strategy performance
            await tracker.update_strategy_performance('test_strategy', {
                'total_signals': 100,
                'successful_signals': 65,
                'avg_latency': 0.05,
                'uptime_ratio': 0.98
            })
            
            # Start A/B test
            test_id = await tracker.start_ab_test(
                'strategy_a', 'strategy_b', PerformanceMetric.ACCURACY
            )
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Get metrics
            metrics = tracker.get_metrics()
            print(f"Performance metrics: {metrics}")
            
        finally:
            await tracker.stop()
    
    asyncio.run(main())