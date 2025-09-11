"""
Health Check Endpoints for Trading Bot
Provides comprehensive health checking for all system components.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """Health status for individual component"""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['last_check'] = self.last_check.isoformat()
        return result

@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    components: List[ComponentHealth]
    check_time: datetime
    uptime_seconds: float
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'components': [comp.to_dict() for comp in self.components],
            'check_time': self.check_time.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'version': self.version,
            'summary': self._get_summary()
        }
    
    def _get_summary(self) -> Dict[str, int]:
        """Get summary of component statuses"""
        summary = {status.value: 0 for status in HealthStatus}
        for component in self.components:
            summary[component.status.value] += 1
        return summary

class HealthChecker:
    """
    Comprehensive health checker for trading bot system.
    
    Monitors:
    - Database connectivity
    - API endpoints
    - External data sources
    - Trading services
    - ML models
    - System resources
    - Background processes
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.check_cache = {}
        self.cache_ttl = 30  # Cache health checks for 30 seconds
        
        # Service configurations
        self.services = {
            'database': {
                'url': 'postgresql://localhost:5432/tradingbot',
                'timeout': 5.0
            },
            'ml_service': {
                'url': 'http://localhost:8001/health',
                'timeout': 10.0
            },
            'rl_service': {
                'url': 'http://localhost:8002/health', 
                'timeout': 10.0
            },
            'binance_api': {
                'url': 'https://api.binance.com/api/v3/ping',
                'timeout': 5.0
            },
            'coinglass_api': {
                'url': 'https://open-api.coinglass.com/public/v2/indicator/funding_rates',
                'timeout': 5.0
            }
        }
        
    async def check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # Mock database check - replace with actual database connection
            await asyncio.sleep(0.1)  # Simulate DB query
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if response time is acceptable
            if response_time > 1000:  # > 1 second
                status = HealthStatus.WARNING
                error_msg = f"Slow database response: {response_time:.2f}ms"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
                
            return ComponentHealth(
                name="database",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_message=error_msg,
                metadata={
                    'connection_pool_size': 10,
                    'active_connections': 3,
                    'query_time_avg_ms': response_time
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_external_api_health(self, service_name: str, url: str, timeout: float) -> ComponentHealth:
        """Check external API health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        error_msg = None
                    elif response.status in [429, 502, 503, 504]:
                        status = HealthStatus.WARNING
                        error_msg = f"API returned status {response.status}"
                    else:
                        status = HealthStatus.CRITICAL
                        error_msg = f"API error: status {response.status}"
                        
                    return ComponentHealth(
                        name=service_name,
                        status=status,
                        last_check=datetime.now(),
                        response_time_ms=response_time,
                        error_message=error_msg,
                        metadata={
                            'status_code': response.status,
                            'url': url
                        }
                    )
                    
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=service_name,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=timeout * 1000,
                error_message=f"Timeout after {timeout}s"
            )
        except Exception as e:
            return ComponentHealth(
                name=service_name,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health status based on resource usage
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if (disk.used / disk.total) * 100 > 90:
                issues.append(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")
                
            if issues:
                status = HealthStatus.WARNING if len(issues) == 1 else HealthStatus.CRITICAL
                error_msg = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
                
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="system_resources",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_message=error_msg,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': (disk.used / disk.total) * 100,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_trading_services(self) -> ComponentHealth:
        """Check trading services health"""
        start_time = time.time()
        
        try:
            # Mock trading service check
            # In real implementation, check:
            # - Portfolio manager status
            # - Risk manager status  
            # - Strategy execution status
            # - Order management system
            
            await asyncio.sleep(0.05)  # Simulate service check
            
            # Example status checks
            services_status = {
                'portfolio_manager': True,
                'risk_manager': True,
                'strategy_executor': True,
                'order_manager': True
            }
            
            failed_services = [name for name, status in services_status.items() if not status]
            
            if failed_services:
                status = HealthStatus.CRITICAL
                error_msg = f"Failed services: {', '.join(failed_services)}"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
                
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="trading_services",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_message=error_msg,
                metadata={
                    'services_status': services_status,
                    'active_strategies': 3,
                    'open_positions': 5,
                    'pending_orders': 2
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="trading_services",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_ml_models(self) -> ComponentHealth:
        """Check ML models and RL agents health"""
        start_time = time.time()
        
        try:
            # Mock ML model health check
            # In real implementation, check:
            # - Model inference latency
            # - Model accuracy metrics
            # - Training status
            # - Resource usage
            
            await asyncio.sleep(0.1)  # Simulate model check
            
            model_health = {
                'price_prediction_model': {'status': 'healthy', 'accuracy': 0.73, 'latency_ms': 45},
                'volume_analysis_model': {'status': 'healthy', 'accuracy': 0.68, 'latency_ms': 32},
                'rl_agent_ensemble': {'status': 'healthy', 'reward': 1250.5, 'actions_per_hour': 12}
            }
            
            issues = []
            for model_name, metrics in model_health.items():
                if metrics['status'] != 'healthy':
                    issues.append(f"{model_name}: {metrics['status']}")
                elif 'accuracy' in metrics and metrics['accuracy'] < 0.6:
                    issues.append(f"{model_name}: low accuracy {metrics['accuracy']:.2f}")
                elif 'latency_ms' in metrics and metrics['latency_ms'] > 100:
                    issues.append(f"{model_name}: high latency {metrics['latency_ms']}ms")
                    
            if issues:
                status = HealthStatus.WARNING
                error_msg = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
                
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="ml_models",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_message=error_msg,
                metadata=model_health
            )
            
        except Exception as e:
            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def check_data_feeds(self) -> ComponentHealth:
        """Check data feed health and freshness"""
        start_time = time.time()
        
        try:
            # Mock data feed health check
            # In real implementation, check:
            # - Last data update timestamps
            # - Data quality metrics
            # - Feed latency
            # - Missing data points
            
            await asyncio.sleep(0.05)
            
            data_feeds = {
                'price_feed': {'last_update': datetime.now() - timedelta(seconds=2), 'quality': 0.99},
                'volume_feed': {'last_update': datetime.now() - timedelta(seconds=1), 'quality': 0.98},
                'orderbook_feed': {'last_update': datetime.now() - timedelta(seconds=3), 'quality': 0.97},
                'news_feed': {'last_update': datetime.now() - timedelta(minutes=5), 'quality': 0.95}
            }
            
            issues = []
            now = datetime.now()
            
            for feed_name, feed_data in data_feeds.items():
                age_seconds = (now - feed_data['last_update']).total_seconds()
                quality = feed_data['quality']
                
                if age_seconds > 300:  # 5 minutes
                    issues.append(f"{feed_name}: stale data ({age_seconds:.0f}s old)")
                elif quality < 0.95:
                    issues.append(f"{feed_name}: low quality ({quality:.2f})")
                    
            if issues:
                status = HealthStatus.WARNING if len(issues) <= 2 else HealthStatus.CRITICAL
                error_msg = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
                
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name="data_feeds",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_message=error_msg,
                metadata={
                    feed: {
                        'age_seconds': (now - data['last_update']).total_seconds(),
                        'quality': data['quality']
                    }
                    for feed, data in data_feeds.items()
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="data_feeds",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def get_comprehensive_health(self, use_cache: bool = True) -> SystemHealth:
        """Get comprehensive system health status"""
        cache_key = "comprehensive_health"
        
        # Check cache first
        if use_cache and cache_key in self.check_cache:
            cached_result, cache_time = self.check_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_result
        
        # Run all health checks concurrently
        health_checks = await asyncio.gather(
            self.check_database_health(),
            self.check_system_resources(),
            self.check_trading_services(),
            self.check_ml_models(),
            self.check_data_feeds(),
            *[
                self.check_external_api_health(name, config['url'], config['timeout'])
                for name, config in self.services.items()
                if name not in ['database']  # Skip database as it's checked separately
            ],
            return_exceptions=True
        )
        
        # Filter out exceptions and collect valid health checks
        components = []
        for check_result in health_checks:
            if isinstance(check_result, ComponentHealth):
                components.append(check_result)
            elif isinstance(check_result, Exception):
                logger.error(f"Health check failed: {check_result}")
                components.append(ComponentHealth(
                    name="unknown_component",
                    status=HealthStatus.CRITICAL,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    error_message=str(check_result)
                ))
        
        # Determine overall system health
        critical_count = sum(1 for c in components if c.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for c in components if c.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 2:
            overall_status = HealthStatus.WARNING
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
            
        system_health = SystemHealth(
            status=overall_status,
            components=components,
            check_time=datetime.now(),
            uptime_seconds=time.time() - self.start_time
        )
        
        # Cache result
        self.check_cache[cache_key] = (system_health, time.time())
        
        return system_health

# FastAPI application for health endpoints
app = FastAPI(title="Trading Bot Health Check API", version="1.0.0")
health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    try:
        # Quick check of critical components
        db_health = await health_checker.check_database_health()
        if db_health.status == HealthStatus.CRITICAL:
            raise HTTPException(status_code=503, detail="Database not ready")
            
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with all components"""
    system_health = await health_checker.get_comprehensive_health()
    
    status_code = 200
    if system_health.status == HealthStatus.CRITICAL:
        status_code = 503
    elif system_health.status == HealthStatus.WARNING:
        status_code = 200  # Still operational but with warnings
        
    return JSONResponse(
        content=system_health.to_dict(),
        status_code=status_code
    )

@app.get("/health/component/{component_name}")
async def component_health(component_name: str):
    """Get health status for specific component"""
    system_health = await health_checker.get_comprehensive_health()
    
    component = next(
        (comp for comp in system_health.components if comp.name == component_name),
        None
    )
    
    if not component:
        raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")
        
    status_code = 200
    if component.status == HealthStatus.CRITICAL:
        status_code = 503
        
    return JSONResponse(
        content=component.to_dict(),
        status_code=status_code
    )

@app.get("/health/metrics")
async def health_metrics():
    """Get health metrics in Prometheus format"""
    system_health = await health_checker.get_comprehensive_health()
    
    metrics = []
    for component in system_health.components:
        status_value = 1 if component.status == HealthStatus.HEALTHY else 0
        metrics.append(f'component_health{{component="{component.name}"}} {status_value}')
        metrics.append(f'component_response_time_ms{{component="{component.name}"}} {component.response_time_ms}')
    
    overall_status_value = 1 if system_health.status == HealthStatus.HEALTHY else 0
    metrics.append(f'system_health_status {overall_status_value}')
    metrics.append(f'system_uptime_seconds {system_health.uptime_seconds}')
    
    return "\n".join(metrics)

@app.post("/health/cache/clear")
async def clear_health_cache():
    """Clear health check cache"""
    health_checker.check_cache.clear()
    return {"status": "cache_cleared", "timestamp": datetime.now().isoformat()}

# Background task to warm up cache
@app.on_event("startup")
async def startup_event():
    """Warm up health check cache on startup"""
    logger.info("Starting health check service...")
    try:
        await health_checker.get_comprehensive_health(use_cache=False)
        logger.info("Health check cache warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to warm up health check cache: {e}")

def start_health_server(host: str = "0.0.0.0", port: int = 8080):
    """Start health check server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_health_server()