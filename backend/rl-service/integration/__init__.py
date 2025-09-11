"""
RL Service Integration Layer
==========================

This package provides the integration layer between the RL system and existing
trading infrastructure, enabling seamless deployment and operation of
reinforcement learning models in production trading environments.

Components:
- rl_service: Main FastAPI service with comprehensive API endpoints
- trading_bridge: Integration with existing TradingService
- data_connector: Integration with DataAggregatorService
- decision_server: Real-time decision serving engine
- monitoring: Performance tracking and alerting system
- api_routes: REST API route definitions

Features:
- Seamless integration with existing services
- Real-time prediction serving with WebSocket support
- A/B testing capabilities for gradual rollout
- Comprehensive monitoring and alerting
- Graceful fallback mechanisms
- Performance optimization and caching
- Production-ready deployment
"""

from .rl_service import RLService, RLServiceConfig, create_rl_service
from .trading_bridge import TradingBridge, create_trading_bridge
from .data_connector import DataConnector, create_data_connector
from .decision_server import DecisionServer, create_decision_server
from .monitoring import RLMonitor, PerformanceTracker, rl_monitor, performance_tracker
from .api_routes import create_api_routes

__version__ = "1.0.0"
__author__ = "RL Trading Integration Team"

__all__ = [
    # Main service
    "RLService",
    "RLServiceConfig", 
    "create_rl_service",
    
    # Integration components
    "TradingBridge",
    "create_trading_bridge",
    "DataConnector", 
    "create_data_connector",
    "DecisionServer",
    "create_decision_server",
    
    # Monitoring
    "RLMonitor",
    "PerformanceTracker",
    "rl_monitor",
    "performance_tracker",
    
    # API
    "create_api_routes",
]

# Integration layer metadata
INTEGRATION_INFO = {
    "version": __version__,
    "description": "RL Service Integration Layer for Trading Systems",
    "supported_services": [
        "TradingService",
        "DataAggregatorService", 
        "AdaptiveThreshold",
        "Composer"
    ],
    "features": [
        "Real-time prediction serving",
        "WebSocket streaming support",
        "A/B testing framework", 
        "Comprehensive monitoring",
        "Graceful fallback mechanisms",
        "Performance optimization",
        "Production deployment"
    ],
    "requirements": {
        "python": ">=3.8",
        "fastapi": ">=0.95.0",
        "uvicorn": ">=0.20.0",
        "aiohttp": ">=3.8.0",
        "numpy": ">=1.21.0",
        "pandas": ">=1.4.0",
        "pydantic": ">=1.10.0",
        "psutil": ">=5.9.0"
    }
}


def get_integration_info():
    """Get integration layer information"""
    return INTEGRATION_INFO


def validate_environment():
    """Validate that the environment is properly configured"""
    import sys
    import importlib
    
    validation_results = {
        "python_version": sys.version_info >= (3, 8),
        "required_packages": {},
        "optional_packages": {},
        "environment_variables": {},
        "overall_status": True
    }
    
    # Check required packages
    required_packages = [
        "fastapi", "uvicorn", "aiohttp", "numpy", 
        "pandas", "pydantic", "psutil"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            validation_results["required_packages"][package] = True
        except ImportError:
            validation_results["required_packages"][package] = False
            validation_results["overall_status"] = False
    
    # Check optional packages
    optional_packages = [
        "torch", "stable_baselines3", "gymnasium", 
        "sklearn", "prometheus_client"
    ]
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            validation_results["optional_packages"][package] = True
        except ImportError:
            validation_results["optional_packages"][package] = False
    
    # Check environment variables
    import os
    env_vars = [
        "TRADING_SERVICE_URL",
        "DATA_AGGREGATOR_URL", 
        "ADAPTIVE_THRESHOLD_URL",
        "RL_MODELS_PATH"
    ]
    
    for var in env_vars:
        validation_results["environment_variables"][var] = var in os.environ
    
    return validation_results


def create_integration_service(config=None):
    """
    Factory function to create a fully configured RL integration service
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RLService instance
    """
    return create_rl_service(config)


# Default configuration for integration layer
DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,
    "debug": False,
    "models_path": "/app/models",
    "max_concurrent_requests": 100,
    "prediction_timeout_seconds": 5.0,
    "ab_testing_enabled": True,
    "rl_traffic_percentage": 0.1,
    "fallback_to_adaptive_threshold": True,
    "adaptive_threshold_url": "http://ml-service:5000",
    "trading_service_url": "http://backend:3000",
    "data_aggregator_url": "http://backend:3000",
    "metrics_retention_hours": 24,
    "performance_check_interval_minutes": 5
}


if __name__ == "__main__":
    # Print integration info when run as script
    import json
    
    print("RL Service Integration Layer")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    print("Environment Validation:")
    print("-" * 20)
    validation = validate_environment()
    
    for category, results in validation.items():
        if category == "overall_status":
            continue
            
        print(f"\n{category.replace('_', ' ').title()}:")
        if isinstance(results, dict):
            for item, status in results.items():
                status_symbol = "✓" if status else "✗"
                print(f"  {status_symbol} {item}")
        else:
            status_symbol = "✓" if results else "✗" 
            print(f"  {status_symbol} {category}")
    
    print(f"\nOverall Status: {'✓ PASS' if validation['overall_status'] else '✗ FAIL'}")
    
    print("\nIntegration Info:")
    print("-" * 15)
    print(json.dumps(INTEGRATION_INFO, indent=2))