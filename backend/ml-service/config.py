"""
Configuration Management System for AdaptiveThreshold ML Service
Handles environment variables, default settings, and configuration validation
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import yaml
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/trading_bot')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', '10'))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    pool_timeout: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    pool_recycle: int = int(os.getenv('DB_POOL_RECYCLE', '3600'))
    echo: bool = os.getenv('DB_ECHO', 'false').lower() == 'true'


@dataclass
class AdaptationConfig:
    """Adaptive threshold configuration parameters"""
    # Learning parameters
    default_learning_rate: float = float(os.getenv('ADAPTATION_LEARNING_RATE', '0.01'))
    min_learning_rate: float = float(os.getenv('ADAPTATION_MIN_LEARNING_RATE', '0.001'))
    max_learning_rate: float = float(os.getenv('ADAPTATION_MAX_LEARNING_RATE', '0.1'))
    
    # Performance tracking
    performance_window: int = int(os.getenv('ADAPTATION_PERFORMANCE_WINDOW', '100'))
    min_trades_for_adaptation: int = int(os.getenv('ADAPTATION_MIN_TRADES', '10'))
    adaptation_frequency_hours: int = int(os.getenv('ADAPTATION_FREQUENCY_HOURS', '24'))
    
    # Threshold bounds
    rsi_bounds: tuple = (
        float(os.getenv('RSI_MIN_THRESHOLD', '50.0')),
        float(os.getenv('RSI_MAX_THRESHOLD', '90.0'))
    )
    confidence_bounds: tuple = (
        float(os.getenv('CONFIDENCE_MIN_THRESHOLD', '0.2')),
        float(os.getenv('CONFIDENCE_MAX_THRESHOLD', '1.0'))
    )
    macd_bounds: tuple = (
        float(os.getenv('MACD_MIN_THRESHOLD', '-0.1')),
        float(os.getenv('MACD_MAX_THRESHOLD', '0.1'))
    )
    volume_bounds: tuple = (
        float(os.getenv('VOLUME_MIN_THRESHOLD', '100000000')),
        float(os.getenv('VOLUME_MAX_THRESHOLD', '10000000000'))
    )
    momentum_bounds: tuple = (
        float(os.getenv('MOMENTUM_MIN_THRESHOLD', '1.0')),
        float(os.getenv('MOMENTUM_MAX_THRESHOLD', '5.0'))
    )
    
    # Default parameter values
    default_parameters: Dict[str, float] = field(default_factory=lambda: {
        'rsi_threshold': float(os.getenv('DEFAULT_RSI_THRESHOLD', '70.0')),
        'confidence_threshold': float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', '0.65')),
        'macd_threshold': float(os.getenv('DEFAULT_MACD_THRESHOLD', '0.0')),
        'volume_threshold': float(os.getenv('DEFAULT_VOLUME_THRESHOLD', '1000000000')),
        'momentum_threshold': float(os.getenv('DEFAULT_MOMENTUM_THRESHOLD', '2.0'))
    })
    
    # Performance score weights
    performance_weights: Dict[str, float] = field(default_factory=lambda: {
        'total_return': float(os.getenv('WEIGHT_TOTAL_RETURN', '0.3')),
        'sharpe_ratio': float(os.getenv('WEIGHT_SHARPE_RATIO', '0.25')),
        'win_rate': float(os.getenv('WEIGHT_WIN_RATE', '0.2')),
        'max_drawdown': float(os.getenv('WEIGHT_MAX_DRAWDOWN', '-0.15')),
        'volatility': float(os.getenv('WEIGHT_VOLATILITY', '-0.1'))
    })
    
    # Safeguards
    max_adaptation_per_cycle: float = float(os.getenv('MAX_ADAPTATION_PER_CYCLE', '0.1'))  # 10% max change
    min_adaptation_threshold: float = float(os.getenv('MIN_ADAPTATION_THRESHOLD', '0.005'))  # 0.5% min change
    cooldown_after_poor_performance_hours: int = int(os.getenv('ADAPTATION_COOLDOWN_HOURS', '48'))


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    file_path: Optional[str] = os.getenv('LOG_FILE_PATH')
    max_file_size: int = int(os.getenv('LOG_MAX_FILE_SIZE', '10485760'))  # 10MB
    backup_count: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # Structured logging
    enable_json_logging: bool = os.getenv('ENABLE_JSON_LOGGING', 'false').lower() == 'true'
    enable_request_id: bool = os.getenv('ENABLE_REQUEST_ID', 'true').lower() == 'true'
    
    # External logging services
    sentry_dsn: Optional[str] = os.getenv('SENTRY_DSN')
    datadog_api_key: Optional[str] = os.getenv('DATADOG_API_KEY')


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    admin_api_key: str = os.getenv('ADMIN_API_KEY', 'change-me-in-production')
    flask_secret_key: str = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
    cors_origins: list = field(default_factory=lambda: 
        os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(','))
    
    # Rate limiting
    enable_rate_limiting: bool = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    rate_limit_per_minute: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Request validation
    max_request_size: int = int(os.getenv('MAX_REQUEST_SIZE', '1048576'))  # 1MB
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', '30'))


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_metrics: bool = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    metrics_port: int = int(os.getenv('METRICS_PORT', '8000'))
    
    # Health check settings
    health_check_timeout: int = int(os.getenv('HEALTH_CHECK_TIMEOUT', '5'))
    
    # Performance monitoring
    enable_performance_tracking: bool = os.getenv('ENABLE_PERFORMANCE_TRACKING', 'true').lower() == 'true'
    performance_sample_rate: float = float(os.getenv('PERFORMANCE_SAMPLE_RATE', '1.0'))
    
    # Alerting thresholds
    alert_on_adaptation_failure: bool = os.getenv('ALERT_ON_ADAPTATION_FAILURE', 'true').lower() == 'true'
    alert_on_db_connection_failure: bool = os.getenv('ALERT_ON_DB_FAILURE', 'true').lower() == 'true'
    alert_on_poor_performance_threshold: float = float(os.getenv('ALERT_POOR_PERFORMANCE_THRESHOLD', '0.2'))


@dataclass
class CacheConfig:
    """Caching configuration"""
    enable_redis: bool = os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true'
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    default_ttl: int = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))  # 1 hour
    
    # Memory cache fallback
    enable_memory_cache: bool = os.getenv('ENABLE_MEMORY_CACHE', 'true').lower() == 'true'
    memory_cache_size: int = int(os.getenv('MEMORY_CACHE_SIZE', '1000'))


@dataclass
class MLServiceConfig:
    """Complete ML Service configuration"""
    environment: str = os.getenv('ENVIRONMENT', 'development')
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', '5000'))
    workers: int = int(os.getenv('WORKERS', '4'))
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate learning rate bounds
        if self.adaptation.min_learning_rate >= self.adaptation.max_learning_rate:
            errors.append("min_learning_rate must be less than max_learning_rate")
        
        if not (self.adaptation.min_learning_rate <= 
                self.adaptation.default_learning_rate <= 
                self.adaptation.max_learning_rate):
            errors.append("default_learning_rate must be between min and max learning rates")
        
        # Validate performance weights sum (should be close to 1.0 or reasonable)
        weight_sum = sum(abs(w) for w in self.adaptation.performance_weights.values())
        if weight_sum < 0.5 or weight_sum > 2.0:
            errors.append(f"Performance weights sum ({weight_sum:.2f}) seems unreasonable")
        
        # Validate threshold bounds
        for param_name in ['rsi', 'confidence', 'macd', 'volume', 'momentum']:
            bounds_attr = f'{param_name}_bounds'
            if hasattr(self.adaptation, bounds_attr):
                min_val, max_val = getattr(self.adaptation, bounds_attr)
                if min_val >= max_val:
                    errors.append(f"{param_name} bounds: min must be less than max")
        
        # Validate port ranges
        if not (1024 <= self.port <= 65535):
            errors.append(f"Port {self.port} is not in valid range (1024-65535)")
        
        if not (1024 <= self.monitoring.metrics_port <= 65535):
            errors.append(f"Metrics port {self.monitoring.metrics_port} is not in valid range")
        
        # Validate environment
        if self.environment not in ['development', 'staging', 'production']:
            logger.warning(f"Unknown environment: {self.environment}")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'yaml'):
        """Save configuration to file"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if format.lower() == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'MLServiceConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLServiceConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        nested_configs = ['database', 'adaptation', 'logging', 'security', 'monitoring', 'cache']
        
        for config_name in nested_configs:
            if config_name in config_dict:
                config_class = {
                    'database': DatabaseConfig,
                    'adaptation': AdaptationConfig,
                    'logging': LoggingConfig,
                    'security': SecurityConfig,
                    'monitoring': MonitoringConfig,
                    'cache': CacheConfig
                }[config_name]
                
                config_dict[config_name] = config_class(**config_dict[config_name])
        
        return cls(**config_dict)


class ConfigManager:
    """Configuration manager with caching and hot reload support"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self._config: Optional[MLServiceConfig] = None
        self._file_mtime: Optional[float] = None
    
    @lru_cache(maxsize=1)
    def get_config(self, reload: bool = False) -> MLServiceConfig:
        """Get configuration with caching and hot reload support"""
        if reload:
            self.get_config.cache_clear()
        
        # Check if config file has been modified
        if self.config_file and self.config_file.exists():
            current_mtime = self.config_file.stat().st_mtime
            if self._file_mtime is None or current_mtime > self._file_mtime:
                self._file_mtime = current_mtime
                try:
                    self._config = MLServiceConfig.load_from_file(self.config_file)
                    logger.info(f"Configuration reloaded from {self.config_file}")
                except Exception as e:
                    logger.error(f"Failed to reload configuration: {e}")
                    if self._config is None:
                        # Fallback to default if no config loaded yet
                        self._config = MLServiceConfig()
        
        # Load from environment if no file config
        if self._config is None:
            self._config = MLServiceConfig()
        
        return self._config
    
    def reload_config(self) -> MLServiceConfig:
        """Force reload configuration"""
        return self.get_config(reload=True)
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        try:
            config = self.get_config()
            config._validate_config()
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigManager(
    config_file=os.getenv('ML_SERVICE_CONFIG_FILE', 'ml_service_config.yaml')
)

# Convenience function to get current configuration
def get_config() -> MLServiceConfig:
    """Get current configuration"""
    return config_manager.get_config()


# Configuration presets for different environments
DEVELOPMENT_CONFIG = {
    'environment': 'development',
    'debug': True,
    'logging': {'level': 'DEBUG'},
    'database': {'echo': True},
    'security': {'enable_rate_limiting': False}
}

PRODUCTION_CONFIG = {
    'environment': 'production',
    'debug': False,
    'logging': {'level': 'INFO', 'enable_json_logging': True},
    'database': {'echo': False},
    'security': {'enable_rate_limiting': True},
    'monitoring': {'enable_metrics': True}
}

TESTING_CONFIG = {
    'environment': 'testing',
    'debug': True,
    'logging': {'level': 'DEBUG'},
    'database': {'url': 'sqlite:///:memory:'},
    'security': {'enable_rate_limiting': False},
    'monitoring': {'enable_metrics': False}
}


def create_config_for_environment(env: str) -> MLServiceConfig:
    """Create configuration for specific environment"""
    base_config = MLServiceConfig()
    
    if env == 'development':
        config_dict = {**base_config.to_dict(), **DEVELOPMENT_CONFIG}
    elif env == 'production':
        config_dict = {**base_config.to_dict(), **PRODUCTION_CONFIG}
    elif env == 'testing':
        config_dict = {**base_config.to_dict(), **TESTING_CONFIG}
    else:
        raise ValueError(f"Unknown environment: {env}")
    
    return MLServiceConfig.from_dict(config_dict)


if __name__ == "__main__":
    # Example usage and configuration generation
    config = MLServiceConfig()
    
    print("=== ML Service Configuration ===")
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug}")
    print(f"Database URL: {config.database.url}")
    print(f"Learning rate: {config.adaptation.default_learning_rate}")
    print(f"Performance window: {config.adaptation.performance_window}")
    
    # Save example configuration
    config.save_to_file('ml_service_config_example.yaml')
    print("\nExample configuration saved to ml_service_config_example.yaml")