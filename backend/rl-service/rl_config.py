"""
RL Environment Configuration for Crypto Trading Bot
Handles configuration for the Gymnasium-compatible trading environment
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Discrete action types for trading environment"""
    HOLD = 0
    BUY_20 = 1
    BUY_40 = 2
    BUY_60 = 3
    BUY_80 = 4
    BUY_100 = 5
    SELL_20 = 6
    SELL_40 = 7
    SELL_60 = 8
    SELL_80 = 9
    SELL_100 = 10


class RewardStrategy(Enum):
    """Reward calculation strategies"""
    SIMPLE_RETURN = "simple_return"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    COMBINED = "combined"


@dataclass
class ObservationConfig:
    """Configuration for observation space features"""
    # Price features (OHLCV)
    lookback_window: int = int(os.getenv('RL_LOOKBACK_WINDOW', '50'))
    price_features: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume'
    ])
    
    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram',
        'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'adx_14', 'stoch_k', 'stoch_d'
    ])
    
    # Market sentiment features
    sentiment_features: List[str] = field(default_factory=lambda: [
        'fear_greed_index', 'sentiment_score', 'news_sentiment',
        'social_sentiment', 'market_cap_rank'
    ])
    
    # Portfolio state features
    portfolio_features: List[str] = field(default_factory=lambda: [
        'cash_balance', 'portfolio_value', 'total_equity',
        'position_size', 'unrealized_pnl', 'realized_pnl',
        'current_allocation', 'drawdown', 'win_rate'
    ])
    
    # Alternative data features
    alternative_features: List[str] = field(default_factory=lambda: [
        'funding_rate', 'open_interest', 'long_short_ratio',
        'whale_activity', 'exchange_inflows', 'exchange_outflows',
        'network_activity', 'active_addresses'
    ])
    
    # Feature normalization
    normalize_features: bool = True
    normalization_method: str = 'robust'  # 'standard', 'minmax', 'robust'
    
    @property
    def total_features(self) -> int:
        """Calculate total number of features"""
        return (
            len(self.price_features) +
            len(self.technical_indicators) +
            len(self.sentiment_features) +
            len(self.portfolio_features) +
            len(self.alternative_features)
        )


@dataclass
class ActionConfig:
    """Configuration for action space"""
    action_type: str = 'discrete'  # 'discrete' or 'continuous'
    
    # Discrete action configuration
    discrete_actions: List[ActionType] = field(default_factory=lambda: list(ActionType))
    
    # Position sizing for discrete actions
    position_sizes: Dict[str, float] = field(default_factory=lambda: {
        'BUY_20': 0.2, 'BUY_40': 0.4, 'BUY_60': 0.6, 'BUY_80': 0.8, 'BUY_100': 1.0,
        'SELL_20': 0.2, 'SELL_40': 0.4, 'SELL_60': 0.6, 'SELL_80': 0.8, 'SELL_100': 1.0
    })
    
    # Continuous action configuration (if used)
    continuous_action_bounds: Tuple[float, float] = (-1.0, 1.0)  # -1 = full sell, +1 = full buy
    
    @property
    def action_space_size(self) -> int:
        """Get action space size"""
        if self.action_type == 'discrete':
            return len(self.discrete_actions)
        else:
            return 1  # Single continuous value


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    strategy: RewardStrategy = RewardStrategy.RISK_ADJUSTED
    
    # Risk-adjusted reward parameters
    risk_free_rate: float = float(os.getenv('RL_RISK_FREE_RATE', '0.02'))  # 2% annual
    sharpe_window: int = int(os.getenv('RL_SHARPE_WINDOW', '30'))
    
    # Reward components and weights
    reward_components: Dict[str, float] = field(default_factory=lambda: {
        'return': 0.4,
        'sharpe': 0.3,
        'drawdown_penalty': -0.2,
        'transaction_cost_penalty': -0.1
    })
    
    # Penalty parameters
    max_drawdown_penalty: float = float(os.getenv('RL_MAX_DRAWDOWN_PENALTY', '0.1'))
    transaction_cost_rate: float = float(os.getenv('RL_TRANSACTION_COST', '0.001'))  # 0.1%
    slippage_rate: float = float(os.getenv('RL_SLIPPAGE_RATE', '0.0005'))  # 0.05%
    
    # Reward scaling and clipping
    reward_scaling: float = float(os.getenv('RL_REWARD_SCALING', '100.0'))
    reward_clip_min: float = float(os.getenv('RL_REWARD_CLIP_MIN', '-10.0'))
    reward_clip_max: float = float(os.getenv('RL_REWARD_CLIP_MAX', '10.0'))


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment behavior"""
    # Episode configuration
    episode_length: int = int(os.getenv('RL_EPISODE_LENGTH', '1000'))  # Number of steps
    max_episode_steps: int = int(os.getenv('RL_MAX_EPISODE_STEPS', '5000'))
    
    # Initial conditions
    initial_balance: float = float(os.getenv('RL_INITIAL_BALANCE', '10000.0'))
    initial_allocation: float = float(os.getenv('RL_INITIAL_ALLOCATION', '0.0'))  # Start in cash
    
    # Trading constraints
    min_trade_amount: float = float(os.getenv('RL_MIN_TRADE_AMOUNT', '10.0'))
    max_position_size: float = float(os.getenv('RL_MAX_POSITION_SIZE', '1.0'))  # 100% of portfolio
    
    # Market simulation
    enable_market_impact: bool = os.getenv('RL_ENABLE_MARKET_IMPACT', 'true').lower() == 'true'
    market_impact_factor: float = float(os.getenv('RL_MARKET_IMPACT_FACTOR', '0.001'))
    
    # Data configuration
    trading_pairs: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD',
        'XRP/USD', 'DOT/USD', 'DOGE/USD', 'AVAX/USD', 'LINK/USD'
    ])
    primary_pair: str = 'BTC/USD'
    data_frequency: str = '1h'  # '1m', '5m', '15m', '1h', '4h', '1d'
    
    # Environment modes
    mode: str = 'train'  # 'train', 'test', 'live'
    enable_live_trading: bool = False
    paper_trading: bool = True


@dataclass
class ModelConfig:
    """Configuration for RL model training"""
    # Algorithm selection
    algorithm: str = os.getenv('RL_ALGORITHM', 'PPO')  # PPO, A2C, SAC, TD3, DQN
    
    # Model architecture
    policy_network: str = 'MlpPolicy'  # MlpPolicy, CnnPolicy, MultiInputPolicy
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation_function: str = 'relu'
    
    # Training parameters
    total_timesteps: int = int(os.getenv('RL_TOTAL_TIMESTEPS', '100000'))
    learning_rate: float = float(os.getenv('RL_LEARNING_RATE', '3e-4'))
    batch_size: int = int(os.getenv('RL_BATCH_SIZE', '64'))
    n_steps: int = int(os.getenv('RL_N_STEPS', '2048'))  # For on-policy algorithms
    
    # Model saving and loading
    save_frequency: int = int(os.getenv('RL_SAVE_FREQUENCY', '10000'))
    model_save_path: str = os.getenv('RL_MODEL_SAVE_PATH', '/tmp/rl_models/')
    
    # Evaluation
    eval_frequency: int = int(os.getenv('RL_EVAL_FREQUENCY', '5000'))
    eval_episodes: int = int(os.getenv('RL_EVAL_EPISODES', '10'))


@dataclass
class DataConfig:
    """Configuration for data sources and processing"""
    # Historical data
    start_date: str = os.getenv('RL_START_DATE', '2023-01-01')
    end_date: str = os.getenv('RL_END_DATE', '2024-12-31')
    train_split: float = float(os.getenv('RL_TRAIN_SPLIT', '0.8'))
    validation_split: float = float(os.getenv('RL_VALIDATION_SPLIT', '0.1'))
    
    # Data sources
    price_data_source: str = 'alpaca'  # 'alpaca', 'binance', 'coinbase'
    sentiment_data_source: str = 'groq'
    alternative_data_sources: List[str] = field(default_factory=lambda: [
        'coinglass', 'bitquery', 'cryptoquant'
    ])
    
    # Data processing
    fill_missing_method: str = 'forward'  # 'forward', 'backward', 'interpolate'
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Caching
    enable_data_cache: bool = True
    cache_duration_hours: int = 24


@dataclass
class RLConfig:
    """Complete RL environment configuration"""
    environment: str = os.getenv('RL_ENVIRONMENT', 'development')
    debug: bool = os.getenv('RL_DEBUG', 'false').lower() == 'true'
    
    # Sub-configurations
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging and monitoring
    log_level: str = os.getenv('RL_LOG_LEVEL', 'INFO')
    enable_tensorboard: bool = os.getenv('RL_ENABLE_TENSORBOARD', 'true').lower() == 'true'
    tensorboard_log_dir: str = os.getenv('RL_TENSORBOARD_LOG_DIR', '/tmp/rl_logs/')
    
    # Random seed for reproducibility
    random_seed: Optional[int] = int(os.getenv('RL_RANDOM_SEED', '42')) if os.getenv('RL_RANDOM_SEED') else None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate observation window
        if self.observation.lookback_window < 1:
            errors.append("Lookback window must be at least 1")
        
        # Validate episode length
        if self.env.episode_length < self.observation.lookback_window:
            errors.append("Episode length must be >= lookback window")
        
        # Validate reward weights
        weight_sum = sum(abs(w) for w in self.reward.reward_components.values())
        if weight_sum < 0.5 or weight_sum > 2.0:
            errors.append(f"Reward component weights sum ({weight_sum:.2f}) seems unreasonable")
        
        # Validate data splits
        total_split = self.data.train_split + self.data.validation_split
        if total_split >= 1.0:
            errors.append("Train + validation split must be < 1.0")
        
        # Validate trading constraints
        if self.env.max_position_size > 1.0 or self.env.max_position_size <= 0:
            errors.append("Max position size must be between 0 and 1")
        
        if errors:
            error_msg = "RL Configuration validation errors:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)
    
    def get_observation_space_size(self) -> int:
        """Calculate total observation space size"""
        return self.observation.total_features * self.observation.lookback_window
    
    def get_action_space_size(self) -> int:
        """Get action space size"""
        return self.action.action_space_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# Global RL configuration instance
def get_rl_config() -> RLConfig:
    """Get RL configuration instance"""
    return RLConfig()


# Environment-specific configurations
DEVELOPMENT_RL_CONFIG = {
    'environment': 'development',
    'debug': True,
    'env': {
        'episode_length': 500,
        'initial_balance': 10000.0,
        'trading_pairs': ['BTC/USD', 'ETH/USD']
    },
    'model': {
        'total_timesteps': 10000,
        'save_frequency': 1000
    }
}

PRODUCTION_RL_CONFIG = {
    'environment': 'production',
    'debug': False,
    'env': {
        'episode_length': 2000,
        'initial_balance': 100000.0,
        'enable_live_trading': True,
        'paper_trading': False
    },
    'model': {
        'total_timesteps': 1000000,
        'save_frequency': 50000
    }
}

TESTING_RL_CONFIG = {
    'environment': 'testing',
    'debug': True,
    'env': {
        'episode_length': 100,
        'initial_balance': 1000.0,
        'trading_pairs': ['BTC/USD']
    },
    'model': {
        'total_timesteps': 1000,
        'save_frequency': 100
    }
}


if __name__ == "__main__":
    # Example usage
    config = RLConfig()
    
    print("=== RL Environment Configuration ===")
    print(f"Environment: {config.environment}")
    print(f"Observation space size: {config.get_observation_space_size()}")
    print(f"Action space size: {config.get_action_space_size()}")
    print(f"Episode length: {config.env.episode_length}")
    print(f"Trading pairs: {config.env.trading_pairs}")
    print(f"Primary pair: {config.env.primary_pair}")
    print(f"Initial balance: ${config.env.initial_balance:,.2f}")
    print(f"Algorithm: {config.model.algorithm}")
    print(f"Total timesteps: {config.model.total_timesteps:,}")