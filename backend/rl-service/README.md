# RL Trading Environment

A comprehensive Gymnasium-compatible reinforcement learning environment for cryptocurrency trading, designed to transform rigid trading rules into adaptive intelligence.

## Overview

This environment provides a realistic simulation of cryptocurrency trading with:
- **15+ Features**: Price data, technical indicators, market sentiment, portfolio state, and alternative data
- **Flexible Actions**: Discrete action space with configurable position sizing (20%, 40%, 60%, 80%, 100%)
- **Realistic Market Dynamics**: Transaction costs, slippage, market impact, and bid-ask spreads
- **Multi-Asset Support**: Trade across 50+ cryptocurrency pairs
- **Production Ready**: Full type hints, comprehensive documentation, and error handling

## Features

### 🎯 Observation Space (15+ Features)
- **Price Data**: OHLCV candlestick data
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands, ATR, ADX, Stochastic
- **Market Sentiment**: Fear & Greed Index, News sentiment from Groq
- **Portfolio State**: Balance, positions, P&L, drawdown, win rate
- **Alternative Data**: Funding rates, whale activity, exchange flows, on-chain metrics

### 🎮 Action Space
- **Discrete Actions**: BUY (20%, 40%, 60%, 80%, 100%), SELL (20%, 40%, 60%, 80%, 100%), HOLD
- **Position Sizing**: Configurable percentage-based position sizing
- **Risk Management**: Built-in position limits and validation

### 💰 Reward Functions
- **Simple Return**: Basic return-based rewards
- **Sharpe Ratio**: Risk-adjusted performance
- **Risk-Adjusted**: Multi-factor reward with drawdown penalties
- **Combined**: Weighted combination of multiple metrics

### 📊 Market Simulation
- **Historical Backtesting**: Test on historical cryptocurrency data
- **Realistic Execution**: Market impact, slippage, and transaction costs
- **Multiple Regimes**: Bull, bear, sideways, high/low volatility markets
- **Live Trading**: Support for paper and live trading (future)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from rl_config import get_rl_config
from environment import TradingEnvironment
from datetime import datetime

# Create environment
config = get_rl_config()
env = TradingEnvironment(config=config, mode='train')

# Load market data
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)
env.load_data(start_date, end_date)

# Training loop
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from environment import TradingEnvironment
from rl_config import get_rl_config

# Create environment
config = get_rl_config()
env = TradingEnvironment(config=config)
env.load_data(start_date, end_date)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained agent
obs, info = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Architecture

```
rl-service/
├── rl_config.py              # Configuration management
├── environment/
│   ├── __init__.py
│   ├── trading_env.py         # Main Gymnasium environment
│   ├── state_processor.py     # Feature engineering
│   ├── portfolio_manager.py   # Portfolio tracking
│   └── market_simulator.py    # Market dynamics
├── tests/
│   ├── __init__.py
│   └── test_environment.py    # Comprehensive tests
├── example_usage.py           # Demo and examples
└── README.md
```

### Core Components

#### TradingEnvironment
- Main Gymnasium-compatible environment class
- Integrates all components for complete trading simulation
- Supports both training and testing modes
- Configurable episode length and trading parameters

#### StateProcessor
- Feature engineering and normalization
- Technical indicator calculation using TA-Lib
- State vector creation for RL agents
- Handles missing values and outliers

#### PortfolioManager
- Position tracking and P&L calculation
- Order management and execution
- Performance metrics calculation
- Risk management and constraints

#### MarketSimulator
- Historical and live data handling
- Realistic market dynamics simulation
- Transaction cost and slippage modeling
- Market impact calculation

## Configuration

The environment is highly configurable through `RLConfig`:

```python
from rl_config import RLConfig

config = RLConfig()

# Episode settings
config.env.episode_length = 1000
config.env.initial_balance = 10000.0
config.env.max_position_size = 1.0

# Observation settings
config.observation.lookback_window = 50
config.observation.normalize_features = True

# Reward settings
config.reward.strategy = RewardStrategy.RISK_ADJUSTED
config.reward.transaction_cost_rate = 0.001

# Trading pairs
config.env.trading_pairs = ['BTC/USD', 'ETH/USD', 'BNB/USD']
config.env.primary_pair = 'BTC/USD'
```

## Environment Variables

Key environment variables for configuration:

```bash
# RL Environment
RL_ENVIRONMENT=development
RL_DEBUG=false
RL_RANDOM_SEED=42

# Episode settings
RL_EPISODE_LENGTH=1000
RL_INITIAL_BALANCE=10000.0
RL_MAX_POSITION_SIZE=1.0

# Data settings
RL_START_DATE=2023-01-01
RL_END_DATE=2024-12-31
RL_TRAIN_SPLIT=0.8

# Reward settings
RL_REWARD_SCALING=100.0
RL_TRANSACTION_COST=0.001
RL_SLIPPAGE_RATE=0.0005

# Model settings
RL_ALGORITHM=PPO
RL_LEARNING_RATE=3e-4
RL_TOTAL_TIMESTEPS=100000
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_environment.py::TestTradingEnvironment -v

# Run with coverage
python -m pytest tests/ --cov=environment --cov-report=html
```

## Example Demo

Run the included demo to see the environment in action:

```bash
python example_usage.py
```

This will:
1. Create the trading environment
2. Load sample market data
3. Run a random agent for several episodes
4. Run a simple strategy demo
5. Display performance metrics and plots

## Performance Metrics

The environment tracks comprehensive performance metrics:

- **Returns**: Total return, Sharpe ratio, volatility
- **Risk**: Maximum drawdown, Value at Risk
- **Trading**: Win rate, profit factor, total trades
- **Execution**: Average trade P&L, transaction costs

## Integration with Existing System

The RL environment integrates with the existing trading bot architecture:

- **Data Sources**: Connects to Alpaca, Binance, and other data providers
- **ML Service**: Works with the existing AdaptiveThreshold ML service
- **Database**: Stores training results and model checkpoints
- **Monitoring**: Integrates with existing performance tracking

## Production Deployment

For production deployment:

1. **Configuration**: Use production config with proper API keys
2. **Data**: Connect to live data feeds
3. **Models**: Load trained models for inference
4. **Monitoring**: Enable comprehensive logging and metrics
5. **Risk Management**: Implement additional safety checks

## Future Enhancements

- **Multi-Asset Trading**: Simultaneous trading across multiple pairs
- **Continuous Actions**: Support for continuous action spaces
- **Advanced Features**: Order book data, microstructure features
- **Ensemble Methods**: Multiple agents and model averaging
- **Live Trading**: Full integration with live trading APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the crypto trading bot system and follows the same licensing terms.

## Support

For questions or issues:
1. Check the test suite for usage examples
2. Review the configuration options
3. Run the demo script to verify setup
4. Check logs for detailed error information

---

**Note**: This environment is designed for educational and research purposes. Live trading involves significant financial risk. Always test thoroughly and start with paper trading.