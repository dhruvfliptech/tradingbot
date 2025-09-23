# Multi-Agent Ensemble Trading System

A sophisticated multi-agent ensemble system for cryptocurrency trading that combines specialized agents with intelligent orchestration to adapt to different market conditions and maximize performance while minimizing risk.

## 🎯 Overview

The ensemble system prevents single points of failure by using multiple specialized trading agents, each optimized for different market conditions. An intelligent orchestrator dynamically selects and combines agent decisions based on real-time market regime detection and performance tracking.

### Key Features

- **4 Specialized Agents** with different objectives and reward functions
- **Real-time Market Regime Detection** using ML and rule-based approaches
- **Multi-Armed Bandit Selection** for intelligent agent selection
- **Dynamic Weight Adjustment** based on performance and market conditions
- **Comprehensive Performance Tracking** with risk-adjusted metrics
- **Explainable Decisions** with full transparency
- **Robust Risk Management** across all market conditions

## 🏗️ Architecture

```
Multi-Agent Ensemble System
├── Specialized Agents
│   ├── Conservative Agent (Capital Preservation)
│   ├── Aggressive Agent (Maximum Returns)
│   ├── Balanced Agent (Risk-Adjusted Returns)
│   └── Contrarian Agent (Mean Reversion)
├── Market Regime Detection
│   ├── Rule-based Detection
│   ├── ML-enhanced Classification
│   └── Real-time Adaptation
├── Meta-Agent Orchestrator
│   ├── Agent Selection & Coordination
│   ├── Decision Combination Strategies
│   └── Risk Management
├── Multi-Armed Bandit Selector
│   ├── Contextual Selection
│   ├── Performance Learning
│   └── Exploration vs Exploitation
└── Performance Tracker
    ├── Individual Agent Metrics
    ├── Ensemble Analytics
    └── Risk Assessment
```

## 🚀 Quick Start

```python
from ensemble import create_default_ensemble_system

# Initialize ensemble system
ensemble = create_default_ensemble_system()

# Initialize with trading environment
ensemble.initialize(env)

# Train all agents
ensemble.train_all_agents(total_timesteps=100000)

# Make predictions
action, decision = ensemble.predict(observation, market_data)

# Update performance
ensemble.update_performance(actual_return, decision, individual_returns)

# Get comprehensive metrics
metrics = ensemble.get_comprehensive_metrics()
```

## 📦 Components

### 1. Specialized Agents (`agents/specialized_agents.py`)

Each agent is optimized for specific market conditions and trading objectives:

#### Conservative Agent
- **Objective**: Capital preservation and steady returns
- **Strategy**: Strict risk controls, low volatility targeting
- **Best In**: Bear markets, high volatility periods
- **Reward Focus**: Minimize drawdown (2.0x), reward consistency (1.0x)

#### Aggressive Agent  
- **Objective**: Maximum returns with higher risk tolerance
- **Strategy**: Large positions, momentum following
- **Best In**: Bull markets, trending conditions
- **Reward Focus**: Maximize returns (2.0x), momentum rewards (1.5x)

#### Balanced Agent
- **Objective**: Optimize risk-adjusted returns (Sharpe ratio)
- **Strategy**: Balanced position sizing, volatility targeting
- **Best In**: All market conditions (fallback)
- **Reward Focus**: Sharpe ratio optimization (1.5x), balanced risk (1.0x)

#### Contrarian Agent
- **Objective**: Mean reversion and contrarian trades
- **Strategy**: Trade against momentum, buy oversold/sell overbought
- **Best In**: Range-bound markets, mean-reverting conditions
- **Reward Focus**: Mean reversion signals (1.8x), contrarian profits (1.5x)

### 2. Market Regime Detection (`regime/market_regime_detector.py`)

Real-time classification of market conditions:

- **Bull Market**: Strong upward trending
- **Bear Market**: Strong downward trending  
- **Sideways**: Range-bound, low volatility
- **High Volatility**: Uncertain direction, high volatility
- **Mean Reverting**: Mean reversion patterns
- **Momentum**: Strong directional momentum

#### Detection Methods
- **Rule-based**: Technical indicators and market structure
- **ML-enhanced**: Random Forest with feature engineering
- **Contextual**: Market volatility, trend strength, volume analysis

### 3. Meta-Agent Orchestrator (`meta/meta_agent_orchestrator.py`)

Central coordination of the ensemble system:

#### Decision Strategies
- **Single Best**: Use highest confidence agent
- **Weighted Ensemble**: Combine based on agent weights
- **Majority Vote**: Democratic decision making
- **Regime Based**: Select optimal agents for current regime
- **Performance Weighted**: Weight by recent performance
- **Adaptive Weight**: Dynamic combination of factors

#### Risk Management
- **Position Sizing**: Adaptive based on market conditions
- **Emergency Stops**: Automatic risk controls
- **Drawdown Limits**: Portfolio protection
- **Volatility Scaling**: Dynamic position adjustment

### 4. Multi-Armed Bandit Selector (`bandit/strategy_selector.py`)

Intelligent agent selection using bandit algorithms:

#### Algorithms
- **Epsilon-Greedy**: Balance exploration vs exploitation
- **UCB (Upper Confidence Bound)**: Confidence-based selection
- **Thompson Sampling**: Bayesian sampling approach
- **Contextual UCB**: Context-aware selection
- **Sliding Window UCB**: Adapt to non-stationary environments

#### Features
- **Contextual Selection**: Based on market regime and conditions
- **Performance Learning**: Continuous adaptation to changing performance
- **Regret Minimization**: Optimal exploration-exploitation balance

### 5. Performance Tracker (`performance/ensemble_tracker.py`)

Comprehensive performance monitoring and analysis:

#### Metrics Tracked
- **Return Metrics**: Total, annualized, period returns
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown Analysis**: Maximum drawdown, recovery time
- **Risk Measures**: VaR, CVaR, Ulcer Index
- **Trading Metrics**: Win rate, profit factor, average win/loss

#### Analytics
- **Performance Attribution**: Agent contribution analysis
- **Regime Performance**: Performance by market condition
- **Risk Assessment**: Automated risk level classification
- **Benchmarking**: Comparison against baseline strategies

## 🔧 Configuration

### Orchestrator Configuration
```python
from ensemble import OrchestratorConfig, DecisionStrategy, RiskLevel

config = OrchestratorConfig(
    enabled_agents=[AgentType.CONSERVATIVE, AgentType.AGGRESSIVE, AgentType.BALANCED],
    decision_strategy=DecisionStrategy.ADAPTIVE_WEIGHT,
    risk_level=RiskLevel.ADAPTIVE,
    max_total_position=1.0,
    regime_confidence_threshold=0.6
)
```

### Bandit Configuration
```python
from ensemble import BanditConfig, BanditAlgorithm

config = BanditConfig(
    algorithm=BanditAlgorithm.CONTEXTUAL_UCB,
    epsilon=0.1,
    ucb_confidence=1.0,
    context_window=20
)
```

### Performance Configuration
```python
from ensemble import PerformanceConfig

config = PerformanceConfig(
    enable_regime_analysis=True,
    enable_attribution_analysis=True,
    risk_free_rate=0.02,
    log_frequency=100
)
```

## 📊 Usage Examples

### Basic Usage
```python
from ensemble import EnsembleSystem, AgentType
from rl_config import get_rl_config
from environment.trading_env import TradingEnvironment

# Create environment
rl_config = get_rl_config()
env = TradingEnvironment(config=rl_config, mode='train')

# Initialize ensemble
ensemble = EnsembleSystem()
ensemble.initialize(env)

# Train agents
training_results = ensemble.train_all_agents(total_timesteps=50000)

# Trading loop
observation = env.reset()
for step in range(1000):
    # Prepare market data
    market_data = {
        'price_history': get_price_history(),
        'volume_history': get_volume_history(),
        'volatility': calculate_volatility(),
        'portfolio_value': get_portfolio_value(),
        'current_drawdown': calculate_drawdown()
    }
    
    # Get ensemble decision
    action, decision = ensemble.predict(observation, market_data)
    
    # Execute trade
    observation, reward, done, info = env.step(action)
    
    # Update performance
    ensemble.update_performance(reward, decision)
    
    if done:
        observation = env.reset()

# Generate report
report = ensemble.generate_performance_report()
print(f"Total Return: {report['ensemble_performance']['total_return']:.2%}")
print(f"Sharpe Ratio: {report['ensemble_performance']['sharpe_ratio']:.3f}")
```

### Conservative Trading Setup
```python
from ensemble import create_conservative_ensemble_system

# Create conservative ensemble (lower risk)
ensemble = create_conservative_ensemble_system()
ensemble.initialize(env)

# Conservative training (more emphasis on stability)
ensemble.train_all_agents(total_timesteps=75000)
```

### Aggressive Trading Setup
```python
from ensemble import create_aggressive_ensemble_system

# Create aggressive ensemble (higher returns)
ensemble = create_aggressive_ensemble_system()
ensemble.initialize(env)

# Train with more exploration
ensemble.train_all_agents(total_timesteps=100000)
```

### Performance Analysis
```python
# Get comprehensive metrics
metrics = ensemble.get_comprehensive_metrics()

# Ensemble performance
ensemble_perf = metrics['ensemble_performance']
print(f"Sharpe Ratio: {ensemble_perf['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {ensemble_perf['max_drawdown']:.2%}")

# Agent comparison
agent_comparison = metrics['agent_comparison']
for agent_name, agent_metrics in agent_comparison.items():
    print(f"{agent_name}: {agent_metrics['total_return']:.2%}")

# Regime analysis
regime_analysis = metrics['regime_analysis']
for regime, regime_metrics in regime_analysis.items():
    print(f"{regime}: {regime_metrics['mean_return']:.4f}")

# Performance attribution
attribution = metrics['attribution_analysis']
for agent, contrib in attribution.items():
    if isinstance(contrib, dict):
        print(f"{agent}: {contrib['total_contribution']:.4f}")
```

### State Management
```python
# Save complete system state
ensemble.save_system_state('/path/to/save/ensemble_state')

# Load system state
ensemble.load_system_state('/path/to/save/ensemble_state')

# Get system summary
summary = ensemble.get_system_summary()
print(f"Agents: {summary['system_info']['num_agents']}")
print(f"Current Regime: {summary['current_regime']['regime']}")
```

## 🎯 Performance Objectives

The ensemble system is designed to achieve:

- **15%+ Annual Returns** with proper risk management
- **Sharpe Ratio > 1.5** through risk-adjusted optimization
- **Maximum Drawdown < 15%** with active risk controls
- **Win Rate > 55%** through intelligent agent selection
- **Volatility < 20%** through diversification and position sizing

## 🛡️ Risk Management

### Multi-Layer Risk Controls
1. **Agent-Level**: Individual risk constraints per agent
2. **Ensemble-Level**: Overall portfolio risk limits
3. **Position-Level**: Maximum position sizing
4. **Drawdown-Level**: Emergency stop mechanisms
5. **Volatility-Level**: Dynamic position scaling

### Risk Assessment
- **Real-time Risk Monitoring**: Continuous risk metric calculation
- **Risk Level Classification**: Automatic risk level assessment
- **Risk Warnings**: Automated alerts for risk threshold breaches
- **Recovery Strategies**: Automatic risk reduction protocols

## 📈 Market Adaptation

### Regime-Based Adaptation
- **Automatic Regime Detection**: Real-time market condition classification
- **Agent Selection**: Optimal agents for current market regime
- **Weight Adjustment**: Performance-based weight updates
- **Strategy Switching**: Dynamic decision strategy selection

### Performance Learning
- **Multi-Armed Bandits**: Intelligent exploration-exploitation balance
- **Contextual Learning**: Market condition-aware selection
- **Performance Tracking**: Continuous agent performance monitoring
- **Adaptive Weights**: Real-time weight optimization

## 🔍 Explainability

Every ensemble decision includes:
- **Selected Agents**: Which agents contributed to the decision
- **Agent Weights**: How much each agent influenced the final decision
- **Market Regime**: Current market condition detected
- **Strategy Used**: Which combination strategy was employed
- **Risk Assessment**: Current risk level and constraints applied
- **Performance Context**: Recent performance metrics

## 🧪 Testing

Run the example to see the system in action:

```bash
cd /Users/greenmachine2.0/Trading\ Bot\ Aug-15/tradingbot/backend/ensemble
python example_usage.py
```

This will:
1. Initialize the ensemble system
2. Train specialized agents
3. Simulate trading across different market regimes
4. Generate comprehensive performance reports
5. Demonstrate regime adaptation capabilities

## 📁 File Structure

```
ensemble/
├── __init__.py                     # Main ensemble system interface
├── README.md                       # This documentation
├── example_usage.py               # Complete example with simulated trading
├── agents/
│   └── specialized_agents.py      # Conservative, Aggressive, Balanced, Contrarian agents
├── regime/
│   └── market_regime_detector.py  # Market condition detection and classification
├── meta/
│   └── meta_agent_orchestrator.py # Central coordination and decision making
├── bandit/
│   └── strategy_selector.py       # Multi-armed bandit agent selection
└── performance/
    └── ensemble_tracker.py        # Performance monitoring and analytics
```

## 🔮 Future Enhancements

- **Additional Agents**: Sector-specific, momentum, arbitrage agents
- **Advanced Regimes**: Volatility clustering, correlation regimes
- **Deep Learning**: Neural network-based regime detection
- **Multi-Asset**: Support for multiple cryptocurrency pairs
- **Real-time Data**: Integration with live market data feeds
- **Backtesting**: Historical performance validation framework

## 📞 Support

For questions or issues:
1. Check the example usage script for implementation patterns
2. Review agent configurations for customization options
3. Examine performance reports for system insights
4. Analyze decision explanations for transparency

The ensemble system is designed to be robust, adaptable, and transparent, providing sophisticated trading capabilities while maintaining explainability and risk control.