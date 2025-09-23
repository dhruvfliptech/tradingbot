# PPO Trading Agents

A comprehensive reinforcement learning system for cryptocurrency trading using Proximal Policy Optimization (PPO) with production-ready features.

## 🌟 Features

### Core Components
- **PPO Agent**: Main trading agent with custom policy networks
- **Policy Networks**: Attention-based and risk-aware neural architectures
- **Training Pipeline**: Advanced training with hyperparameter optimization
- **Ensemble Agent**: Multi-agent system for different market conditions
- **Explainability**: Decision analysis and feature importance

### Production Features
- **Model Versioning**: Track and manage different model versions
- **A/B Testing**: Compare model performance and gradual rollouts
- **Risk Management**: Built-in risk constraints and monitoring
- **Explainability**: Understand agent decisions with multiple analysis methods
- **Market Regime Detection**: Automatic switching between specialized agents

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from agents import PPOAgent, PPOConfig
from environment.trading_env import TradingEnvironment
from rl_config import get_rl_config

# Create environment
rl_config = get_rl_config()
env = TradingEnvironment(config=rl_config, mode='train')
env.load_data(start_date, end_date)

# Create and train agent
config = PPOConfig(
    policy_type='attention',
    total_timesteps=100000,
    enable_versioning=True
)
agent = PPOAgent(env, config)
agent.train()

# Make predictions
obs, _ = env.reset()
action, info = agent.predict(obs)
```

## 📋 Components

### 1. PPO Agent (`ppo_agent.py`)

Main PPO agent with production features:

```python
from agents import PPOAgent, PPOConfig

config = PPOConfig(
    policy_type='attention',  # 'standard', 'attention', 'risk_aware'
    learning_rate=3e-4,
    total_timesteps=1000000,
    enable_versioning=True,
    enable_ab_testing=True,
    performance_threshold=0.15  # 15% improvement for deployment
)

agent = PPOAgent(env, config)
```

**Key Features:**
- Custom policy networks with attention mechanisms
- Model versioning and deployment management
- Performance tracking and A/B testing
- Risk-aware action selection
- Gradual rollout capabilities

### 2. Policy Networks (`policy_network.py`)

Custom neural network architectures:

```python
from agents import TradingPolicyNetwork, AttentionPolicy, RiskAwarePolicy

# Attention-based policy
policy = AttentionPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda x: 3e-4
)

# Risk-aware policy with uncertainty estimation
policy = RiskAwarePolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda x: 3e-4,
    config=NetworkConfig(risk_aversion=0.1)
)
```

**Features:**
- Multi-head attention for temporal patterns
- Risk-aware networks with uncertainty estimation
- Custom feature extractors for trading signals
- Ensemble policy networks

### 3. Training Pipeline (`trainer.py`)

Advanced training with optimization:

```python
from agents import PPOTrainer, TrainingConfig

config = TrainingConfig(
    total_timesteps=1000000,
    enable_early_stopping=True,
    enable_tensorboard=True,
    enable_hyperopt=True,  # Optuna optimization
    patience=10,
    checkpoint_frequency=50000
)

trainer = PPOTrainer(env_factory, config)
results = trainer.train(ppo_config)
```

**Features:**
- Hyperparameter optimization with Optuna
- Early stopping with multiple criteria
- TensorBoard logging and monitoring
- Advanced checkpointing
- Cross-validation support

### 4. Ensemble Agent (`ensemble_agent.py`)

Multi-agent system for different market conditions:

```python
from agents import EnsembleAgent, EnsembleConfig

config = EnsembleConfig(
    n_agents=4,
    agent_types=['bull', 'bear', 'sideways', 'high_volatility'],
    weighting_method='confidence'
)

ensemble = EnsembleAgent(env, config)
ensemble.train_ensemble(historical_data, base_config)
```

**Features:**
- Specialized agents for different market regimes
- Automatic market regime detection
- Dynamic weight allocation
- Performance-based agent selection

### 5. Explainability (`explainer.py`)

Comprehensive decision analysis:

```python
from agents import AgentExplainer

explainer = AgentExplainer(agent, feature_names, env)

# Explain a decision
explanation = explainer.explain_decision(observation)
print(f"Action: {explanation.action_name}")
print(f"Confidence: {explanation.confidence:.2%}")
print(f"Explanation: {explanation.explanation_text}")

# Analyze feature importance
importance = explainer.analyze_feature_importance(
    methods=['permutation', 'shap', 'attention']
)
```

**Features:**
- Multiple feature importance methods
- Counterfactual analysis
- Risk attribution
- Natural language explanations
- Visualization tools

## 🏗️ Architecture

### Agent Hierarchy
```
PPOAgent (Base)
├── Policy Networks
│   ├── Standard Policy
│   ├── Attention Policy
│   └── Risk-Aware Policy
├── Training Pipeline
│   ├── Hyperparameter Optimization
│   ├── Early Stopping
│   └── Checkpointing
└── Production Features
    ├── Model Versioning
    ├── A/B Testing
    └── Deployment

EnsembleAgent
├── Market Regime Detector
├── Specialized Agents
│   ├── Bull Market Agent
│   ├── Bear Market Agent
│   ├── Sideways Agent
│   └── High Volatility Agent
└── Dynamic Weighting
```

### Data Flow
```
Market Data → Environment → Observation
                              ↓
Observation → Policy Network → Action Probabilities
                              ↓
Action Selection → Environment Step → Reward
                              ↓
Reward → Training Update → Policy Improvement
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### Financial Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst portfolio decline
- **Total Return**: Cumulative performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Technical Metrics
- **Training Loss**: Model convergence
- **Policy Entropy**: Action diversity
- **Value Function Error**: Critic accuracy
- **Gradient Norm**: Training stability

### Risk Metrics
- **Value at Risk (VaR)**: Potential losses
- **Conditional VaR**: Expected tail losses
- **Beta**: Market correlation
- **Volatility**: Return variability

## 🔧 Configuration

### Environment Variables
```bash
# Training parameters
RL_LEARNING_RATE=3e-4
RL_TOTAL_TIMESTEPS=1000000
RL_BATCH_SIZE=64

# Model management
RL_MODEL_SAVE_PATH=/models/rl
RL_ENABLE_TENSORBOARD=true
RL_ENABLE_VERSIONING=true

# Risk management
RL_MAX_DRAWDOWN_THRESHOLD=0.2
RL_RISK_FREE_RATE=0.02

# Performance thresholds
RL_PERFORMANCE_THRESHOLD=0.15
RL_ROLLOUT_PERCENTAGE=0.1
```

### Configuration Files

**PPO Config:**
```python
PPOConfig(
    # Algorithm parameters
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    
    # Policy configuration
    policy_type='attention',
    policy_kwargs={
        'attention_heads': 8,
        'dropout_rate': 0.1
    },
    
    # Production features
    enable_versioning=True,
    enable_ab_testing=True,
    performance_threshold=0.15
)
```

**Training Config:**
```python
TrainingConfig(
    total_timesteps=1000000,
    eval_frequency=10000,
    enable_early_stopping=True,
    enable_hyperopt=True,
    enable_tensorboard=True,
    patience=10
)
```

## 📈 Monitoring and Logging

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir=/tmp/rl_logs/tensorboard

# View metrics:
# - Training loss and rewards
# - Policy entropy and KL divergence
# - Value function accuracy
# - Custom trading metrics
```

### Performance Tracking
```python
# Get comprehensive metrics
metrics = agent.get_metrics()

# Performance comparison
comparison = agent.get_performance_comparison()

# Check deployment criteria
ready = agent.meets_deployment_criteria()
```

## 🚀 Deployment

### Model Versioning
```python
# Register a new version
version = agent.version_manager.register_version(
    version="1.2.0",
    model_path=model_path,
    metadata={"sharpe_ratio": 1.5}
)

# Deploy with gradual rollout
agent.deploy_model(version, rollout_percentage=0.1)
```

### A/B Testing
```python
# Create baseline performance tracker
tracker = PerformanceTracker(baseline_metrics)

# Update with new metrics
tracker.update_metrics(current_metrics)

# Check deployment criteria
meets_criteria = tracker.meets_deployment_criteria(threshold=0.15)
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agents --cov-report=html
```

### Integration Tests
```bash
# Test complete training pipeline
python -m pytest tests/test_integration.py

# Test ensemble training
python -m pytest tests/test_ensemble.py
```

### Example Usage
```bash
# Run comprehensive examples
python agents/example_usage.py

# Test specific components
python agents/ppo_agent.py
python agents/ensemble_agent.py
python agents/explainer.py
```

## 📚 Advanced Usage

### Custom Policy Networks
```python
from agents.policy_network import NetworkConfig

config = NetworkConfig(
    hidden_dims=[512, 512, 256],
    attention_heads=16,
    dropout_rate=0.1,
    risk_aversion=0.2,
    uncertainty_estimation=True
)

policy = create_policy_network(
    'risk_aware', 
    observation_space, 
    action_space,
    config=config
)
```

### Hyperparameter Optimization
```python
from agents.trainer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(env_factory, training_config)
study = optimizer.create_study()

# Run optimization
results = optimizer.optimize(n_trials=100, timeout=3600)
best_params = results['best_params']
```

### Custom Ensemble Configurations
```python
# Create custom market regime detector
detector = MarketRegimeDetector(n_regimes=5)
detector.fit(historical_data, custom_labels)

# Custom ensemble with specialized agents
ensemble_config = EnsembleConfig(
    agent_types=['momentum', 'mean_reversion', 'volatility', 'sentiment'],
    weighting_method='performance',
    confidence_threshold=0.8
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black agents/
flake8 agents/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Stable-Baselines3 for RL implementations
- Optuna for hyperparameter optimization
- SHAP for explainability analysis
- OpenAI for PPO algorithm research

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example usage files

---

**Built with ❤️ for profitable crypto trading**