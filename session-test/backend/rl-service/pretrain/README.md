# RL Pre-training Pipeline

A comprehensive pre-training pipeline that leverages Composer MCP's 1000+ strategies to give RL agents a strong baseline through supervised learning and transfer learning.

## Overview

This pipeline extracts successful trading patterns from Composer's extensive strategy library, analyzes and categorizes them, then uses supervised learning to pre-train RL agents before reinforcement learning begins. This approach significantly improves convergence speed and final performance compared to random initialization.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Composer MCP   │───▶│ Pattern Extract │───▶│ Pattern Analysis│
│   Strategies    │    │     & Store     │    │ & Categorization│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validation    │◀───│ Transfer Learn  │◀───│ Supervised Pre- │
│  & Evaluation   │    │   to RL Agent   │    │    training     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Composer Extractor (`composer_extractor.py`)
- Connects to Composer MCP service
- Extracts successful trading patterns from 1000+ strategies
- Categorizes patterns by type (momentum, mean reversion, breakout, trend following)
- Associates patterns with market regimes (bull, bear, sideways, volatile)
- Stores patterns in local SQLite database

**Key Features:**
- Async pattern extraction for scalability
- Pattern quality filtering based on confidence and performance
- Market state representation with technical indicators
- Comprehensive trade analysis and reward calculation

### 2. Pattern Analyzer (`pattern_analyzer.py`)
- Analyzes extracted patterns using ML techniques
- Performs clustering to identify similar patterns
- Calculates feature importance and correlations
- Detects anomalous patterns using isolation forest
- Generates actionable recommendations

**Key Features:**
- Multiple clustering algorithms (K-means, DBSCAN)
- Statistical analysis and normality testing
- Pattern performance evaluation
- Visualization and export capabilities

### 3. Pre-training Pipeline (`pretrain_pipeline.py`)
- Main orchestrator for the entire pipeline
- Implements supervised learning on extracted patterns
- Manages training data preparation and augmentation
- Handles model training with weighted sampling
- Coordinates all pipeline components

**Key Features:**
- Configurable training parameters
- Weighted sampling based on pattern quality
- Early stopping and learning rate scheduling
- Comprehensive training history tracking

### 4. Transfer Learning (`transfer_learning.py`)
- Transfers knowledge from supervised model to RL agent
- Supports multiple transfer strategies (fine-tuning, feature extraction, progressive)
- Implements knowledge distillation and feature alignment
- Manages differential learning rates for transferred layers

**Key Features:**
- Multiple transfer learning strategies
- Knowledge distillation loss functions
- Progressive layer unfreezing
- Catastrophic forgetting prevention

### 5. Validation (`validation.py`)
- Validates pre-trained agent performance
- Compares against multiple baselines (random, buy-hold, vanilla RL)
- Performs statistical significance testing
- Generates comprehensive performance reports

**Key Features:**
- Multiple baseline comparisons
- Statistical significance testing
- Risk-adjusted performance metrics
- Automated report generation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Composer MCP service is accessible at the configured URL

3. Verify RL environment components are available

## Usage

### Quick Start

```python
import asyncio
from pretrain import PretrainingPipeline, PretrainingConfig

# Configure pipeline
config = PretrainingConfig(
    max_strategies=100,
    min_confidence=0.7,
    num_epochs=50,
    batch_size=256,
    validation_episodes=200
)

# Run pipeline
async def main():
    pipeline = PretrainingPipeline(config)
    results = await pipeline.run_pipeline()
    print(f"Pipeline completed with {results['training_results']['epoch'][-1]} epochs")

asyncio.run(main())
```

### Individual Components

```python
# Extract patterns
from pretrain import ComposerExtractor

async with ComposerExtractor() as extractor:
    pattern_count = await extractor.run_extraction_pipeline(max_strategies=50)
    patterns = extractor.load_patterns(min_confidence=0.6)

# Analyze patterns
from pretrain import PatternAnalyzer

analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(patterns)
analyzer.visualize_patterns(analysis, save_path="analysis.png")

# Transfer to RL agent
from pretrain import TransferLearningManager

transfer_manager = TransferLearningManager()
results = transfer_manager.transfer_knowledge(
    source_model_path="pretrained_model.pt",
    target_agent=rl_agent,
    transfer_strategy='fine_tuning'
)

# Validate performance
from pretrain import PretrainValidator

validator = PretrainValidator()
validation_results = await validator.validate_pretrained_performance(
    agent_path="transferred_agent.pt",
    validation_episodes=100
)
```

### Configuration Options

#### PretrainingConfig
```python
config = PretrainingConfig(
    # Data extraction
    max_strategies=100,           # Number of strategies to extract from
    min_confidence=0.6,           # Minimum pattern confidence threshold
    min_performance_threshold=0.1, # Minimum strategy performance
    
    # Pattern analysis
    n_clusters=8,                 # Number of pattern clusters
    clustering_method='kmeans',   # Clustering algorithm
    
    # Training parameters
    batch_size=256,               # Training batch size
    learning_rate=1e-4,           # Learning rate
    num_epochs=100,               # Maximum training epochs
    validation_split=0.2,         # Validation set proportion
    
    # Model architecture
    state_dim=64,                 # State vector dimension
    action_dim=3,                 # Number of actions (hold, buy, sell)
    hidden_dims=[512, 256, 128],  # Hidden layer dimensions
    
    # Transfer learning
    transfer_strategy='fine_tuning', # Transfer strategy
    freeze_layers=[],             # Layers to freeze during transfer
    
    # Validation
    validation_episodes=200,      # Episodes for validation
    validation_envs=['BTC-USD', 'ETH-USD'], # Assets to test on
    
    # Output
    output_dir="/tmp/rl_pretrain" # Output directory
)
```

## Pattern Categories

The pipeline automatically categorizes extracted patterns into four main types:

### 1. Momentum Patterns
- **Description**: Capitalize on price momentum and trending behavior
- **Indicators**: Moving averages, RSI, momentum indicators
- **Market Conditions**: Trending markets, moderate to high volatility
- **Typical Duration**: Short to medium term

### 2. Mean Reversion Patterns  
- **Description**: Profit from price reversions to mean values
- **Indicators**: Bollinger Bands, RSI, Z-score
- **Market Conditions**: Sideways markets, oversold/overbought conditions
- **Typical Duration**: Short term

### 3. Breakout Patterns
- **Description**: Capture breakout movements from consolidation
- **Indicators**: Volume, volatility, support/resistance levels
- **Market Conditions**: Low volatility consolidation periods
- **Typical Duration**: Medium term

### 4. Trend Following Patterns
- **Description**: Follow established market trends
- **Indicators**: Trend lines, moving averages, ADX
- **Market Conditions**: Strong trending markets
- **Typical Duration**: Medium to long term

## Market Regimes

Patterns are associated with specific market regimes:

- **Bull**: Upward trending, positive sentiment, low to medium volatility
- **Bear**: Downward trending, negative sentiment, high volatility  
- **Sideways**: Neutral trend, range-bound, low volatility
- **Volatile**: Mixed trends, high uncertainty, high volatility

## Performance Metrics

The pipeline evaluates performance using comprehensive metrics:

### Risk-Adjusted Returns
- Sharpe Ratio
- Sortino Ratio  
- Calmar Ratio
- Information Ratio

### Risk Metrics
- Maximum Drawdown
- Value at Risk (VaR 95%, 99%)
- Expected Shortfall
- Volatility

### Trade Statistics
- Win Rate
- Profit Factor
- Average Win/Loss
- Trade Duration
- Consecutive Win/Loss Streaks

## Output Artifacts

The pipeline generates comprehensive outputs:

### Models
- `pretrained_model.pt` - Supervised pre-trained model
- `transferred_rl_agent.pt` - RL agent with transferred knowledge
- `best_model.pt` - Best performing model checkpoint

### Analysis Reports
- `pattern_analysis.json` - Detailed pattern analysis
- `pattern_analysis.png` - Visualization plots
- `validation_report.txt` - Performance validation report
- `pipeline_results.json` - Complete pipeline results

### Data
- `composer_patterns.db` - SQLite database of extracted patterns
- Training/validation datasets
- Feature importance rankings
- Cluster analysis results

## Advanced Features

### Knowledge Distillation
The transfer learning component implements knowledge distillation to effectively transfer learned representations:

```python
# Knowledge distillation loss
kd_loss = KnowledgeDistillationLoss(temperature=3.0, alpha=0.3)
loss = kd_loss(student_logits, teacher_logits, true_labels)
```

### Progressive Transfer
Gradually unfreeze layers during transfer learning:

```python
config = TransferConfig(
    strategy='progressive',
    progressive_unfreeze_schedule=[5, 10, 15]  # Unfreeze at these epochs
)
```

### Weighted Sampling
Patterns are weighted based on confidence and performance:

```python
# Weight calculation considers:
# - Pattern confidence score
# - Trade PnL and success rate  
# - Strategy overall performance (Sharpe ratio, win rate)
weight = confidence * reward_weight * strategy_weight
```

### Statistical Validation
Comprehensive statistical testing ensures robustness:

```python
# Performed automatically in validation
- Shapiro-Wilk normality tests
- Confidence interval calculations  
- Statistical significance testing
- Effect size measurements
```

## Best Practices

### 1. Data Quality
- Filter patterns by minimum confidence (≥0.6 recommended)
- Ensure sufficient pattern diversity across market regimes
- Validate strategy performance before pattern extraction

### 2. Training
- Use weighted sampling to emphasize high-quality patterns
- Implement early stopping to prevent overfitting
- Monitor validation metrics throughout training

### 3. Transfer Learning
- Start with fine-tuning strategy for most cases
- Use feature extraction for domain adaptation
- Apply progressive unfreezing for complex transfers

### 4. Validation
- Test across multiple assets and time periods
- Compare against relevant baselines
- Perform statistical significance testing
- Monitor out-of-sample performance

## Troubleshooting

### Common Issues

1. **Composer MCP Connection Failed**
   - Verify MCP service URL and connectivity
   - Check authentication credentials
   - Ensure network connectivity

2. **Insufficient Patterns Extracted**
   - Lower minimum confidence threshold
   - Increase maximum strategies parameter
   - Check strategy performance filters

3. **Poor Transfer Learning Results**
   - Verify model architecture compatibility
   - Check transfer strategy selection
   - Adjust learning rate scaling

4. **Validation Failures**
   - Ensure trading environment is properly configured
   - Check data availability for validation periods
   - Verify baseline agent implementations

### Performance Optimization

1. **Memory Usage**
   - Reduce batch size if running out of memory
   - Use gradient accumulation for large batches
   - Clear pattern cache periodically

2. **Training Speed**
   - Use GPU acceleration when available
   - Implement data loading optimization
   - Reduce validation frequency during training

3. **Storage**
   - Compress pattern database periodically
   - Implement pattern archiving for old data
   - Use efficient serialization formats

## Examples

See `example_usage.py` for comprehensive examples demonstrating:
- Complete pipeline execution
- Individual component usage
- Custom configuration setups
- Results analysis and visualization

## Contributing

1. Follow the existing code structure and documentation standards
2. Add unit tests for new components
3. Update README and docstrings for new features
4. Ensure compatibility with existing pipeline components

## License

This project is part of the Trading Bot system and follows the same licensing terms.