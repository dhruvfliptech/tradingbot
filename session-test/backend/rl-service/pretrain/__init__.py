"""
RL Pre-training Pipeline Package

This package provides a comprehensive pre-training pipeline that leverages
Composer MCP's 1000+ strategies to give RL agents a strong baseline through
supervised learning and knowledge transfer.

Modules:
- composer_extractor: Extract trading patterns from Composer strategies
- pattern_analyzer: Analyze and categorize trading patterns
- pretrain_pipeline: Main orchestration of the pre-training process
- transfer_learning: Transfer knowledge from supervised model to RL agent
- validation: Validate pre-trained agent performance

Key Classes:
- ComposerExtractor: Extracts patterns from Composer MCP
- PatternAnalyzer: Analyzes pattern types and market regimes
- PretrainingPipeline: Main pipeline orchestrator
- TransferLearningManager: Handles knowledge transfer
- PretrainValidator: Validates performance improvements

Example Usage:
    from pretrain import PretrainingPipeline, PretrainingConfig
    
    config = PretrainingConfig(
        max_strategies=100,
        min_confidence=0.7,
        num_epochs=50
    )
    
    pipeline = PretrainingPipeline(config)
    results = await pipeline.run_pipeline()
"""

from .composer_extractor import (
    ComposerExtractor,
    StrategyPattern,
    MarketState
)

from .pattern_analyzer import (
    PatternAnalyzer,
    PatternAnalysis,
    PatternCluster
)

from .pretrain_pipeline import (
    PretrainingPipeline,
    PretrainingConfig,
    PretrainingDataset,
    SupervisedPretrainer
)

from .transfer_learning import (
    TransferLearningManager,
    TransferConfig,
    KnowledgeDistillationLoss,
    FeatureAlignmentLoss
)

from .validation import (
    PretrainValidator,
    ValidationConfig,
    ValidationResult,
    ComparisonResult
)

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "RL Pre-training Pipeline using Composer MCP Strategies"

__all__ = [
    # Extractor components
    'ComposerExtractor',
    'StrategyPattern', 
    'MarketState',
    
    # Analyzer components
    'PatternAnalyzer',
    'PatternAnalysis',
    'PatternCluster',
    
    # Pipeline components
    'PretrainingPipeline',
    'PretrainingConfig',
    'PretrainingDataset', 
    'SupervisedPretrainer',
    
    # Transfer learning components
    'TransferLearningManager',
    'TransferConfig',
    'KnowledgeDistillationLoss',
    'FeatureAlignmentLoss',
    
    # Validation components
    'PretrainValidator',
    'ValidationConfig',
    'ValidationResult',
    'ComparisonResult'
]