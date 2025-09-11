#!/usr/bin/env python3
"""
Example Usage of RL Pre-training Pipeline

This script demonstrates how to use the complete pre-training pipeline
to extract patterns from Composer MCP strategies and pre-train an RL agent.

Run with: python example_usage.py
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from pathlib import Path

# Import pipeline components
from pretrain_pipeline import PretrainingPipeline, PretrainingConfig
from composer_extractor import ComposerExtractor
from pattern_analyzer import PatternAnalyzer
from transfer_learning import TransferLearningManager, TransferConfig
from validation import PretrainValidator, ValidationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pretrain_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_complete_pipeline():
    """Run the complete pre-training pipeline from start to finish"""
    logger.info("=" * 80)
    logger.info("STARTING RL PRE-TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Configuration
    output_dir = Path("/tmp/rl_pretrain_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Pipeline configuration
    config = PretrainingConfig(
        # Data extraction settings
        max_strategies=50,  # Reduced for demo
        min_confidence=0.6,
        min_performance_threshold=0.05,
        
        # Training settings
        batch_size=128,
        learning_rate=1e-4,
        num_epochs=30,  # Reduced for demo
        validation_split=0.2,
        
        # Model architecture
        state_dim=64,
        action_dim=3,
        hidden_dims=[512, 256, 128],
        
        # Transfer learning
        transfer_strategy='fine_tuning',
        freeze_layers=['hidden1'],
        
        # Validation
        validation_episodes=50,  # Reduced for demo
        validation_envs=['BTC-USD', 'ETH-USD'],
        
        # Output
        output_dir=str(output_dir)
    )
    
    try:
        # Create and run pipeline
        pipeline = PretrainingPipeline(config)
        results = await pipeline.run_pipeline()
        
        # Print summary
        print_pipeline_summary(results)
        
        # Save detailed results
        save_pipeline_artifacts(results, output_dir)
        
        logger.info("Pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

async def run_individual_components_demo():
    """Demonstrate individual components of the pipeline"""
    logger.info("=" * 80)
    logger.info("DEMONSTRATING INDIVIDUAL COMPONENTS")
    logger.info("=" * 80)
    
    output_dir = Path("/tmp/rl_pretrain_components")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Pattern Extraction Demo
    await demo_pattern_extraction(output_dir)
    
    # 2. Pattern Analysis Demo
    await demo_pattern_analysis(output_dir)
    
    # 3. Transfer Learning Demo
    demo_transfer_learning(output_dir)
    
    # 4. Validation Demo
    await demo_validation(output_dir)

async def demo_pattern_extraction(output_dir: Path):
    """Demonstrate pattern extraction from Composer"""
    logger.info("1. Pattern Extraction Demo")
    logger.info("-" * 40)
    
    try:
        async with ComposerExtractor() as extractor:
            # Extract patterns from a few strategies
            pattern_count = await extractor.run_extraction_pipeline(
                max_strategies=10,  # Small demo
                min_performance_threshold=0.1
            )
            
            logger.info(f"Extracted {pattern_count} patterns")
            
            # Load and examine patterns
            patterns = extractor.load_patterns(min_confidence=0.5)
            logger.info(f"Loaded {len(patterns)} patterns for analysis")
            
            # Show pattern distribution
            if patterns:
                pattern_types = {}
                market_regimes = {}
                
                for pattern in patterns[:100]:  # Examine first 100
                    pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
                    market_regimes[pattern.market_regime] = market_regimes.get(pattern.market_regime, 0) + 1
                
                logger.info(f"Pattern types: {pattern_types}")
                logger.info(f"Market regimes: {market_regimes}")
                
                # Save sample patterns
                sample_patterns_file = output_dir / "sample_patterns.json"
                with open(sample_patterns_file, 'w') as f:
                    sample_data = {
                        'total_patterns': len(patterns),
                        'pattern_distribution': pattern_types,
                        'regime_distribution': market_regimes,
                        'sample_patterns': [
                            {
                                'strategy_id': p.strategy_id,
                                'pattern_type': p.pattern_type,
                                'market_regime': p.market_regime,
                                'reward': p.reward,
                                'confidence': p.confidence
                            }
                            for p in patterns[:10]  # First 10 patterns
                        ]
                    }
                    json.dump(sample_data, f, indent=2, default=str)
                
                logger.info(f"Sample patterns saved to {sample_patterns_file}")
    
    except Exception as e:
        logger.error(f"Pattern extraction demo failed: {e}")

async def demo_pattern_analysis(output_dir: Path):
    """Demonstrate pattern analysis"""
    logger.info("2. Pattern Analysis Demo")
    logger.info("-" * 40)
    
    try:
        # Load patterns
        extractor = ComposerExtractor()
        patterns = extractor.load_patterns(min_confidence=0.5)
        
        if not patterns:
            logger.warning("No patterns available for analysis demo")
            return
        
        # Analyze patterns
        analyzer = PatternAnalyzer(n_clusters=4)  # Smaller for demo
        analysis = analyzer.analyze_patterns(patterns[:200])  # Analyze subset
        
        logger.info(f"Analysis completed for {analysis.total_patterns} patterns")
        logger.info(f"Found {len(analysis.clusters)} pattern clusters")
        logger.info(f"Detected {len(analysis.anomalies)} anomalous patterns")
        
        # Show key insights
        logger.info("Key Performance Metrics:")
        for metric, value in analysis.performance_summary.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("Recommendations:")
        for i, rec in enumerate(analysis.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # Export analysis
        analysis_file = output_dir / "pattern_analysis.json"
        analyzer.export_analysis(analysis, str(analysis_file))
        
        # Create visualizations
        viz_file = output_dir / "pattern_analysis.png"
        analyzer.visualize_patterns(analysis, str(viz_file))
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Pattern analysis demo failed: {e}")

def demo_transfer_learning(output_dir: Path):
    """Demonstrate transfer learning"""
    logger.info("3. Transfer Learning Demo")
    logger.info("-" * 40)
    
    try:
        # Create transfer learning manager
        transfer_config = TransferConfig(
            strategy='fine_tuning',
            freeze_layers=['layer1'],
            learning_rate_scale=0.1
        )
        
        transfer_manager = TransferLearningManager(transfer_config)
        
        # Simulate transfer (would need actual models in practice)
        logger.info("Transfer learning configuration:")
        logger.info(f"  Strategy: {transfer_config.strategy}")
        logger.info(f"  Learning rate scale: {transfer_config.learning_rate_scale}")
        logger.info(f"  Frozen layers: {transfer_config.freeze_layers}")
        
        # Save transfer config
        config_file = output_dir / "transfer_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'strategy': transfer_config.strategy,
                'freeze_layers': transfer_config.freeze_layers,
                'learning_rate_scale': transfer_config.learning_rate_scale,
                'adaptation_epochs': transfer_config.adaptation_epochs
            }, f, indent=2)
        
        logger.info(f"Transfer learning demo completed - config saved to {config_file}")
    
    except Exception as e:
        logger.error(f"Transfer learning demo failed: {e}")

async def demo_validation(output_dir: Path):
    """Demonstrate validation"""
    logger.info("4. Validation Demo")
    logger.info("-" * 40)
    
    try:
        # Create validation configuration
        val_config = ValidationConfig(
            validation_episodes=20,  # Small for demo
            validation_symbols=['BTC-USD'],
            compare_against_random=True,
            compare_against_buy_hold=True
        )
        
        validator = PretrainValidator(val_config)
        
        logger.info("Validation configuration:")
        logger.info(f"  Episodes: {val_config.validation_episodes}")
        logger.info(f"  Symbols: {val_config.validation_symbols}")
        logger.info(f"  Baseline comparisons: Random, Buy & Hold")
        
        # Simulate validation results (would run actual validation in practice)
        mock_results = {
            'symbols_tested': val_config.validation_symbols,
            'total_episodes': val_config.validation_episodes,
            'average_metrics': {
                'sharpe_ratio': {'mean': 1.2, 'std': 0.3},
                'max_drawdown': {'mean': 0.15, 'std': 0.05},
                'win_rate': {'mean': 0.65, 'std': 0.1}
            },
            'recommendations': [
                "Good risk-adjusted performance achieved",
                "Consider position sizing optimization",
                "Monitor drawdown levels in live trading"
            ]
        }
        
        # Create validation report
        report = validator.create_validation_report(mock_results)
        report_file = output_dir / "validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation demo completed - report saved to {report_file}")
        
        # Save validation config
        config_file = output_dir / "validation_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'validation_episodes': val_config.validation_episodes,
                'validation_symbols': val_config.validation_symbols,
                'metrics_evaluated': val_config.metrics_to_evaluate
            }, f, indent=2)
    
    except Exception as e:
        logger.error(f"Validation demo failed: {e}")

def print_pipeline_summary(results: dict):
    """Print a summary of pipeline results"""
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    # Extraction results
    if 'extraction_results' in results:
        extraction = results['extraction_results']
        print(f"Pattern Extraction:")
        print(f"  Total patterns extracted: {extraction.get('total_patterns_extracted', 'N/A')}")
        print(f"  Extraction completed: {extraction.get('extraction_completed', False)}")
    
    # Analysis results
    if 'analysis_results' in results:
        analysis = results['analysis_results']
        if hasattr(analysis, 'total_patterns'):
            print(f"\nPattern Analysis:")
            print(f"  Patterns analyzed: {analysis.total_patterns}")
            print(f"  Clusters found: {len(analysis.clusters) if hasattr(analysis, 'clusters') else 'N/A'}")
            print(f"  Anomalies detected: {len(analysis.anomalies) if hasattr(analysis, 'anomalies') else 'N/A'}")
    
    # Training results
    if 'training_results' in results:
        training = results['training_results']
        if 'epoch' in training and training['epoch']:
            print(f"\nSupervised Pre-training:")
            print(f"  Epochs completed: {training['epoch'][-1] + 1}")
            print(f"  Final training accuracy: {training['train_acc'][-1]:.3f}")
            if 'val_acc' in training and training['val_acc']:
                print(f"  Final validation accuracy: {training['val_acc'][-1]:.3f}")
    
    # Transfer results
    if 'transfer_results' in results:
        transfer = results['transfer_results']
        print(f"\nTransfer Learning:")
        print(f"  Transfer completed: {transfer.get('transfer_completed', False)}")
        if 'transferred_layers' in transfer:
            print(f"  Layers transferred: {len(transfer['transferred_layers'])}")
    
    # Validation results
    if 'validation_results' in results:
        validation = results['validation_results']
        print(f"\nValidation:")
        print(f"  Validation completed: {validation.get('validation_completed', False)}")
    
    print("=" * 80)

def save_pipeline_artifacts(results: dict, output_dir: Path):
    """Save pipeline artifacts and results"""
    logger.info("Saving pipeline artifacts...")
    
    # Save complete results
    results_file = output_dir / "complete_results.json"
    try:
        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                serializable_results[key] = value.__dict__
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Complete results saved to {results_file}")
    except Exception as e:
        logger.warning(f"Could not save complete results: {e}")
    
    # Save execution summary
    summary_file = output_dir / "execution_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"RL Pre-training Pipeline Execution Summary\n")
        f.write(f"Executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        for phase, result in results.items():
            f.write(f"{phase.replace('_', ' ').title()}:\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {result}\n")
            f.write("\n")
    
    logger.info(f"Execution summary saved to {summary_file}")

async def main():
    """Main function to run examples"""
    print("RL Pre-training Pipeline Examples")
    print("=" * 50)
    print("1. Complete Pipeline Demo")
    print("2. Individual Components Demo")
    print("3. Both")
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == "1":
        await run_complete_pipeline()
    elif choice == "2":
        await run_individual_components_demo()
    elif choice == "3":
        await run_individual_components_demo()
        await run_complete_pipeline()
    else:
        print("Invalid choice. Running individual components demo...")
        await run_individual_components_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise