#!/usr/bin/env python
"""Run complete experiment pipeline.

This script runs:
1. Data preprocessing
2. Baseline pre-training and fine-tuning
3. Low-rank pre-training and fine-tuning (multiple ranks)
4. Comparison and visualization

Usage:
    python -m experiments.run_all --reviews path/to/reviews.json --metadata path/to/meta.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data.preprocessing import preprocess_amazon_beauty, load_processed_data
from data.dataset import create_dataloaders
from models.s3rec import S3RecModel, S3RecLowRankModel
from trainers.pretrain import PretrainTrainer
from trainers.finetune import FinetuneTrainer
from utils.helpers import set_seed, save_results, print_model_summary
from utils.visualization import (
    plot_training_curves,
    plot_rank_comparison,
    plot_parameter_reduction,
    create_results_table
)


def run_experiment(
    model_class,
    data: dict,
    config: dict,
    experiment_name: str,
    output_dir: Path,
    device: torch.device
) -> dict:
    """Run a single experiment (pretrain + finetune).
    
    Returns:
        Dictionary with metrics, training time, and parameter count
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print('='*60)
    
    start_time = time.time()
    
    # Create model
    if model_class == S3RecLowRankModel:
        rank = config.get('lowrank', {}).get('rank', 16)
        model = S3RecLowRankModel(
            num_items=data['num_items'],
            num_attributes=data['num_attributes'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            max_seq_length=config['sequence']['max_length'],
            dropout=config['model']['dropout'],
            rank=rank
        )
    else:
        model = S3RecModel(
            num_items=data['num_items'],
            num_attributes=data['num_attributes'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            max_seq_length=config['sequence']['max_length'],
            dropout=config['model']['dropout']
        )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Create dataloaders
    pretrain_loaders = create_dataloaders(
        data,
        batch_size=config['pretrain']['batch_size'],
        max_seq_length=config['sequence']['max_length'],
        pretrain=True
    )
    
    finetune_loaders = create_dataloaders(
        data,
        batch_size=config['finetune']['batch_size'],
        max_seq_length=config['sequence']['max_length'],
        pretrain=False
    )
    
    # Update config for this experiment
    exp_config = config.copy()
    exp_config['output'] = {
        'results_dir': str(output_dir / experiment_name),
        'checkpoint_dir': str(output_dir / experiment_name / 'checkpoints')
    }
    exp_config['logging'] = {
        'use_tensorboard': True,
        'log_dir': str(output_dir / experiment_name / 'logs')
    }
    
    # Pre-training
    print("\n--- Pre-training ---")
    pretrain_trainer = PretrainTrainer(model, exp_config, device)
    pretrain_history = pretrain_trainer.train(pretrain_loaders['pretrain'])
    
    pretrain_time = time.time() - start_time
    
    # Fine-tuning
    print("\n--- Fine-tuning ---")
    finetune_start = time.time()
    finetune_trainer = FinetuneTrainer(model, exp_config, device)
    finetune_history = finetune_trainer.train(
        finetune_loaders['train'],
        finetune_loaders['valid'],
        finetune_loaders['test']
    )
    
    finetune_time = time.time() - finetune_start
    total_time = time.time() - start_time
    
    # Evaluate
    test_metrics = finetune_trainer.evaluate(finetune_loaders['test'])
    
    return {
        'name': experiment_name,
        'test_metrics': test_metrics,
        'param_count': param_count,
        'pretrain_time': pretrain_time,
        'finetune_time': finetune_time,
        'total_time': total_time,
        'pretrain_history': pretrain_history,
        'finetune_history': finetune_history
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run complete S3Rec experiment pipeline"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        required=True,
        help="Path to Amazon reviews JSON"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to Amazon metadata JSON"
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs='+',
        default=[8, 16, 32],
        help="Low-rank dimensions to test"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
        help="Pre-training epochs"
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=50,
        help="Fine-tuning epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use existing data)"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'model': {
            'hidden_size': args.hidden_size,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.2
        },
        'sequence': {
            'max_length': 50,
            'mask_ratio': 0.2
        },
        'pretrain': {
            'epochs': args.pretrain_epochs,
            'batch_size': args.batch_size,
            'learning_rate': 0.001,
            'aap_weight': 1.0,
            'mip_weight': 0.2,
            'map_weight': 1.0,
            'sp_weight': 0.5,
            'save_every': 10
        },
        'finetune': {
            'epochs': args.finetune_epochs,
            'batch_size': args.batch_size,
            'learning_rate': 0.001,
            'patience': 10
        }
    }
    
    # Step 1: Preprocessing
    data_path = output_dir / "data" / "beauty_processed.pkl"
    
    if not args.skip_preprocess or not data_path.exists():
        print("\n" + "="*60)
        print("Step 1: Data Preprocessing")
        print("="*60)
        
        preprocess_amazon_beauty(
            reviews_path=args.reviews,
            metadata_path=args.metadata,
            output_dir=str(output_dir / "data")
        )
    else:
        print(f"\nSkipping preprocessing, using existing data: {data_path}")
    
    # Load data
    data = load_processed_data(str(data_path))
    
    # Step 2: Run experiments
    print("\n" + "="*60)
    print("Step 2: Running Experiments")
    print("="*60)
    
    all_results = {}
    
    # Baseline experiment
    baseline_result = run_experiment(
        S3RecModel,
        data,
        config,
        "baseline",
        output_dir,
        device
    )
    all_results['Baseline'] = baseline_result
    
    # Low-rank experiments
    for rank in args.ranks:
        config['lowrank'] = {'rank': rank}
        
        result = run_experiment(
            S3RecLowRankModel,
            data,
            config,
            f"lowrank_r{rank}",
            output_dir,
            device
        )
        all_results[f'r={rank}'] = result
    
    # Step 3: Generate comparison
    print("\n" + "="*60)
    print("Step 3: Results Comparison")
    print("="*60)
    
    # Metrics comparison
    metrics_comparison = {
        name: result['test_metrics']
        for name, result in all_results.items()
    }
    
    print("\n" + create_results_table(metrics_comparison))
    
    # Parameter comparison
    param_comparison = {
        name: result['param_count']
        for name, result in all_results.items()
    }
    
    print("\nParameter Counts:")
    for name, count in param_comparison.items():
        print(f"  {name}: {count:,}")
    
    # Time comparison
    print("\nTraining Times:")
    for name, result in all_results.items():
        print(f"  {name}: {result['total_time']:.1f}s")
    
    # Step 4: Generate plots
    print("\n" + "="*60)
    print("Step 4: Generating Visualizations")
    print("="*60)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Rank comparison plot
    plot_rank_comparison(
        metrics_comparison,
        metric='ndcg@10',
        title='NDCG@10 vs Model Configuration',
        save_path=str(figures_dir / 'rank_comparison.png')
    )
    
    # Parameter reduction plot
    plot_parameter_reduction(
        param_comparison,
        title='Parameter Reduction Analysis',
        save_path=str(figures_dir / 'parameter_reduction.png')
    )
    
    # Training curves for each experiment
    for name, result in all_results.items():
        if 'pretrain_history' in result:
            plot_training_curves(
                result['pretrain_history'],
                title=f'Pre-training Loss Curves ({name})',
                save_path=str(figures_dir / f'training_curves_{name}.png')
            )
    
    # Save all results
    save_results(
        {name: {
            'metrics': r['test_metrics'],
            'params': r['param_count'],
            'time': r['total_time']
        } for name, r in all_results.items()},
        str(output_dir),
        'all_results.json'
    )
    
    # Create CSV comparison
    create_results_table(
        metrics_comparison,
        save_path=str(output_dir / 'results_comparison.csv')
    )
    
    print("\n" + "="*60)
    print("✅ All Experiments Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Results: {output_dir / 'all_results.json'}")
    print(f"  - Comparison CSV: {output_dir / 'results_comparison.csv'}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    baseline_ndcg = all_results['Baseline']['test_metrics']['ndcg@10']
    best_lowrank = max(
        [(name, r['test_metrics']['ndcg@10']) 
         for name, r in all_results.items() if name != 'Baseline'],
        key=lambda x: x[1]
    )
    
    print(f"Baseline NDCG@10: {baseline_ndcg:.4f}")
    print(f"Best Low-rank ({best_lowrank[0]}): {best_lowrank[1]:.4f}")
    print(f"Improvement: {(best_lowrank[1] - baseline_ndcg) / baseline_ndcg * 100:.2f}%")
    
    baseline_params = all_results['Baseline']['param_count']
    best_params = all_results[best_lowrank[0]]['param_count']
    print(f"Parameter reduction: {(baseline_params - best_params) / baseline_params * 100:.2f}%")


if __name__ == "__main__":
    main()

