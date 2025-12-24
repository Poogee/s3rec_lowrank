#!/usr/bin/env python
"""Fine-tuning script for S3Rec models.

Usage:
    python -m experiments.finetune --checkpoint results/checkpoints/pretrained.pt
    python -m experiments.finetune --checkpoint results/checkpoints/pretrained.pt --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data.preprocessing import load_processed_data
from data.dataset import create_dataloaders
from models.s3rec import S3RecModel, S3RecLowRankModel, create_model
from trainers.finetune import FinetuneTrainer
from utils.helpers import set_seed, load_config, print_model_summary, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune S3Rec model for next-item prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/beauty_processed.pkl",
        help="Path to processed data file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint (optional)"
    )
    parser.add_argument(
        "--lowrank",
        action="store_true",
        help="Use low-rank AAP model"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Low-rank dimension"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 2, 'dropout': 0.2},
            'lowrank': {'enabled': args.lowrank, 'rank': args.rank},
            'sequence': {'max_length': 50},
            'finetune': {
                'epochs': 50, 'batch_size': 256, 'learning_rate': 0.001,
                'patience': 10
            },
            'evaluation': {'sample_size': 99},
            'output': {'results_dir': args.output, 'checkpoint_dir': f'{args.output}/checkpoints'},
            'logging': {'use_tensorboard': True, 'log_dir': f'{args.output}/logs'}
        }
    
    # Override config
    if args.lowrank:
        config['lowrank'] = {'enabled': True, 'rank': args.rank}
    if args.epochs:
        config['finetune']['epochs'] = args.epochs
    if args.batch_size:
        config['finetune']['batch_size'] = args.batch_size
    if args.lr:
        config['finetune']['learning_rate'] = args.lr
    if args.patience:
        config['finetune']['patience'] = args.patience
    
    print("=" * 60)
    print("S3Rec Fine-tuning")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data...")
    data_path = Path(__file__).parent.parent / args.data
    
    if not data_path.exists():
        alt_paths = [
            Path(args.data),
            Path(__file__).parent.parent / "data" / "processed" / "beauty_processed.pkl"
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path = alt_path
                break
        else:
            print(f"Error: Data file not found.")
            sys.exit(1)
    
    data = load_processed_data(str(data_path))
    
    print(f"  Users: {data['num_users']:,}")
    print(f"  Items: {data['num_items']:,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data,
        batch_size=config['finetune']['batch_size'],
        max_seq_length=config['sequence']['max_length'],
        pretrain=False,
        neg_sample_size=config.get('evaluation', {}).get('sample_size', 99)
    )
    
    # Create model
    print("\nCreating model...")
    use_lowrank = config.get('lowrank', {}).get('enabled', args.lowrank)
    
    model = create_model(
        num_items=data['num_items'],
        num_attributes=data['num_attributes'],
        config=config,
        use_lowrank=use_lowrank
    )
    
    model_type = "Low-rank" if use_lowrank else "Baseline"
    print(f"  Model type: {model_type}")
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ Loaded pre-trained weights")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    print_model_summary(model)
    
    # Create trainer
    device = None
    if args.device != "auto":
        device = torch.device(args.device)
    
    trainer = FinetuneTrainer(model, config, device)
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    history = trainer.train(
        dataloaders['train'],
        dataloaders['valid'],
        dataloaders['test']
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Test Results")
    print("=" * 60)
    
    test_metrics = trainer.evaluate(dataloaders['test'])
    
    for metric, value in sorted(test_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "s3rec_lowrank" if use_lowrank else "s3rec_baseline"
    if use_lowrank:
        model_name += f"_r{config.get('lowrank', {}).get('rank', args.rank)}"
    
    results = {
        'model': model_name,
        'test_metrics': test_metrics,
        'config': config
    }
    
    save_results(results, str(output_dir), f"{model_name}_results.json")
    
    # Save final model
    final_path = output_dir / "checkpoints" / f"{model_name}_finetuned.pt"
    trainer.save(str(final_path))
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Results saved to: {output_dir}")
    
    return test_metrics


if __name__ == "__main__":
    main()

