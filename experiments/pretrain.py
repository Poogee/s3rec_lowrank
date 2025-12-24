#!/usr/bin/env python
"""Pre-training script for S3Rec models.

Usage:
    python -m experiments.pretrain --config config/default_config.yaml
    python -m experiments.pretrain --config config/default_config.yaml --lowrank --rank 16
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
from trainers.pretrain import PretrainTrainer
from utils.helpers import set_seed, load_config, merge_configs, print_model_summary


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train S3Rec model with self-supervised objectives"
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
        "--lowrank",
        action="store_true",
        help="Use low-rank AAP factorization"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Low-rank dimension (if --lowrank is set)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of pre-training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
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
        # Use default config
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 2, 'dropout': 0.2},
            'lowrank': {'enabled': args.lowrank, 'rank': args.rank},
            'sequence': {'max_length': 50, 'mask_ratio': 0.2},
            'pretrain': {
                'epochs': 100, 'batch_size': 256, 'learning_rate': 0.001,
                'aap_weight': 1.0, 'mip_weight': 0.2, 'map_weight': 1.0, 'sp_weight': 0.5,
                'save_every': 10
            },
            'output': {'results_dir': args.output, 'checkpoint_dir': f'{args.output}/checkpoints'},
            'logging': {'use_tensorboard': True, 'log_dir': f'{args.output}/logs'}
        }
    
    # Override with command line arguments
    if args.lowrank:
        config['lowrank'] = {'enabled': True, 'rank': args.rank}
    if args.epochs:
        config['pretrain']['epochs'] = args.epochs
    if args.batch_size:
        config['pretrain']['batch_size'] = args.batch_size
    if args.lr:
        config['pretrain']['learning_rate'] = args.lr
    
    print("=" * 60)
    print("S3Rec Pre-training")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data_path = Path(__file__).parent.parent / args.data
    
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(args.data),
            Path(__file__).parent.parent / "data" / "processed" / "beauty_processed.pkl"
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path = alt_path
                break
        else:
            print(f"Error: Data file not found. Run preprocessing first.")
            print(f"  python -m experiments.preprocess --reviews <path> --metadata <path>")
            sys.exit(1)
    
    data = load_processed_data(str(data_path))
    
    print(f"  Users: {data['num_users']:,}")
    print(f"  Items: {data['num_items']:,}")
    print(f"  Attributes: {data['num_attributes']:,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data,
        batch_size=config['pretrain']['batch_size'],
        max_seq_length=config['sequence']['max_length'],
        mask_ratio=config['sequence']['mask_ratio'],
        pretrain=True
    )
    
    print(f"  Training samples: {len(dataloaders['pretrain'].dataset):,}")
    
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
    
    if use_lowrank:
        rank = config.get('lowrank', {}).get('rank', args.rank)
        print(f"  Low-rank dimension: {rank}")
    
    print_model_summary(model)
    
    # Create trainer and train
    device = None
    if args.device != "auto":
        device = torch.device(args.device)
    
    trainer = PretrainTrainer(model, config, device)
    
    print("\nStarting pre-training...")
    history = trainer.train(dataloaders['pretrain'])
    
    # Save final model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "s3rec_lowrank" if use_lowrank else "s3rec_baseline"
    if use_lowrank:
        model_name += f"_r{config.get('lowrank', {}).get('rank', args.rank)}"
    
    final_path = output_dir / "checkpoints" / f"{model_name}_pretrained.pt"
    trainer.save(str(final_path))
    
    print(f"\n✅ Pre-training complete!")
    print(f"Model saved to: {final_path}")
    
    return history


if __name__ == "__main__":
    main()

