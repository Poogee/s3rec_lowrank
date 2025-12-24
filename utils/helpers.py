"""General helper utilities for S3Rec.

Includes:
- Random seed setting
- Configuration loading and merging
- Result saving utilities
- Model summary printing
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union
import copy

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loaded configuration from {config_path}")
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def save_results(
    results: Dict,
    save_dir: str,
    filename: str = "results.json"
):
    """Save results to JSON file.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save results
        filename: Output filename
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_file = save_path / filename
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {output_file}")


def load_results(path: str) -> Dict:
    """Load results from JSON file.
    
    Args:
        path: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(path, 'r') as f:
        results = json.load(f)
    return results


def print_model_summary(model: nn.Module, detailed: bool = False):
    """Print model architecture summary.
    
    Args:
        model: PyTorch model
        detailed: Whether to print detailed parameter info
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    param_info = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            
        param_info.append({
            'name': name,
            'shape': list(param.shape),
            'params': num_params,
            'trainable': param.requires_grad
        })
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimate
    param_size = total_params * 4  # float32
    print(f"Estimated model size: {param_size / 1024 / 1024:.2f} MB")
    
    if detailed:
        print("\n" + "-" * 60)
        print("Parameter Details:")
        print("-" * 60)
        
        for info in param_info:
            status = "✓" if info['trainable'] else "✗"
            print(f"  {status} {info['name']}: {info['shape']} ({info['params']:,})")
    
    print("=" * 60)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_info': param_info
    }


def get_device(device: Optional[str] = None) -> torch.device:
    """Get appropriate device for training.
    
    Args:
        device: Specified device ('cuda', 'cpu', 'auto' or None)
        
    Returns:
        torch.device object
    """
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = 'cpu'
            print("Using CPU")
    
    return torch.device(device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by layer type.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_type = type(module).__name__
                if layer_type not in counts:
                    counts[layer_type] = 0
                counts[layer_type] += params
    
    counts['total'] = sum(counts.values())
    return counts


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create directory for experiment outputs.
    
    Args:
        base_dir: Base results directory
        experiment_name: Name of experiment
        
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0
        
    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"{self.name}: {format_time(self.elapsed)}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str
):
    """Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Evaluation metrics
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """Load training checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        optimizer: Optional optimizer to restore
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with epoch and metrics
    """
    if device is None:
        device = get_device()
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }

