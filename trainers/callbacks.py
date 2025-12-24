"""Training callbacks for S3Rec.

Implements:
- EarlyStopping: Stop training when metrics plateau
- ModelCheckpoint: Save model checkpoints
- TensorBoardLogger: Log metrics to TensorBoard
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import os

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping based on validation metrics.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
        verbose: Whether to print messages
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                    
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """Save model checkpoints during training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        save_best_only: Only save when score improves
        save_every: Save every N epochs (if save_best_only=False)
        mode: 'min' or 'max' for score comparison
        verbose: Whether to print messages
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        save_every: int = 10,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.best_path = None
        
    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        score: Optional[float] = None,
        prefix: str = "model"
    ) -> Optional[str]:
        """Save checkpoint if appropriate.
        
        Args:
            model: Model to save
            epoch: Current epoch
            score: Validation score (required if save_best_only=True)
            prefix: Filename prefix
            
        Returns:
            Path to saved checkpoint or None
        """
        save_path = None
        
        if self.save_best_only:
            if score is None:
                raise ValueError("score required when save_best_only=True")
                
            should_save = False
            if self.best_score is None:
                should_save = True
            elif self.mode == 'max' and score > self.best_score:
                should_save = True
            elif self.mode == 'min' and score < self.best_score:
                should_save = True
                
            if should_save:
                # Remove old best checkpoint
                if self.best_path and self.best_path.exists():
                    self.best_path.unlink()
                    
                self.best_score = score
                save_path = self.checkpoint_dir / f"{prefix}_best.pt"
                self._save_model(model, save_path, epoch, score)
                self.best_path = save_path
        else:
            if (epoch + 1) % self.save_every == 0:
                save_path = self.checkpoint_dir / f"{prefix}_epoch{epoch+1}.pt"
                self._save_model(model, save_path, epoch, score)
                
        return str(save_path) if save_path else None
    
    def _save_model(
        self,
        model: nn.Module,
        path: Path,
        epoch: int,
        score: Optional[float]
    ):
        """Save model state dict."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score
        }
        torch.save(checkpoint, path)
        
        if self.verbose:
            score_str = f" (score: {score:.4f})" if score else ""
            print(f"Saved checkpoint to {path}{score_str}")
    
    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best checkpoint."""
        if self.best_path and self.best_path.exists():
            checkpoint = torch.load(self.best_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from {self.best_path}")
        return model


class TensorBoardLogger:
    """TensorBoard logging for training metrics.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name of experiment
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "s3rec"
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.writer = None
        self._init_writer()
        
    def _init_writer(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = self.log_dir / self.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_path))
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
            
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step or epoch
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)
            
    def log_pretrain_losses(
        self,
        epoch: int,
        aap_loss: float,
        mip_loss: float,
        map_loss: float,
        sp_loss: float,
        total_loss: float
    ):
        """Log pre-training losses."""
        losses = {
            'AAP': aap_loss,
            'MIP': mip_loss,
            'MAP': map_loss,
            'SP': sp_loss,
            'Total': total_loss
        }
        self.log_scalars('pretrain/loss', losses, epoch)
        
    def log_finetune_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "valid"
    ):
        """Log fine-tuning evaluation metrics."""
        self.log_metrics(metrics, epoch, prefix=f"finetune/{phase}")
        
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()


class MetricsTracker:
    """Track and save training metrics history.
    
    Args:
        save_dir: Directory to save metrics
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'pretrain': [],
            'train': [],
            'valid': [],
            'test': []
        }
        
    def add(self, phase: str, epoch: int, metrics: Dict[str, float]):
        """Add metrics for an epoch.
        
        Args:
            phase: One of 'pretrain', 'train', 'valid', 'test'
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        entry = {'epoch': epoch, **metrics}
        self.history[phase].append(entry)
        
    def get_best(self, phase: str, metric: str, mode: str = 'max') -> Dict:
        """Get best metrics entry.
        
        Args:
            phase: Phase to search
            metric: Metric to optimize
            mode: 'min' or 'max'
            
        Returns:
            Best metrics entry
        """
        if not self.history[phase]:
            return {}
            
        if mode == 'max':
            return max(self.history[phase], key=lambda x: x.get(metric, float('-inf')))
        else:
            return min(self.history[phase], key=lambda x: x.get(metric, float('inf')))
    
    def save(self, filename: str = "metrics_history.json"):
        """Save metrics history to JSON file."""
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved metrics history to {save_path}")
        
    def load(self, filename: str = "metrics_history.json"):
        """Load metrics history from JSON file."""
        load_path = self.save_dir / filename
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded metrics history from {load_path}")

