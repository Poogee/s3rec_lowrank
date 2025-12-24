from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import os

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    
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
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    
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
    
    def _save_model(self, model: nn.Module, path: Path, epoch: int, score: Optional[float]):
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
        if self.best_path and self.best_path.exists():
            checkpoint = torch.load(self.best_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from {self.best_path}")
        return model


class TensorBoardLogger:
    
    def __init__(self, log_dir: str, experiment_name: str = "s3rec"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.writer = None
        self._init_writer()
        
    def _init_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = self.log_dir / self.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_path))
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            
    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)
            
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
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
        losses = {
            'AAP': aap_loss,
            'MIP': mip_loss,
            'MAP': map_loss,
            'SP': sp_loss,
            'Total': total_loss
        }
        self.log_scalars('pretrain/loss', losses, epoch)
        
    def log_finetune_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "valid"):
        self.log_metrics(metrics, epoch, prefix=f"finetune/{phase}")
        
    def close(self):
        if self.writer:
            self.writer.close()


class MetricsTracker:
    
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
        entry = {'epoch': epoch, **metrics}
        self.history[phase].append(entry)
        
    def get_best(self, phase: str, metric: str, mode: str = 'max') -> Dict:
        if not self.history[phase]:
            return {}
            
        if mode == 'max':
            return max(self.history[phase], key=lambda x: x.get(metric, float('-inf')))
        else:
            return min(self.history[phase], key=lambda x: x.get(metric, float('inf')))
    
    def save(self, filename: str = "metrics_history.json"):
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved metrics history to {save_path}")
        
    def load(self, filename: str = "metrics_history.json"):
        load_path = self.save_dir / filename
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded metrics history from {load_path}")
