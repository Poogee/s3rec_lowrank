"""Pre-training trainer for S3Rec.

Implements pre-training with 4 self-supervised objectives:
- AAP: Associated Attribute Prediction
- MIP: Masked Item Prediction
- MAP: Masked Attribute Prediction
- SP: Segment Prediction
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint, TensorBoardLogger, MetricsTracker


class PretrainTrainer:
    """Trainer for S3Rec pre-training.
    
    Args:
        model: S3Rec model
        config: Configuration dictionary
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)
        
        # Training config
        pretrain_config = config.get('pretrain', {})
        self.epochs = pretrain_config.get('epochs', 100)
        self.batch_size = pretrain_config.get('batch_size', 256)
        self.lr = pretrain_config.get('learning_rate', 0.001)
        self.weight_decay = pretrain_config.get('weight_decay', 0.0)
        self.lowrank_weight_decay = pretrain_config.get('lowrank_weight_decay', None)
        
        # Loss weights
        self.aap_weight = pretrain_config.get('aap_weight', 1.0)
        self.mip_weight = pretrain_config.get('mip_weight', 0.2)
        self.map_weight = pretrain_config.get('map_weight', 1.0)
        self.sp_weight = pretrain_config.get('sp_weight', 0.5)
        
        # Logging
        self.log_freq = pretrain_config.get('log_freq', 1)
        self.save_every = pretrain_config.get('save_every', 10)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Callbacks
        output_config = config.get('output', {})
        self.checkpoint = ModelCheckpoint(
            output_config.get('checkpoint_dir', 'results/checkpoints'),
            save_best_only=False,
            save_every=self.save_every
        )
        
        logging_config = config.get('logging', {})
        if logging_config.get('use_tensorboard', True):
            self.logger = TensorBoardLogger(
                logging_config.get('log_dir', 'results/logs'),
                experiment_name='pretrain'
            )
        else:
            self.logger = None
            
        self.metrics_tracker = MetricsTracker(
            output_config.get('results_dir', 'results')
        )
        
        # Print info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model on device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def _setup_optimizer(self):
        """Setup optimizer with Adam.
        
        Supports adaptive weight decay: different weight_decay for low-rank parameters
        (U, V matrices in AAP/MAP heads) and other parameters.
        """
        adam_beta1 = self.config.get('pretrain', {}).get('adam_beta1', 0.9)
        adam_beta2 = self.config.get('pretrain', {}).get('adam_beta2', 0.999)
        
        # If lowrank_weight_decay is specified, use parameter groups
        if self.lowrank_weight_decay is not None and self.lowrank_weight_decay != self.weight_decay:
            # Separate low-rank parameters from others
            lowrank_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Check if this is a low-rank parameter (U or V in AAP/MAP heads)
                    if 'aap_head.U' in name or 'aap_head.V' in name or \
                       'map_head.U' in name or 'map_head.V' in name:
                        lowrank_params.append(param)
                    else:
                        other_params.append(param)
            
            # Create parameter groups with different weight_decay
            param_groups = [
                {'params': other_params, 'weight_decay': self.weight_decay},
                {'params': lowrank_params, 'weight_decay': self.lowrank_weight_decay}
            ]
            
            self.optimizer = Adam(
                param_groups,
                lr=self.lr,
                betas=(adam_beta1, adam_beta2)
            )
            
            print(f"Using adaptive weight decay: {self.weight_decay} (others), "
                  f"{self.lowrank_weight_decay} (low-rank)")
        else:
            # Standard optimizer with single weight_decay
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(adam_beta1, adam_beta2),
                weight_decay=self.weight_decay
            )
        
    def train(self, dataloader: DataLoader) -> Dict[str, list]:
        """Run pre-training.
        
        Args:
            dataloader: Pre-training data loader
            
        Returns:
            Dictionary of training history
        """
        print("=" * 60)
        print("Starting Pre-training")
        print("=" * 60)
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Loss weights: AAP={self.aap_weight}, MIP={self.mip_weight}, "
              f"MAP={self.map_weight}, SP={self.sp_weight}")
        print("=" * 60)
        
        history = {
            'aap_loss': [],
            'mip_loss': [],
            'map_loss': [],
            'sp_loss': [],
            'total_loss': [],
            'epoch_time': []
        }
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train one epoch
            losses = self._train_epoch(dataloader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['aap_loss'].append(losses['aap'])
            history['mip_loss'].append(losses['mip'])
            history['map_loss'].append(losses['map'])
            history['sp_loss'].append(losses['sp'])
            history['total_loss'].append(losses['total'])
            history['epoch_time'].append(epoch_time)
            
            # Logging
            if (epoch + 1) % self.log_freq == 0:
                print(f"\nEpoch {epoch + 1}/{self.epochs} ({epoch_time:.2f}s)")
                print(f"  AAP: {losses['aap']:.6f}, MIP: {losses['mip']:.6f}, "
                      f"MAP: {losses['map']:.6f}, SP: {losses['sp']:.6f}")
                print(f"  Total: {losses['total']:.6f}")
                
            # TensorBoard logging
            if self.logger:
                self.logger.log_pretrain_losses(
                    epoch, losses['aap'], losses['mip'],
                    losses['map'], losses['sp'], losses['total']
                )
                
            # Save checkpoint
            self.checkpoint(self.model, epoch, prefix="pretrain")
            
            # Track metrics
            self.metrics_tracker.add('pretrain', epoch, losses)
            
        # Save final checkpoint and metrics
        self.checkpoint.save_every = 1  # Ensure final save
        self.checkpoint(self.model, self.epochs - 1, prefix="pretrain_final")
        self.metrics_tracker.save()
        
        if self.logger:
            self.logger.close()
            
        print("\n" + "=" * 60)
        print("Pre-training Complete!")
        print("=" * 60)
        
        return history
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_aap = 0.0
        total_mip = 0.0
        total_map = 0.0
        total_sp = 0.0
        num_samples = 0
        
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True
        )
        
        for batch in pbar:
            # Move batch to device
            batch = tuple(t.to(self.device) for t in batch)
            (attributes, masked_item_seq, pos_items, neg_items,
             masked_segment_seq, pos_segment, neg_segment) = batch
            
            # Forward pass
            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes, masked_item_seq, pos_items, neg_items,
                masked_segment_seq, pos_segment, neg_segment
            )
            
            # Weighted loss
            joint_loss = (
                self.aap_weight * aap_loss +
                self.mip_weight * mip_loss +
                self.map_weight * map_loss +
                self.sp_weight * sp_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            joint_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            batch_size = masked_item_seq.size(0)
            total_aap += aap_loss.item()
            total_mip += mip_loss.item()
            total_map += map_loss.item()
            total_sp += sp_loss.item()
            num_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{joint_loss.item():.4f}",
                'aap': f"{aap_loss.item():.4f}"
            })
        
        # Compute averages
        num_batches = len(dataloader)
        avg_losses = {
            'aap': total_aap / num_batches,
            'mip': total_mip / num_batches,
            'map': total_map / num_batches,
            'sp': total_sp / num_batches
        }
        avg_losses['total'] = (
            self.aap_weight * avg_losses['aap'] +
            self.mip_weight * avg_losses['mip'] +
            self.map_weight * avg_losses['map'] +
            self.sp_weight * avg_losses['sp']
        )
        
        return avg_losses
    
    def save(self, path: str):
        """Save model and optimizer state.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load(self, path: str):
        """Load model and optimizer state.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")

