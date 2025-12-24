"""Fine-tuning trainer for S3Rec.

Fine-tunes the pre-trained model for next-item prediction
using pairwise ranking loss.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint, TensorBoardLogger, MetricsTracker
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import compute_metrics


class FinetuneTrainer:
    """Trainer for S3Rec fine-tuning.
    
    Args:
        model: S3Rec model (pre-trained)
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
        finetune_config = config.get('finetune', {})
        self.epochs = finetune_config.get('epochs', 50)
        self.batch_size = finetune_config.get('batch_size', 256)
        self.lr = finetune_config.get('learning_rate', 0.001)
        self.weight_decay = finetune_config.get('weight_decay', 0.0)
        self.lowrank_weight_decay = finetune_config.get('lowrank_weight_decay', None)
        self.log_freq = finetune_config.get('log_freq', 1)
        self.patience = finetune_config.get('patience', 10)
        
        # Evaluation config
        eval_config = config.get('evaluation', {})
        self.full_sort = eval_config.get('full_sort', False)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Callbacks
        output_config = config.get('output', {})
        self.checkpoint = ModelCheckpoint(
            output_config.get('checkpoint_dir', 'results/checkpoints'),
            save_best_only=True,
            mode='max'
        )
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            mode='max'
        )
        
        logging_config = config.get('logging', {})
        if logging_config.get('use_tensorboard', True):
            self.logger = TensorBoardLogger(
                logging_config.get('log_dir', 'results/logs'),
                experiment_name='finetune'
            )
        else:
            self.logger = None
            
        self.metrics_tracker = MetricsTracker(
            output_config.get('results_dir', 'results')
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        print(f"Model on device: {self.device}")
        print(f"Fine-tuning epochs: {self.epochs}")
        print(f"Early stopping patience: {self.patience}")
        
    def _setup_optimizer(self):
        """Setup optimizer.
        
        Supports adaptive weight decay: different weight_decay for low-rank parameters
        (U, V matrices in AAP/MAP heads) and other parameters.
        """
        adam_beta1 = self.config.get('finetune', {}).get('adam_beta1', 0.9)
        adam_beta2 = self.config.get('finetune', {}).get('adam_beta2', 0.999)
        
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
        
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """Run fine-tuning.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Dictionary of training history
        """
        print("=" * 60)
        print("Starting Fine-tuning")
        print("=" * 60)
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print("=" * 60)
        
        history = {
            'train_loss': [],
            'valid_metrics': [],
            'epoch_time': []
        }
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss = self._train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            valid_metrics = self.evaluate(valid_loader)
            history['valid_metrics'].append(valid_metrics)
            
            epoch_time = time.time() - epoch_start
            history['epoch_time'].append(epoch_time)
            
            # Primary metric for early stopping (NDCG@10)
            primary_metric = valid_metrics.get('ndcg@10', valid_metrics.get('ndcg@5', 0))
            
            # Logging
            if (epoch + 1) % self.log_freq == 0:
                print(f"\nEpoch {epoch + 1}/{self.epochs} ({epoch_time:.2f}s)")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Valid: Hit@10={valid_metrics.get('hit@10', 0):.4f}, "
                      f"NDCG@10={valid_metrics.get('ndcg@10', 0):.4f}, "
                      f"MRR={valid_metrics.get('mrr', 0):.4f}")
                
            # TensorBoard logging
            if self.logger:
                self.logger.log_scalar('finetune/train_loss', train_loss, epoch)
                self.logger.log_finetune_metrics(epoch, valid_metrics, 'valid')
                
            # Track metrics
            self.metrics_tracker.add('train', epoch, {'loss': train_loss})
            self.metrics_tracker.add('valid', epoch, valid_metrics)
            
            # Save checkpoint
            self.checkpoint(self.model, epoch, score=primary_metric, prefix="finetune")
            
            # Early stopping
            if self.early_stopping(primary_metric, epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
                
        # Load best model
        self.checkpoint.load_best(self.model)
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        
        valid_metrics = self.evaluate(valid_loader)
        print(f"Validation: Hit@10={valid_metrics.get('hit@10', 0):.4f}, "
              f"NDCG@10={valid_metrics.get('ndcg@10', 0):.4f}, "
              f"MRR={valid_metrics.get('mrr', 0):.4f}")
        
        if test_loader:
            test_metrics = self.evaluate(test_loader)
            print(f"Test: Hit@10={test_metrics.get('hit@10', 0):.4f}, "
                  f"NDCG@10={test_metrics.get('ndcg@10', 0):.4f}, "
                  f"MRR={test_metrics.get('mrr', 0):.4f}")
            self.metrics_tracker.add('test', 0, test_metrics)
            
        # Save metrics
        self.metrics_tracker.save()
        
        if self.logger:
            self.logger.close()
            
        print("\n" + "=" * 60)
        print("Fine-tuning Complete!")
        print("=" * 60)
        
        return history
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True
        )
        
        for batch in pbar:
            # Move batch to device
            batch = tuple(t.to(self.device) for t in batch)
            
            if len(batch) == 5:
                user_ids, input_ids, target_pos, target_neg, answers = batch
            else:
                user_ids, input_ids, target_pos, target_neg, answers, _ = batch
            
            # Forward pass
            sequence_output = self.model.finetune(input_ids)
            
            # Compute pairwise ranking loss
            loss = self._cross_entropy_loss(sequence_output, target_pos, target_neg)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / num_batches
    
    def _cross_entropy_loss(
        self,
        seq_output: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.
        
        Args:
            seq_output: Sequence representations [B, L, H]
            pos_ids: Positive item IDs [B, L]
            neg_ids: Negative item IDs [B, L]
            
        Returns:
            Loss value
        """
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_output.view(-1, seq_output.size(2))
        
        pos_logits = torch.sum(pos * seq_emb, dim=-1)
        neg_logits = torch.sum(neg * seq_emb, dim=-1)
        
        istarget = (pos_ids > 0).view(-1).float()
        
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        
        return loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                
                if len(batch) == 6:
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                else:
                    # Training loader format
                    continue
                    
                # Get sequence output
                sequence_output = self.model.finetune(input_ids)
                sequence_output = sequence_output[:, -1, :]  # Last position
                
                # Combine answer and negative samples
                test_items = torch.cat([answers, sample_negs], dim=-1)  # [B, 1+99]
                
                # Get item embeddings and compute scores
                test_item_emb = self.model.item_embeddings(test_items)  # [B, 100, H]
                
                # Compute logits: [B, H] @ [B, H, 100] -> [B, 100]
                logits = torch.bmm(
                    test_item_emb,
                    sequence_output.unsqueeze(-1)
                ).squeeze(-1)
                
                all_predictions.append(logits.cpu().numpy())
                
        if not all_predictions:
            return {'hit@1': 0, 'hit@5': 0, 'hit@10': 0, 'ndcg@5': 0, 'ndcg@10': 0, 'mrr': 0}
            
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions)
        
        return metrics
    
    def predict(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate top-k predictions for input sequence.
        
        Args:
            input_ids: Input item sequence [B, L]
            top_k: Number of top items to return
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            sequence_output = self.model.finetune(input_ids)
            sequence_output = sequence_output[:, -1, :]  # [B, H]
            
            # Score all items
            item_embeddings = self.model.item_embeddings.weight  # [num_items, H]
            scores = torch.matmul(sequence_output, item_embeddings.T)  # [B, num_items]
            
            # Get top-k
            top_scores, top_items = torch.topk(scores, top_k, dim=-1)
            
        return top_items, top_scores
    
    def save(self, path: str):
        """Save model and optimizer state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")

