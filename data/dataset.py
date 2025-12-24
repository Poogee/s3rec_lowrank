"""Dataset classes for S3Rec pre-training and fine-tuning.

Implements:
- PretrainDataset: Masked item/segment sequences with attribute labels
- FinetuneDataset: Sequential recommendation with leave-one-out splitting
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def neg_sample(item_set: set, num_items: int) -> int:
    """Sample a negative item not in the item set.
    
    Args:
        item_set: Set of positive items to avoid
        num_items: Total number of items (exclusive upper bound)
        
    Returns:
        Negative item ID
    """
    item = random.randint(1, num_items - 1)
    while item in item_set:
        item = random.randint(1, num_items - 1)
    return item


class PretrainDataset(Dataset):
    """Dataset for S3Rec pre-training with 4 self-supervised objectives.
    
    Implements:
    - AAP: Associated Attribute Prediction (for non-masked positions)
    - MIP: Masked Item Prediction
    - MAP: Masked Attribute Prediction (for masked positions)
    - SP: Segment Prediction
    
    Args:
        user_sequences: List of user item sequences
        long_sequence: Concatenated sequence for negative segment sampling
        item2attributes: Mapping of item ID to attribute IDs
        num_items: Total number of items
        num_attributes: Total number of attributes
        max_seq_length: Maximum sequence length
        mask_ratio: Probability of masking an item
        mask_id: ID to use for masked items
    """
    
    def __init__(
        self,
        user_sequences: List[List[int]],
        long_sequence: List[int],
        item2attributes: Dict[int, List[int]],
        num_items: int,
        num_attributes: int,
        max_seq_length: int = 50,
        mask_ratio: float = 0.2,
        mask_id: Optional[int] = None
    ):
        self.user_sequences = user_sequences
        self.long_sequence = long_sequence
        self.item2attributes = item2attributes
        self.num_items = num_items
        self.num_attributes = num_attributes
        self.max_len = max_seq_length
        self.mask_ratio = mask_ratio
        self.mask_id = mask_id if mask_id is not None else num_items
        
        # Split sequences into training subsequences
        self.part_sequences = []
        self._split_sequences()
        
    def _split_sequences(self):
        """Split each user sequence into multiple training subsequences."""
        for seq in self.user_sequences:
            if len(seq) < 3:  # Need at least 3 items
                continue
            # Use [:-2] for pre-training (leave last 2 for val/test)
            input_seq = seq[:-2]
            for i in range(1, len(input_seq) + 1):
                self.part_sequences.append(input_seq[:i])
                
    def __len__(self) -> int:
        return len(self.part_sequences)
    
    def _get_attributes(self, item_id: int) -> List[int]:
        """Get binary attribute vector for an item."""
        attrs = [0] * self.num_attributes
        if item_id in self.item2attributes:
            for attr_id in self.item2attributes[item_id]:
                if attr_id < self.num_attributes:
                    attrs[attr_id] = 1
        return attrs
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        sequence = self.part_sequences[index]
        
        # Create item sets for negative sampling
        item_set = set(sequence)
        
        # === Masked Item Prediction ===
        masked_item_sequence = []
        neg_items = []
        
        for item in sequence[:-1]:
            if random.random() < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        
        # Always mask the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))
        
        # === Segment Prediction ===
        if len(sequence) < 2:
            masked_segment_sequence = list(sequence)
            pos_segment = list(sequence)
            neg_segment = list(sequence)
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, max(1, len(self.long_sequence) - sample_length))
            
            pos_segment = sequence[start_id:start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            
            # Create masked segment sequence
            masked_segment_sequence = (
                sequence[:start_id] +
                [self.mask_id] * sample_length +
                sequence[start_id + sample_length:]
            )
            
            # Pad segments to match sequence length
            pos_segment = (
                [self.mask_id] * start_id +
                pos_segment +
                [self.mask_id] * (len(sequence) - start_id - sample_length)
            )
            neg_segment = (
                [self.mask_id] * start_id +
                neg_segment +
                [self.mask_id] * (len(sequence) - start_id - sample_length)
            )
        
        # Ensure all sequences have same length
        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)
        
        # === Padding ===
        pad_len = self.max_len - len(sequence)
        
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + list(sequence)
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment
        
        # Truncate if necessary
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]
        
        # === Attribute Labels ===
        attributes = [self._get_attributes(item) for item in pos_items]
        
        # Verify dimensions
        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        
        return (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )


class FinetuneDataset(Dataset):
    """Dataset for S3Rec fine-tuning (next-item prediction).
    
    Uses leave-one-out evaluation:
    - train: sequence[:-3] -> sequence[1:-2]
    - valid: sequence[:-2] -> sequence[-2]
    - test: sequence[:-1] -> sequence[-1]
    
    Args:
        user_sequences: List of user item sequences
        num_items: Total number of items
        max_seq_length: Maximum sequence length
        data_type: One of 'train', 'valid', 'test'
        neg_samples: Negative samples for evaluation (optional)
    """
    
    def __init__(
        self,
        user_sequences: List[List[int]],
        num_items: int,
        max_seq_length: int = 50,
        data_type: str = 'train',
        neg_samples: Optional[List[List[int]]] = None
    ):
        assert data_type in {'train', 'valid', 'test'}
        
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_seq_length
        self.data_type = data_type
        self.neg_samples = neg_samples
        
    def __len__(self) -> int:
        return len(self.user_sequences)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        user_id = index
        items = self.user_sequences[index]
        
        # Leave-one-out splitting
        if self.data_type == 'train':
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # Not used during training
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:  # test
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        
        # Handle short sequences
        if len(input_ids) == 0:
            input_ids = [items[0]] if items else [0]
            target_pos = [items[1]] if len(items) > 1 else [0]
        
        # Negative sampling
        seq_set = set(items)
        target_neg = [neg_sample(seq_set, self.num_items) for _ in input_ids]
        
        # Padding
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg
        
        # Truncate
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        
        result = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
        
        # Add negative samples for evaluation
        if self.neg_samples is not None:
            test_samples = self.neg_samples[index]
            result = result + (torch.tensor(test_samples, dtype=torch.long),)
            
        return result


def generate_negative_samples(
    user_sequences: List[List[int]],
    num_items: int,
    num_samples: int = 99
) -> List[List[int]]:
    """Generate negative samples for evaluation.
    
    Args:
        user_sequences: List of user item sequences
        num_items: Total number of items
        num_samples: Number of negative samples per user
        
    Returns:
        List of negative sample lists for each user
    """
    neg_samples = []
    
    for items in user_sequences:
        item_set = set(items)
        user_negs = []
        while len(user_negs) < num_samples:
            neg = random.randint(1, num_items - 1)
            if neg not in item_set and neg not in user_negs:
                user_negs.append(neg)
        neg_samples.append(user_negs)
        
    return neg_samples


def create_dataloaders(
    data: Dict,
    batch_size: int = 256,
    max_seq_length: int = 50,
    mask_ratio: float = 0.2,
    num_workers: int = 4,
    pretrain: bool = True,
    neg_sample_size: int = 99
) -> Dict[str, DataLoader]:
    """Create DataLoaders for pre-training or fine-tuning.
    
    Args:
        data: Processed data dictionary
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        mask_ratio: Masking ratio for pre-training
        num_workers: Number of data loading workers
        pretrain: Whether to create pre-training or fine-tuning loaders
        neg_sample_size: Number of negative samples for evaluation
        
    Returns:
        Dictionary of DataLoaders
    """
    user_sequences = data['user_sequences']
    num_items = data['num_items']
    num_attributes = data['num_attributes']
    
    if pretrain:
        # Pre-training dataset
        pretrain_dataset = PretrainDataset(
            user_sequences=user_sequences,
            long_sequence=data['long_sequence'],
            item2attributes=data['item2attributes'],
            num_items=num_items,
            num_attributes=num_attributes,
            max_seq_length=max_seq_length,
            mask_ratio=mask_ratio,
            mask_id=num_items  # Use num_items as mask token
        )
        
        return {
            'pretrain': DataLoader(
                pretrain_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        }
    else:
        # Generate negative samples for evaluation
        neg_samples = generate_negative_samples(
            user_sequences, num_items, neg_sample_size
        )
        
        # Create datasets for train/valid/test
        train_dataset = FinetuneDataset(
            user_sequences, num_items, max_seq_length, 'train'
        )
        valid_dataset = FinetuneDataset(
            user_sequences, num_items, max_seq_length, 'valid', neg_samples
        )
        test_dataset = FinetuneDataset(
            user_sequences, num_items, max_seq_length, 'test', neg_samples
        )
        
        return {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            ),
            'valid': DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        }

