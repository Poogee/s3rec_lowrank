import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def neg_sample(item_set: set, num_items: int) -> int:
    item = random.randint(1, num_items - 1)
    while item in item_set:
        item = random.randint(1, num_items - 1)
    return item


class PretrainDataset(Dataset):
    
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
        
        self.part_sequences = []
        self._split_sequences()
        
    def _split_sequences(self):
        for seq in self.user_sequences:
            if len(seq) < 3:
                continue
            input_seq = seq[:-2]
            for i in range(1, len(input_seq) + 1):
                self.part_sequences.append(input_seq[:i])
                
    def __len__(self) -> int:
        return len(self.part_sequences)
    
    def _get_attributes(self, item_id: int) -> List[int]:
        attrs = [0] * self.num_attributes
        if item_id in self.item2attributes:
            for attr_id in self.item2attributes[item_id]:
                if attr_id < self.num_attributes:
                    attrs[attr_id] = 1
        return attrs
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        sequence = self.part_sequences[index]
        item_set = set(sequence)
        
        masked_item_sequence = []
        neg_items = []
        
        for item in sequence[:-1]:
            if random.random() < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))
        
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
            
            masked_segment_sequence = (
                sequence[:start_id] +
                [self.mask_id] * sample_length +
                sequence[start_id + sample_length:]
            )
            
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
        
        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)
        
        pad_len = self.max_len - len(sequence)
        
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + list(sequence)
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment
        
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]
        
        attributes = [self._get_attributes(item) for item in pos_items]
        
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
        
        if self.data_type == 'train':
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        
        if len(input_ids) == 0:
            input_ids = [items[0]] if items else [0]
            target_pos = [items[1]] if len(items) > 1 else [0]
        
        seq_set = set(items)
        target_neg = [neg_sample(seq_set, self.num_items) for _ in input_ids]
        
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg
        
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
        
        if self.neg_samples is not None:
            test_samples = self.neg_samples[index]
            result = result + (torch.tensor(test_samples, dtype=torch.long),)
            
        return result


def generate_negative_samples(
    user_sequences: List[List[int]],
    num_items: int,
    num_samples: int = 99
) -> List[List[int]]:
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
    user_sequences = data['user_sequences']
    num_items = data['num_items']
    num_attributes = data['num_attributes']
    
    if pretrain:
        pretrain_dataset = PretrainDataset(
            user_sequences=user_sequences,
            long_sequence=data['long_sequence'],
            item2attributes=data['item2attributes'],
            num_items=num_items,
            num_attributes=num_attributes,
            max_seq_length=max_seq_length,
            mask_ratio=mask_ratio,
            mask_id=num_items
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
        neg_samples = generate_negative_samples(user_sequences, num_items, neg_sample_size)
        
        train_dataset = FinetuneDataset(user_sequences, num_items, max_seq_length, 'train')
        valid_dataset = FinetuneDataset(user_sequences, num_items, max_seq_length, 'valid', neg_samples)
        test_dataset = FinetuneDataset(user_sequences, num_items, max_seq_length, 'test', neg_samples)
        
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        }
