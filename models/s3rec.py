from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LayerNorm, TransformerEncoder
from .lowrank_aap import LowRankAAP, FullRankAAP


class S3RecModel(nn.Module):
    
    def __init__(
        self,
        num_items: int,
        num_attributes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02
    ):
        super().__init__()
        
        self.num_items = num_items
        self.num_attributes = num_attributes
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.initializer_range = initializer_range
        
        self.mask_id = num_items
        
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(num_attributes, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            activation=hidden_act,
            dropout=dropout
        )
        
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.aap_head = FullRankAAP(hidden_size)
        self.mip_head = nn.Linear(hidden_size, hidden_size)
        self.map_head = FullRankAAP(hidden_size)
        self.sp_head = nn.Linear(hidden_size, hidden_size)
        
        self.criterion = nn.BCELoss(reduction='none')
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def add_position_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        
        item_emb = self.item_embeddings(sequence)
        pos_emb = self.position_embeddings(position_ids)
        
        emb = item_emb + pos_emb
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        
        return emb
    
    def encode(
        self,
        sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (sequence == 0).float() * -1e9
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        seq_emb = self.add_position_embedding(sequence)
        encoded_layers = self.encoder(seq_emb, attention_mask)
        
        return encoded_layers[-1]
    
    def associated_attribute_prediction(
        self,
        sequence_output: torch.Tensor,
        attribute_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.aap_head(sequence_output, attribute_embeddings)
    
    def masked_item_prediction(
        self,
        sequence_output: torch.Tensor,
        target_items: torch.Tensor
    ) -> torch.Tensor:
        seq_output = self.mip_head(sequence_output.view(-1, self.hidden_size))
        target_items = target_items.view(-1, self.hidden_size)
        
        score = torch.sum(seq_output * target_items, dim=-1)
        return torch.sigmoid(score)
    
    def masked_attribute_prediction(
        self,
        sequence_output: torch.Tensor,
        attribute_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.map_head(sequence_output, attribute_embeddings)
    
    def segment_prediction(
        self,
        context: torch.Tensor,
        segment: torch.Tensor
    ) -> torch.Tensor:
        context = self.sp_head(context)
        score = torch.sum(context * segment, dim=-1)
        return torch.sigmoid(score)
    
    def pretrain(
        self,
        attributes: torch.Tensor,
        masked_item_sequence: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        masked_segment_sequence: torch.Tensor,
        pos_segment: torch.Tensor,
        neg_segment: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_output = self.encode(masked_item_sequence)
        attribute_embeddings = self.attribute_embeddings.weight
        
        aap_score = self.associated_attribute_prediction(sequence_output, attribute_embeddings)
        aap_loss = self.criterion(aap_score, attributes.view(-1, self.num_attributes).float())
        aap_mask = (masked_item_sequence != self.mask_id).float() * (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))
        
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance))
        mip_mask = (masked_item_sequence == self.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())
        
        map_score = self.masked_attribute_prediction(sequence_output, attribute_embeddings)
        map_loss = self.criterion(map_score, attributes.view(-1, self.num_attributes).float())
        map_mask = (masked_item_sequence == self.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))
        
        segment_output = self.encode(masked_segment_sequence)
        segment_context = segment_output[:, -1, :]
        
        pos_segment_output = self.encode(pos_segment)
        pos_segment_emb = pos_segment_output[:, -1, :]
        
        neg_segment_output = self.encode(neg_segment)
        neg_segment_emb = neg_segment_output[:, -1, :]
        
        pos_sp_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_sp_score = self.segment_prediction(segment_context, neg_segment_emb)
        
        sp_distance = torch.sigmoid(pos_sp_score - neg_sp_score)
        sp_loss = torch.sum(self.criterion(sp_distance, torch.ones_like(sp_distance)))
        
        return aap_loss, mip_loss, map_loss, sp_loss
    
    def finetune(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids > 0).long()
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        max_len = attention_mask.size(-1)
        causal_mask = torch.triu(torch.ones((1, max_len, max_len), device=input_ids.device), diagonal=1)
        causal_mask = (causal_mask == 0).unsqueeze(1)
        
        combined_mask = extended_mask * causal_mask.long()
        combined_mask = combined_mask.float()
        combined_mask = (1.0 - combined_mask) * -1e9
        
        seq_emb = self.add_position_embedding(input_ids)
        encoded_layers = self.encoder(seq_emb, combined_mask)
        
        return encoded_layers[-1]
    
    def get_parameter_count(self) -> Dict[str, int]:
        counts = {
            'item_embeddings': self.item_embeddings.weight.numel(),
            'attribute_embeddings': self.attribute_embeddings.weight.numel(),
            'position_embeddings': self.position_embeddings.weight.numel(),
            'encoder': self.encoder.get_parameter_count(),
            'aap_head': sum(p.numel() for p in self.aap_head.parameters()),
            'mip_head': sum(p.numel() for p in self.mip_head.parameters()),
            'map_head': sum(p.numel() for p in self.map_head.parameters()),
            'sp_head': sum(p.numel() for p in self.sp_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


class S3RecLowRankModel(S3RecModel):
    
    def __init__(
        self,
        num_items: int,
        num_attributes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        rank: int = 16,
        aap_rank: Optional[int] = None,
        map_rank: Optional[int] = None,
        lowrank_init_method: str = "xavier"
    ):
        super().__init__(
            num_items=num_items,
            num_attributes=num_attributes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range
        )
        
        self.aap_rank = aap_rank if aap_rank is not None else rank
        self.map_rank = map_rank if map_rank is not None else rank
        self.rank = rank
        self.lowrank_init_method = lowrank_init_method
        
        self.aap_head = LowRankAAP(hidden_size, self.aap_rank, init_method=lowrank_init_method)
        self.map_head = LowRankAAP(hidden_size, self.map_rank, init_method=lowrank_init_method)
        
    def get_aap_analysis(self) -> Dict:
        return {
            'aap': self.aap_head.get_parameter_count(),
            'map': self.map_head.get_parameter_count(),
            'aap_rank': self.aap_rank,
            'map_rank': self.map_rank,
            'rank': self.rank,
            'hidden_size': self.hidden_size,
            'init_method': self.lowrank_init_method
        }
    
    def reconstruct_aap_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        aap_weight = self.aap_head.reconstruct_weight_matrix()
        map_weight = self.map_head.reconstruct_weight_matrix()
        return aap_weight, map_weight
    
    def get_parameter_count(self) -> Dict[str, int]:
        counts = {
            'item_embeddings': self.item_embeddings.weight.numel(),
            'attribute_embeddings': self.attribute_embeddings.weight.numel(),
            'position_embeddings': self.position_embeddings.weight.numel(),
            'encoder': self.encoder.get_parameter_count(),
            'aap_head_lowrank': sum(p.numel() for p in self.aap_head.parameters()),
            'mip_head': sum(p.numel() for p in self.mip_head.parameters()),
            'map_head_lowrank': sum(p.numel() for p in self.map_head.parameters()),
            'sp_head': sum(p.numel() for p in self.sp_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts
    
    def get_parameter_reduction(self) -> Dict:
        lowrank_params = self.get_parameter_count()
        
        full_aap_params = self.hidden_size * self.hidden_size + self.hidden_size
        full_map_params = self.hidden_size * self.hidden_size + self.hidden_size
        
        lowrank_aap = sum(p.numel() for p in self.aap_head.parameters())
        lowrank_map = sum(p.numel() for p in self.map_head.parameters())
        
        full_total = lowrank_params['total'] - lowrank_aap - lowrank_map + full_aap_params + full_map_params
        
        return {
            'lowrank_total': lowrank_params['total'],
            'fullrank_total': full_total,
            'reduction': (full_total - lowrank_params['total']) / full_total,
            'reduction_percent': f"{(full_total - lowrank_params['total']) / full_total * 100:.2f}%",
            'aap_reduction': f"{(full_aap_params - lowrank_aap) / full_aap_params * 100:.2f}%",
            'aap_rank': self.aap_rank,
            'map_rank': self.map_rank,
            'rank': self.rank
        }


def create_model(
    num_items: int,
    num_attributes: int,
    config: dict,
    use_lowrank: bool = True
) -> nn.Module:
    model_config = config.get('model', {})
    
    if use_lowrank:
        lowrank_config = config.get('lowrank', {})
        rank = lowrank_config.get('rank', 16)
        aap_rank = lowrank_config.get('aap_rank', None)
        map_rank = lowrank_config.get('map_rank', None)
        init_method = lowrank_config.get('init_method', 'xavier')
        
        return S3RecLowRankModel(
            num_items=num_items,
            num_attributes=num_attributes,
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            num_heads=model_config.get('num_heads', 2),
            max_seq_length=config.get('sequence', {}).get('max_length', 50),
            dropout=model_config.get('dropout', 0.2),
            hidden_act=model_config.get('hidden_act', 'gelu'),
            initializer_range=model_config.get('initializer_range', 0.02),
            rank=rank,
            aap_rank=aap_rank,
            map_rank=map_rank,
            lowrank_init_method=init_method
        )
    else:
        return S3RecModel(
            num_items=num_items,
            num_attributes=num_attributes,
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            num_heads=model_config.get('num_heads', 2),
            max_seq_length=config.get('sequence', {}).get('max_length', 50),
            dropout=model_config.get('dropout', 0.2),
            hidden_act=model_config.get('hidden_act', 'gelu'),
            initializer_range=model_config.get('initializer_range', 0.02)
        )


if __name__ == "__main__":
    print("=" * 60)
    print("S3Rec Model Test")
    print("=" * 60)
    
    num_items = 12102
    num_attributes = 1221
    batch_size = 4
    seq_len = 50
    
    baseline = S3RecModel(num_items, num_attributes, hidden_size=64)
    lowrank = S3RecLowRankModel(num_items, num_attributes, hidden_size=64, rank=16)
    
    print("\nParameter Counts:")
    print(f"Baseline: {baseline.get_parameter_count()['total']:,}")
    print(f"Low-rank: {lowrank.get_parameter_count()['total']:,}")
    
    print("\nParameter Reduction:")
    reduction = lowrank.get_parameter_reduction()
    print(f"  Total reduction: {reduction['reduction_percent']}")
    print(f"  AAP reduction: {reduction['aap_reduction']}")
    
    print("\nForward Pass Test:")
    attributes = torch.randint(0, 2, (batch_size, seq_len, num_attributes))
    masked_seq = torch.randint(1, num_items, (batch_size, seq_len))
    pos_items = torch.randint(1, num_items, (batch_size, seq_len))
    neg_items = torch.randint(1, num_items, (batch_size, seq_len))
    
    aap, mip, map_loss, sp = baseline.pretrain(
        attributes, masked_seq, pos_items, neg_items,
        masked_seq, pos_items, neg_items
    )
    print(f"Baseline losses: AAP={aap.item():.4f}, MIP={mip.item():.4f}, MAP={map_loss.item():.4f}, SP={sp.item():.4f}")
    
    aap, mip, map_loss, sp = lowrank.pretrain(
        attributes, masked_seq, pos_items, neg_items,
        masked_seq, pos_items, neg_items
    )
    print(f"Low-rank losses: AAP={aap.item():.4f}, MIP={mip.item():.4f}, MAP={map_loss.item():.4f}, SP={sp.item():.4f}")
    
    input_ids = torch.randint(1, num_items, (batch_size, seq_len))
    output_baseline = baseline.finetune(input_ids)
    output_lowrank = lowrank.finetune(input_ids)
    print(f"\nFinetune output shape: {output_baseline.shape}")
    
    print("\n✅ All tests passed!")
