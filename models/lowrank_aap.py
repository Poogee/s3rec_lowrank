from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAAP(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        use_bias: bool = False,
        init_method: str = "xavier"
    ):
        super().__init__()
        
        assert rank < hidden_size, f"Rank {rank} must be less than hidden_size {hidden_size}"
        
        self.hidden_size = hidden_size
        self.rank = rank
        self.use_bias = use_bias
        self.init_method = init_method
        
        self.U = nn.Linear(hidden_size, rank, bias=use_bias)
        self.V = nn.Linear(hidden_size, rank, bias=use_bias)
        
        self._init_weights()
        
    def _init_weights(self):
        if self.init_method == "orthogonal":
            nn.init.orthogonal_(self.U.weight)
            nn.init.orthogonal_(self.V.weight)
        else:
            scale = 1.0 / math.sqrt(self.rank)
            nn.init.xavier_uniform_(self.U.weight, gain=scale)
            nn.init.xavier_uniform_(self.V.weight, gain=scale)
        
        if self.use_bias:
            nn.init.zeros_(self.U.bias)
            nn.init.zeros_(self.V.bias)
            
    def forward(
        self,
        sequence_output: torch.Tensor,
        attribute_embeddings: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden = sequence_output.shape
        num_attrs = attribute_embeddings.shape[0]
        
        item_proj = self.U(sequence_output)
        item_proj = item_proj.view(-1, self.rank)
        
        attr_proj = self.V(attribute_embeddings)
        
        scores = torch.matmul(item_proj, attr_proj.T)
        
        return torch.sigmoid(scores)
    
    def reconstruct_weight_matrix(self) -> torch.Tensor:
        U_weight = self.U.weight.T
        V_weight = self.V.weight
        return torch.matmul(U_weight, V_weight)
    
    def get_parameter_count(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        full_rank_params = self.hidden_size * self.hidden_size
        
        if self.use_bias:
            full_rank_params += self.hidden_size
            
        reduction = 1 - total_params / full_rank_params
        
        return {
            'lowrank_params': total_params,
            'fullrank_params': full_rank_params,
            'reduction_ratio': reduction,
            'reduction_percent': f"{reduction * 100:.1f}%",
            'rank': self.rank,
            'hidden_size': self.hidden_size
        }
    
    def compute_reconstruction_error(self, full_rank_weight: torch.Tensor) -> float:
        reconstructed = self.reconstruct_weight_matrix()
        error = torch.norm(full_rank_weight - reconstructed, p='fro')
        return error.item()
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, rank={self.rank}, use_bias={self.use_bias}'


class FullRankAAP(nn.Module):
    
    def __init__(self, hidden_size: int, use_bias: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        self.W = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.use_bias:
            nn.init.zeros_(self.W.bias)
            
    def forward(
        self,
        sequence_output: torch.Tensor,
        attribute_embeddings: torch.Tensor
    ) -> torch.Tensor:
        projected = self.W(sequence_output)
        projected = projected.view(-1, self.hidden_size, 1)
        
        scores = torch.matmul(attribute_embeddings, projected)
        
        return torch.sigmoid(scores.squeeze(-1))
    
    def get_weight_matrix(self) -> torch.Tensor:
        return self.W.weight.T
    
    def get_parameter_count(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'fullrank_params': total_params,
            'hidden_size': self.hidden_size
        }
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, use_bias={self.use_bias}'


def compare_aap_modules(hidden_size: int = 64, ranks: list = [8, 16, 32]) -> dict:
    full_rank = FullRankAAP(hidden_size)
    full_params = full_rank.get_parameter_count()['fullrank_params']
    
    results = {
        'hidden_size': hidden_size,
        'full_rank_params': full_params,
        'comparisons': []
    }
    
    for rank in ranks:
        low_rank = LowRankAAP(hidden_size, rank)
        stats = low_rank.get_parameter_count()
        
        results['comparisons'].append({
            'rank': rank,
            'lowrank_params': stats['lowrank_params'],
            'reduction': stats['reduction_percent']
        })
        
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Low-rank AAP Module Analysis")
    print("=" * 60)
    
    hidden_size = 64
    batch_size = 32
    seq_len = 50
    num_attrs = 1221
    
    for rank in [8, 16, 32]:
        print(f"\n--- Rank {rank} ---")
        
        lowrank = LowRankAAP(hidden_size, rank)
        fullrank = FullRankAAP(hidden_size)
        
        items = torch.randn(batch_size, seq_len, hidden_size)
        attrs = torch.randn(num_attrs, hidden_size)
        
        scores_lr = lowrank(items, attrs)
        scores_fr = fullrank(items, attrs)
        
        print(f"Output shape: {scores_lr.shape}")
        print(f"Low-rank params: {lowrank.get_parameter_count()}")
        print(f"Full-rank params: {fullrank.get_parameter_count()['fullrank_params']}")
        
    print("\n" + "=" * 60)
    comparison = compare_aap_modules(hidden_size=64, ranks=[4, 8, 16, 32])
    
    print(f"Hidden size: {comparison['hidden_size']}")
    print(f"Full-rank parameters: {comparison['full_rank_params']:,}")
    for comp in comparison['comparisons']:
        print(f"  Rank {comp['rank']:2d}: {comp['lowrank_params']:,} params ({comp['reduction']} reduction)")
