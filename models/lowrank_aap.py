"""Low-rank Associated Attribute Prediction (AAP) Module.

The key innovation of this work: factorize the d×d weight matrix W_AAP
into U·V^T where U,V ∈ R^(d×r) with r << d.

Standard AAP:
    score = sigmoid(item^T · W · attribute)
    where W ∈ R^(d×d), requiring d² parameters
    
Low-rank AAP:
    score = sigmoid(item^T · U · V^T · attribute)
    where U,V ∈ R^(d×r), requiring only 2dr parameters
    
Parameter reduction: 1 - 2dr/(d²) = 1 - 2r/d
For d=64, r=16: 1 - 32/64 = 50% reduction in AAP parameters
For d=64, r=8: 1 - 16/64 = 75% reduction in AAP parameters
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAAP(nn.Module):
    """Low-rank Associated Attribute Prediction module.
    
    Factorizes W_AAP ≈ U · V^T where:
    - U ∈ R^(hidden_size × rank)
    - V ∈ R^(hidden_size × rank)
    
    Score computation:
    - item_proj = U^T · item          # (rank,)
    - attr_proj = V^T · attribute     # (rank,)
    - score = sigmoid(item_proj · attr_proj)  # scalar
    
    Batch computation:
    - For sequence: items [B, L, H] → scores [B*L, num_attributes]
    
    Args:
        hidden_size: Dimension of item/attribute embeddings (d)
        rank: Low-rank dimension (r), must be r < d
        use_bias: Whether to use bias terms
        
    Attributes:
        U: First projection matrix [hidden_size, rank]
        V: Second projection matrix [hidden_size, rank]
    """
    
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        use_bias: bool = False
    ):
        super().__init__()
        
        assert rank < hidden_size, f"Rank {rank} must be less than hidden_size {hidden_size}"
        
        self.hidden_size = hidden_size
        self.rank = rank
        self.use_bias = use_bias
        
        # Low-rank factors
        self.U = nn.Linear(hidden_size, rank, bias=use_bias)
        self.V = nn.Linear(hidden_size, rank, bias=use_bias)
        
        # Initialize with Xavier/Glorot for better gradient flow
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with scaled Xavier initialization."""
        # Scale by 1/sqrt(rank) to maintain variance
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
        """Compute AAP scores using low-rank factorization.
        
        Args:
            sequence_output: Item representations [B, L, H]
            attribute_embeddings: Attribute embeddings [num_attrs, H]
            
        Returns:
            scores: Attribute prediction scores [B*L, num_attrs]
        """
        batch_size, seq_len, hidden = sequence_output.shape
        num_attrs = attribute_embeddings.shape[0]
        
        # Project items through U: [B, L, H] -> [B, L, r]
        item_proj = self.U(sequence_output)  # [B, L, r]
        item_proj = item_proj.view(-1, self.rank)  # [B*L, r]
        
        # Project attributes through V: [num_attrs, H] -> [num_attrs, r]
        attr_proj = self.V(attribute_embeddings)  # [num_attrs, r]
        
        # Compute scores: [B*L, r] × [r, num_attrs] -> [B*L, num_attrs]
        scores = torch.matmul(item_proj, attr_proj.T)  # [B*L, num_attrs]
        
        return torch.sigmoid(scores)
    
    def reconstruct_weight_matrix(self) -> torch.Tensor:
        """Reconstruct the full W = U·V^T matrix for analysis.
        
        Returns:
            W: Reconstructed weight matrix [hidden_size, hidden_size]
        """
        # U.weight: [r, H], V.weight: [r, H]
        # W = U^T · V where U, V have shape [r, H]
        # Result: [H, r] × [r, H] = [H, H]
        U_weight = self.U.weight.T  # [H, r]
        V_weight = self.V.weight    # [r, H]
        
        return torch.matmul(U_weight, V_weight)
    
    def get_parameter_count(self) -> dict:
        """Get parameter count statistics.
        
        Returns:
            Dictionary with parameter counts
        """
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
        """Compute reconstruction error compared to full-rank matrix.
        
        Args:
            full_rank_weight: Full-rank weight matrix [H, H]
            
        Returns:
            Frobenius norm of difference
        """
        reconstructed = self.reconstruct_weight_matrix()
        error = torch.norm(full_rank_weight - reconstructed, p='fro')
        return error.item()
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, rank={self.rank}, use_bias={self.use_bias}'


class FullRankAAP(nn.Module):
    """Full-rank AAP module for baseline comparison.
    
    Standard AAP with W ∈ R^(d×d).
    
    Args:
        hidden_size: Dimension of embeddings
        use_bias: Whether to use bias
    """
    
    def __init__(self, hidden_size: int, use_bias: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # Full-rank projection
        self.W = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize with Xavier initialization."""
        nn.init.xavier_uniform_(self.W.weight)
        if self.use_bias:
            nn.init.zeros_(self.W.bias)
            
    def forward(
        self,
        sequence_output: torch.Tensor,
        attribute_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute AAP scores using full-rank matrix.
        
        Args:
            sequence_output: Item representations [B, L, H]
            attribute_embeddings: Attribute embeddings [num_attrs, H]
            
        Returns:
            scores: Attribute prediction scores [B*L, num_attrs]
        """
        # Project sequence: [B, L, H] -> [B, L, H]
        projected = self.W(sequence_output)
        projected = projected.view(-1, self.hidden_size, 1)  # [B*L, H, 1]
        
        # Score: [num_attrs, H] × [B*L, H, 1] -> [B*L, num_attrs, 1]
        scores = torch.matmul(attribute_embeddings, projected)
        
        return torch.sigmoid(scores.squeeze(-1))  # [B*L, num_attrs]
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Get the full weight matrix.
        
        Returns:
            W: Weight matrix [hidden_size, hidden_size]
        """
        return self.W.weight.T  # Transpose to match U·V^T shape
    
    def get_parameter_count(self) -> dict:
        """Get parameter count statistics.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'fullrank_params': total_params,
            'hidden_size': self.hidden_size
        }
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, use_bias={self.use_bias}'


def compare_aap_modules(hidden_size: int = 64, ranks: list = [8, 16, 32]) -> dict:
    """Compare parameter counts between full-rank and low-rank AAP.
    
    Args:
        hidden_size: Embedding dimension
        ranks: List of low-rank dimensions to compare
        
    Returns:
        Comparison dictionary
    """
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
    # Test and demonstrate the module
    print("=" * 60)
    print("Low-rank AAP Module Analysis")
    print("=" * 60)
    
    hidden_size = 64
    batch_size = 32
    seq_len = 50
    num_attrs = 1221
    
    # Test different ranks
    for rank in [8, 16, 32]:
        print(f"\n--- Rank {rank} ---")
        
        # Create modules
        lowrank = LowRankAAP(hidden_size, rank)
        fullrank = FullRankAAP(hidden_size)
        
        # Test forward pass
        items = torch.randn(batch_size, seq_len, hidden_size)
        attrs = torch.randn(num_attrs, hidden_size)
        
        scores_lr = lowrank(items, attrs)
        scores_fr = fullrank(items, attrs)
        
        print(f"Output shape: {scores_lr.shape}")
        print(f"Low-rank params: {lowrank.get_parameter_count()}")
        print(f"Full-rank params: {fullrank.get_parameter_count()['fullrank_params']}")
        
    # Full comparison
    print("\n" + "=" * 60)
    print("Parameter Comparison")
    print("=" * 60)
    comparison = compare_aap_modules(hidden_size=64, ranks=[4, 8, 16, 32])
    
    print(f"Hidden size: {comparison['hidden_size']}")
    print(f"Full-rank parameters: {comparison['full_rank_params']:,}")
    print("\nLow-rank comparisons:")
    for comp in comparison['comparisons']:
        print(f"  Rank {comp['rank']:2d}: {comp['lowrank_params']:,} params ({comp['reduction']} reduction)")

