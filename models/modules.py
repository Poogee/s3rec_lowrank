"""Transformer encoder modules for S3Rec.

Implements:
- LayerNorm: Layer normalization (TF-style)
- SelfAttention: Multi-head self-attention with masking
- FeedForward: Position-wise feed-forward network
- TransformerLayer: Single Transformer layer
- TransformerEncoder: Stack of Transformer layers
"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit activation.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "swish": swish,
    "tanh": torch.tanh,
}


class LayerNorm(nn.Module):
    """Layer Normalization (TensorFlow style with epsilon inside sqrt).
    
    Args:
        hidden_size: Dimension to normalize over
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional masking.
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
            
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size
        
        # QKV projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # Normalization and dropout
        self.layer_norm = LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention.
        
        [B, L, H] -> [B, num_heads, L, head_size]
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        """Forward pass.
        
        Args:
            hidden_states: Input tensor [B, L, H]
            attention_mask: Mask tensor [B, 1, 1, L] or [B, 1, L, L]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_probs) if output_attentions else (output,)
        """
        # QKV projections
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores: [B, heads, L, L]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        
        # Apply mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Compute context
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        
        # Output projection with residual connection
        output = self.dense(context)
        output = self.output_dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        if output_attentions:
            return output, attention_probs
        return (output,)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Args:
        hidden_size: Model dimension
        intermediate_size: FFN intermediate dimension
        activation: Activation function name
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
            
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        if activation in ACT2FN:
            self.activation = ACT2FN[activation]
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            hidden_states: Input tensor [B, L, H]
            
        Returns:
            Output tensor [B, L, H]
        """
        output = self.dense1(hidden_states)
        output = self.activation(output)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        return output


class TransformerLayer(nn.Module):
    """Single Transformer encoder layer.
    
    Consists of:
    - Multi-head self-attention
    - Position-wise feed-forward network
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = SelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, activation, dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        """Forward pass.
        
        Args:
            hidden_states: Input tensor [B, L, H]
            attention_mask: Attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of outputs
        """
        attention_output = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        
        if output_attentions:
            attention_output, attention_probs = attention_output[0], attention_output[1]
        else:
            attention_output = attention_output[0]
            
        output = self.ffn(attention_output)
        
        if output_attentions:
            return output, attention_probs
        return (output,)


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers.
    
    Args:
        num_layers: Number of Transformer layers
        hidden_size: Model dimension
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        layer = TransformerLayer(
            hidden_size, num_heads, intermediate_size, activation, dropout
        )
        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_layers: bool = True,
        output_attentions: bool = False
    ) -> tuple:
        """Forward pass through all layers.
        
        Args:
            hidden_states: Input tensor [B, L, H]
            attention_mask: Attention mask
            output_all_layers: Whether to return all layer outputs
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (layer_outputs, attention_weights) or just layer_outputs
        """
        all_outputs = []
        all_attentions = []
        
        for layer in self.layers:
            layer_output = layer(
                hidden_states, attention_mask, output_attentions
            )
            
            if output_attentions:
                hidden_states, attention_probs = layer_output[0], layer_output[1]
                all_attentions.append(attention_probs)
            else:
                hidden_states = layer_output[0]
                
            if output_all_layers:
                all_outputs.append(hidden_states)
                
        if not output_all_layers:
            all_outputs.append(hidden_states)
            
        if output_attentions:
            return all_outputs, all_attentions
        return all_outputs
    
    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

