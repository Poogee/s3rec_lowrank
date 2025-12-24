import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "swish": swish,
    "tanh": torch.tanh,
}


class LayerNorm(nn.Module):
    
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
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        
        output = self.dense(context)
        output = self.output_dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        if output_attentions:
            return output, attention_probs
        return (output,)


class FeedForward(nn.Module):
    
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
        output = self.dense1(hidden_states)
        output = self.activation(output)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        return output


class TransformerLayer(nn.Module):
    
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
        attention_output = self.attention(hidden_states, attention_mask, output_attentions)
        
        if output_attentions:
            attention_output, attention_probs = attention_output[0], attention_output[1]
        else:
            attention_output = attention_output[0]
            
        output = self.ffn(attention_output)
        
        if output_attentions:
            return output, attention_probs
        return (output,)


class TransformerEncoder(nn.Module):
    
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
        
        layer = TransformerLayer(hidden_size, num_heads, intermediate_size, activation, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_layers: bool = True,
        output_attentions: bool = False
    ) -> tuple:
        all_outputs = []
        all_attentions = []
        
        for layer in self.layers:
            layer_output = layer(hidden_states, attention_mask, output_attentions)
            
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
        return sum(p.numel() for p in self.parameters())
