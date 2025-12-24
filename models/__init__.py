"""Model module for S3Rec with Low-rank AAP.

This module provides:
- S3RecModel: Base S3Rec model with standard full-rank AAP
- S3RecLowRankModel: S3Rec with Low-rank AAP factorization
- LowRankAAP: Low-rank Associated Attribute Prediction module
- Transformer encoder modules
"""

from .lowrank_aap import LowRankAAP, FullRankAAP
from .modules import (
    LayerNorm,
    SelfAttention,
    FeedForward,
    TransformerLayer,
    TransformerEncoder,
)
from .s3rec import S3RecModel, S3RecLowRankModel

__all__ = [
    "S3RecModel",
    "S3RecLowRankModel",
    "LowRankAAP",
    "FullRankAAP",
    "LayerNorm",
    "SelfAttention",
    "FeedForward",
    "TransformerLayer",
    "TransformerEncoder",
]

