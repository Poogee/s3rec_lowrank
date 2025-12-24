"""Training module for S3Rec with Low-rank AAP.

This module provides:
- PretrainTrainer: Pre-training with 4 self-supervised objectives
- FinetuneTrainer: Fine-tuning for next-item prediction
- Training utilities and callbacks
"""

from .pretrain import PretrainTrainer
from .finetune import FinetuneTrainer
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
)

__all__ = [
    "PretrainTrainer",
    "FinetuneTrainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "TensorBoardLogger",
]

