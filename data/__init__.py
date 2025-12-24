"""Data module for S3Rec with Low-rank AAP.

This module provides:
- Data preprocessing from raw Amazon JSON files
- Dataset classes for pre-training and fine-tuning
- DataLoader utilities with negative sampling
"""

from .preprocessing import (
    preprocess_amazon_beauty,
    load_processed_data,
    DataStats,
)
from .dataset import (
    PretrainDataset,
    FinetuneDataset,
    create_dataloaders,
)

__all__ = [
    "preprocess_amazon_beauty",
    "load_processed_data",
    "DataStats",
    "PretrainDataset",
    "FinetuneDataset",
    "create_dataloaders",
]

