"""Evaluation metrics for sequential recommendation.

Implements:
- Hit Ratio @ K (Hit@K)
- Normalized Discounted Cumulative Gain @ K (NDCG@K)
- Mean Reciprocal Rank (MRR)
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np


def hit_at_k(predictions: np.ndarray, k: int = 10) -> float:
    """Compute Hit Ratio @ K.
    
    Hit@K = 1 if the ground truth item is in top-K predictions, else 0.
    Averaged over all users.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
                    First column is the ground truth item score.
        k: Number of top items to consider
        
    Returns:
        Hit@K value (0 to 1)
    """
    # Get ranking of the first item (ground truth)
    # Higher score = better, so we need to count items with higher scores
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    
    # Hit if rank < k (0-indexed)
    hits = (ranks < k).astype(float)
    
    return hits.mean()


def ndcg_at_k(predictions: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K.
    
    NDCG@K = DCG@K / IDCG@K
    where DCG = rel_i / log2(rank_i + 2)
    
    For binary relevance (single relevant item), 
    NDCG = 1/log2(rank+2) if rank < k, else 0.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
                    First column is the ground truth item score.
        k: Number of top items to consider
        
    Returns:
        NDCG@K value (0 to 1)
    """
    # Get ranking of the first item (ground truth)
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    
    # Compute NDCG
    ndcg_values = np.zeros(len(predictions))
    valid_mask = ranks < k
    
    # DCG for relevant item at position (rank+1)
    ndcg_values[valid_mask] = 1.0 / np.log2(ranks[valid_mask] + 2)
    
    return ndcg_values.mean()


def mrr(predictions: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.
    
    MRR = 1/(rank+1) averaged over all users.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
                    First column is the ground truth item score.
        
    Returns:
        MRR value (0 to 1)
    """
    # Get ranking of the first item (ground truth)
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    
    # Reciprocal rank (1-indexed)
    rr = 1.0 / (ranks + 1)
    
    return rr.mean()


def get_metric_at_k(
    predictions: np.ndarray,
    metric: str,
    k: int = 10
) -> float:
    """Get a specific metric at K.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
        metric: One of 'hit', 'ndcg', 'mrr'
        k: Number of top items to consider
        
    Returns:
        Metric value
    """
    metric = metric.lower()
    
    if metric in ('hit', 'hr', 'recall'):
        return hit_at_k(predictions, k)
    elif metric == 'ndcg':
        return ndcg_at_k(predictions, k)
    elif metric == 'mrr':
        return mrr(predictions)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_metrics(
    predictions: np.ndarray,
    ks: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute all metrics at multiple K values.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
                    First column is the ground truth item score.
        ks: List of K values for Hit@K and NDCG@K
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    for k in ks:
        metrics[f'hit@{k}'] = hit_at_k(predictions, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, k)
        
    metrics['mrr'] = mrr(predictions)
    
    return metrics


def precision_at_k(predictions: np.ndarray, k: int = 10) -> float:
    """Compute Precision @ K.
    
    For single relevant item, Precision@K = Hit@K / K.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
        k: Number of top items to consider
        
    Returns:
        Precision@K value
    """
    return hit_at_k(predictions, k) / k


def recall_at_k(predictions: np.ndarray, k: int = 10) -> float:
    """Compute Recall @ K.
    
    For single relevant item, Recall@K = Hit@K.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
        k: Number of top items to consider
        
    Returns:
        Recall@K value (same as Hit@K for single relevant item)
    """
    return hit_at_k(predictions, k)


def map_at_k(predictions: np.ndarray, k: int = 10) -> float:
    """Compute Mean Average Precision @ K.
    
    For single relevant item, MAP@K = Precision@rank if rank < k, else 0.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
        k: Number of top items to consider
        
    Returns:
        MAP@K value
    """
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    
    ap_values = np.zeros(len(predictions))
    valid_mask = ranks < k
    
    # Average Precision at the rank of the relevant item
    ap_values[valid_mask] = 1.0 / (ranks[valid_mask] + 1)
    
    return ap_values.mean()


def auc(predictions: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC).
    
    AUC = proportion of negative items ranked below the positive item.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
                    First column is the ground truth item score.
        
    Returns:
        AUC value (0 to 1)
    """
    # Count negative items with lower score than positive
    num_items = predictions.shape[1]
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    
    # AUC = (num_items - 1 - rank) / (num_items - 1)
    auc_values = (num_items - 1 - ranks) / (num_items - 1)
    
    return auc_values.mean()


def compute_all_metrics(
    predictions: np.ndarray,
    ks: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Compute comprehensive metrics.
    
    Args:
        predictions: Prediction scores [num_users, num_items]
        ks: List of K values
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    for k in ks:
        metrics[f'hit@{k}'] = hit_at_k(predictions, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, k)
        metrics[f'precision@{k}'] = precision_at_k(predictions, k)
        metrics[f'recall@{k}'] = recall_at_k(predictions, k)
        metrics[f'map@{k}'] = map_at_k(predictions, k)
        
    metrics['mrr'] = mrr(predictions)
    metrics['auc'] = auc(predictions)
    
    return metrics


def format_metrics(metrics: Dict[str, float], decimal: int = 4) -> str:
    """Format metrics dictionary for printing.
    
    Args:
        metrics: Dictionary of metric names to values
        decimal: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in sorted(metrics.items()):
        lines.append(f"  {name}: {value:.{decimal}f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Create test predictions [100 users, 100 items]
    # First column is ground truth item score
    predictions = np.random.randn(100, 100)
    
    # Make ground truth have higher score for some users
    predictions[:50, 0] = predictions[:50].max(axis=1) + 0.1
    
    print("Test Metrics:")
    print(format_metrics(compute_all_metrics(predictions)))
    
    # Verify expected behavior
    print("\n" + "=" * 40)
    print("Verification Tests")
    print("=" * 40)
    
    # Perfect predictions (ground truth always highest)
    perfect = np.random.randn(100, 100)
    perfect[:, 0] = 10.0  # Ground truth has highest score
    
    print("\nPerfect predictions:")
    print(f"  Hit@10: {hit_at_k(perfect, 10):.4f} (expected: 1.0000)")
    print(f"  NDCG@10: {ndcg_at_k(perfect, 10):.4f} (expected: 1.0000)")
    print(f"  MRR: {mrr(perfect):.4f} (expected: 1.0000)")
    
    # Worst predictions (ground truth always lowest)
    worst = np.random.randn(100, 100)
    worst[:, 0] = -10.0  # Ground truth has lowest score
    
    print("\nWorst predictions:")
    print(f"  Hit@10: {hit_at_k(worst, 10):.4f} (expected: 0.0000)")
    print(f"  NDCG@10: {ndcg_at_k(worst, 10):.4f} (expected: 0.0000)")

