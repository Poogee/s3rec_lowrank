import math
from typing import Dict, List, Optional, Union

import numpy as np


def hit_at_k(predictions: np.ndarray, k: int = 10) -> float:
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    hits = (ranks < k).astype(float)
    return hits.mean()


def ndcg_at_k(predictions: np.ndarray, k: int = 10) -> float:
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    ndcg_values = np.zeros(len(predictions))
    valid_mask = ranks < k
    ndcg_values[valid_mask] = 1.0 / np.log2(ranks[valid_mask] + 2)
    return ndcg_values.mean()


def mrr(predictions: np.ndarray) -> float:
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    rr = 1.0 / (ranks + 1)
    return rr.mean()


def get_metric_at_k(predictions: np.ndarray, metric: str, k: int = 10) -> float:
    metric = metric.lower()
    
    if metric in ('hit', 'hr', 'recall'):
        return hit_at_k(predictions, k)
    elif metric == 'ndcg':
        return ndcg_at_k(predictions, k)
    elif metric == 'mrr':
        return mrr(predictions)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_metrics(predictions: np.ndarray, ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    metrics = {}
    
    for k in ks:
        metrics[f'hit@{k}'] = hit_at_k(predictions, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, k)
        
    metrics['mrr'] = mrr(predictions)
    
    return metrics


def precision_at_k(predictions: np.ndarray, k: int = 10) -> float:
    return hit_at_k(predictions, k) / k


def recall_at_k(predictions: np.ndarray, k: int = 10) -> float:
    return hit_at_k(predictions, k)


def map_at_k(predictions: np.ndarray, k: int = 10) -> float:
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    ap_values = np.zeros(len(predictions))
    valid_mask = ranks < k
    ap_values[valid_mask] = 1.0 / (ranks[valid_mask] + 1)
    return ap_values.mean()


def auc(predictions: np.ndarray) -> float:
    num_items = predictions.shape[1]
    ranks = (predictions > predictions[:, 0:1]).sum(axis=1)
    auc_values = (num_items - 1 - ranks) / (num_items - 1)
    return auc_values.mean()


def compute_all_metrics(predictions: np.ndarray, ks: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
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
    lines = []
    for name, value in sorted(metrics.items()):
        lines.append(f"  {name}: {value:.{decimal}f}")
    return "\n".join(lines)


if __name__ == "__main__":
    np.random.seed(42)
    
    predictions = np.random.randn(100, 100)
    predictions[:50, 0] = predictions[:50].max(axis=1) + 0.1
    
    print("Test Metrics:")
    print(format_metrics(compute_all_metrics(predictions)))
    
    print("\n" + "=" * 40)
    print("Verification Tests")
    print("=" * 40)
    
    perfect = np.random.randn(100, 100)
    perfect[:, 0] = 10.0
    
    print("\nPerfect predictions:")
    print(f"  Hit@10: {hit_at_k(perfect, 10):.4f}")
    print(f"  NDCG@10: {ndcg_at_k(perfect, 10):.4f}")
    print(f"  MRR: {mrr(perfect):.4f}")
    
    worst = np.random.randn(100, 100)
    worst[:, 0] = -10.0
    
    print("\nWorst predictions:")
    print(f"  Hit@10: {hit_at_k(worst, 10):.4f}")
    print(f"  NDCG@10: {ndcg_at_k(worst, 10):.4f}")
