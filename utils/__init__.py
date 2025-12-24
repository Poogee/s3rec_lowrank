"""Utilities module for S3Rec with Low-rank AAP.

This module provides:
- Evaluation metrics (Hit@K, NDCG@K, MRR)
- Visualization utilities
- Configuration utilities
- General helpers
"""

from .metrics import (
    compute_metrics,
    hit_at_k,
    ndcg_at_k,
    mrr,
    get_metric_at_k,
)
from .visualization import (
    plot_training_curves,
    plot_rank_comparison,
    plot_parameter_reduction,
    plot_attention_heatmap,
    plot_embedding_tsne,
)
from .helpers import (
    set_seed,
    load_config,
    merge_configs,
    save_results,
    print_model_summary,
)

__all__ = [
    # Metrics
    "compute_metrics",
    "hit_at_k",
    "ndcg_at_k",
    "mrr",
    "get_metric_at_k",
    # Visualization
    "plot_training_curves",
    "plot_rank_comparison",
    "plot_parameter_reduction",
    "plot_attention_heatmap",
    "plot_embedding_tsne",
    # Helpers
    "set_seed",
    "load_config",
    "merge_configs",
    "save_results",
    "print_model_summary",
]

