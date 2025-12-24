"""Visualization utilities for S3Rec with Low-rank AAP.

Implements publication-quality plots for:
- Training curves
- Rank comparison
- Parameter reduction
- Attention heatmaps
- Embedding visualization
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


# Publication-quality style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B1F2B',      # Dark
    'baseline': '#666666',     # Gray
    'lowrank': '#2E86AB',      # Blue
}


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Pre-training Loss Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot training loss curves for all objectives.
    
    Args:
        history: Dictionary with keys like 'aap_loss', 'mip_loss', etc.
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('aap_loss', history.get('total_loss', []))) + 1)
    
    # Left plot: Individual losses
    ax1 = axes[0]
    
    loss_keys = ['aap_loss', 'mip_loss', 'map_loss', 'sp_loss']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    labels = ['AAP', 'MIP', 'MAP', 'SP']
    
    for key, color, label in zip(loss_keys, colors, labels):
        if key in history:
            ax1.plot(epochs, history[key], color=color, label=label, linewidth=2)
            
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Individual Objective Losses')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Right plot: Total loss
    ax2 = axes[1]
    
    if 'total_loss' in history:
        ax2.plot(epochs, history['total_loss'], color=COLORS['neutral'], linewidth=2.5)
        ax2.fill_between(epochs, history['total_loss'], alpha=0.3, color=COLORS['neutral'])
        
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Joint Training Loss')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_rank_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'ndcg@10',
    title: str = "Performance vs Low-rank Dimension",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot performance comparison across different ranks.
    
    Args:
        results: Dict mapping experiment names to metrics dicts
                e.g., {'r=8': {'ndcg@10': 0.35}, 'r=16': {'ndcg@10': 0.37}}
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    names = list(results.keys())
    values = [results[name].get(metric, 0) for name in names]
    
    # Create bar plot
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=COLORS['primary'], edgecolor='white', linewidth=1.5)
    
    # Highlight best
    best_idx = np.argmax(values)
    bars[best_idx].set_color(COLORS['accent'])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.annotate(
            f'{val:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9,
            fontweight='bold' if i == best_idx else 'normal'
        )
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add horizontal line for baseline if exists
    if 'baseline' in results or 'Baseline' in results:
        baseline_key = 'baseline' if 'baseline' in results else 'Baseline'
        baseline_val = results[baseline_key].get(metric, 0)
        ax.axhline(y=baseline_val, color=COLORS['baseline'], linestyle='--', 
                   linewidth=2, label=f'Baseline ({baseline_val:.4f})')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_parameter_reduction(
    param_counts: Dict[str, int],
    title: str = "Parameter Reduction Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """Plot parameter count comparison.
    
    Args:
        param_counts: Dict mapping model names to parameter counts
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    names = list(param_counts.keys())
    counts = [param_counts[name] for name in names]
    
    # Left: Bar chart
    ax1 = axes[0]
    x = np.arange(len(names))
    colors = [COLORS['baseline'] if 'baseline' in n.lower() else COLORS['lowrank'] 
              for n in names]
    
    bars = ax1.bar(x, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.annotate(
            f'{count:,}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Parameter Count')
    ax1.set_title('Total Parameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M' if x >= 1e6 else f'{x/1e3:.1f}K'))
    
    # Right: Pie chart for reduction
    ax2 = axes[1]
    
    if len(counts) >= 2:
        baseline_params = max(counts)
        reduced_params = min(counts)
        reduction = baseline_params - reduced_params
        
        sizes = [reduced_params, reduction]
        labels = ['Remaining', 'Reduced']
        colors_pie = [COLORS['primary'], COLORS['secondary']]
        explode = (0, 0.05)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax2.set_title(f'Parameter Reduction\n({reduction:,} params saved)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot attention weight heatmap.
    
    Args:
        attention_weights: Attention matrix [seq_len, seq_len]
        title: Plot title
        x_labels: Labels for x-axis (items)
        y_labels: Labels for y-axis (queries)
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Labels
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Embedding Visualization (t-SNE)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30
) -> plt.Figure:
    """Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding matrix [num_items, hidden_size]
        labels: Optional cluster labels for coloring
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        
    Returns:
        Matplotlib figure
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn required for t-SNE. Install with: pip install scikit-learn")
        return None
        
    # Compute t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        # Color by labels
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.7, s=10
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            alpha=0.7, s=10, c=COLORS['primary']
        )
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_reconstruction_error(
    ranks: List[int],
    errors: List[float],
    title: str = "Low-rank Reconstruction Error",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot reconstruction error vs rank.
    
    Args:
        ranks: List of rank values
        errors: Corresponding reconstruction errors
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(ranks, errors, marker='o', color=COLORS['primary'], 
            linewidth=2, markersize=8)
    ax.fill_between(ranks, errors, alpha=0.3, color=COLORS['primary'])
    
    ax.set_xlabel('Rank (r)')
    ax.set_ylabel('Frobenius Norm Error')
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_speed_comparison(
    times: Dict[str, float],
    title: str = "Training Speed Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot training time comparison.
    
    Args:
        times: Dict mapping model names to training times
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(times.keys())
    values = list(times.values())
    
    x = np.arange(len(names))
    colors = [COLORS['baseline'] if 'baseline' in n.lower() else COLORS['lowrank'] 
              for n in names]
    
    bars = ax.barh(x, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(
            f'{val:.1f}s',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=9
        )
    
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Model')
    ax.set_title(title)
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        
    return fig


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['hit@10', 'ndcg@10', 'mrr'],
    save_path: Optional[str] = None
) -> str:
    """Create formatted results table.
    
    Args:
        results: Dict of experiment names to metrics dicts
        metrics: List of metrics to include
        save_path: Path to save CSV (optional)
        
    Returns:
        Formatted table string
    """
    # Create header
    header = ['Model'] + [m.upper() for m in metrics]
    
    # Create rows
    rows = []
    for name, result in results.items():
        row = [name]
        for metric in metrics:
            value = result.get(metric, 0)
            row.append(f'{value:.4f}')
        rows.append(row)
    
    # Format as table
    col_widths = [max(len(str(row[i])) for row in [header] + rows) 
                  for i in range(len(header))]
    
    lines = []
    
    # Header
    header_line = ' | '.join(h.ljust(w) for h, w in zip(header, col_widths))
    lines.append(header_line)
    lines.append('-' * len(header_line))
    
    # Rows
    for row in rows:
        row_line = ' | '.join(str(r).ljust(w) for r, w in zip(row, col_widths))
        lines.append(row_line)
    
    table = '\n'.join(lines)
    
    if save_path:
        # Save as CSV
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows([[name] + [results[name].get(m, 0) for m in metrics] 
                             for name in results])
        print(f"Saved results to {save_path}")
    
    return table

