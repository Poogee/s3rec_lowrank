# S3Rec Low-rank AAP Analysis Notebooks

This directory contains Jupyter notebooks for analyzing and visualizing the S3Rec with Low-rank AAP experiments.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_data_exploration.ipynb](01_data_exploration.ipynb) | Amazon Beauty dataset exploration and statistics |
| 2 | [02_preprocessing_demo.ipynb](02_preprocessing_demo.ipynb) | Step-by-step data preprocessing demonstration |
| 3 | [03_model_architecture.ipynb](03_model_architecture.ipynb) | Model architecture visualization and diagrams |
| 4 | [04_lowrank_analysis.ipynb](04_lowrank_analysis.ipynb) | Low-rank AAP properties and parameter analysis |
| 5 | [05_training_curves.ipynb](05_training_curves.ipynb) | Training loss curves and convergence analysis |
| 6 | [06_ablation_study.ipynb](06_ablation_study.ipynb) | Ablation study: rank sensitivity, task importance |
| 7 | [07_results_comparison.ipynb](07_results_comparison.ipynb) | Final results comparison and summary |

## Quick Start

```bash
# Navigate to notebooks directory
cd s3rec_lowrank/notebooks

# Start Jupyter
jupyter notebook
```

## Requirements

- matplotlib
- seaborn  
- pandas
- numpy
- torch
- jupyter

## Output

Generated figures are saved to `../results/figures/`:
- `sequence_lengths.png` - Distribution of user sequence lengths
- `architecture.png` - S3Rec architecture diagram
- `attention_patterns.png` - Bi/uni-directional attention visualization
- `lowrank_decomposition.png` - Low-rank matrix factorization
- `lowrank_analysis.png` - Parameter reduction analysis
- `pretrain_losses.png` - Pre-training loss curves
- `finetune_curves.png` - Fine-tuning progress
- `rank_sensitivity.png` - Performance vs rank
- `task_ablation.png` - Pre-training task ablation
- `final_comparison.png` - Final results comparison
