# S3Rec Low-rank AAP Analysis Notebooks

This directory contains Jupyter notebooks for analyzing and visualizing the S3Rec with Low-rank AAP experiments.

## Notebooks

1. **01_data_exploration.py** - Amazon Beauty dataset exploration
2. **02_preprocessing_demo.py** - Data preprocessing demonstration
3. **03_model_architecture.py** - Model architecture visualization
4. **04_lowrank_analysis.py** - Low-rank properties analysis
5. **05_training_curves.py** - Training curves and metrics
6. **06_ablation_study.py** - Ablation study analysis
7. **07_results_comparison.py** - Final results and comparisons

## Running the Notebooks

To convert Python scripts to Jupyter notebooks:

```bash
pip install jupytext
jupytext --to notebook *.py
```

Or run them directly as Python scripts:

```bash
python 01_data_exploration.py
```

## Requirements

- matplotlib
- seaborn
- pandas
- numpy
- torch
- jupyter (for interactive notebooks)

