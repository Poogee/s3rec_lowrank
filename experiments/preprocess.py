#!/usr/bin/env python
"""Data preprocessing script for Amazon Beauty dataset.

Usage:
    python -m experiments.preprocess --reviews path/to/reviews.json --metadata path/to/meta.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import preprocess_amazon_beauty, DataStats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon Beauty dataset for S3Rec"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        required=True,
        help="Path to Amazon reviews JSON file (reviews_Beauty_5.json)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to Amazon metadata JSON file (meta_Beauty.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--user-core",
        type=int,
        default=5,
        help="Minimum interactions per user (K-core)"
    )
    parser.add_argument(
        "--item-core",
        type=int,
        default=5,
        help="Minimum interactions per item (K-core)"
    )
    parser.add_argument(
        "--attribute-core",
        type=int,
        default=0,
        help="Minimum items per attribute"
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=0.0,
        help="Minimum rating to include (filter low ratings)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("S3Rec Low-rank AAP - Data Preprocessing")
    print("=" * 60)
    print(f"Reviews file: {args.reviews}")
    print(f"Metadata file: {args.metadata}")
    print(f"Output directory: {args.output}")
    print(f"K-core: user={args.user_core}, item={args.item_core}")
    print("=" * 60)
    
    # Run preprocessing
    stats = preprocess_amazon_beauty(
        reviews_path=args.reviews,
        metadata_path=args.metadata,
        output_dir=args.output,
        user_core=args.user_core,
        item_core=args.item_core,
        attribute_core=args.attribute_core,
        min_rating=args.min_rating
    )
    
    print("\n✅ Preprocessing complete!")
    print(f"\nDataset ready at: {args.output}")
    
    return stats


if __name__ == "__main__":
    main()

