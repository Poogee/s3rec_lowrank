"""Data preprocessing for Amazon Beauty dataset.

This module handles:
- Loading raw Amazon review and metadata JSON files
- K-core filtering (5-core for both users and items)
- Attribute extraction (categories + brands)
- ID mapping and vocabulary building
- Train/val/test splitting (leave-one-out)
"""

import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


@dataclass
class DataStats:
    """Statistics about the processed dataset."""
    num_users: int
    num_items: int
    num_attributes: int
    num_interactions: int
    avg_seq_length: float
    avg_attributes_per_item: float
    sparsity: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Dataset Statistics:\n"
            f"  Users: {self.num_users:,}\n"
            f"  Items: {self.num_items:,}\n"
            f"  Attributes: {self.num_attributes:,}\n"
            f"  Interactions: {self.num_interactions:,}\n"
            f"  Avg Sequence Length: {self.avg_seq_length:.2f}\n"
            f"  Avg Attributes/Item: {self.avg_attributes_per_item:.2f}\n"
            f"  Sparsity: {self.sparsity:.4f}%"
        )


def load_amazon_reviews(file_path: str, min_rating: float = 0.0) -> List[Tuple[str, str, int]]:
    """Load Amazon reviews from JSON file.
    
    Args:
        file_path: Path to reviews JSON file
        min_rating: Minimum rating threshold (0-5)
        
    Returns:
        List of (user_id, item_id, timestamp) tuples
    """
    interactions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line.strip())
                rating = float(review.get('overall', 0))
                
                if rating <= min_rating:
                    continue
                    
                user = review['reviewerID']
                item = review['asin']
                timestamp = int(review.get('unixReviewTime', 0))
                
                interactions.append((user, item, timestamp))
            except (json.JSONDecodeError, KeyError):
                continue
                
    return interactions


def load_amazon_metadata(file_path: str, valid_items: Set[str]) -> Dict[str, Dict]:
    """Load Amazon product metadata.
    
    Args:
        file_path: Path to metadata JSON file
        valid_items: Set of valid item ASINs
        
    Returns:
        Dictionary mapping ASIN to metadata dict
    """
    metadata = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading metadata"):
            try:
                # Try JSON first, then Python dict format
                line = line.strip()
                try:
                    info = json.loads(line)
                except json.JSONDecodeError:
                    # Handle Python dict format (single quotes)
                    info = eval(line)
                
                asin = info.get('asin')
                
                if asin and asin in valid_items:
                    metadata[asin] = info
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
                
    return metadata


def get_user_sequences(
    interactions: List[Tuple[str, str, int]]
) -> Dict[str, List[str]]:
    """Convert interactions to user sequences sorted by time.
    
    Args:
        interactions: List of (user, item, timestamp) tuples
        
    Returns:
        Dictionary mapping user_id to list of item_ids (chronological)
    """
    user_items = defaultdict(list)
    
    for user, item, timestamp in interactions:
        user_items[user].append((item, timestamp))
    
    # Sort by timestamp and extract items
    user_sequences = {}
    for user, item_times in user_items.items():
        item_times.sort(key=lambda x: x[1])
        user_sequences[user] = [item for item, _ in item_times]
        
    return user_sequences


def filter_k_core(
    user_sequences: Dict[str, List[str]],
    user_core: int = 5,
    item_core: int = 5
) -> Dict[str, List[str]]:
    """Apply k-core filtering to user-item interactions.
    
    Args:
        user_sequences: Dictionary of user -> item sequences
        user_core: Minimum interactions per user
        item_core: Minimum interactions per item
        
    Returns:
        Filtered user sequences
    """
    def check_k_core(seqs: Dict[str, List[str]]) -> Tuple[Dict, Dict, bool]:
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        
        for user, items in seqs.items():
            for item in items:
                user_count[user] += 1
                item_count[item] += 1
                
        is_valid = True
        for user, count in user_count.items():
            if count < user_core:
                is_valid = False
                break
                
        if is_valid:
            for item, count in item_count.items():
                if count < item_core:
                    is_valid = False
                    break
                    
        return user_count, item_count, is_valid
    
    user_count, item_count, is_k_core = check_k_core(user_sequences)
    
    iterations = 0
    while not is_k_core:
        iterations += 1
        
        # Remove users with too few interactions
        users_to_remove = [u for u, c in user_count.items() if c < user_core]
        for user in users_to_remove:
            if user in user_sequences:
                del user_sequences[user]
        
        # Remove items with too few interactions
        items_to_remove = {i for i, c in item_count.items() if c < item_core}
        for user in list(user_sequences.keys()):
            user_sequences[user] = [i for i in user_sequences[user] if i not in items_to_remove]
            if len(user_sequences[user]) < user_core:
                del user_sequences[user]
        
        user_count, item_count, is_k_core = check_k_core(user_sequences)
        
        if iterations > 100:
            print(f"Warning: K-core filtering did not converge after {iterations} iterations")
            break
    
    print(f"K-core filtering completed in {iterations} iterations")
    return user_sequences


def create_id_mappings(
    user_sequences: Dict[str, List[str]]
) -> Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, int]]:
    """Create ID mappings for users and items.
    
    Args:
        user_sequences: Dictionary of raw user/item IDs
        
    Returns:
        Tuple of (mapped_sequences, user2id, item2id)
    """
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    
    # Start from 1 (0 is padding)
    user_id = 1
    item_id = 1
    
    mapped_sequences = {}
    
    for user, items in user_sequences.items():
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
            
        mapped_items = []
        for item in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            mapped_items.append(item2id[item])
            
        mapped_sequences[user2id[user]] = mapped_items
        
    return mapped_sequences, user2id, item2id, id2user, id2item


def extract_attributes(
    metadata: Dict[str, Dict],
    item2id: Dict[str, int],
    attribute_core: int = 0
) -> Tuple[Dict[int, List[int]], int]:
    """Extract attributes (categories + brands) from metadata.
    
    Args:
        metadata: Item metadata dictionary
        item2id: Item ID mapping
        attribute_core: Minimum items per attribute
        
    Returns:
        Tuple of (item2attributes, num_attributes)
    """
    # Count attribute frequencies
    attribute_counts = defaultdict(int)
    
    for asin, info in metadata.items():
        # Extract categories (skip main category)
        categories = info.get('categories', [])
        for cat_list in categories:
            for cat in cat_list[1:]:  # Skip main category
                attribute_counts[cat] += 1
                
        # Extract brand
        brand = info.get('brand')
        if brand:
            attribute_counts[brand] += 1
    
    print(f"Total unique attributes before filtering: {len(attribute_counts)}")
    
    # Filter attributes by core
    valid_attributes = {a for a, c in attribute_counts.items() if c >= attribute_core}
    
    # Create attribute ID mapping (start from 1, 0 is padding)
    attribute2id = {}
    attribute_id = 1
    
    item2attributes = {}
    attribute_lens = []
    
    for asin, info in metadata.items():
        if asin not in item2id:
            continue
            
        item_id = item2id[asin]
        item_attrs = []
        
        # Add categories
        categories = info.get('categories', [])
        for cat_list in categories:
            for cat in cat_list[1:]:
                if cat in valid_attributes:
                    if cat not in attribute2id:
                        attribute2id[cat] = attribute_id
                        attribute_id += 1
                    if attribute2id[cat] not in item_attrs:
                        item_attrs.append(attribute2id[cat])
                        
        # Add brand
        brand = info.get('brand')
        if brand and brand in valid_attributes:
            if brand not in attribute2id:
                attribute2id[brand] = attribute_id
                attribute_id += 1
            if attribute2id[brand] not in item_attrs:
                item_attrs.append(attribute2id[brand])
                
        item2attributes[item_id] = item_attrs
        attribute_lens.append(len(item_attrs))
    
    num_attributes = len(attribute2id) + 1  # +1 for padding
    
    print(f"Attributes after filtering: {len(attribute2id)}")
    if attribute_lens:
        print(f"Avg attributes per item: {np.mean(attribute_lens):.2f}")
        print(f"Min/Max attributes: {np.min(attribute_lens)}/{np.max(attribute_lens)}")
    else:
        print("Warning: No attributes found for items")
    
    return item2attributes, num_attributes, attribute2id


def generate_rating_matrix(
    user_sequences: Dict[int, List[int]],
    num_users: int,
    num_items: int,
    exclude_last: int = 0
) -> csr_matrix:
    """Generate sparse rating matrix for evaluation.
    
    Args:
        user_sequences: User-item sequences
        num_users: Total number of users
        num_items: Total number of items
        exclude_last: Number of items to exclude from end of sequence
        
    Returns:
        Sparse rating matrix
    """
    row, col, data = [], [], []
    
    for user_id, items in user_sequences.items():
        end_idx = len(items) - exclude_last if exclude_last > 0 else len(items)
        for item in items[:end_idx]:
            row.append(user_id - 1)  # Convert to 0-indexed
            col.append(item)
            data.append(1)
            
    return csr_matrix(
        (data, (row, col)),
        shape=(num_users, num_items)
    )


def preprocess_amazon_beauty(
    reviews_path: str,
    metadata_path: str,
    output_dir: str,
    user_core: int = 5,
    item_core: int = 5,
    attribute_core: int = 0,
    min_rating: float = 0.0
) -> DataStats:
    """Full preprocessing pipeline for Amazon Beauty dataset.
    
    Args:
        reviews_path: Path to reviews JSON file
        metadata_path: Path to metadata JSON file
        output_dir: Directory to save processed data
        user_core: Minimum interactions per user
        item_core: Minimum interactions per item
        attribute_core: Minimum items per attribute
        min_rating: Minimum rating threshold
        
    Returns:
        Dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("S3Rec Low-rank AAP Data Preprocessing")
    print("=" * 60)
    
    # Step 1: Load reviews
    print("\n[1/6] Loading reviews...")
    interactions = load_amazon_reviews(reviews_path, min_rating)
    print(f"  Loaded {len(interactions):,} interactions")
    
    # Step 2: Create user sequences
    print("\n[2/6] Creating user sequences...")
    user_sequences = get_user_sequences(interactions)
    print(f"  {len(user_sequences):,} users before filtering")
    
    # Step 3: K-core filtering
    print(f"\n[3/6] Applying {user_core}/{item_core}-core filtering...")
    user_sequences = filter_k_core(user_sequences, user_core, item_core)
    print(f"  {len(user_sequences):,} users after filtering")
    
    # Step 4: Create ID mappings
    print("\n[4/6] Creating ID mappings...")
    mapped_seqs, user2id, item2id, id2user, id2item = create_id_mappings(user_sequences)
    num_users = len(user2id)
    num_items = len(item2id) + 1  # +1 for padding
    print(f"  {num_users:,} users, {num_items - 1:,} items")
    
    # Step 5: Load metadata and extract attributes
    print("\n[5/6] Extracting attributes...")
    valid_items = set(item2id.keys())
    metadata = load_amazon_metadata(metadata_path, valid_items)
    item2attributes, num_attributes, attribute2id = extract_attributes(
        metadata, item2id, attribute_core
    )
    
    # Ensure all items have attribute entries
    for item_id in range(1, num_items):
        if item_id not in item2attributes:
            item2attributes[item_id] = []
    
    # Step 6: Generate rating matrices and save
    print("\n[6/6] Generating matrices and saving...")
    
    # Create long sequence for negative sampling in SP
    long_sequence = []
    for user_id in sorted(mapped_seqs.keys()):
        long_sequence.extend(mapped_seqs[user_id])
    
    # Convert to list format
    user_sequences_list = [
        mapped_seqs.get(uid, []) 
        for uid in range(1, num_users + 1)
    ]
    
    # Generate rating matrices
    valid_matrix = generate_rating_matrix(mapped_seqs, num_users, num_items, exclude_last=2)
    test_matrix = generate_rating_matrix(mapped_seqs, num_users, num_items, exclude_last=1)
    
    # Calculate statistics
    seq_lengths = [len(seq) for seq in user_sequences_list]
    attr_counts = [len(item2attributes.get(i, [])) for i in range(1, num_items)]
    num_interactions = sum(seq_lengths)
    sparsity = (1 - num_interactions / (num_users * (num_items - 1))) * 100
    
    stats = DataStats(
        num_users=num_users,
        num_items=num_items,
        num_attributes=num_attributes,
        num_interactions=num_interactions,
        avg_seq_length=np.mean(seq_lengths),
        avg_attributes_per_item=np.mean(attr_counts),
        sparsity=sparsity
    )
    
    # Save all data
    save_data = {
        'user_sequences': user_sequences_list,
        'long_sequence': long_sequence,
        'item2attributes': item2attributes,
        'num_users': num_users,
        'num_items': num_items,
        'num_attributes': num_attributes,
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'attribute2id': attribute2id,
        'valid_matrix': valid_matrix,
        'test_matrix': test_matrix,
        'stats': stats.to_dict()
    }
    
    output_file = output_path / "beauty_processed.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Also save in text format for compatibility
    txt_file = output_path / "Beauty.txt"
    with open(txt_file, 'w') as f:
        for uid, items in enumerate(user_sequences_list, 1):
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")
    
    # Save item2attributes as JSON
    json_file = output_path / "Beauty_item2attributes.json"
    with open(json_file, 'w') as f:
        json.dump({str(k): v for k, v in item2attributes.items()}, f)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(stats)
    print(f"\nSaved to: {output_path}")
    
    return stats


def load_processed_data(data_path: str) -> Dict:
    """Load preprocessed data from pickle file.
    
    Args:
        data_path: Path to processed pickle file
        
    Returns:
        Dictionary containing all processed data
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Amazon Beauty dataset")
    parser.add_argument("--reviews", type=str, required=True,
                        help="Path to reviews JSON file")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata JSON file")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory")
    parser.add_argument("--user-core", type=int, default=5,
                        help="Minimum interactions per user")
    parser.add_argument("--item-core", type=int, default=5,
                        help="Minimum interactions per item")
    
    args = parser.parse_args()
    
    preprocess_amazon_beauty(
        reviews_path=args.reviews,
        metadata_path=args.metadata,
        output_dir=args.output,
        user_core=args.user_core,
        item_core=args.item_core
    )

