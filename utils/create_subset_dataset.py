"""
Create a subset of the bucketed dataset while preserving bucket structure.

This script samples from each bucket proportionally to maintain the
same bucket distribution as the original dataset.

Usage:
    python create_subset_dataset.py \
        --data_path /path/to/tokenized_bucketed_mono \
        --out_path /path/to/subset \
        --frac 0.25

The script preserves:
    - Bucket structure (samples proportionally from each bucket)
    - Validation and test splits (copied unchanged)
    - Random seed for reproducibility
"""

import os
import json
import argparse
import random
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from collections import defaultdict


def detect_buckets(dataset, tokenizer, max_len=128, num_buckets=5):
    """
    Detect bucket boundaries and assign each row to a bucket.
    Uses actual tokenization for precise bucket assignment.
    Returns a dict mapping bucket_id -> list of row indices.
    """
    bucket_indices = defaultdict(list)
    
    for i, row in enumerate(tqdm(dataset, desc="Detecting buckets")):
        text = row["text"]
        # Use actual tokenization for precise length
        token_len = len(tokenizer.encode(text))
        
        # Assign to bucket based on max_len and num_buckets
        # This matches the logic in 2+_mono_and_bucket.py
        bucket_id = min(num_buckets - 1, token_len * num_buckets // max_len)
        bucket_indices[bucket_id].append(i)
    
    return bucket_indices


def sample_from_buckets(bucket_indices, frac, seed=42):
    """
    Sample a fraction of indices from each bucket.
    Returns combined list of selected indices.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    selected = []
    stats = {}
    
    for bucket_id, indices in sorted(bucket_indices.items()):
        n_original = len(indices)
        n_keep = int(n_original * frac)
        
        # Shuffle and select
        shuffled = indices.copy()
        random.shuffle(shuffled)
        selected.extend(shuffled[:n_keep])
        
        stats[f"bucket_{bucket_id}_original"] = n_original
        stats[f"bucket_{bucket_id}_selected"] = n_keep
        print(f"  Bucket {bucket_id}: {n_original} -> {n_keep} rows")
    
    return selected, stats


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset of bucketed dataset preserving bucket structure"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to tokenized_bucketed_mono dataset"
    )
    parser.add_argument(
        "--out_path", required=True,
        help="Output path for subset dataset"
    )
    parser.add_argument(
        "--frac", type=float, default=0.25,
        help="Fraction to keep (default: 0.25 = 25%%)"
    )
    parser.add_argument(
        "--max_len", type=int, default=128,
        help="Max sequence length used in bucketing"
    )
    parser.add_argument(
        "--num_buckets", type=int, default=5,
        help="Number of buckets"
    )
    parser.add_argument(
        "--tok_path", required=True,
        help="Path to tokenizer (for accurate bucket detection)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Validate
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    
    if args.frac <= 0 or args.frac > 1:
        raise ValueError("frac must be between 0 and 1")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    ds = load_from_disk(args.data_path)
    
    if "train" not in ds:
        raise ValueError("Dataset must contain 'train' split")
    
    train = ds["train"]
    n_original = len(train)
    
    print(f"\nOriginal train size: {n_original:,} rows")
    print(f"Target fraction: {args.frac * 100:.1f}%")
    print(f"Expected output: ~{int(n_original * args.frac):,} rows")
    
    # Load tokenizer for accurate bucket detection
    print("\nLoading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tok_path)
    
    # Detect buckets
    print("\nStep 1: Detecting bucket structure...")
    bucket_indices = detect_buckets(
        train,
        tokenizer,
        max_len=args.max_len, 
        num_buckets=args.num_buckets
    )
    
    print(f"\nFound {len(bucket_indices)} buckets:")
    for bid, indices in sorted(bucket_indices.items()):
        print(f"  Bucket {bid}: {len(indices):,} rows")
    
    # Sample from each bucket
    print(f"\nStep 2: Sampling {args.frac * 100:.1f}% from each bucket...")
    selected_indices, bucket_stats = sample_from_buckets(
        bucket_indices, 
        args.frac, 
        seed=args.seed
    )
    
    # Sort indices to maintain relative order within each section
    selected_indices.sort()
    
    print(f"\nTotal selected: {len(selected_indices):,} rows")
    
    # Create subset
    print("\nStep 3: Creating subset...")
    subset_train = train.select(tqdm(selected_indices, desc="Selecting"))
    
    # Build output dataset
    out = {"train": subset_train}
    
    # Copy validation and test splits unchanged
    for split in ds:
        if split != "train":
            out[split] = ds[split]
            print(f"  Copied {split}: {len(ds[split]):,} rows")
    
    subset_ds = DatasetDict(out)
    
    # Save
    print(f"\nStep 4: Saving to {args.out_path}...")
    os.makedirs(args.out_path, exist_ok=True)
    subset_ds.save_to_disk(args.out_path)
    
    # Save stats
    stats = {
        "original_rows": n_original,
        "selected_rows": len(selected_indices),
        "fraction": args.frac,
        "seed": args.seed,
        **bucket_stats
    }
    
    stats_path = os.path.join(args.out_path, "subset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original train: {n_original:,} rows")
    print(f"Subset train:   {len(selected_indices):,} rows ({args.frac * 100:.1f}%)")
    print(f"Saved to: {args.out_path}")
    print(f"Stats: {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
