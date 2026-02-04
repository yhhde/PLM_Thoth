import argparse
import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def parse_args():
    p = argparse.ArgumentParser(description="Pretokenize dataset splits.")
    p.add_argument("--in_path", required=True, help="Path to processed dataset")
    p.add_argument("--out_path", required=True, help="Path to save tokenized dataset")
    p.add_argument("--tok_path", required=True, help="Tokenizer directory")
    p.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    ds = load_from_disk(args.in_path)

    tokenized_splits = {}
    for split in ds.keys():
        print("Tokenizing split:", split)
        tokenized_splits[split] = ds[split].map(
            lambda batch: tokenize_batch(batch, tokenizer, args.max_len),
            batched=True,
            remove_columns=ds[split].column_names,
        )

    tokenized = DatasetDict(tokenized_splits)
    tokenized.save_to_disk(args.out_path)

    print("Saved all pretokenized splits to:", args.out_path)
