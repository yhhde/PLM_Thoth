import os
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm


# -------------------------
# utils
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def hist_ascii(lengths, title, bins=20):
    hist, edges = np.histogram(lengths, bins=bins)
    lines = [f"\n=== {title} ==="]
    m = max(hist) if max(hist) > 0 else 1
    for i in range(bins):
        bar = "#" * int(40 * hist[i] / m)
        lines.append(f"{int(edges[i]):4d}-{int(edges[i+1]):4d}: {bar}")
    return "\n".join(lines)


# -------------------------
# main logic
# -------------------------

def main(args):
    from collections import defaultdict
    from datasets import Dataset, DatasetDict
    from tqdm import tqdm
    import json, os

    stats = defaultdict(int)

    print("Loading dataset...")
    raw = load_from_disk(args.in_path)["train"]

    print("Loading tokenizer...")
    tok = PreTrainedTokenizerFast.from_pretrained(args.tok_path)

    lengths_before = []
    lengths_after = []

    cleaned_texts = []
    cleaned_lengths = []

    seen = set()

    print("Cleaning, deduping, pruning, tokenizing...")
    for row in tqdm(raw, desc="Processing rows"):
        en = row["translation"]["en"].strip()
        fr = row["translation"]["fr"].strip()

        # raw length (count once)
        raw_len = len(tok.encode(en + " " + fr))
        lengths_before.append(raw_len)
        stats["total_tokens_raw"] += raw_len
        stats["total_rows_raw"] += 1

        # empty / partial
        if not en or not fr:
            stats["removed_empty"] += 1
            continue

        # untranslated
        if en == fr:
            stats["removed_untranslated"] += 1
            continue

        # duplicates
        pair = (en, fr)
        if pair in seen:
            stats["removed_duplicates"] += 1
            continue
        seen.add(pair)

        # build final text once
        text = f"<en> {en} <fr> {fr}"
        tok_ids = tok.encode(text)
        tok_len = len(tok_ids)

        # too long
        if tok_len > args.max_len:
            stats["removed_too_long"] += 1
            continue

        # keep
        cleaned_texts.append(text)
        cleaned_lengths.append(tok_len)

        stats["total_rows_cleaned"] += 1
        stats["total_tokens_cleaned"] += tok_len

    # ---- hist BEFORE ----
    hist_before = hist_ascii(lengths_before, "Row lengths in RAW dataset")
    print(hist_before)

    print("Creating Dataset object...")
    ds = Dataset.from_dict({"text": cleaned_texts})

    print("Shuffling dataset...")
    ds = ds.shuffle(seed=42)

    # ---- splits ----
    print("Splitting dataset...")
    val = ds.select(range(args.val_size))
    test = ds.select(range(args.val_size, args.val_size + args.test_size))
    train = ds.select(range(args.val_size + args.test_size, len(ds)))

    out = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    # ---- split stats (no tokenization here) ----
    def split_token_sum(start, end):
        return sum(cleaned_lengths[start:end])

    stats["rows_train"] = len(train)
    stats["rows_val"] = len(val)
    stats["rows_test"] = len(test)

    stats["tokens_val"] = split_token_sum(0, args.val_size)
    stats["tokens_test"] = split_token_sum(
        args.val_size, args.val_size + args.test_size
    )
    stats["tokens_train"] = split_token_sum(
        args.val_size + args.test_size, len(cleaned_lengths)
    )

    # ---- hist AFTER (train only) ----
    train_lengths = cleaned_lengths[args.val_size + args.test_size:]
    hist_after = hist_ascii(train_lengths, "Row lengths in CLEANED train split")
    print(hist_after)

    # ---- save ----
    ensure_dir(args.out_path)
    print("Saving dataset to disk...")
    out.save_to_disk(args.out_path)

    print("Writing stats and histograms...")
    with open(os.path.join(args.out_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(args.out_path, "histograms.txt"), "w") as f:
        f.write(hist_before)
        f.write("\n\n")
        f.write(hist_after)

    print("\n=== FINAL STATS ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Clean, dedupe, prune and split dataset")
    p.add_argument("--in_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--tok_path", required=True)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--val_size", type=int, default=4096)
    p.add_argument("--test_size", type=int, default=4096)
    main(p.parse_args())
