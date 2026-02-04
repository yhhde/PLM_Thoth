import os
import json
import argparse
import random
from collections import defaultdict

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_row(txt):
    en = txt.split("<en>")[1].split("<fr>")[0].strip()
    fr = txt.split("<fr>")[1].strip()
    return en, fr


def hist_ascii(lengths, title, edges, global_max):
    hist, _ = np.histogram(lengths, bins=edges)
    lines = [f"=== {title} ==="]

    for i, c in enumerate(hist):
        if c == 0:
            continue
        bar = "#" * int(40 * c / global_max)
        lines.append(f"{edges[i]:4d}-{edges[i+1]:4d}: {bar}")

    return "\n".join(lines)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    stats = defaultdict(int)

    ds = load_from_disk(args.in_path)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tok_path)

    train = ds["train"]
    N = len(train)

    # ----------------------------------------------
    # Token count BEFORE mono creation
    # ----------------------------------------------
    print("Counting tokens before monolingual expansion...")
    tokens_before = 0
    base_lengths = []

    for row in tqdm(train):
        l = len(tok.encode(row["text"]))
        tokens_before += l
        base_lengths.append(l)

    stats["total_rows_before"] = N
    stats["total_tokens_before"] = tokens_before
    
    # ----------------------------------------------
    # Decide mono vs bilingual
    # ----------------------------------------------
    idxs = list(range(N))
    random.shuffle(idxs)

    n_mono = int(args.p_mono * N)
    mono_idxs = set(idxs[:n_mono])
    bi_idxs = idxs[n_mono:]

    # force exact 50/50 direction split
    half = len(bi_idxs) // 2
    bi_en_fr = set(bi_idxs[:half])
    bi_fr_en = set(bi_idxs[half:])

    final_rows = []

    print("Creating monolingual and bilingual rows...")
    for i in tqdm(idxs):
        en, fr = parse_row(train[i]["text"])

        if i in mono_idxs:
            final_rows.append(f"<en> {en}")
            final_rows.append(f"<fr> {fr}")
            stats["mono_en"] += 1
            stats["mono_fr"] += 1
        elif i in bi_en_fr:
            final_rows.append(f"<en> {en} <fr> {fr}")
            stats["bi_en_fr"] += 1
        else:
            final_rows.append(f"<fr> {fr} <en> {en}")
            stats["bi_fr_en"] += 1

    print("Shuffling rows...")
    random.shuffle(final_rows)

    # ----------------------------------------------
    # Lengths after mono expansion
    # ----------------------------------------------
    lengths_after_mono = [len(tok.encode(x)) for x in final_rows]

    # ----------------------------------------------
    # GLOBAL bins and scale (single source of truth)
    # ----------------------------------------------
    NBINS = 20
    global_min = min(lengths_after_mono)
    global_max = max(lengths_after_mono)
    bin_edges = np.linspace(global_min, global_max, NBINS + 1, dtype=int)

    global_hist_max = max(
        np.histogram(lengths_after_mono, bins=bin_edges)[0]
    )

    # ----------------------------------------------
    # Global histogram
    # ----------------------------------------------
    hist_after_mono = hist_ascii(
        lengths_after_mono,
        "Row lengths incl. MONOLINGUAL sentences",
        edges=bin_edges,
        global_max=global_hist_max,
    )
    print(hist_after_mono)

    # ----------------------------------------------
    # Bucketing
    # ----------------------------------------------
    print("Bucketing...")
    buckets = [[] for _ in range(args.buckets)]
    bucket_lengths = [[] for _ in range(args.buckets)]

    for row in final_rows:
        l = len(tok.encode(row))
        b = min(args.buckets - 1, l * args.buckets // args.max_len)
        buckets[b].append(row)
        bucket_lengths[b].append(l)

    ordered = []
    for i, b in enumerate(buckets):
        stats[f"bucket_{i}_rows"] = len(b)
        ordered.extend(b)

    stats["total_rows"] = len(ordered)
    stats["total_tokens"] = sum(len(tok.encode(x)) for x in ordered)
    

    # ----------------------------------------------
    # Bucket histograms (same bins, same scale)
    # ----------------------------------------------
    lines = ["=== Row lengths by BUCKETS ==="]

    for i, lengths in enumerate(bucket_lengths):
        if not lengths:
            continue

        hist, _ = np.histogram(lengths, bins=bin_edges)

        lines.append(
            f"--- Bucket {i} (len {min(lengths)}–{max(lengths)}, n={len(lengths)}) ---"
        )

        for j, c in enumerate(hist):
            if c == 0:
                continue
            bar = "#" * int(40 * c / global_hist_max)
            lines.append(f"{bin_edges[j]:4d}-{bin_edges[j+1]:4d}: {bar}")

    bucket_plot = "\n".join(lines)
    print(bucket_plot)

    # ----------------------------------------------
    # Save dataset
    # ----------------------------------------------
    out_train = Dataset.from_dict({"text": ordered})

    out = DatasetDict({"train": out_train})
    for split in ds:
        if split != "train":
            out[split] = ds[split]

    out.save_to_disk(args.out_path)

    # ----------------------------------------------
    # Save stats + plots
    # ----------------------------------------------
    with open(os.path.join(args.out_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(args.out_path, "histograms.txt"), "w") as f:
        f.write(hist_after_mono)
        f.write("\n\n")
        f.write(bucket_plot)

    print("\n=== FINAL STATS ===")
    print(json.dumps(stats, indent=2))


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Create monolingual data and bucket by length")
    p.add_argument("--in_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--tok_path", required=True)
    p.add_argument("--p_mono", type=float, default=0.3)
    p.add_argument("--buckets", type=int, default=5)
    p.add_argument("--max_len", type=int, default=128)
    main(p.parse_args())
