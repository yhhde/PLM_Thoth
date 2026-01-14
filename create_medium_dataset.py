import argparse
import os
from datasets import load_from_disk, DatasetDict
from tqdm.auto import tqdm

def main():
    p = argparse.ArgumentParser("Create a medium-sized dataset split.")
    p.add_argument("--data_path", required=True, help="Path to preprocessed dataset (DatasetDict).")
    p.add_argument("--out_path", required=True, help="Where to save the medium subset.")
    p.add_argument("--frac", type=float, default=0.1, help="Fraction of training data to keep.")
    args = p.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"{args.data_path} not found.")

    print("Loading dataset...")
    ds = load_from_disk(args.data_path)

    if "train" not in ds:
        raise ValueError("DatasetDict must contain a 'train' split.")

    train_len = len(ds["train"])
    keep_n = int(train_len * args.frac)

    print(f"Creating medium dataset: keeping {keep_n} of {train_len} training rows.")

    # Shuffle and take subset with progress bar
    print("Shuffling train split...")
    train_shuffled = ds["train"].shuffle(seed=42)

    print("Selecting subset...")
    idxs = list(range(keep_n))
    medium_train = train_shuffled.select(tqdm(idxs, desc="Selecting"))

    out = {"train": medium_train}

    # Copy all other splits untouched
    for split in ds:
        if split != "train":
            out[split] = ds[split]

    medium_ds = DatasetDict(out)
    print(medium_ds)

    print(f"Saving to {args.out_path}")
    medium_ds.save_to_disk(args.out_path)

    print("Done.")

if __name__ == "__main__":
    main()
