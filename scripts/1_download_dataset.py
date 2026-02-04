import argparse
import os
from datasets import load_dataset, load_from_disk, DatasetDict

def main():
    p = argparse.ArgumentParser("Download UNPC EN-FR and create a smaller subset")
    p.add_argument("--data_path", required=True, help="Where to download/store the dataset")
    p.add_argument("--size", type=int, default=10000, help="If non-zero, create a small subset of this many rows")
    args = p.parse_args()

    data_path = args.data_path
    small_size = args.size

    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} does not exist.")

    # Try loading from disk first
    try:
        print(f"Attempting to load existing dataset from {data_path}...")
        ds = load_from_disk(data_path)
        print("Loaded existing dataset.")
    except Exception:
        print("Dataset not found on disk. Downloading UNPC en-fr dataset...")
        ds = load_dataset(
            "Helsinki-NLP/un_pc",
            "en-fr",
            cache_dir=data_path
        )
        print("Download complete. Saving...")
        ds.save_to_disk(data_path)
        print("Dataset saved.")

    # Create small subset
    if small_size is not None:
        print(f"Creating a small subset of {small_size} rows from the train split...")
        small_train = ds["train"].shuffle(seed=42).select(range(small_size))

        small_ds = DatasetDict({"train": small_train})
        for split in ds:
            if split != "train":
                small_ds[split] = ds[split]

        out_path = data_path + "_small"
        print(f"Saving small dataset to {out_path}...")
        small_ds.save_to_disk(out_path)
        print("Done.")

if __name__ == "__main__":
    main()

