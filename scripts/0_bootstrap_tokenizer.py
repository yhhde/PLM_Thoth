# Bootstrap tokenizer training directly from raw data
# This solves the circular dependency between preprocessing and tokenizer training

import argparse
import os
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser(
        description="Train tokenizer directly from raw UNPC data (bootstrap step)"
    )
    p.add_argument("--raw_path", required=True, help="Path to raw dataset")
    p.add_argument("--tok_out", required=True, help="Where to save tokenizer")
    p.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    p.add_argument(
        "--max_samples",
        type=int,
        default=500000,
        help="Max samples to use for training (0 = all)",
    )
    args = p.parse_args()

    os.makedirs(args.tok_out, exist_ok=True)

    print(f"Loading raw dataset from {args.raw_path}...")
    ds = load_from_disk(args.raw_path)["train"]
    total = len(ds)
    print(f"Total samples in dataset: {total:,}")

    # Determine how many samples to use
    n_samples = args.max_samples if args.max_samples > 0 else total
    n_samples = min(n_samples, total)
    print(f"Using {n_samples:,} samples for tokenizer training...")

    # Format text in the same way as preprocessing
    print("Formatting text...")
    texts = []
    skipped = 0

    for i, row in enumerate(tqdm(ds, total=n_samples)):
        if i >= n_samples:
            break

        en = row["translation"]["en"].strip()
        fr = row["translation"]["fr"].strip()

        # Skip empty or untranslated pairs
        if not en or not fr:
            skipped += 1
            continue
        if en == fr:
            skipped += 1
            continue

        # Format: <en> English text <fr> French text
        texts.append(f"<en> {en} <fr> {fr}")

    print(f"Collected {len(texts):,} valid text samples (skipped {skipped:,})")

    # Initialize tokenizer
    print("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<en>", "<fr>"],
        show_progress=True,
    )

    # Train tokenizer
    print(f"Training tokenizer with vocab_size={args.vocab_size}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Wrap in HuggingFace interface
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    # Save tokenizer
    print(f"Saving tokenizer to {args.tok_out}...")
    tok.save_pretrained(args.tok_out)

    # Verify
    print("\n=== Verification ===")
    test_tok = PreTrainedTokenizerFast.from_pretrained(args.tok_out)

    test_text = "<en> The meeting was held in New York. <fr> La réunion a eu lieu à New York."
    encoded = test_tok(test_text)
    print(f"Test text: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Token count: {len(encoded['input_ids'])}")
    print(f"Decoded: {test_tok.decode(encoded['input_ids'])}")

    print("\n=== Special Tokens ===")
    print(f"pad_token: {test_tok.pad_token} (id={test_tok.pad_token_id})")
    print(f"unk_token: {test_tok.unk_token} (id={test_tok.unk_token_id})")
    print(f"bos_token: {test_tok.bos_token} (id={test_tok.bos_token_id})")
    print(f"eos_token: {test_tok.eos_token} (id={test_tok.eos_token_id})")

    print("\nTokenizer training complete!")


if __name__ == "__main__":
    main()

