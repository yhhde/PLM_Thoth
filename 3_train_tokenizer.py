import os
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
import argparse

# ---------------------------------------------------------
# Path helpers
# ---------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_tokenizer(
    dataset_path,
    tokenizer_out_path,
    vocab_size=50000,
):
    # Output dir must exist, input must already exist
    ensure_dir(tokenizer_out_path)

    # Load dataset
    ds = load_from_disk(dataset_path)
    train_data = ds["train"]

    # Pull all text rows (already formatted in preprocessing)
    print("Loading raw text into memory...")
    text_list = train_data["text"]  # HF loads the column lazily, cheap
    print(f"Loaded {len(text_list):,} rows.")

    # Wrap with tqdm for user-friendly progress updates
    text_iter = tqdm(text_list, desc="Feeding text to tokenizer", leave=True)

    # Init tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<en>", "<fr>"],
        show_progress=True,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(text_iter, trainer=trainer)

    # Wrap in HF interface
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    tok.save_pretrained(tokenizer_out_path)

    # quick sanity test
    test_tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_out_path)
    print(test_tok("<en> Hello. <fr> Bonjour."))

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a tokenizer.")
    p.add_argument("--data_path", required=True, help="Processed dataset path")
    p.add_argument("--tok_out", required=True, help="Where to save tokenizer")
    p.add_argument("--vocab_size", type=int, default=50000, help="Tokenizer vocab size")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        dataset_path=args.data_path,
        tokenizer_out_path=args.tok_out,
        vocab_size=args.vocab_size,
    )

