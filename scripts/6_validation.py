#!/usr/bin/env python3
"""
Compact validation for pretokenized bilingual dataset.

Assumes:
- dataset saved with load_from_disk and splits (train/validation/test)
- each example has 'input_ids' and 'attention_mask' for the full pair:
    <en> EN-TOKENS <fr> FR-TOKENS (padded to fixed length)
- special token ids: EN_ID and FR_ID (default 4 and 5)
- model is Custom GPT2 and tokenizer is PreTrainedTokenizerFast

Downstream tasks:
1) Conditional PPL -- metrics: PPL(EN|FR), PPL(FR|EN)
2) Discrimination using Conditional Probability -- metrics: Accuracy, Recall, F1

Outputs JSON results and prints a small detokenized sample.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import PreTrainedTokenizerFast

from model import GPT2

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------
# Model Wrapper for HF-style API
# ---------------------------

class ModelWrapper:
    """
    Wrapper to make custom GPT2 model compatible with HuggingFace-style API.
    """
    def __init__(self, model, pad_token_id=1):
        self.model = model
        self.config = {"tokenizer": {"pad_token_id": pad_token_id}}

    def __call__(self, input_ids, attention_mask=None):
        """Forward pass returning object with logits attribute."""
        logits, _ = self.model(input_ids, attention_mask)
        return type('Output', (), {'logits': logits})()

    @property
    def transformer(self):
        """Access underlying transformer (kept for compatibility, not used here)."""
        return self.model.transformer

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self


# ---------------------------
# Utilities
# ---------------------------

def split_pair_ids(ids, en_id, fr_id, pad_id=1):
    """Given token ids for full pair, return (src_list, tgt_list) with padding stripped.
       If tags not found, return ([], [])."""
    ids_t = torch.tensor(ids, dtype=torch.long)
    en_pos = (ids_t == en_id).nonzero(as_tuple=False)
    fr_pos = (ids_t == fr_id).nonzero(as_tuple=False)
    if en_pos.numel() == 0 or fr_pos.numel() == 0:
        return [], []
    en_last = int(en_pos[-1].item())
    fr_last = int(fr_pos[-1].item())
    if en_last >= fr_last:
        return [], []
    src = [x for x in ids_t[en_last + 1:fr_last].tolist() if x != pad_id]
    tgt = [x for x in ids_t[fr_last + 1:].tolist() if x != pad_id]
    return src, tgt


def build_pair_ids(src, tgt, en_id, fr_id):
    """Construct full pair ids list: [en_id] + src + [fr_id] + tgt"""
    return [en_id] + list(src) + [fr_id] + list(tgt)


# ---------------------------
# Table formatter
# ---------------------------

def judge(score, bad, good, best, higher_is_better=True):
    if score is None or not np.isfinite(score):
        return "❌"

    if higher_is_better:
        if score >= good:
            return "✅"
        elif score >= good * 0.8:
            return "⚠"
        else:
            return "❌"
    else:
        if score <= good:
            return "✅"
        elif score <= good * 1.5:
            return "⚠️"
        else:
            return "❌"


def format_results_table(results):
    lines = []

    def header(title):
        lines.append(title)
        lines.append("-" * 60)
        lines.append(
            f"{'Metric':<15} | {'Score':>8} | {'Bad':>5} | {'Good':>5} | {'Best':>5} | {'':>1}"
        )
        lines.append("-" * 60)

    def row(metric, score, bad, good, best, higher_is_better=True):
        score_val = score
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        badge = judge(score_val, bad, good, best, higher_is_better)
        lines.append(
            f"{metric:<15} | {score_str:>8} | {bad:>5} | {good:>5} | {best:>5} | {badge:>1}"
        )

    def row_no_ref(metric, score):
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        lines.append(
            f"{metric:<15} | {score_str:>8} | {'':>5} | {'':>5} | {'':>5} | {'':>1}"
        )

    # --------------------------------------------------
    # Task 1: Conditional Perplexity
    # --------------------------------------------------
    header("Task 1: Conditional Perplexity")

    row(
        "PPL FR | EN",
        results.get("ppl_fr_given_en"),
        bad=50000,
        good=40,
        best=10,
        higher_is_better=False,
    )
    row(
        "PPL EN | FR",
        results.get("ppl_en_given_fr"),
        bad=50000,
        good=40,
        best=10,
        higher_is_better=False,
    )

    lines.append("")

    # --------------------------------------------------
    # Task 2: Discrimination using Conditional Probability
    # --------------------------------------------------
    header("Task 2: Discrimination (Conditional Probability)")

    row(
        "Accuracy",
        results.get("disc_accuracy"),
        bad=0.50,
        good=0.75,
        best=0.90,
        higher_is_better=True,
    )
    row(
        "Recall",
        results.get("disc_recall"),
        bad=0.50,
        good=0.75,
        best=0.90,
        higher_is_better=True,
    )
    row(
        "F1",
        results.get("disc_f1"),
        bad=0.50,
        good=0.75,
        best=0.90,
        higher_is_better=True,
    )
    row_no_ref("Best threshold", results.get("disc_best_threshold"))
    row_no_ref("Avg real score", results.get("avg_real_score"))
    row_no_ref("Avg fake score", results.get("avg_fake_score"))
    row_no_ref("Score gap", results.get("score_gap"))

    return "\n".join(lines)


# ---------------------------
# PPL
# ---------------------------

def conditional_ppl_from_pair(model, input_ids, attention_mask, tgt_id, device):
    """Compute PPL(target|context) on a single example already tokenized as full pair.
       Masks out padding using attention_mask."""
    ids = input_ids[0]
    att = attention_mask[0]

    # find target tag position
    tgt_pos = (ids == tgt_id).nonzero(as_tuple=False)
    if tgt_pos.numel() == 0:
        return float("inf")
    tgt_start = int(tgt_pos[-1].item())  # last occurrence
    if tgt_start + 1 >= ids.size(0):
        return float("inf")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logp = F.log_softmax(out.logits, dim=-1)  # [1, T, V]

    # positions to evaluate = tokens after tgt_start where attention_mask == 1
    pos = torch.arange(tgt_start + 1, ids.size(0), device=device)
    valid = att[pos].bool()
    if valid.sum().item() == 0:
        return float("inf")

    pos = pos[valid]
    toks = ids[pos]
    tok_logp = logp[0, pos - 1, toks]  # predict toks at step pos using logits at pos-1
    nll = -tok_logp.mean().item()
    return math.exp(nll)


def eval_ppl(model, ds, en_id, fr_id, device, max_samples):
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    pad_id = model.config["tokenizer"]["pad_token_id"]
    max_len = 128

    ppls_fwd, ppls_rev = [], []
    for ex in tqdm(ds, desc="PPL"):
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if not src or not tgt:
            continue

        # PPL(FR|EN): <en> src <fr> tgt — evaluate tgt given src context
        fwd_ids = [en_id] + src + [fr_id] + tgt
        if len(fwd_ids) > max_len:
            fwd_ids = fwd_ids[:max_len]
        fwd_t = torch.tensor(fwd_ids, device=device).unsqueeze(0)
        ppls_fwd.append(conditional_ppl_from_pair(model, fwd_t, torch.ones_like(fwd_t), fr_id, device))

        # PPL(EN|FR): <fr> tgt <en> src — reversed so EN tokens can attend to FR context
        rev_ids = [fr_id] + tgt + [en_id] + src
        if len(rev_ids) > max_len:
            rev_ids = rev_ids[:max_len]
        rev_t = torch.tensor(rev_ids, device=device).unsqueeze(0)
        ppls_rev.append(conditional_ppl_from_pair(model, rev_t, torch.ones_like(rev_t), en_id, device))

    fwd = [x for x in ppls_fwd if np.isfinite(x)]
    rev = [x for x in ppls_rev if np.isfinite(x)]
    return {
        "ppl_fr_given_en": float(np.mean(fwd)) if fwd else float("inf"),
        "ppl_en_given_fr": float(np.mean(rev)) if rev else float("inf"),
        "num_samples": len(fwd),
    }


# ---------------------------
# Discrimination (Conditional Probability -> Acc/Recall/F1)
# ---------------------------

def get_lang_positions(ids, lang_token_id):
    """Return last position of a language token ID in the sequence."""
    pos = (ids == lang_token_id).nonzero(as_tuple=False).view(-1)
    return pos[-1].item() if pos.numel() > 0 else None


def score_pair_safe(model, ids, attn, lang_token_id, device):
    """
    Compute average log-probability of target tokens (length-normalized),
    excluding padding via attention_mask, and using correct next-token alignment.
    """
    ids = ids[0]  # [T]
    attn = attn   # [1, T]

    tgt_start = get_lang_positions(ids, lang_token_id)
    if tgt_start is None or tgt_start + 1 >= len(ids):
        return -1e9  # no target tokens

    span = torch.arange(tgt_start + 1, len(ids), device=device)

    # Mask out padding / invalid positions
    valid = attn[0, span].bool()
    if valid.sum().item() == 0:
        return -1e9

    span = span[valid]
    token_ids = ids[span]

    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), attention_mask=attn)
        log_probs = F.log_softmax(out.logits, dim=-1)

    vocab_size = log_probs.size(-1)
    if torch.any(token_ids >= vocab_size):
        token_ids = torch.clamp(token_ids, max=vocab_size - 1)

    # Correct next-token alignment: token at span is predicted by logits at span-1
    pred_pos = span - 1
    tok_logp = log_probs[0, pred_pos, token_ids]

    # LENGTH NORMALIZATION: average per real target token
    return tok_logp.mean().item()


def eval_discrimination(model, ds, en_id, fr_id, device, max_samples=None):
    """
    Discrimination using conditional probability (length-normalized avg log P(tgt|src)).
    Creates negative pairs by taking src_i with tgt_(i+1).
    Reports Accuracy / Recall / F1 using a threshold chosen to maximize F1.
    """
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    pos_scores, neg_scores = [], []
    N = len(ds)
    pad_id = model.config["tokenizer"]["pad_token_id"]

    def make_tensor(ids_list, target_len):
        # pad / truncate to fixed length target_len
        if len(ids_list) >= target_len:
            ids_trim = ids_list[:target_len]
            att = [1] * target_len
        else:
            pad_len = target_len - len(ids_list)
            ids_trim = ids_list + [pad_id] * pad_len
            att = [1] * len(ids_list) + [0] * pad_len
        ids_tensor = torch.tensor(ids_trim, device=device).unsqueeze(0)
        att_tensor = torch.tensor(att, device=device).unsqueeze(0)
        return ids_tensor, att_tensor

    for i, ex in enumerate(tqdm(ds, desc="Discrimination (cond-prob)")):
        ex2 = ds[(i + 1) % N]

        # Full length from dataset (pre-tokenized to fixed length)
        target_len = len(ex["input_ids"])

        # -------- Real pair: src_i + tgt_i --------
        src_i, tgt_i = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if len(src_i) == 0 or len(tgt_i) == 0:
            continue  # skip malformed example

        real_ids_list = build_pair_ids(src_i, tgt_i, en_id, fr_id)
        real_ids, real_att = make_tensor(real_ids_list, target_len)
        pos_scores.append(score_pair_safe(model, real_ids, real_att, fr_id, device))

        # -------- Fake pair: src_i + tgt_(i+1) --------
        _, tgt_j = split_pair_ids(ex2["input_ids"], en_id, fr_id, pad_id)
        if len(tgt_j) == 0:
            continue  # skip if the next one is malformed

        fake_ids_list = build_pair_ids(src_i, tgt_j, en_id, fr_id)
        fake_ids, fake_att = make_tensor(fake_ids_list, target_len)
        neg_scores.append(score_pair_safe(model, fake_ids, fake_att, fr_id, device))

    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int64)
    scores = np.array(pos_scores + neg_scores, dtype=np.float32)

    if len(scores) == 0 or len(np.unique(y_true)) < 2:
        return {
            "disc_accuracy": 0.0,
            "disc_recall": 0.0,
            "disc_f1": 0.0,
            "disc_best_threshold": 0.0,
            "avg_real_score": float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0,
            "avg_fake_score": float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0,
            "num_samples": int(len(pos_scores)),
        }

    # Choose threshold to maximize F1 (predict positive iff score >= threshold)
    uniq = np.unique(scores)
    thresholds = np.concatenate(([uniq.min() - 1e-6], uniq, [uniq.max() + 1e-6]))

    best_f1 = -1.0
    best_thr = float(thresholds[0])
    best_acc = 0.0
    best_rec = 0.0

    for thr in thresholds:
        y_pred = (scores >= thr).astype(np.int64)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = float(f1v)
            best_thr = float(thr)
            best_acc = float(accuracy_score(y_true, y_pred))
            best_rec = float(recall_score(y_true, y_pred, zero_division=0))

    avg_real = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0
    avg_fake = float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0

    return {
        "disc_accuracy": best_acc,
        "disc_recall": best_rec,
        "disc_f1": best_f1,
        "disc_best_threshold": best_thr,
        "avg_real_score": avg_real,
        "avg_fake_score": avg_fake,
        "score_gap": avg_real - avg_fake,
        "num_samples": int(len(pos_scores)),
    }


# ---------------------------
# Detokenize / Samples
# ---------------------------

def detokenize_and_generate(model, tok, ds, en_id, fr_id, device, n=5):
    print("\n### SAMPLE EXAMPLES\n")
    for i in range(min(n, len(ds))):
        ex = ds[i]
        ids = ex["input_ids"]
        txt_in = tok.decode(ids, skip_special_tokens=False)
        print(f"Example {i}: {txt_in}")
        print()


# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model downstream.")
    p.add_argument("--model_path", required=True, help="Path to model directory")
    p.add_argument("--data_path", required=True, help="Tokenized dataset path")
    p.add_argument("--split", default="test", help="Dataset split")
    p.add_argument("--max_n", type=int, default=None, help="Max samples to evaluate")
    p.add_argument("--device", type=int, default=0, help="CUDA device")
    p.add_argument("--en_id", type=int, default=4, help="Token ID for <en>")
    p.add_argument("--fr_id", type=int, default=5, help="Token ID for <fr>")
    p.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for results (defaults to model_path)")
    p.add_argument("--model_id", type=str, default=None,
                    help="Short model ID for output filenames (e.g. r0v0)")
    return p.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(f"cuda:{args.device}")

    # Load tokenizer from model output directory
    tok = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    # Load model checkpoint
    ckpt = os.path.join(args.model_path, "best_checkpoint", "model.pt")
    raw_model = GPT2.load(ckpt).to(device).eval()

    # Wrap model for HF-style API compatibility
    model = ModelWrapper(raw_model, pad_token_id=pad_id)

    ds = load_from_disk(args.data_path)[args.split]
    EN_ID, FR_ID = args.en_id, args.fr_id

    print("\nRunning evaluation on split:", args.split)

    # Task 1
    res_ppl = eval_ppl(model, ds, EN_ID, FR_ID, device, args.max_n)

    # Task 2
    res_disc = eval_discrimination(model, ds, EN_ID, FR_ID, device, args.max_n)

    results = {**res_ppl, **res_disc}

    # --------------------------------------------------
    # Determine output paths
    # --------------------------------------------------
    out_dir = args.out_dir if args.out_dir else args.model_path
    os.makedirs(out_dir, exist_ok=True)

    model_id = args.model_id if args.model_id else os.path.basename(args.model_path.rstrip("/"))
    json_out = os.path.join(out_dir, f"{model_id}_validation.json")
    txt_out = os.path.join(out_dir, f"{model_id}_validation.txt")

    # --------------------------------------------------
    # Save JSON (machine-readable)
    # --------------------------------------------------
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------
    # Save pretty text report (human-readable)
    # --------------------------------------------------
    report_header = (
        "=" * 60 + "\n"
        f"EVALUATION REPORT: {model_id}\n"
        + "=" * 60 + "\n\n"
    )
    table = format_results_table(results)
    with open(txt_out, "w") as f:
        f.write(report_header + table)

    # --------------------------------------------------
    # Print nicely
    # --------------------------------------------------
    print()
    print(table)
    print()
    print("Saved JSON results to:", json_out)
    print("Saved text report to:", txt_out)

    detokenize_and_generate(model, tok, ds, EN_ID, FR_ID, device, n=5)


if __name__ == "__main__":
    main()
