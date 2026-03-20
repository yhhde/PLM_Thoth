#!/usr/bin/env python3
"""
Supplementary validation: Bitext Retrieval (MRR) + Discrimination (AUC).

Tasks:
  1) Bitext Retrieval — rank candidate translations by P(FR|EN), report Pass@1/5 and MRR
  2) Discrimination — AUC using random negatives (offset-by-1 target pairing)

All pad tokens are stripped before scoring.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from model import GPT2

np.random.seed(42)
torch.manual_seed(42)


# ------------------------------------------------------------------ #
#  Model wrapper                                                      #
# ------------------------------------------------------------------ #

class ModelWrapper:
    def __init__(self, model, pad_token_id=1):
        self.model = model
        self.config = dict(model.config) if hasattr(model, "config") else {}
        if "tokenizer" not in self.config:
            self.config["tokenizer"] = {}
        self.config["tokenizer"]["pad_token_id"] = pad_token_id

    def __call__(self, input_ids, attention_mask=None):
        logits, _ = self.model(input_ids, attention_mask)
        return type("Output", (), {"logits": logits})()

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self


# ------------------------------------------------------------------ #
#  Utilities                                                          #
# ------------------------------------------------------------------ #

def resolve_device(device_arg: str) -> torch.device:
    s = str(device_arg).strip()
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def split_pair_ids(ids, en_id, fr_id, pad_id=1):
    """Extract (src, tgt) from token ids, stripping pad tokens."""
    ids_t = torch.tensor(ids, dtype=torch.long)
    en_pos = (ids_t == en_id).nonzero(as_tuple=False)
    fr_pos = (ids_t == fr_id).nonzero(as_tuple=False)
    if en_pos.numel() == 0 or fr_pos.numel() == 0:
        return [], []
    en_last = int(en_pos[-1].item())
    fr_last = int(fr_pos[-1].item())
    if en_last >= fr_last:
        return [], []
    src = [t for t in ids_t[en_last + 1:fr_last].tolist() if t != pad_id]
    tgt = [t for t in ids_t[fr_last + 1:].tolist() if t != pad_id]
    return src, tgt


def build_pair_ids(src, tgt, en_id, fr_id):
    return [en_id] + list(src) + [fr_id] + list(tgt)


def score_pair_logprob(model, pair_ids, fr_id, max_seq_len, device):
    """Length-normalized conditional log-prob P(tgt|src) on a clean (no-pad) pair."""
    if len(pair_ids) > max_seq_len:
        pair_ids = pair_ids[:max_seq_len]
    ids = torch.tensor(pair_ids, device=device).unsqueeze(0)
    att = torch.ones_like(ids)

    tgt_pos = (ids[0] == fr_id).nonzero(as_tuple=False)
    if tgt_pos.numel() == 0:
        return -1e9
    tgt_start = int(tgt_pos[-1].item())
    if tgt_start + 1 >= ids.size(1):
        return -1e9

    span = torch.arange(tgt_start + 1, ids.size(1), device=device)
    if span.numel() == 0:
        return -1e9

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=att)
        logp = F.log_softmax(out.logits, dim=-1)

    tok_logp = logp[0, span - 1, ids[0, span]]
    return tok_logp.mean().item()


# ------------------------------------------------------------------ #
#  Task 1: Bitext Retrieval                                           #
# ------------------------------------------------------------------ #

def eval_retrieval(model, ds, en_id, fr_id, pad_id, max_seq_len, device,
                   max_samples=None, pool_size=20):
    """
    Pool-based bitext retrieval: for each EN query, rank pool_size FR candidates
    by P(FR|EN). Reports Pass@1, Pass@5, MRR.
    """
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    all_src, all_tgt = [], []
    for ex in ds:
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if src and tgt:
            all_src.append(src)
            all_tgt.append(tgt)

    n = len(all_src)
    if n < pool_size:
        return {"pass@1": 0.0, "pass@5": 0.0, "mrr": 0.0,
                "num_pools": 0, "num_queries": 0}

    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    num_pools = n // pool_size
    ranks = []

    for pool_i in tqdm(range(num_pools), desc=f"Retrieval (pool={pool_size})"):
        pool = indices[pool_i * pool_size:(pool_i + 1) * pool_size]
        for q_pos, q_idx in enumerate(pool):
            src_q = all_src[int(q_idx)]
            scores = []
            for c_idx in pool:
                tgt_c = all_tgt[int(c_idx)]
                pair = build_pair_ids(src_q, tgt_c, en_id, fr_id)
                scores.append(score_pair_logprob(model, pair, fr_id, max_seq_len, device))
            scores_arr = np.asarray(scores, dtype=np.float64)
            rank = int((scores_arr > scores_arr[q_pos]).sum()) + 1
            ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "pass@1": float((ranks_arr == 1).mean()),
        "pass@5": float((ranks_arr <= 5).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
        "num_pools": num_pools,
        "num_queries": len(ranks),
    }


# ------------------------------------------------------------------ #
#  Task 2: Discrimination (AUC)                                       #
# ------------------------------------------------------------------ #

def eval_discrimination(model, ds, en_id, fr_id, pad_id, max_seq_len, device,
                        max_samples=None):
    """
    Discrimination via AUC using random negatives (offset-by-1 target pairing).
    Positive: src_i paired with tgt_i.
    Negative: src_i paired with tgt_{i+1}.
    """
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    N = len(ds)

    all_targets = []
    for ex in ds:
        _, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        all_targets.append(tgt)

    def score(src, tgt):
        pair = build_pair_ids(src, tgt, en_id, fr_id)
        return score_pair_logprob(model, pair, fr_id, max_seq_len, device)

    pos_scores = []
    neg_scores = []

    for i, ex in enumerate(tqdm(ds, desc="Discrimination")):
        src_i, tgt_i = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if not src_i or not tgt_i:
            continue

        real = score(src_i, tgt_i)
        if real <= -1e8:
            continue
        pos_scores.append(real)

        tgt_j = all_targets[(i + 1) % N]
        if tgt_j:
            s = score(src_i, tgt_j)
            if s > -1e8:
                neg_scores.append(s)

    if not pos_scores or not neg_scores:
        return {"disc_auc": 0.5, "disc_gap": 0.0,
                "avg_real_score": 0.0, "num_samples_disc": 0}

    n = min(len(pos_scores), len(neg_scores))
    y_true = np.array([1] * n + [0] * n, dtype=np.int64)
    all_scores = np.array(pos_scores[:n] + neg_scores[:n], dtype=np.float32)
    auc = float(roc_auc_score(y_true, all_scores)) if len(np.unique(y_true)) >= 2 else 0.5
    gap = float(np.mean(pos_scores[:n]) - np.mean(neg_scores[:n]))

    return {
        "disc_auc": auc,
        "disc_gap": gap,
        "avg_real_score": float(np.mean(pos_scores)),
        "num_samples_disc": len(pos_scores),
    }


# ------------------------------------------------------------------ #
#  Output formatting                                                  #
# ------------------------------------------------------------------ #

def judge(score, good, higher_is_better=True):
    if score is None or not np.isfinite(score):
        return "?"
    if higher_is_better:
        if score >= good:
            return "PASS"
        elif score >= good * 0.8:
            return "WARN"
        return "FAIL"
    if score <= good:
        return "PASS"
    elif score <= good * 1.5:
        return "WARN"
    return "FAIL"


def format_report(results, model_name):
    auc = results.get("disc_auc", 0.5)
    gap = results.get("disc_gap", 0.0)
    lines = [
        "=" * 60,
        f"SUPPLEMENTARY VALIDATION: {model_name}",
        "=" * 60,
        "",
        "Task 1: Bitext Retrieval (pool=20)",
        "-" * 60,
        f"  Pass@1    : {results.get('pass@1', 0):.4f}  [{judge(results.get('pass@1', 0), 0.30)}]",
        f"  Pass@5    : {results.get('pass@5', 0):.4f}  [{judge(results.get('pass@5', 0), 0.60)}]",
        f"  MRR       : {results.get('mrr', 0):.4f}  [{judge(results.get('mrr', 0), 0.40)}]",
        f"  Queries   : {results.get('num_queries', 0)}",
        "",
        "Task 2: Discrimination (AUC)",
        "-" * 60,
        f"  AUC       : {auc:.4f}  [{judge(auc, 0.75)}]",
        f"  Gap       : {gap:.4f}",
        f"  Samples   : {results.get('num_samples_disc', 0)}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Supplementary validation: Retrieval (MRR) + Discrimination (AUC)."
    )
    p.add_argument("--model_path", required=True, help="Path to model directory")
    p.add_argument("--data_path", required=True, help="Path to tokenized dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--max_n_ret", type=int, default=4096, help="Max samples for retrieval")
    p.add_argument("--max_n_disc", type=int, default=4096, help="Max samples for discrimination")
    p.add_argument("--pool_size", type=int, default=20, help="Retrieval pool size")
    p.add_argument("--output_dir", default=None, help="Output directory (default: model_path)")
    p.add_argument("--device", default="0")
    p.add_argument("--en_id", type=int, default=4)
    p.add_argument("--fr_id", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = resolve_device(args.device)

    tok = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    ckpt = os.path.join(args.model_path, "best_checkpoint", "model.pt")
    raw_model = GPT2.load(ckpt).to(device).eval()
    model = ModelWrapper(raw_model, pad_token_id=pad_id)
    max_seq_len = model.config.get("model", model.config).get("max_seq_len", 128)

    ds = load_from_disk(args.data_path)[args.split]
    model_name = os.path.basename(os.path.normpath(args.model_path))
    short_id = model_name.split("_")[0] if "_" in model_name else model_name

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    res_ret = eval_retrieval(
        model, ds, args.en_id, args.fr_id, pad_id, max_seq_len, device,
        max_samples=args.max_n_ret, pool_size=args.pool_size,
    )
    res_disc = eval_discrimination(
        model, ds, args.en_id, args.fr_id, pad_id, max_seq_len, device,
        max_samples=args.max_n_disc,
    )

    results = {**res_ret, **res_disc}

    out_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(out_dir, exist_ok=True)

    json_out = os.path.join(out_dir, f"{short_id}_retrieval_disc.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    report = format_report(results, model_name)
    txt_out = os.path.join(out_dir, f"{short_id}_retrieval_disc.txt")
    with open(txt_out, "w") as f:
        f.write(report)

    print()
    print(report)
    print(f"\nJSON -> {json_out}")
    print(f"TXT  -> {txt_out}")


if __name__ == "__main__":
    main()
