#!/usr/bin/env python3
"""
Tasks:
  1) Conditional Perplexity — PPL(FR|EN) and PPL(EN|FR), pad-stripped
  2) Bitext Retrieval — pool-based ranking by P(FR|EN), reports Pass@1/5 and MRR
  3) Discrimination — AUC via random negatives (offset-by-1 pairing)
"""

import argparse
import json
import math
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


def score_pair_logprob(model, pair_ids, tgt_tag_id, max_seq_len, device):
    if len(pair_ids) > max_seq_len:
        pair_ids = pair_ids[:max_seq_len]
    ids = torch.tensor(pair_ids, device=device).unsqueeze(0)
    att = torch.ones_like(ids)

    tgt_pos = (ids[0] == tgt_tag_id).nonzero(as_tuple=False)
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
#  Task 1: Conditional Perplexity                                     #
# ------------------------------------------------------------------ #

def eval_ppl(model, ds, en_id, fr_id, pad_id, max_seq_len, device, max_samples=None):
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    ppls_fwd, ppls_rev = [], []
    for ex in tqdm(ds, desc="PPL"):
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if not src or not tgt:
            continue

        fwd_ids = build_pair_ids(src, tgt, en_id, fr_id)
        if len(fwd_ids) > max_seq_len:
            fwd_ids = fwd_ids[:max_seq_len]
        fwd_t = torch.tensor(fwd_ids, device=device).unsqueeze(0)
        fwd_att = torch.ones_like(fwd_t)

        tgt_pos = (fwd_t[0] == fr_id).nonzero(as_tuple=False)
        if tgt_pos.numel() == 0:
            continue
        tgt_start = int(tgt_pos[-1].item())
        if tgt_start + 1 >= fwd_t.size(1):
            continue

        with torch.no_grad():
            out = model(input_ids=fwd_t, attention_mask=fwd_att)
            logp = F.log_softmax(out.logits, dim=-1)
        span = torch.arange(tgt_start + 1, fwd_t.size(1), device=device)
        nll_fwd = -logp[0, span - 1, fwd_t[0, span]].mean().item()
        ppls_fwd.append(math.exp(nll_fwd))

        rev_ids = [fr_id] + tgt + [en_id] + src
        if len(rev_ids) > max_seq_len:
            rev_ids = rev_ids[:max_seq_len]
        rev_t = torch.tensor(rev_ids, device=device).unsqueeze(0)
        rev_att = torch.ones_like(rev_t)

        en_pos = (rev_t[0] == en_id).nonzero(as_tuple=False)
        if en_pos.numel() == 0:
            continue
        en_start = int(en_pos[-1].item())
        if en_start + 1 >= rev_t.size(1):
            continue

        with torch.no_grad():
            out_rev = model(input_ids=rev_t, attention_mask=rev_att)
            logp_rev = F.log_softmax(out_rev.logits, dim=-1)
        span_rev = torch.arange(en_start + 1, rev_t.size(1), device=device)
        nll_rev = -logp_rev[0, span_rev - 1, rev_t[0, span_rev]].mean().item()
        ppls_rev.append(math.exp(nll_rev))

    fwd = [x for x in ppls_fwd if np.isfinite(x)]
    rev = [x for x in ppls_rev if np.isfinite(x)]
    return {
        "ppl_fr_given_en": float(np.mean(fwd)) if fwd else float("inf"),
        "ppl_en_given_fr": float(np.mean(rev)) if rev else float("inf"),
        "num_samples_ppl": len(fwd),
    }


# ------------------------------------------------------------------ #
#  Task 2: Bitext Retrieval                                           #
# ------------------------------------------------------------------ #

def eval_retrieval(model, ds_subset, en_id, fr_id, pad_id, max_seq_len, device,
                   pool_size, negative_mode="random", hard_neg_ratio=0.5) -> dict:
    all_src, all_tgt = [], []
    for ex in ds_subset:
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if src and tgt:
            all_src.append(src)
            all_tgt.append(tgt)

    n = len(all_src)
    if n < pool_size:
        return {"pass@1": 0.0, "pass@5": 0.0, "mrr": 0.0, "num_queries": 0}

    ranks = []

    if negative_mode == "random":
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        num_pools = n // pool_size
        for pool_i in tqdm(range(num_pools), desc=f"RET(pool={pool_size},random)", leave=False):
            pool = indices[pool_i * pool_size : (pool_i + 1) * pool_size]
            for q_pos, q_idx in enumerate(pool):
                src_q = all_src[int(q_idx)]
                scores = []
                for c_idx in pool:
                    tgt_c = all_tgt[int(c_idx)]
                    pair = build_pair_ids(src_q, tgt_c, en_id, fr_id)
                    s = score_pair_logprob(model, pair, fr_id, max_seq_len, device)
                    scores.append(s)
                scores_arr = np.asarray(scores, dtype=np.float64)
                correct_score = scores_arr[q_pos]
                rank = int((scores_arr > correct_score).sum()) + 1
                ranks.append(rank)
    else:
        # Hard negative logic
        tgt_lens = np.asarray([len(x) for x in all_tgt], dtype=np.int64)
        sorted_idx = np.argsort(tgt_lens)
        sorted_lens = tgt_lens[sorted_idx]

        def get_length_matched(q_idx: int, k: int) -> list:
            if k <= 0: return []
            q_len = tgt_lens[q_idx]
            pos = int(np.searchsorted(sorted_lens, q_len))
            out = []
            step = 0
            while len(out) < k and step < n:
                for c in [pos - step, pos + step]:
                    if 0 <= c < n:
                        cand = int(sorted_idx[c])
                        if cand != q_idx and cand not in out:
                            out.append(cand)
                    if len(out) >= k: break
                step += 1
            return out[:k]

        num_hard = min(max(0, int(round((pool_size - 1) * hard_neg_ratio))), pool_size - 1)
        num_rand = (pool_size - 1) - num_hard
        all_indices = np.arange(n, dtype=np.int64)

        for q_idx in tqdm(range(n), desc=f"RET(pool={pool_size},hard)", leave=False):
            src_q = all_src[q_idx]
            neg = get_length_matched(q_idx, num_hard)

            if num_rand > 0:
                mask = np.ones(n, dtype=bool)
                mask[q_idx] = False
                if neg: mask[np.asarray(neg, dtype=np.int64)] = False
                available = all_indices[mask]
                if available.size > 0:
                    take = min(num_rand, int(available.size))
                    neg.extend(int(x) for x in np.random.choice(available, take, replace=False))

            if len(neg) < (pool_size - 1):
                used = set(neg + [q_idx])
                backfill = [int(i) for i in all_indices if int(i) not in used]
                neg.extend(backfill[: pool_size - 1 - len(neg)])

            cand = [q_idx] + neg[: pool_size - 1]
            np.random.shuffle(cand)
            pos_of_gold = cand.index(q_idx)

            scores = []
            for c_idx in cand:
                tgt_c = all_tgt[c_idx]
                pair = build_pair_ids(src_q, tgt_c, en_id, fr_id)
                s = score_pair_logprob(model, pair, fr_id, max_seq_len, device)
                scores.append(s)
            scores_arr = np.asarray(scores, dtype=np.float64)
            correct_score = scores_arr[pos_of_gold]
            rank = int((scores_arr > correct_score).sum()) + 1
            ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "pass@1": float((ranks_arr == 1).mean()),
        "pass@5": float((ranks_arr <= 5).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
        "num_queries": len(ranks),
    }


# ------------------------------------------------------------------ #
#  Task 3: Discrimination (AUC)                                       #
# ------------------------------------------------------------------ #

def eval_discrimination(model, ds, en_id, fr_id, pad_id, max_seq_len, device,
                        max_samples=None):
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    N = len(ds)
    all_targets = []
    for ex in ds:
        _, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        all_targets.append(tgt)

    pos_scores, neg_scores = [], []
    for i, ex in enumerate(tqdm(ds, desc="Discrimination")):
        src_i, tgt_i = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if not src_i or not tgt_i:
            continue
        real = score_pair_logprob(model, build_pair_ids(src_i, tgt_i, en_id, fr_id),
                                  fr_id, max_seq_len, device)
        if real <= -1e8:
            continue
        pos_scores.append(real)

        tgt_j = all_targets[(i + 1) % N]
        if tgt_j:
            fake = score_pair_logprob(model, build_pair_ids(src_i, tgt_j, en_id, fr_id),
                                      fr_id, max_seq_len, device)
            if fake > -1e8:
                neg_scores.append(fake)

    if not pos_scores or not neg_scores:
        return {"disc_auc": 0.5, "avg_real_score": 0.0, "num_samples_disc": 0}

    n = min(len(pos_scores), len(neg_scores))
    y_true = np.array([1] * n + [0] * n, dtype=np.int64)
    all_scores = np.array(pos_scores[:n] + neg_scores[:n], dtype=np.float32)
    auc = float(roc_auc_score(y_true, all_scores)) if len(np.unique(y_true)) >= 2 else 0.5

    return {
        "disc_auc": auc,
        "avg_real_score": float(np.mean(pos_scores)),
        "num_samples_disc": len(pos_scores),
    }


# ------------------------------------------------------------------ #
#  Output formatting                                                  #
# ------------------------------------------------------------------ #

def judge_ppl(score, good=40):
    if score is None or not np.isfinite(score):
        return "❌"
    return "✅" if score <= good else ("⚠️" if score <= good * 1.5 else "❌")


def judge_high(score, good):
    if score is None or not np.isfinite(score):
        return "?"
    return "[PASS]" if score >= good else ("[WARN]" if score >= good * 0.8 else "[FAIL]")


def format_report(results, model_id):
    ppl_fwd = results.get("ppl_fr_given_en", float("inf"))
    ppl_rev = results.get("ppl_en_given_fr", float("inf"))
    mode_str = results.get("ret_negative_mode", "random")

    lines = [
        "EVALUATION REPORT: " + model_id,
        "=" * 60,
        "",
        "Task 1: Conditional Perplexity",
        "-" * 60,
        f"{'Metric':<15} | {'Score':>8} | {'Bad':>5} | {'Good':>5} | {'Best':>5} |",
        "-" * 60,
        f"{'PPL FR | EN':<15} | {ppl_fwd:>8.4f} | {'50000':>5} | {'40':>5} | {'10':>5} | {judge_ppl(ppl_fwd)}",
        f"{'PPL EN | FR':<15} | {ppl_rev:>8.4f} | {'50000':>5} | {'40':>5} | {'10':>5} | {judge_ppl(ppl_rev)}",
        "",
        f"Task 2: Bitext Retrieval (pool={results.get('pool_size', 20)}, mode={mode_str})",
        "-" * 60,
        f"  Pass@1  : {results.get('pass@1', 0):.4f}  {judge_high(results.get('pass@1', 0), 0.30)}",
        f"  Pass@5  : {results.get('pass@5', 0):.4f}  {judge_high(results.get('pass@5', 0), 0.60)}",
        f"  MRR     : {results.get('mrr', 0):.4f}  {judge_high(results.get('mrr', 0), 0.40)}",
        f"  Queries : {results.get('num_queries', 0)}",
        "",
        "Task 3: Discrimination (AUC)",
        "-" * 60,
        f"  AUC     : {results.get('disc_auc', 0.5):.4f}  {judge_high(results.get('disc_auc', 0.5), 0.75)}",
        f"  Samples : {results.get('num_samples_disc', 0)}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Unified validation: PPL + Retrieval + AUC.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max_n", type=int, default=4096)
    p.add_argument("--max_n_ret", type=int, default=4096)
    p.add_argument("--pool_size", type=int, default=20)
    p.add_argument("--ret_negative_mode", choices=["random", "hard"], default="random")
    p.add_argument("--hard_neg_ratio", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--en_id", type=int, default=4)
    p.add_argument("--fr_id", type=int, default=5)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=None)
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

    model_id = args.model_id
    if not model_id:
        model_id = os.path.basename(os.path.normpath(args.model_path))

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id}")
    print(f"{'='*60}")

    res_ppl = eval_ppl(model, ds, args.en_id, args.fr_id, pad_id,
                       max_seq_len, device, max_samples=args.max_n)

    if args.max_n_ret and len(ds) > args.max_n_ret:
        ds_ret = ds.select(np.random.choice(len(ds), args.max_n_ret, replace=False))
    else:
        ds_ret = ds

    res_ret = eval_retrieval(
        model, ds_ret, args.en_id, args.fr_id, pad_id, max_seq_len, device,
        pool_size=args.pool_size,
        negative_mode=args.ret_negative_mode,
        hard_neg_ratio=args.hard_neg_ratio
    )

    res_disc = eval_discrimination(model, ds, args.en_id, args.fr_id, pad_id,
                                   max_seq_len, device, max_samples=args.max_n)

    results = {**res_ppl, **res_ret, **res_disc, "pool_size": args.pool_size,
               "ret_negative_mode": args.ret_negative_mode}

    out_dir = args.out_dir if args.out_dir else args.model_path
    os.makedirs(out_dir, exist_ok=True)

    json_out = os.path.join(out_dir, f"{model_id}_validation.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    report = format_report(results, model_id)
    txt_out = os.path.join(out_dir, f"{model_id}_validation.txt")
    with open(txt_out, "w") as f:
        f.write(report)

    print()
    print(report)
    print(f"\nJSON -> {json_out}")
    print(f"TXT  -> {txt_out}")


if __name__ == "__main__":
    main()
