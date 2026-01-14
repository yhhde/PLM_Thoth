#6_validation.py

#!/usr/bin/env python3
"""
Compact validation for pretokenized bilingual dataset.

Assumes:
- dataset saved with load_from_disk and splits (train/validation/test)
- each example has 'input_ids' and 'attention_mask' for the full pair:
    <en> EN-TOKENS <fr> FR-TOKENS (padded to fixed length)
- special token ids: EN_ID and FR_ID (default 4 and 5)
- model is Custom GPT2 and tokenizer is PreTrainedTokenizerFast

Outputs JSON results and prints a small detokenized sample.
"""

import argparse, json, os, math, numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
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
        """Access underlying transformer for embedding extraction."""
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

def split_pair_ids(ids, en_id, fr_id):
    """Given a 1D tensor/list of token ids (full pair), return (src_list, tgt_list).
       If tags not found, return ([],[])."""
    ids_t = torch.tensor(ids, dtype=torch.long)
    en_pos = (ids_t == en_id).nonzero(as_tuple=False)
    fr_pos = (ids_t == fr_id).nonzero(as_tuple=False)
    if en_pos.numel() == 0 or fr_pos.numel() == 0:
        return [], []
    en_last = int(en_pos[-1].item())
    fr_last = int(fr_pos[-1].item())
    if en_last >= fr_last:
        return [], []
    src = ids_t[en_last+1:fr_last].tolist()
    tgt = ids_t[fr_last+1:].tolist()
    return src, tgt
    
# ---------------------------
# Table formater
# ---------------------------

def build_pair_ids(src, tgt, en_id, fr_id):
    """Construct full pair ids list: [en_id] + src + [fr_id] + tgt"""
    return [en_id] + list(src) + [fr_id] + list(tgt)
    
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
    # Task 2: Bitext Retrieval (Embedding)
    # --------------------------------------------------
    header("Task 2a: Bitext Retrieval (Embedding)")

    row(
        "Pass@1",
        results.get("pass@1"),
        bad=0.01,
        good=0.30,
        best=0.65,
        higher_is_better=True,
    )

    row(
        "Pass@5",
        results.get("pass@5"),
        bad=0.05,
        good=0.60,
        best=0.90,
        higher_is_better=True,
    )

    row(
        "MRR",
        results.get("mrr"),
        bad=0.05,
        good=0.40,
        best=0.75,
        higher_is_better=True,
    )

    lines.append("")

    # --------------------------------------------------
    # Task 2b: Bitext Retrieval (Conditional Probability)
    # --------------------------------------------------
    header("Task 2b: Bitext Retrieval (Cond. Prob.)")

    row(
        "Pass@1 (PPL)",
        results.get("pass@1_ppl"),
        bad=0.01,
        good=0.30,
        best=0.65,
        higher_is_better=True,
    )

    row(
        "Pass@5 (PPL)",
        results.get("pass@5_ppl"),
        bad=0.05,
        good=0.60,
        best=0.90,
        higher_is_better=True,
    )

    row(
        "MRR (PPL)",
        results.get("mrr_ppl"),
        bad=0.05,
        good=0.40,
        best=0.75,
        higher_is_better=True,
    )

    lines.append("")

    # --------------------------------------------------
    # Task 3: Bitext Alignment Discrimination
    # --------------------------------------------------
    header("Task 3: Bitext Alignment Discrimination ")

    row(
        "AUC",
        results.get("auc"),
        bad=0.50,
        good=0.75,
        best=0.90,
        higher_is_better=True,
    )

    row_no_ref("Avg real score", results.get("avg_real_score"))
    row_no_ref("Avg fake score", results.get("avg_fake_score"))

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
    pos = torch.arange(tgt_start+1, ids.size(0), device=device)
    valid = att[pos].bool()
    if valid.sum().item() == 0:
        return float("inf")

    pos = pos[valid]
    toks = ids[pos]
    tok_logp = logp[0, pos-1, toks]  # predict toks at step pos using logits at pos-1
    nll = -tok_logp.mean().item()
    return math.exp(nll)

def eval_ppl(model, ds, en_id, fr_id, device, max_samples):
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    ppls_tgt_given_src = []
    ppls_src_given_tgt = []
    for ex in tqdm(ds, desc="PPL"):
        inp = torch.tensor(ex["input_ids"], device=device).unsqueeze(0)
        att = torch.tensor(ex["attention_mask"], device=device).unsqueeze(0)
        # tgt|src : target is FR (fr_id) given EN
        ppls_tgt_given_src.append(conditional_ppl_from_pair(model, inp, att, fr_id, device))
        # src|tgt : target is EN (en_id) given FR
        ppls_src_given_tgt.append(conditional_ppl_from_pair(model, inp, att, en_id, device))

    # filter Inf
    fwd = [x for x in ppls_tgt_given_src if np.isfinite(x)]
    rev = [x for x in ppls_src_given_tgt if np.isfinite(x)]
    return {
        "ppl_fr_given_en": float(np.mean(fwd)) if len(fwd)>0 else float("inf"),
        "ppl_en_given_fr": float(np.mean(rev)) if len(rev)>0 else float("inf"),
        "num_samples": len(fwd)
    }

# ---------------------------
# Retrieval (embeddings)
# ---------------------------

def embed_sequence_ids(model, ids_list, device):
    """Build embedding for a short sequence of ids (list). Use model.transformer and mean-pool."""
    if len(ids_list) == 0:
        return None
    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    att = torch.ones_like(ids, device=device)
    with torch.no_grad():
        out = model.transformer(input_ids=ids, attention_mask=att)
        hs = out.last_hidden_state  # [1, L, D]
        mask = att.unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled[0].cpu()

def eval_retrieval(model, ds, en_id, fr_id, device, max_samples):
    if max_samples and len(ds) > max_samples:
        ds = ds.select(np.random.choice(len(ds), max_samples, replace=False))

    src_embs, tgt_embs = [], []
    for ex in tqdm(ds, desc="Embed"):
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id)
        e_src = embed_sequence_ids(model, [en_id] + src, device)   # include en tag for context
        e_tgt = embed_sequence_ids(model, [fr_id] + tgt, device)
        if e_src is None or e_tgt is None:
            continue
        src_embs.append(e_src)
        tgt_embs.append(e_tgt)

    if len(src_embs) == 0:
        return {"pass@1": 0.0, "pass@5": 0.0, "mrr": 0.0, "num_samples": 0}

    src_mat = torch.stack(src_embs)   # [N, D]
    tgt_mat = torch.stack(tgt_embs)   # [N, D]
    sim = (src_mat @ tgt_mat.T).numpy()

    ranks = []
    for i in range(sim.shape[0]):
        rank = np.where(np.argsort(-sim[i]) == i)[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {
        "pass@1": float(np.mean(ranks == 1)),
        "pass@5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "num_samples": len(ranks),
    }

# ---------------------------
# Retrieval (conditional probability) - Alternative method
# ---------------------------

def compute_conditional_logprob(model, full_ids, tgt_start_id, device, pad_id=1):
    """
    Compute log P(target | source) from a full sequence.
    
    Args:
        full_ids: list of token ids [en_id, src_tokens..., fr_id, tgt_tokens...]
        tgt_start_id: token id that marks start of target (e.g., fr_id)
        device: torch device
        pad_id: padding token id
    
    Returns:
        average log probability of target tokens
    """
    ids = torch.tensor([full_ids], device=device)
    att = torch.ones_like(ids)
    
    # Find target start position
    tgt_positions = (ids[0] == tgt_start_id).nonzero(as_tuple=False)
    if tgt_positions.numel() == 0:
        return -float('inf')
    
    tgt_start = tgt_positions[-1].item()
    
    # No target tokens after the marker
    if tgt_start + 1 >= len(full_ids):
        return -float('inf')
    
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=att)
        logp = F.log_softmax(out.logits, dim=-1)
    
    # Compute log prob of target tokens
    target_positions = torch.arange(tgt_start + 1, len(full_ids), device=device)
    target_tokens = ids[0, target_positions]
    
    # Filter out padding tokens
    valid_mask = target_tokens != pad_id
    if valid_mask.sum() == 0:
        return -float('inf')
    
    target_positions = target_positions[valid_mask]
    target_tokens = target_tokens[valid_mask]
    
    # logp[0, pos-1] predicts token at pos
    token_logprobs = logp[0, target_positions - 1, target_tokens]
    
    return token_logprobs.mean().item()


def eval_retrieval_by_ppl(model, ds, en_id, fr_id, device, max_samples=500, 
                          num_candidates=50, pad_id=1):
    """
    Memory-efficient retrieval evaluation using conditional probability ranking.
    
    For each query EN sentence, rank candidate FR sentences by P(FR | EN).
    
    Args:
        model: the language model
        ds: dataset with 'input_ids' column
        en_id: token id for <en>
        fr_id: token id for <fr>
        device: torch device
        max_samples: number of queries to evaluate (default 500)
        num_candidates: number of candidates per query (default 50)
        pad_id: padding token id
    
    Memory safe for 12GB GPU with default parameters.
    
    Returns:
        dict with pass@1, pass@5, mrr, num_samples
    """
    import random
    
    # Sample queries if needed
    N = len(ds)
    if max_samples and N > max_samples:
        query_indices = np.random.choice(N, max_samples, replace=False).tolist()
    else:
        query_indices = list(range(N))
    
    # Extract all (src, tgt) pairs
    all_pairs = []
    for ex in ds:
        src, tgt = split_pair_ids(ex["input_ids"], en_id, fr_id)
        if len(src) > 0 and len(tgt) > 0:
            all_pairs.append((src, tgt))
    
    if len(all_pairs) < 2:
        return {"pass@1_ppl": 0.0, "pass@5_ppl": 0.0, "mrr_ppl": 0.0, "num_samples_ppl": 0}
    
    N_pairs = len(all_pairs)
    ranks = []
    
    for i in tqdm(query_indices, desc="Retrieval by PPL"):
        if i >= N_pairs:
            continue
            
        query_src = all_pairs[i][0]  # EN tokens
        
        # Sample candidates (always include correct answer at position 0)
        candidate_indices = [i]  # correct answer
        other_indices = [j for j in range(N_pairs) if j != i]
        
        # Sample min(num_candidates-1, available) other candidates
        n_sample = min(num_candidates - 1, len(other_indices))
        candidate_indices.extend(random.sample(other_indices, n_sample))
        
        # Compute scores for all candidates
        scores = []
        for j in candidate_indices:
            candidate_tgt = all_pairs[j][1]  # FR tokens
            
            # Build: <en> query_src <fr> candidate_tgt
            full_ids = [en_id] + query_src + [fr_id] + candidate_tgt
            
            score = compute_conditional_logprob(model, full_ids, fr_id, device, pad_id)
            scores.append(score)
        
        # Find rank of correct answer (index 0 in candidate_indices)
        scores = np.array(scores)
        sorted_indices = np.argsort(-scores)  # descending
        rank = np.where(sorted_indices == 0)[0][0] + 1  # 1-indexed
        ranks.append(rank)
    
    if len(ranks) == 0:
        return {"pass@1_ppl": 0.0, "pass@5_ppl": 0.0, "mrr_ppl": 0.0, "num_samples_ppl": 0}
    
    ranks = np.array(ranks)
    return {
        "pass@1_ppl": float(np.mean(ranks == 1)),
        "pass@5_ppl": float(np.mean(ranks <= 5)),
        "mrr_ppl": float(np.mean(1.0 / ranks)),
        "num_samples_ppl": len(ranks),
    }

# ---------------------------
# Discrimination (AUC)
# ---------------------------

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

def get_lang_positions(ids, lang_token_id):
    """Return last position of a language token ID in the sequence"""
    pos = (ids == lang_token_id).nonzero(as_tuple=False).view(-1)
    return pos[-1].item() if pos.numel() > 0 else None

def score_pair_safe(model, ids, attn, lang_token_id, device):
    """
    Compute average log-probability of target tokens safely,
    avoiding out-of-vocab errors.
    """
    ids = ids[0]
    tgt_start = get_lang_positions(ids, lang_token_id)
    if tgt_start is None or tgt_start + 1 >= len(ids):
        return -1e9  # no target tokens

    span = torch.arange(tgt_start + 1, len(ids), device=device)
    token_ids = ids[span]

    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), attention_mask=attn)
        log_probs = F.log_softmax(out.logits, dim=-1)

    vocab_size = log_probs.size(-1)

    # Safety check: clip any token ID >= vocab_size
    if torch.any(token_ids >= vocab_size):
        token_ids = torch.clamp(token_ids, max=vocab_size - 1)

    return log_probs[0, span, token_ids].mean().item()

def eval_discrimination(model, ds, en_id, fr_id, device, max_samples=None):
    """
    Evaluate bitext alignment discrimination using safe indexing.
    Creates negative pairs by reusing the source from example i
    and the target from example (i+1).
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

    for i, ex in enumerate(tqdm(ds, desc="Discrimination")):
        ex2 = ds[(i + 1) % N]

        # Full length from dataset (pre-tokenized to fixed length)
        target_len = len(ex["input_ids"])

        # -------- Real pair: src_i + tgt_i --------
        src_i, tgt_i = split_pair_ids(ex["input_ids"], en_id, fr_id)
        if len(src_i) == 0 or len(tgt_i) == 0:
            continue  # skip malformed example

        real_ids_list = build_pair_ids(src_i, tgt_i, en_id, fr_id)
        real_ids, real_att = make_tensor(real_ids_list, target_len)
        pos_scores.append(score_pair_safe(model, real_ids, real_att, fr_id, device))

        # -------- Fake pair: src_i + tgt_(i+1) --------
        src_j, tgt_j = split_pair_ids(ex2["input_ids"], en_id, fr_id)
        if len(tgt_j) == 0:
            continue  # skip if the next one is malformed

        fake_ids_list = build_pair_ids(src_i, tgt_j, en_id, fr_id)
        fake_ids, fake_att = make_tensor(fake_ids_list, target_len)
        neg_scores.append(score_pair_safe(model, fake_ids, fake_att, fr_id, device))

    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    # Compute AUC safely
    auc = 0.5
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, scores)

    return {
        "auc": float(auc),
        "avg_real_score": float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0,
        "avg_fake_score": float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0,
        "num_samples": len(pos_scores),
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

    res_ppl = eval_ppl(model, ds, EN_ID, FR_ID, device, args.max_n)
    res_ret = eval_retrieval(model, ds, EN_ID, FR_ID, device, args.max_n)
    
    # New: Conditional probability based retrieval (better for GPT-2)
    res_ret_ppl = eval_retrieval_by_ppl(
        model, ds, EN_ID, FR_ID, device, 
        max_samples=min(500, len(ds)), 
        num_candidates=50,
        pad_id=pad_id
    )
    
    res_disc = eval_discrimination(model, ds, EN_ID, FR_ID, device, args.max_n)

    results = {**res_ppl, **res_ret, **res_ret_ppl, **res_disc}

    # --------------------------------------------------
    # Save JSON (machine-readable)
    # --------------------------------------------------
    json_out = os.path.join(args.model_path, "validation_results.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------
    # Save pretty text report (human-readable)
    # --------------------------------------------------
    table = format_results_table(results)

    txt_out = os.path.join(args.model_path, "validation_results.txt")
    with open(txt_out, "w") as f:
        f.write(table)
    
    # --------------------------------------------------
    # Create unified experiment results
    # --------------------------------------------------
    unified_results = {"validation": results}
    
    # Load training summary if exists
    training_summary_path = os.path.join(args.model_path, "training_summary.json")
    if os.path.exists(training_summary_path):
        with open(training_summary_path, "r") as f:
            training_summary = json.load(f)
        unified_results["training"] = training_summary
    
    # Load config if exists
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        unified_results["config"] = {
            "run_name": config.get("run", {}).get("name"),
            "model": config.get("model", {}),
            "training": config.get("training", {}),
        }
    
    # Save unified results
    unified_out = os.path.join(args.model_path, "experiment_results.json")
    with open(unified_out, "w") as f:
        json.dump(unified_results, f, indent=2)

    # --------------------------------------------------
    # Print nicely
    # --------------------------------------------------
    print()
    print(table)
    print()
    print("Saved JSON results to:", json_out)
    print("Saved text report to:", txt_out)
    print("Saved unified results to:", unified_out)

    detokenize_and_generate(model, tok, ds, EN_ID, FR_ID, device, n=5)


if __name__ == "__main__":
    main()
