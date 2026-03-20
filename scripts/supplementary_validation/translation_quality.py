#!/usr/bin/env python3
"""
Supplementary validation: Translation Quality (chrF, COMET, LLM-as-Judge).

TSV naming (unified): blocked = no-repeat-3-gram -> {id}_blocked_translations.tsv;
  free = no ngram blocking -> {id}_free_translations.tsv.
  Results: translation_quality.json (blocked), translation_quality_ablation.json (free).

Modes:
  1) chrf  — greedy decode, chrF, write {id}_blocked_translations.tsv or _free_translations.tsv
  2) comet — score both TSV types, merge into translation_quality[._ablation].json
  3) llm   — paired ablation when both TSVs exist, else single-condition

Usage:
    # Blocked (no-repeat-3-gram, default)
    python translation_quality.py --mode chrf --model_path ... --data_path ...

    # Free (ablation)
    python translation_quality.py --mode chrf --no_repeat_ngram_size 0 --model_path ... --data_path ...

    # Step 2: COMET scoring on generated TSVs
    python translation_quality.py --mode comet --input_dir /path/to/results

    # Step 3: LLM-as-judge scoring on generated TSVs
    python translation_quality.py --mode llm --input_dir /path/to/results \\
        --scorer phi --device 0
"""

import argparse
import csv
import glob
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ================================================================== #
#  TSV naming: blocked = no-repeat-3-gram, free = no blocking       #
# ================================================================== #

BLOCKED_TSV_SUFFIX = "_blocked_translations.tsv"
FREE_TSV_SUFFIX = "_free_translations.tsv"


def _extended_layout_dirs(input_dir):
    """If input_dir has extended layout (tsv/blocked, tsv/free, translation_quality/blocked, translation_quality/free), return (blocked_tsv_dir, free_tsv_dir, blocked_out_dir, free_out_dir). Else all four = input_dir."""
    b_tsv = os.path.join(input_dir, "tsv", "blocked")
    f_tsv = os.path.join(input_dir, "tsv", "free")
    if os.path.isdir(b_tsv) or os.path.isdir(f_tsv):
        return (
            b_tsv if os.path.isdir(b_tsv) else input_dir,
            f_tsv if os.path.isdir(f_tsv) else input_dir,
            os.path.join(input_dir, "translation_quality", "blocked"),
            os.path.join(input_dir, "translation_quality", "free"),
        )
    return (input_dir, input_dir, input_dir, input_dir)


# ================================================================== #
#  TSV I/O                                                            #
# ================================================================== #

def load_tsv(path):
    """Load translation TSV. Auto-detects 3-col (EN, REF, GEN) or 6-col (+ gen_len, ref_len, rep_rate)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            en, ref, gen = row["EN"].strip(), row["REF"].strip(), row["GEN"].strip()
            gen_len = int(row["gen_len"]) if "gen_len" in row else len(gen)
            ref_len = int(row["ref_len"]) if "ref_len" in row else len(ref)
            rep_rate = float(row["rep_rate"]) if "rep_rate" in row else 0.0
            samples.append({
                "en": en, "ref": ref, "gen": gen,
                "gen_len": gen_len, "ref_len": ref_len, "rep_rate": rep_rate,
            })
    return samples


# ================================================================== #
#  Report formatting                                          #
# ================================================================== #

def judge(score, good, higher=True):
    if score is None or not np.isfinite(score):
        return "?"
    if higher:
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


def format_translation_quality_report(results, model_name):
    """Generate a human-readable TXT report from the combined JSON results."""
    nrng = results.get("no_repeat_ngram_size", 0)
    nrng_label = f"no-repeat-{nrng}-gram" if nrng > 0 else "no ngram blocking"
    lines = [
        "=" * 60,
        f"TRANSLATION QUALITY: {model_name} ({nrng_label})",
        "=" * 60,
        "",
        "Task: Translation Quality (EN->FR, greedy decode)",
        "-" * 60,
        f"  chrF        : {results.get('chrf', 0):.2f}  [{judge(results.get('chrf', 0), 30)}]",
    ]
    ci_lo = results.get("chrf_ci_low")
    ci_hi = results.get("chrf_ci_high")
    if ci_lo is not None and ci_hi is not None:
        lines.append(f"  chrF 95% CI : [{ci_lo:.2f}, {ci_hi:.2f}]")
    lines.append(f"  chrF-LP     : {results.get('chrf_lp', 0):.2f}  [{judge(results.get('chrf_lp', 0), 25)}]")

    comet = results.get("comet")
    if comet is not None:
        lines.append(f"  COMET       : {comet:.4f}  [{judge(comet, 0.75)}]")
        comet_ci_lo = results.get("comet_ci_low")
        comet_ci_hi = results.get("comet_ci_high")
        if comet_ci_lo is not None and comet_ci_hi is not None:
            lines.append(f"  COMET 95% CI: [{comet_ci_lo:.4f}, {comet_ci_hi:.4f}]")

    lines.extend([
        f"  Rep rate    : {results.get('rep_rate', 0):.4f}",
        f"  Avg gen len : {results.get('avg_gen_len', 0):.1f}",
        f"  Avg ref len : {results.get('avg_ref_len', 0):.1f}",
        f"  Len ratio   : {results.get('len_ratio', 0):.2f}",
        f"  Samples     : {results.get('num_samples', 0)}",
    ])
    return "\n".join(lines)


def bootstrap_ci(values, n_boot=1000, ci_alpha=0.95, seed=42):
    """Bootstrap confidence interval for the mean (used for chrF and COMET 95% CI)."""
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = arr[rng.integers(0, arr.size, size=arr.size)].mean()
    lo_q = (1.0 - ci_alpha) / 2.0
    return float(np.quantile(means, lo_q)), float(np.quantile(means, 1.0 - lo_q))


def save_translation_quality_results(results, out_dir, short_id, model_name,
                                     ablation=False):
    """Save combined JSON + TXT for translation quality."""
    suffix = "_ablation" if ablation else ""
    json_path = os.path.join(out_dir, f"{short_id}_translation_quality{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt_path = os.path.join(out_dir, f"{short_id}_translation_quality{suffix}.txt")
    report = format_translation_quality_report(results, model_name)
    with open(txt_path, "w") as f:
        f.write(report)

    return json_path, txt_path


# ================================================================== #
#  Mode 1: chrF (requires model + dataset)                            #
# ================================================================== #

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


def greedy_decode(model, prompt_ids, device, max_new_tokens=80, stop_ids=None,
                  no_repeat_ngram_size=0):
    """Autoregressive greedy decoding with optional no-repeat-ngram blocking."""
    if stop_ids is None:
        stop_ids = {1, 3}
    max_len = model.config.get("model", model.config).get("max_seq_len", 128)

    def repeats_ngram(seq, token, n):
        if n <= 1 or len(seq) < n - 1:
            return False
        cand = tuple(seq[-(n - 1):] + [token])
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n]) == cand:
                return True
        return False

    generated = []
    current_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(current_ids) >= max_len:
                break
            inp = torch.tensor(current_ids, device=device).unsqueeze(0)
            att = torch.ones_like(inp)
            out = model(input_ids=inp, attention_mask=att)
            logits = out.logits[0, -1, :]
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                top_k = min(128, int(logits.size(0)))
                topv, topi = torch.topk(logits, k=top_k)
                nxt = None
                for idx in topi.tolist():
                    if not repeats_ngram(current_ids, int(idx), no_repeat_ngram_size):
                        nxt = int(idx)
                        break
                if nxt is None:
                    nxt = int(topi[0].item())
            else:
                nxt = int(logits.argmax().item())
            if nxt in stop_ids:
                break
            generated.append(nxt)
            current_ids.append(nxt)
    return generated


def compute_rep_rate(token_ids, n=3):
    """Fraction of n-grams that are duplicates."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def run_chrf(args):
    """Generate translations via greedy decode and score with chrF."""
    from datasets import load_from_disk
    from transformers import PreTrainedTokenizerFast
    from sacrebleu.metrics import CHRF
    from model import GPT2

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(f"cuda:{args.device}")

    tok = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    ckpt = os.path.join(args.model_path, "best_checkpoint", "model.pt")
    raw_model = GPT2.load(ckpt).to(device).eval()
    model = ModelWrapper(raw_model, pad_token_id=pad_id)

    ds = load_from_disk(args.data_path)[args.split]
    en_id, fr_id = args.en_id, args.fr_id
    stop_ids = {pad_id, 3, en_id}

    max_n = args.max_n_chrf
    if max_n and len(ds) > max_n:
        ds = ds.select(np.random.choice(len(ds), max_n, replace=False))

    hypotheses, references = [], []
    gen_lens, ref_lens, rep_rates = [], [], []
    samples = []

    for ex in tqdm(ds, desc="chrF (greedy decode)"):
        src_en, tgt_fr = split_pair_ids(ex["input_ids"], en_id, fr_id, pad_id)
        if not src_en or not tgt_fr:
            continue

        prompt = [en_id] + src_en + [fr_id]
        max_new = int(len(src_en) * 1.5) + 10
        gen_ids = greedy_decode(model, prompt, device,
                                max_new_tokens=max_new, stop_ids=stop_ids,
                                no_repeat_ngram_size=args.no_repeat_ngram_size)

        hypothesis = tok.decode(gen_ids, skip_special_tokens=True).strip()
        reference = tok.decode(tgt_fr, skip_special_tokens=True).strip()
        if not hypothesis:
            hypothesis = "<empty>"

        hypotheses.append(hypothesis)
        references.append(reference)
        gen_lens.append(len(gen_ids))
        ref_lens.append(len(tgt_fr))
        rep_rates.append(compute_rep_rate(gen_ids))
        samples.append({
            "en": tok.decode(src_en, skip_special_tokens=True).strip(),
            "ref": reference,
            "gen": hypothesis,
            "gen_len": len(gen_ids),
            "ref_len": len(tgt_fr),
            "rep_rate": rep_rates[-1],
        })

    if not hypotheses:
        print("No valid samples for chrF evaluation.")
        return

    chrf_metric = CHRF()
    chrf_score = chrf_metric.corpus_score(hypotheses, [references]).score
    sent_scores = [float(chrf_metric.sentence_score(h, [r]).score)
                   for h, r in zip(hypotheses, references)]
    ci_lo, ci_hi = bootstrap_ci(sent_scores, n_boot=1000, ci_alpha=0.95, seed=args.seed + 303)

    avg_gen = float(np.mean(gen_lens))
    avg_ref = float(np.mean(ref_lens))
    len_ratio = avg_gen / avg_ref if avg_ref > 0 else 1.0
    lp = min(1.0, avg_ref / avg_gen) if avg_gen > 0 else 1.0

    results = {
        "chrf": round(chrf_score, 2),
        "chrf_ci_low": round(ci_lo, 2) if ci_lo is not None else None,
        "chrf_ci_high": round(ci_hi, 2) if ci_hi is not None else None,
        "chrf_lp": round(chrf_score * lp, 2),
        "rep_rate": round(float(np.mean(rep_rates)), 4),
        "avg_gen_len": round(avg_gen, 1),
        "avg_ref_len": round(avg_ref, 1),
        "len_ratio": round(len_ratio, 2),
        "num_samples": len(hypotheses),
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    out_dir = args.output_dir if args.output_dir else args.model_path
    model_name = os.path.basename(os.path.normpath(args.model_path))
    short_id = model_name.split("_")[0] if "_" in model_name else model_name
    is_blocked = args.no_repeat_ngram_size >= 1
    tsv_suffix = BLOCKED_TSV_SUFFIX if is_blocked else FREE_TSV_SUFFIX

    _, _, blocked_out, free_out = _extended_layout_dirs(out_dir)
    if os.path.isdir(os.path.join(out_dir, "tsv", "blocked")) or os.path.isdir(os.path.join(out_dir, "tsv", "free")):
        tsv_dir = os.path.join(out_dir, "tsv", "blocked") if is_blocked else os.path.join(out_dir, "tsv", "free")
        result_dir = blocked_out if is_blocked else free_out
        os.makedirs(tsv_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
    else:
        tsv_dir = result_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    tsv_out = os.path.join(tsv_dir, f"{short_id}{tsv_suffix}")
    with open(tsv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["EN", "REF", "GEN", "gen_len", "ref_len", "rep_rate"])
        for s in samples:
            writer.writerow([
                s["en"], s["ref"], s["gen"],
                s["gen_len"], s["ref_len"], f"{s['rep_rate']:.4f}",
            ])

    # Save JSON + TXT (blocked -> translation_quality; free -> translation_quality_ablation)
    json_out, txt_out = save_translation_quality_results(
        results, result_dir, short_id, model_name, ablation=not is_blocked)

    nrng = args.no_repeat_ngram_size
    print(f"\nchrF = {results['chrf']:.2f}  (LP: {results['chrf_lp']:.2f})")
    print(f"no_repeat_ngram_size: {nrng} ({'enabled' if nrng > 0 else 'disabled'})")
    print(f"Repetition rate: {results['rep_rate']:.4f}")
    print(f"Length ratio: {results['len_ratio']:.2f}")
    print(f"Samples: {results['num_samples']}")
    print(f"JSON -> {json_out}")
    print(f"TXT  -> {txt_out}")
    print(f"TSV  -> {tsv_out}")


# ================================================================== #
#  Mode 2: COMET (reads TSVs)                                        #
# ================================================================== #

def run_comet(args):
    """Score blocked and/or free translation TSVs with COMET; merge into translation_quality[._ablation].json."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("Please install unbabel-comet: pip install unbabel-comet")
        return

    input_dir = args.input_dir
    blocked_tsv_dir, free_tsv_dir, blocked_out_dir, free_out_dir = _extended_layout_dirs(input_dir)
    out_dirs = (blocked_out_dir, free_out_dir)
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    blocked_files = {f.replace(BLOCKED_TSV_SUFFIX, ""): f for f in os.listdir(blocked_tsv_dir)
                     if f.endswith(BLOCKED_TSV_SUFFIX)}
    free_files = {f.replace(FREE_TSV_SUFFIX, ""): f for f in os.listdir(free_tsv_dir)
                  if f.endswith(FREE_TSV_SUFFIX)}
    model_ids = sorted(set(blocked_files.keys()) | set(free_files.keys()))
    if not model_ids:
        print(f"No *{BLOCKED_TSV_SUFFIX} or *{FREE_TSV_SUFFIX} found under {input_dir}")
        return

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    comet_model_name = args.comet_model
    print(f"Loading COMET: {comet_model_name}")
    comet_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(comet_path)
    comet_model.eval()

    tsv_dirs = (blocked_tsv_dir, free_tsv_dir)
    for model_id in model_ids:
        for ablation, suffix, file_key, tsv_dir in [(False, BLOCKED_TSV_SUFFIX, blocked_files, blocked_tsv_dir),
                                                    (True, FREE_TSV_SUFFIX, free_files, free_tsv_dir)]:
            if model_id not in file_key:
                continue
            tsv_path = os.path.join(tsv_dir, file_key[model_id])
            print(f"\n>>> {model_id}{suffix}")
            raw_samples = load_tsv(tsv_path)
            if not raw_samples:
                print("  No samples, skipping.")
                continue

            comet_data = [{"src": s["en"], "ref": s["ref"], "mt": s["gen"]} for s in raw_samples]
            output = comet_model.predict(
                comet_data, batch_size=args.batch_size,
                gpus=1 if torch.cuda.is_available() else 0,
            )
            overall = float(output.system_score if hasattr(output, "system_score") else output["system_score"])
            seg_scores = getattr(output, "scores", None) if not isinstance(output, dict) else output.get("scores")
            if seg_scores and len(seg_scores) >= 2:
                c_lo, c_hi = bootstrap_ci(seg_scores, n_boot=1000, ci_alpha=0.95, seed=args.seed + 404)
                comet_ci_low = round(c_lo, 4) if c_lo is not None else None
                comet_ci_high = round(c_hi, 4) if c_hi is not None else None
            else:
                comet_ci_low = comet_ci_high = None

            out_dir = free_out_dir if ablation else blocked_out_dir
            existing_json = os.path.join(out_dir, f"{model_id}_translation_quality{'_ablation' if ablation else ''}.json")
            if os.path.exists(existing_json):
                with open(existing_json) as f:
                    results = json.load(f)
            else:
                results = {}
            results["comet"] = round(overall, 4)
            results["comet_model"] = comet_model_name
            results["comet_ci_low"] = comet_ci_low
            results["comet_ci_high"] = comet_ci_high
            save_translation_quality_results(results, out_dir, model_id, model_id, ablation=ablation)

            ci_str = f"  95% CI = [{comet_ci_low:.4f}, {comet_ci_high:.4f}]" if (comet_ci_low and comet_ci_high) else ""
            out_label = f"{model_id}_translation_quality{'_ablation' if ablation else ''}.json/.txt"
            print(f"  COMET = {overall:.4f}{ci_str}  ({len(raw_samples)} samples)  -> {out_label}")


# ================================================================== #
#  Mode 3: LLM-as-Judge (reads TSVs)                                 #
# ================================================================== #

SCORER_MODELS = {
    "mistral": {"name": "mistralai/Mistral-7B-Instruct-v0.2", "memory_gb": 14},
    "croissant": {"name": "croissantllm/CroissantLLMChat-v0.1", "memory_gb": 14},
    "qwen": {"name": "Qwen/Qwen2.5-7B-Instruct", "memory_gb": 14},
    "phi": {"name": "microsoft/Phi-3-mini-4k-instruct", "memory_gb": 8},
    "phi2": {"name": "microsoft/phi-2", "memory_gb": 6},
}

SCORING_PROMPT = """You are an expert evaluator for English-to-French machine translation.

**Source (English):**
{en}

**Reference (French):**
{ref}

**Model Output (French):**
{gen}

Evaluate the model output on these criteria (1-5 scale):
1. **Accuracy**: Does it correctly convey the meaning of the source?
2. **Fluency**: Is the French grammatically correct and natural?
3. **Completeness**: Does it translate the full sentence without missing parts?
4. **Conciseness**: Does it avoid unnecessary repetition or extra content?

Respond ONLY with a JSON object:
{{"accuracy": X, "fluency": X, "completeness": X, "conciseness": X, "overall": X, "comment": "brief explanation"}}"""


class LLMScorer:
    def __init__(self, scorer_key="phi", device="cuda:0", use_4bit=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if scorer_key not in SCORER_MODELS:
            raise ValueError(f"Unknown scorer: {scorer_key}. Available: {list(SCORER_MODELS.keys())}")

        info = SCORER_MODELS[scorer_key]
        model_name = info["name"]
        print(f"Loading scorer: {scorer_key} ({model_name}, ~{info['memory_gb']}GB)")

        self.device = device
        self.scorer_key = scorer_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_4bit:
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=qconfig,
                device_map=device, trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                attn_implementation="eager", trust_remote_code=True,
            ).to(device)
        self.model.eval()
        print("  Scorer loaded.")

    def score(self, en, ref, gen, max_new_tokens=200):
        prompt = SCORING_PROMPT.format(en=en, ref=ref, gen=gen)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return {"error": "No JSON found", "raw": response[:200]}
        except json.JSONDecodeError as e:
            return {"error": str(e), "raw": response[:200]}


def select_samples(samples, top_n=50):
    """Select representative samples: worst by repetition, worst by length, best overall (single-condition)."""
    for s in samples:
        s["len_ratio"] = s["gen_len"] / max(1, s["ref_len"])
        s["quality_score"] = s["rep_rate"] + abs(1.0 - s["len_ratio"])

    worst_rep = sorted(samples, key=lambda x: x["rep_rate"], reverse=True)[:top_n]
    worst_len = sorted(samples, key=lambda x: x["len_ratio"], reverse=True)[:top_n]
    best = sorted(samples, key=lambda x: x["quality_score"])[:top_n]

    seen = set()
    selected = []
    for group_name, group in [("worst_rep", worst_rep), ("worst_len", worst_len), ("best", best)]:
        for s in group:
            if s["en"] not in seen:
                seen.add(s["en"])
                s["selection_reason"] = group_name
                selected.append(s)
    return selected


def select_keys_from_common(blocked_by_en, free_by_en, top_n=10, seed=42):
    """Stratified selection from sentences present in BOTH conditions (aligned with llm_scoring_colab.ipynb).
    Groups: high_ratio (most over-generated), mid_ratio (closest to ref length), random."""
    rng = np.random.default_rng(seed)
    common_ens = set(blocked_by_en.keys()) & set(free_by_en.keys())
    pool = []
    for en in common_ens:
        s = dict(blocked_by_en[en])
        s["len_ratio"] = s["gen_len"] / max(1, s["ref_len"])
        pool.append(s)

    high = sorted(pool, key=lambda x: x["len_ratio"], reverse=True)[:top_n]
    mid = sorted(pool, key=lambda x: abs(x["len_ratio"] - 1.0))[:top_n]
    used_ens = {s["en"] for s in high} | {s["en"] for s in mid}
    rest = [s for s in pool if s["en"] not in used_ens]
    rand = list(rng.choice(len(rest), size=min(top_n, len(rest)), replace=False)) if rest else []
    rand = [rest[i] for i in rand] if rand else []

    selected = {}
    for group_name, group in [("high_ratio", high), ("mid_ratio", mid), ("random", rand)]:
        for s in group:
            if s["en"] not in selected:
                selected[s["en"]] = group_name
    return selected


def _aggregate_llm(scored, dims=None):
    dims = dims or ["accuracy", "fluency", "completeness", "conciseness", "overall"]
    valid = [r for r in scored if "error" not in r.get("llm_scores", {})]
    if not valid:
        return {}, {}
    avgs = {d: round(sum(r["llm_scores"].get(d, 0) for r in valid) / len(valid), 2) for d in dims}
    by_reason = {}
    for r in valid:
        gk = r.get("selection_reason", "unknown")
        by_reason.setdefault(gk, []).append(r["llm_scores"])
    reason_avgs = {
        gk: {"count": len(v), "avg_overall": round(sum(s.get("overall", 0) for s in v) / len(v), 2)}
        for gk, v in by_reason.items()
    }
    return avgs, reason_avgs


def run_llm(args):
    """Score translation TSVs with an LLM judge. When both blocked and free TSVs exist: paired
    ablation (same EN, stratified selection). Else: single-condition scoring with select_samples."""
    input_dir = args.input_dir
    out_dir = args.output_dir if args.output_dir else input_dir
    blocked_tsv_dir, free_tsv_dir, blocked_out_dir, free_out_dir = _extended_layout_dirs(input_dir)
    scorer_key = args.scorer
    top_n = args.top_n
    seed = getattr(args, "seed", 42)

    blocked_files = {f.replace(BLOCKED_TSV_SUFFIX, ""): f for f in os.listdir(blocked_tsv_dir)
                     if f.endswith(BLOCKED_TSV_SUFFIX)}
    free_files = {f.replace(FREE_TSV_SUFFIX, ""): f for f in os.listdir(free_tsv_dir)
                  if f.endswith(FREE_TSV_SUFFIX)}
    model_ids_ablation = sorted(set(blocked_files.keys()) & set(free_files.keys()))

    if model_ids_ablation:
        # Paired ablation: same EN in both conditions (aligned with notebook)
        all_blocked = {}
        all_free = {}
        global_ens = None
        for mid in model_ids_ablation:
            b = load_tsv(os.path.join(blocked_tsv_dir, blocked_files[mid]))
            f = load_tsv(os.path.join(free_tsv_dir, free_files[mid]))
            b_by_en = {s["en"]: s for s in b}
            f_by_en = {s["en"]: s for s in f}
            all_blocked[mid] = b_by_en
            all_free[mid] = f_by_en
            common = set(b_by_en.keys()) & set(f_by_en.keys())
            global_ens = common if global_ens is None else global_ens & common

        print(f"Found {len(model_ids_ablation)} models with BOTH conditions (blocked + free)")
        print(f"Global intersection: {len(global_ens)} source sentences")

        first_blocked = {en: all_blocked[model_ids_ablation[0]][en] for en in global_ens}
        first_free = {en: all_free[model_ids_ablation[0]][en] for en in global_ens}
        selected_keys = select_keys_from_common(first_blocked, first_free, top_n=top_n, seed=seed)
        print(f"Selected {len(selected_keys)} for scoring (high_ratio / mid_ratio / random, top_n={top_n})")

        scorer = LLMScorer(scorer_key=scorer_key, device=f"cuda:{args.device}", use_4bit=args.use_4bit)
        dims = ["accuracy", "fluency", "completeness", "conciseness", "overall"]
        all_summaries = []

        for model_id in model_ids_ablation:
            blocked_by_en = all_blocked[model_id]
            free_by_en = all_free[model_id]

            def score_condition(samples_by_en, condition_tag):
                to_score = []
                for en_key, reason in selected_keys.items():
                    if en_key in samples_by_en:
                        s = dict(samples_by_en[en_key])
                        s["selection_reason"] = reason
                        s["condition"] = condition_tag
                        to_score.append(s)
                scored = []
                for s in tqdm(to_score, desc=f"  {model_id} {condition_tag}"):
                    r = scorer.score(en=s["en"], ref=s["ref"], gen=s["gen"])
                    scored.append({**s, "llm_scores": r})
                return scored

            scored_blocked = score_condition(blocked_by_en, "blocked")
            scored_free = score_condition(free_by_en, "free")

            avgs_b, reason_b = _aggregate_llm(scored_blocked, dims)
            avgs_f, reason_f = _aggregate_llm(scored_free, dims)
            delta = {}
            if avgs_b and avgs_f and "error" not in avgs_b and "error" not in avgs_f:
                delta = {d: round(avgs_b[d] - avgs_f[d], 2) for d in dims}

            summary = {
                "model_id": model_id,
                "num_paired": len(selected_keys),
                "blocked": {"averages": avgs_b, "by_selection": reason_b},
                "free": {"averages": avgs_f, "by_selection": reason_f},
                "delta_blocked_minus_free": delta,
            }
            all_summaries.append(summary)

            out_base = os.path.join(out_dir, f"{model_id}_llm_ablation_{scorer_key}")
            with open(out_base + ".json", "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "blocked": scored_blocked, "free": scored_free},
                          f, indent=2, ensure_ascii=False)
            lines = [
                "=" * 55,
                f"LLM ABLATION: {model_id} ({scorer_key})",
                "=" * 55,
                f"Paired samples: {len(selected_keys)}",
                "",
                "--- BLOCKED (no_repeat_ngram_size=3) ---",
            ]
            if avgs_b and "error" not in avgs_b:
                for d in dims:
                    lines.append(f"  {d.capitalize():<15} | {avgs_b.get(d, 0):>5.2f}")
            lines.append("\n--- FREE (original greedy) ---")
            if avgs_f and "error" not in avgs_f:
                for d in dims:
                    lines.append(f"  {d.capitalize():<15} | {avgs_f.get(d, 0):>5.2f}")
            if delta:
                lines.append("\n--- DELTA (blocked - free) ---")
                for d in dims:
                    lines.append(f"  {d.capitalize():<15} | {delta.get(d, 0):>+5.2f}")
            with open(out_base + ".txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  -> {out_base}.json & .txt")

        abl_path = os.path.join(out_dir, f"llm_ablation_summary_{scorer_key}.json")
        with open(abl_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"Summary -> {abl_path}")
        return

    # Single-condition fallback: any *_blocked_translations.tsv or *_free_translations.tsv
    all_selected = []
    model_id = None
    total_raw = 0
    single_condition_out_dir = out_dir
    for suffix, tsv_dir in [(BLOCKED_TSV_SUFFIX, blocked_tsv_dir), (FREE_TSV_SUFFIX, free_tsv_dir)]:
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(suffix)]
        if not tsv_files:
            continue
        tsv_path = os.path.join(tsv_dir, tsv_files[0])
        mode = "blocked" if suffix == BLOCKED_TSV_SUFFIX else "free"
        if model_id is None:
            model_id = tsv_files[0].replace(suffix, "")
            single_condition_out_dir = blocked_out_dir if mode == "blocked" else free_out_dir

        samples = load_tsv(tsv_path)
        total_raw += len(samples)
        selected = select_samples(samples, top_n=top_n)
        for s in selected:
            s["decode_mode"] = mode
        all_selected.extend(selected)

    if not all_selected:
        print(f"No TSV samples found. Run --mode chrf first or provide *{BLOCKED_TSV_SUFFIX} / *{FREE_TSV_SUFFIX}.")
        return

    print(f"Total samples: {total_raw}, selected for scoring: {len(all_selected)}")
    scorer = LLMScorer(scorer_key=scorer_key, device=f"cuda:{args.device}", use_4bit=args.use_4bit)
    scored = []
    for s in tqdm(all_selected, desc=f"LLM scoring ({model_id})"):
        result = scorer.score(en=s["en"], ref=s["ref"], gen=s["gen"])
        scored.append({**s, "llm_scores": result})

    valid = [r for r in scored if "error" not in r["llm_scores"]]
    dims = ["accuracy", "fluency", "completeness", "conciseness", "overall"]
    avgs = {}
    if valid:
        for d in dims:
            vals = [r["llm_scores"].get(d, 0) for r in valid]
            avgs[d] = round(sum(vals) / len(vals), 2)

    summary = {
        "model_id": model_id,
        "scorer": scorer_key,
        "total_samples": total_raw,
        "scored": len(all_selected),
        "valid": len(valid),
        "averages": avgs if avgs else {"error": "No valid scores"},
    }
    llm_json = os.path.join(single_condition_out_dir, f"{model_id}_llm_scores_{scorer_key}.json")
    with open(llm_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": scored}, f, indent=2, ensure_ascii=False)

    ablation_single = (all_selected[0].get("decode_mode") == "free") if all_selected else False
    existing_json = os.path.join(single_condition_out_dir, f"{model_id}_translation_quality{'_ablation' if ablation_single else ''}.json")
    if os.path.exists(existing_json):
        with open(existing_json) as f:
            results = json.load(f)
        results[f"llm_{scorer_key}_overall"] = avgs.get("overall")
        results[f"llm_{scorer_key}_scorer"] = scorer_key
        save_translation_quality_results(results, single_condition_out_dir, model_id, model_id, ablation=ablation_single)

    lines = ["=" * 60, f"LLM SCORING: {model_id} (scorer: {scorer_key})", "=" * 60,
             f"Scored: {len(all_selected)}, Valid: {len(valid)}", ""]
    if avgs:
        for d in dims:
            lines.append(f"  {d.capitalize():<15}: {avgs.get(d, 0):.2f}")
    print("\n".join(lines))
    print(f"Detailed -> {llm_json}")


# ================================================================== #
#  CLI                                                                #
# ================================================================== #

def main():
    p = argparse.ArgumentParser(
        description="Translation quality evaluation: chrF, COMET, LLM-as-Judge, Bootstrap CI."
    )
    p.add_argument("--mode", required=True, choices=["chrf", "comet", "llm"],
                    help="Evaluation mode")

    # chrF mode
    p.add_argument("--model_path", default=None, help="Model directory (chrF mode)")
    p.add_argument("--data_path", default=None, help="Tokenized dataset (chrF mode)")
    p.add_argument("--split", default="test")
    p.add_argument("--max_n_chrf", type=int, default=1024, help="Max samples for chrF")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3,
                    help="Block repeated n-grams during decoding (0 to disable, for ablation)")
    p.add_argument("--output_dir", default=None, help="Output directory")

    # COMET / LLM / Bootstrap mode
    p.add_argument("--input_dir", default=None, help="Directory with TSVs (comet/llm/bootstrap)")
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch_size", type=int, default=16)

    # LLM mode
    p.add_argument("--scorer", default="phi", choices=list(SCORER_MODELS.keys()))
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--top_n", type=int, default=50, help="Samples per selection group")


    p.add_argument("--seed", type=int, default=42, help="Random seed; chrF CI uses seed+303, COMET CI uses seed+404.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--en_id", type=int, default=4)
    p.add_argument("--fr_id", type=int, default=5)

    args = p.parse_args()

    if args.mode == "chrf":
        if not args.model_path or not args.data_path:
            p.error("--mode chrf requires --model_path and --data_path")
        run_chrf(args)
    elif args.mode == "comet":
        if not args.input_dir:
            p.error("--mode comet requires --input_dir")
        run_comet(args)
    elif args.mode == "llm":
        if not args.input_dir:
            p.error("--mode llm requires --input_dir")
        run_llm(args)


if __name__ == "__main__":
    main()
