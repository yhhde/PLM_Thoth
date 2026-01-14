#5_train_model.py

"""
Training script for custom GPT-2 model.

Works in two modes:
1) Standalone: python training_loop.py --config config.json --device 0
2) Orchestrated: Called by run_experiments.py with generated config.json

ALL paths, hyperparameters, and metadata MUST come from config.json.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from model import GPT2


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_paths(config):
    """
    Resolve ${USER} and ${paths.xxx} references inside config["paths"].
    """
    paths = config["paths"]
    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

    def resolve(val):
        if not isinstance(val, str):
            return val
        val = val.replace("${USER}", user)
        for _ in range(10):
            changed = False
            for k, v in paths.items():
                if isinstance(v, str) and f"${{{k}}}" in val:
                    val = val.replace(f"${{{k}}}", v)
                    changed = True
            if not changed:
                break
        return val

    def walk(d):
        return {k: walk(v) if isinstance(v, dict) else resolve(v) for k, v in d.items()}

    config["paths"] = walk(paths)
    return config


def collate_pretokenized(batch, pad_id):
    ids = [torch.tensor(b["input_ids"]) for b in batch]
    att = [torch.tensor(b["attention_mask"]) for b in batch]
    return {
        "input_ids": pad_sequence(ids, batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence(att, batch_first=True, padding_value=0),
    }


def eval_one_epoch(model, loader, device, pad_id):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            labels = ids.clone()
            labels[labels == pad_id] = -100

            _, loss = model(ids, attention_mask=att, labels=labels)
            n = (labels != -100).sum().item()

            total_loss += loss.item() * n
            total_tokens += n

    return total_loss / max(1, total_tokens)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(config, device_idx, resume_path=None):
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device(f"cuda:{device_idx}")
    
    # Print GPU information
    print(f"\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(device_idx)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device_idx).total_memory / 1e9:.2f} GB")
    print("=" * 30 + "\n")

    # ---- Config sections ----
    run_cfg   = config["run"]
    paths     = config["paths"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    log_cfg   = config["logging"]
    
    required = ["d_model", "n_head", "n_layer", "max_seq_len", "d_ff"]
    for k in required:
        if k not in model_cfg:
            raise ValueError(f"Missing model parameter: {k}")

    run_name = run_cfg["name"]

    # ---- Paths (flat, explicit, boring) ----
    data_path = paths["data"]
    tok_path  = paths["tokenizer"]
    out_root  = paths["output"]
    wandb_dir = paths["wandb"]

    out_path = os.path.join(out_root, run_name)

    ensure_dir(out_path)
    ensure_dir(os.path.join(out_path, "checkpoints"))
    ensure_dir(os.path.join(out_path, "best_checkpoint"))
    ensure_dir(wandb_dir)

    # ---- Tokenizer ----
    tok = PreTrainedTokenizerFast.from_pretrained(tok_path)

    pad_id = config["tokenizer"]["pad_token_id"]
    pad_token = tok.convert_ids_to_tokens(pad_id)

    tok.pad_token = pad_token
    tok.pad_token_id = pad_id

    tok.save_pretrained(out_path)

    # ---- Dataset ----
    ds = load_from_disk(data_path)
    train_ds = ds["train"]
    val_ds   = ds["validation"]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_pretokenized(b, pad_id),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_pretokenized(b, pad_id),
    )

    # ---- Model ----
    # vocab size belongs to tokenizer, not architecture
    model_cfg["vocab_size"] = len(tok)

    # Architecture consistency check (correct, no longer clumsy)
    cfg_arch   = model_cfg["arch_name"]
    model_arch = GPT2.architecture_name

    if cfg_arch != model_arch:
        raise RuntimeError(
            "Architecture mismatch!\n"
            f"Config arch_name: {cfg_arch}\n"
            f"Model arch_name:  {model_arch}\n"
            "Refusing to train."
        )

    if resume_path:
        model = GPT2.load(resume_path, config)
    else:
        model = GPT2(config)

    model.to(device)
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Architecture: {model_arch}")

    # ---- Optimizer ----
    opt = model.configure_optimizers(
        weight_decay=train_cfg["weight_decay"],
        learning_rate=train_cfg["learning_rate"],
        betas=tuple(train_cfg["adam_betas"]),
        device_type="cuda",
    )

    steps_per_epoch = len(train_loader)
    max_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = int(train_cfg["warmup_ratio"] * max_steps)

    sched = get_linear_schedule_with_warmup(opt, warmup_steps, max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"])

    # ---- WandB ----
    import wandb
    os.environ["WANDB_DIR"] = wandb_dir

    run = wandb.init(
        entity=log_cfg["wandb_entity"],
        project=log_cfg["wandb_project"],
        name=run_name,
        config=config,
        dir=wandb_dir,
    )

    # ---- Loop ----
    best_val = float("inf")
    no_improve = 0
    step = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        prog = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in prog:
            opt.zero_grad()

            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            labels = ids.clone()
            labels[labels == pad_id] = -100

            with torch.cuda.amp.autocast(enabled=train_cfg["mixed_precision"]):
                _, loss = model(ids, attention_mask=att, labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["max_grad_norm"]
            )

            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            prog.set_postfix(loss=loss.item())

            if step % log_cfg["checkpoint_steps"] == 0:
                val_loss = eval_one_epoch(model, val_loader, device, pad_id)

                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": sched.get_last_lr()[0],
                })

                model.save(os.path.join(out_path, "last_checkpoint.pt"))

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    model.save(os.path.join(out_path, "best_checkpoint", "model.pt"))
                else:
                    no_improve += 1

                if no_improve >= log_cfg["patience"]:
                    print("Early stopping.")
                    model.save(os.path.join(out_path, "final_checkpoint.pt"))
                    run.finish()
                    return

    model.save(os.path.join(out_path, "final_checkpoint.pt"))
    run.finish()


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cfg = resolve_paths(cfg)

    # Mandatory sanity checks
    assert "run" in cfg and "name" in cfg["run"]
    assert "model" in cfg and "arch_name" in cfg["model"]
    assert "paths" in cfg

    train(cfg, args.device, args.resume)

