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
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from model import GPT2


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# --------------------------------------------------
# Logging
# --------------------------------------------------

import logging
import sys
from datetime import datetime

def setup_logging(log_dir: str, run_name: str):
    """
    Setup logging to both console and file.
    
    Creates a log file at: {log_dir}/{run_name}/training.log
    """
    log_file = os.path.join(log_dir, "training.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"=" * 60)
    logging.info(f"Training started: {run_name}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"=" * 60)
    
    return logger


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
    
    # Set random seed for reproducibility
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
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
    
    # ---- Setup logging ----
    setup_logging(out_path, run_name)
    
    logging.info(f"Output path: {out_path}")
    
    # ---- Save config copy ----
    config_copy_path = os.path.join(out_path, "config.json")
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Config saved to: {config_copy_path}")

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
    
    # ---- Training history CSV ----
    history_file = os.path.join(out_path, "training_history.csv")
    with open(history_file, "w") as f:
        f.write("step,epoch,train_loss,val_loss,lr,best_val,no_improve\n")
    logging.info(f"Training history will be saved to: {history_file}")

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

                # Log to wandb
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": sched.get_last_lr()[0],
                })
                
                # Log to file
                logging.info(
                    f"Step {step} | Epoch {epoch} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {sched.get_last_lr()[0]:.2e} | "
                    f"No Improve: {no_improve}/{log_cfg['patience']}"
                )
                
                # Save to CSV
                with open(history_file, "a") as f:
                    f.write(f"{step},{epoch},{loss.item():.6f},{val_loss:.6f},"
                            f"{sched.get_last_lr()[0]:.8f},{best_val:.6f},{no_improve}\n")

                model.save(os.path.join(out_path, "last_checkpoint.pt"))

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    model.save(os.path.join(out_path, "best_checkpoint", "model.pt"))
                    logging.info(f"New best model saved! Val Loss: {val_loss:.4f}")
                else:
                    no_improve += 1

                if no_improve >= log_cfg["patience"]:
                    logging.info(f"Early stopping triggered after {step} steps (patience={log_cfg['patience']})")
                    model.save(os.path.join(out_path, "final_checkpoint.pt"))
                    logging.info(f"Final model saved to {out_path}/final_checkpoint.pt")
                    
                    # Save training summary
                    training_summary = {
                        "run_name": run_name,
                        "status": "early_stopped",
                        "total_steps": step,
                        "total_epochs": epoch,
                        "final_train_loss": loss.item(),
                        "final_val_loss": val_loss,
                        "best_val_loss": best_val,
                        "early_stopped_at_step": step,
                        "patience": log_cfg["patience"],
                    }
                    summary_path = os.path.join(out_path, "training_summary.json")
                    with open(summary_path, "w") as f:
                        json.dump(training_summary, f, indent=2)
                    logging.info(f"Training summary saved to: {summary_path}")
                    
                    run.finish()
                    return

    # Save training summary for completed training
    training_summary = {
        "run_name": run_name,
        "status": "completed",
        "total_steps": step,
        "total_epochs": train_cfg["epochs"],
        "final_train_loss": loss.item(),
        "final_val_loss": val_loss,
        "best_val_loss": best_val,
    }
    summary_path = os.path.join(out_path, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    logging.info(f"Training summary saved to: {summary_path}")
    
    logging.info(f"Training completed after {step} steps")
    model.save(os.path.join(out_path, "final_checkpoint.pt"))
    logging.info(f"Final model saved to {out_path}/final_checkpoint.pt")
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

