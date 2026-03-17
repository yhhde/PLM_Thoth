"""
Training script for custom GPT-2 model with Cosine LR Schedule.
Uses cosine annealing scheduler  as recommended by pretrainLLM
project for better convergence.

Key differences from prev version of 5_train_model.py:
- Uses get_cosine_schedule_with_warmup instead of linear
- Adds lr_scheduler_type config option ("cosine" or "linear")
- Default warmup ratio is 0.03 (3%) as per pretrainLLM
- Progress-based checkpoint saving at 10%, 20%, ..., 100% intervals

Works in two modes:
1) Standalone: python 5_train_model.py --config config.json --device 0
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
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
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


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, 
                    best_val, no_improve, total_loss, total_accuracy, batch_count,
                    save_intervals_idx):
    """
    Save complete training state for resumption.
    
    Saves:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Scaler state (for AMP)
    - Training progress (epoch, step, metrics)
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "no_improve": no_improve,
        "total_loss": total_loss,
        "total_accuracy": total_accuracy,
        "batch_count": batch_count,
        "save_intervals_idx": save_intervals_idx,
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path} (epoch={epoch}, step={step})")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """
    Load complete training state for resumption.
    
    Returns:
        dict with training state variables (epoch, step, best_val, etc.)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    scaler.load_state_dict(checkpoint["scaler"])
    
    state = {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "best_val": checkpoint["best_val"],
        "no_improve": checkpoint["no_improve"],
        "total_loss": checkpoint["total_loss"],
        "total_accuracy": checkpoint["total_accuracy"],
        "batch_count": checkpoint["batch_count"],
        "save_intervals_idx": checkpoint["save_intervals_idx"],
    }
    
    logging.info(f"Checkpoint loaded: {path}")
    logging.info(f"  Resuming from epoch={state['epoch']}, step={state['step']}")
    logging.info(f"  Best val loss so far: {state['best_val']:.4f}")
    
    return state


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


def mask_first_segment(labels, input_ids, en_id, fr_id):
    """
    Mask the first segment (before second language tag) in labels.
    For bilingual data: only compute loss on the second language segment.
    For monolingual data: compute loss on all tokens (no masking).
    
    Args:
        labels: [B, T] tensor of label token ids
        input_ids: [B, T] tensor of input token ids (shifted, so use this to find tags)
        en_id: token id for <en>
        fr_id: token id for <fr>
    
    Returns:
        Modified labels tensor with first segment masked as -100
    """
    B, T = labels.size()
    
    for b in range(B):
        ids = input_ids[b]
        
        # Find positions of language tags in input_ids
        # Note: input_ids is already shifted (ids[:, :-1]), and labels is (ids[:, 1:])
        # So we need to look at input_ids to find where the second language tag is
        lang_mask = (ids == en_id) | (ids == fr_id)
        lang_positions = lang_mask.nonzero(as_tuple=False).view(-1)
        
        if len(lang_positions) >= 2:
            # Bilingual: mask everything before the second language tag
            # The label at position i corresponds to predicting input at position i+1
            # So we mask labels[0:second_tag_pos] to ignore first segment
            second_tag_pos = lang_positions[1].item()
            labels[b, :second_tag_pos] = -100
        # else: monolingual, keep all labels (no masking)
    
    return labels


@torch.no_grad()
def compute_accuracy(logits, labels, ignore_index=-100):
    """
    Compute token-level accuracy, ignoring padding and masked tokens.
    
    Args:
        logits: [B, T, V] model output logits
        labels: [B, T] target labels (-100 for ignored positions)
        ignore_index: value to ignore in accuracy calculation
    
    Returns:
        accuracy: float between 0 and 1
    """
    predictions = torch.argmax(logits, dim=-1)  # [B, T]
    
    # Create mask for valid positions (not ignored)
    mask = labels != ignore_index
    
    # Count correct predictions only where mask is True
    correct = (predictions == labels) & mask
    
    # Calculate accuracy
    total_valid = mask.sum().item()
    if total_valid == 0:
        return 0.0
    
    accuracy = correct.sum().item() / total_valid
    return accuracy


def eval_one_epoch(model, loader, device, pad_id, mask_first=False, en_id=4, fr_id=5):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            # Shift for next-token prediction
            input_ids = ids[:, :-1]
            attention_mask = att[:, :-1]

            labels = ids[:, 1:].clone()
            labels[labels == pad_id] = -100
            
            # Optionally mask first segment loss
            if mask_first:
                labels = mask_first_segment(labels, input_ids, en_id, fr_id)

            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
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
    ensure_dir(os.path.join(out_path, "checkpoints", "progress"))  # For progress checkpoints
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
    
    # ---- Mask first segment config ----
    en_id = config["tokenizer"].get("en_token_id", 4)
    fr_id = config["tokenizer"].get("fr_token_id", 5)
    mask_first_segment_loss = train_cfg.get("mask_first_segment_loss", False)
    
    if mask_first_segment_loss:
        logging.info(f"Mask first segment loss: ENABLED (en_id={en_id}, fr_id={fr_id})")
    else:
        logging.info(f"Mask first segment loss: DISABLED")

    # ---- Dataset ----
    ds = load_from_disk(data_path)
    train_ds = ds["train"]
    val_ds   = ds["validation"]

    # Create generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        generator=g,  # Ensures reproducible shuffle order
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

    # Always create model fresh; if resuming, load_checkpoint will load weights later
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

    # ---- LR Scheduler (Cosine or Linear) ----
    lr_scheduler_type = train_cfg.get("lr_scheduler_type", "cosine")  # Default to cosine
    use_sched = train_cfg.get("use_lr_scheduling", True)
    
    if use_sched:
        if lr_scheduler_type == "cosine":
            sched = get_cosine_schedule_with_warmup(opt, warmup_steps, max_steps)
            logging.info(f"LR Scheduler: COSINE with {warmup_steps} warmup steps ({train_cfg['warmup_ratio']*100:.1f}%)")
        else:
            sched = get_linear_schedule_with_warmup(opt, warmup_steps, max_steps)
            logging.info(f"LR Scheduler: LINEAR with {warmup_steps} warmup steps ({train_cfg['warmup_ratio']*100:.1f}%)")
    else:
        sched = None
        logging.info("LR Scheduler: DISABLED")

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"])
    if train_cfg["mixed_precision"]:
        logging.info("Mixed Precision: ENABLED (FP16)")
    else:
        logging.info("Mixed Precision: DISABLED")

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
        f.write("step,epoch,batch_loss,avg_loss,batch_acc,avg_acc,val_loss,lr,best_val,no_improve\n")
    logging.info(f"Training history will be saved to: {history_file}")
    
    # ---- High-frequency logging (pretrainLLM style) ----
    # Detailed metrics CSV for plotting
    metrics_file = os.path.join(out_path, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("step,epoch,batch_loss,avg_loss,batch_acc,avg_acc,lr\n")
    logging.info(f"Training metrics will be saved to: {metrics_file}")
    
    # Log interval (configurable, default every 10 batches like pretrainLLM)
    log_interval = log_cfg.get("log_interval", 10)
    logging.info(f"Logging interval: every {log_interval} batches")
    
    # Accuracy computation toggle (default True for backward compatibility)
    compute_acc = log_cfg.get("compute_accuracy", True)
    if compute_acc:
        logging.info("Accuracy computation: ENABLED (slower training)")
    else:
        logging.info("Accuracy computation: DISABLED (faster training)")

    # ---- Progress-based checkpoint saving (pretrainLLM style) ----
    total_batches = steps_per_epoch * train_cfg["epochs"]
    # Save at 10%, 20%, ..., 100% progress
    save_intervals = [int(total_batches * (i / 10)) for i in range(1, 11)]
    save_intervals_idx = 0
    logging.info(f"Progress checkpoints will be saved at: {[f'{(i+1)*10}%' for i in range(10)]}")
    logging.info(f"Total batches: {total_batches}, Save intervals: {save_intervals}")

    # ---- Loop ----
    best_val = float("inf")
    no_improve = 0
    step = 0
    start_epoch = 0
    start_batch = 0
    
    # Cumulative metrics (pretrainLLM style)
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    
    # ---- Resume from checkpoint ----
    if resume_path:
        logging.info(f"Attempting to resume from: {resume_path}")
        try:
            state = load_checkpoint(resume_path, model, opt, sched, scaler, device)
            start_epoch = state["epoch"]
            step = state["step"]
            best_val = state["best_val"]
            no_improve = state["no_improve"]
            total_loss = state["total_loss"]
            total_accuracy = state["total_accuracy"]
            batch_count = state["batch_count"]
            save_intervals_idx = state["save_intervals_idx"]
            
            # Calculate which batch to start from within the epoch
            start_batch = step % steps_per_epoch
            if start_batch == 0 and step > 0:
                # We finished an epoch, start next epoch from batch 0
                start_epoch += 1
                start_batch = 0
            
            logging.info(f"Resume successful: starting from epoch={start_epoch}, batch={start_batch}, step={step}")
        except Exception as e:
            logging.warning(f"Could not load checkpoint state, trying model-only load: {e}")
            # Fallback to model-only load (backward compatibility)
            model = GPT2.load(resume_path, config)
            model.to(device)
            logging.info("Loaded model weights only (no optimizer/scheduler state)")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        prog = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(prog):
            # Skip batches we've already processed (for resume)
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            
            opt.zero_grad()

            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            # Shift for next-token prediction
            input_ids = ids[:, :-1]
            attention_mask = att[:, :-1]

            labels = ids[:, 1:].clone()
            labels[labels == pad_id] = -100
            
            # Optionally mask first segment loss
            if mask_first_segment_loss:
                labels = mask_first_segment(labels, input_ids, en_id, fr_id)

            with torch.cuda.amp.autocast(enabled=train_cfg["mixed_precision"]):
                _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["max_grad_norm"]
            )

            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            step += 1
            batch_count += 1
            
            # Compute accuracy for this batch (if enabled)
            if compute_acc:
                with torch.no_grad():
                    logits, _ = model(input_ids, attention_mask=attention_mask, labels=None)
                    batch_acc = compute_accuracy(logits, labels)
            else:
                batch_acc = 0.0  # Placeholder when disabled
            
            # Update cumulative metrics
            total_loss += loss.item()
            total_accuracy += batch_acc
            avg_loss = total_loss / batch_count
            avg_acc = total_accuracy / batch_count if compute_acc else 0.0
            
            prog.set_postfix(loss=loss.item(), acc=batch_acc, avg_loss=avg_loss)

            # ---- Progress-based checkpoint saving (pretrainLLM style) ----
            if save_intervals_idx < len(save_intervals) and step == save_intervals[save_intervals_idx]:
                progress_pct = (save_intervals_idx + 1) * 10
                progress_ckpt_name = f"checkpoint_{progress_pct:02d}_percent.pt"
                progress_ckpt_path = os.path.join(out_path, "checkpoints", "progress", progress_ckpt_name)
                
                # Save complete training state (not just model weights)
                save_checkpoint(
                    progress_ckpt_path, model, opt, sched, scaler, 
                    epoch, step, best_val, no_improve, 
                    total_loss, total_accuracy, batch_count, save_intervals_idx + 1
                )
                
                # Also save model-only version for inference
                model_only_path = os.path.join(out_path, "checkpoints", "progress", f"model_{progress_pct:02d}_percent.pt")
                model.save(model_only_path)
                
                # Run validation at progress checkpoint
                progress_val_loss = eval_one_epoch(model, val_loader, device, pad_id, 
                                                    mask_first=mask_first_segment_loss, 
                                                    en_id=en_id, fr_id=fr_id)
                model.train()  # Switch back to training mode
                
                logging.info(
                    f"Progress checkpoint saved: {progress_pct}% | "
                    f"Step {step}/{total_batches} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg Acc: {avg_acc:.4f} | "
                    f"Val Loss: {progress_val_loss:.4f}"
                )
                
                # Log to wandb
                current_lr = sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]["lr"]
                wandb.log({
                    "progress_checkpoint": progress_pct,
                    "progress_train_loss": loss.item(),
                    "progress_avg_loss": avg_loss,
                    "progress_avg_acc": avg_acc,
                    "progress_val_loss": progress_val_loss,
                    "progress_lr": current_lr,
                })
                
                # Save "latest" checkpoint for easy resume
                latest_ckpt_path = os.path.join(out_path, "checkpoints", "latest_checkpoint.pt")
                save_checkpoint(
                    latest_ckpt_path, model, opt, sched, scaler,
                    epoch, step, best_val, no_improve,
                    total_loss, total_accuracy, batch_count, save_intervals_idx + 1
                )
                
                save_intervals_idx += 1
            
            # ---- High-frequency logging (pretrainLLM style) ----
            if step % log_interval == 0:
                current_lr = sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]["lr"]
                
                # Log to wandb (like pretrainLLM)
                wandb.log({
                    "batch_loss": loss.item(),
                    "avg_loss": avg_loss,
                    "batch_accuracy": batch_acc,
                    "avg_accuracy": avg_acc,
                    "lr": current_lr,
                    "step": step,
                })
                
                # Save to metrics CSV
                with open(metrics_file, "a") as f:
                    f.write(f"{step},{epoch},{loss.item():.6f},{avg_loss:.6f},{batch_acc:.6f},{avg_acc:.6f},{current_lr:.8f}\n")

            if step % log_cfg["checkpoint_steps"] == 0:
                val_loss = eval_one_epoch(model, val_loader, device, pad_id, 
                                          mask_first=mask_first_segment_loss, 
                                          en_id=en_id, fr_id=fr_id)

                # Log to wandb
                current_lr = sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]["lr"]
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": current_lr,
                })
                
                # Log to file
                current_lr = sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]["lr"]
                logging.info(
                    f"Step {step} | Epoch {epoch} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"No Improve: {no_improve}/{log_cfg['patience']}"
                )
                
                # Save to CSV (updated format with accuracy)
                with open(history_file, "a") as f:
                    f.write(f"{step},{epoch},{loss.item():.6f},{avg_loss:.6f},{batch_acc:.6f},{avg_acc:.6f},{val_loss:.6f},"
                            f"{current_lr:.8f},{best_val:.6f},{no_improve}\n")

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
                        "lr_scheduler_type": lr_scheduler_type,
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
        "lr_scheduler_type": lr_scheduler_type,
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
    
    # Clean up GPU memory for subsequent validation
    del model, opt, scaler
    if sched is not None:
        del sched
    torch.cuda.empty_cache()
    logging.info("GPU memory cleared for validation")


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
