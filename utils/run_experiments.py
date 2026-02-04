# run_experiments.py

import json
import argparse
import subprocess
import copy
import os


# --------------------------------------------------
# Utilities
# --------------------------------------------------

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


def deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            deep_update(base[k], v)
        else:
            base[k] = v


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def require(cfg, path):
    cur = cfg
    for p in path.split("."):
        if p not in cur:
            raise RuntimeError(f"Missing required config field: {path}")
        cur = cur[p]


def validate_config(cfg):
    required = [
        # run
        "run.name",

        # paths
        "paths.data",
        "paths.tokenizer",
        "paths.output",
        "paths.wandb",

        # model
        "model.arch_name",
        "model.d_model",
        "model.n_head",
        "model.n_layer",
        "model.max_seq_len",
        "model.d_ff",
        "model.dropout.embed",
        "model.dropout.attn",
        "model.dropout.resid",
        "model.dropout.ff",

        # tokenizer
        "tokenizer.vocab_size",
        "tokenizer.special_tokens",

        # training
        "training.batch_size",
        "training.epochs",
        "training.learning_rate",
        "training.adam_betas",
        "training.weight_decay",
        "training.warmup_ratio",
        "training.max_grad_norm",
        "training.mixed_precision",

        # logging
        "logging.wandb_entity",
        "logging.wandb_project",
        "logging.checkpoint_steps",
        "logging.patience",
    ]

    for r in required:
        require(cfg, r)


def run_cmd(cmd):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="Experiments JSONL")
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--train_script", default="5_train_model.py")
    p.add_argument("--eval_script", default="6_validation.py")
    p.add_argument("--eval_checkpoint", default="best",
                   choices=["best", "last", "final"])
    args = p.parse_args()

    runs = load_jsonl(args.jsonl)
    if len(runs) < 1:
        raise RuntimeError("JSONL must contain at least one run")

    baseline = runs[0]

    for run_override in runs:
        cfg = copy.deepcopy(baseline)
        deep_update(cfg, run_override)
        validate_config(cfg)
        
        # Resolve ${USER} and nested path references
        cfg = resolve_paths(cfg)

        run_name = cfg["run"]["name"]
        out_root = cfg["paths"]["output"]
        out_dir = os.path.join(out_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        cfg_path = os.path.join(out_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n=== TRAINING: {run_name} (GPU {args.device}) ===")

        run_cmd([
            "python",
            args.train_script,
            "--config", cfg_path,
            "--device", str(args.device),
        ])


        print(f"\n=== EVALUATION: {run_name} ({args.eval_checkpoint}) ===")

        run_cmd([
            "python",
            args.eval_script,
            "--model_path", out_dir,
            "--data_path", cfg["paths"]["data"],
            "--device", str(args.device),
        ])



if __name__ == "__main__":
    main()
