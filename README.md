# PLM_Thoth

A bilingual (English-French) GPT-2 pretraining pipeline built from scratch for the "Pretraining Language Models" course.

## Overview

PLM_Thoth is a complete pretraining system that trains a GPT-2 model on the UN Parallel Corpus (UNPC) English-French dataset. The project includes:

- Custom BPE tokenizer training
- Data preprocessing with monolingual augmentation and bidirectional translation
- GPT-2 model implementation (Thoth_v1 architecture)
- Training with W&B logging and early stopping
- Comprehensive validation tasks (PPL, Retrieval, Discrimination)

## Project Structure

```
PLM_Thoth/
├── 0_bootstrap_tokenizer.py   # Train tokenizer from raw data
├── 1_download_dataset.py      # Download UNPC EN-FR dataset
├── 2_preprocess_and_split.py  # Clean, dedupe, and split data
├── 2+_mono_and_bucket.py      # Monolingual augmentation + length bucketing (optional)
├── 3_train_tokenizer.py       # Train tokenizer on processed data
├── 4_pretokenize.py           # Pretokenize dataset for faster training
├── 5_train_model.py           # Main training loop
├── 6_validation.py            # Evaluate trained model
├── model.py                   # GPT-2 model architecture (Thoth_v1)
├── run_experiments.py         # Batch experiment runner
├── experiments_*.jsonl        # Experiment configurations
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ~50GB disk space for full dataset

### Setup

```bash
# Clone the repository
git clone https://github.com/yhhde/PLM_Thoth.git
cd PLM_Thoth

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start (Full Pipeline)

```bash
# Step 0: Bootstrap tokenizer from raw data
python 0_bootstrap_tokenizer.py \
    --raw_path /path/to/unpc_raw \
    --tok_out /path/to/tokenizer \
    --vocab_size 50000

# Step 1: Download dataset
python 1_download_dataset.py \
    --data_path /path/to/datasets/unpc_raw \
    --size 0  # Use 0 for full dataset, or specify subset size

# Step 2: Preprocess and split
python 2_preprocess_and_split.py \
    --in_path /path/to/unpc_raw \
    --out_path /path/to/processed \
    --tok_path /path/to/tokenizer \
    --max_len 128

# Step 2+ (Optional): Add monolingual data and bucketing
python 2+_mono_and_bucket.py \
    --in_path /path/to/processed \
    --out_path /path/to/processed_augmented \
    --tok_path /path/to/tokenizer \
    --p_mono 0.3

# Step 3: Train tokenizer on processed data
python 3_train_tokenizer.py \
    --data_path /path/to/processed \
    --tok_out /path/to/tokenizer \
    --vocab_size 50000

# Step 4: Pretokenize dataset
python 4_pretokenize.py \
    --in_path /path/to/processed \
    --out_path /path/to/tokenized \
    --tok_path /path/to/tokenizer \
    --max_len 128

# Step 5: Train model
python 5_train_model.py \
    --config config.json \
    --device 0

# Step 6: Evaluate model
python 6_validation.py \
    --model_path /path/to/model_output \
    --data_path /path/to/tokenized \
    --split test
```

### Running Experiments

Use `run_experiments.py` to batch multiple training runs:

```bash
python run_experiments.py \
    --jsonl experiments_v0_v19.jsonl \
    --device 0
```

## Configuration

Experiments are configured via JSONL files. Each line is a JSON object with:

```json
{
  "run": {"name": "v0_baseline"},
  "paths": {
    "data": "/path/to/tokenized",
    "tokenizer": "/path/to/tokenizer",
    "output": "/path/to/output",
    "wandb": "/path/to/wandb"
  },
  "model": {
    "arch_name": "Thoth_v1",
    "d_model": 768,
    "n_head": 12,
    "n_layer": 12,
    "max_seq_len": 128,
    "d_ff": 3072,
    "dropout": {"embed": 0.1, "attn": 0.1, "resid": 0.1, "ff": 0.1}
  },
  "training": {
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.1
  }
}
```

## Model Architecture

**Thoth_v1** is a GPT-2 style decoder-only transformer with:

- Pre-LayerNorm architecture
- Multi-Head Attention with causal masking
- GELU activation in feed-forward layers
- Configurable dropout rates

Default configuration:
- 12 layers, 12 attention heads
- 768 hidden dimension, 3072 FFN dimension
- 50K vocabulary size
- 128 max sequence length

## Validation Tasks

The validation script (`6_validation.py`) evaluates three tasks:

| Task | Metric | Description |
|------|--------|-------------|
| Conditional PPL | PPL FR\|EN, PPL EN\|FR | Perplexity of target language given source |
| Bitext Retrieval | Pass@1, Pass@5, MRR | Cross-lingual embedding retrieval |
| Alignment Discrimination | AUC | Distinguish real vs. fake translation pairs |

## Data Processing

### Input Format
Raw UNPC data: `{"translation": {"en": "...", "fr": "..."}}`

### Processed Format
Text format: `<en> English text <fr> French text`

### Augmentation (Optional)
- 30% monolingual: `<en> English` and `<fr> French` (separate)
- 50% EN→FR: `<en> ... <fr> ...`
- 50% FR→EN: `<fr> ... <en> ...`

## Results

See `experiments_v0_v19.jsonl` for hyperparameter search configurations covering:
- Dropout rates (0.05, 0.1, 0.2)
- Weight decay (0.005, 0.01, 0.1)
- Learning rate warmup (0%, 10%)
- Data augmentation (with/without mono+bucket)

## Citation

```bibtex
@misc{plm_thoth,
  title={PLM_Thoth: Bilingual GPT-2 Pretraining Pipeline},
  author={Your Team},
  year={2025},
  url={https://github.com/yhhde/PLM_Thoth}
}
```

## License

MIT License

## Acknowledgments

- UN Parallel Corpus (UNPC) dataset from Helsinki-NLP
- Course: Pretraining Language Models, University of the Saarland
