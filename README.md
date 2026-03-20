# PLM_Thoth
Bilingual (English–French) GPT-2 pretraining project for the UdS Pretraining LLMs seminar.

## Model Variants

Six GPT-2 variants were trained on the full UN Parallel Corpus (EN–FR), each exploring a different hyperparameter or regularization strategy:

| ID | Name | Description |
|----|------|-------------|
| `r0v0` | **GPT-Base**   | Baseline — fixed LR 1e-5, no scheduling, no extra regularization |
| `r0v1` | **GPT-LR**     | Higher learning rate (5e-4) with linear warmup, no cosine decay |
| `r0v2` | **GPT-Cosine 1**  | Cosine-annealing LR schedule (η_min = 1e-6) |
| `r0v3` | **GPT-Cosine 2**  | Cosine-annealing with higher peak LR (5e-4, η_min = 1e-6) |
| `r0v4` | **GPT-Reg 1**  | Label smoothing (0.1) + increased dropout (0.2) |
| `r0v5` | **GPT-Reg 2**  | GPT-Reg 1 + cosine-annealing schedule |

## Model Architecture

- **Type**: GPT-2 (Pre-LayerNorm)
- **Parameters**: ~124M
- **Vocab Size**: 50,000 (BPE, jointly trained on EN + FR)
- **Max Sequence Length**: 128

## Project Structure

```
PLM_Thoth/
├── model.py                          # Core GPT-2 model definition
├── requirements.txt                  # Python dependencies
├── README.md
│
├── configs/                          # Experiment JSONL configurations
│   ├── experiments_r0v0_full.jsonl   # Full-run configs (r0v0–r0v5)
│   ├── experiments_r0v1_full.jsonl
│   ├── experiments_r1.jsonl          # Round 1 (subset experiments)
│   ├── experiments_round2.jsonl      # Round 2
│   ├── experiments_round3_*.jsonl    # Round 3
│   └── experiments_r4_*.jsonl        # Round 4
│
├── scripts/                          # Main pipeline (run in order)
│   ├── 0_bootstrap_tokenizer.py      # Download pretrained tokenizer
│   ├── 1_download_dataset.py         # Download UN Parallel Corpus
│   ├── 2_preprocess_and_split.py     # Clean and split dataset
│   ├── 3_mono_and_bucket.py          # Monolingual format + bucketing
│   ├── 4_pretokenize.py              # Pre-tokenize dataset
│   ├── 5_train_tokenizer.py          # Train BPE tokenizer
│   ├── 6_train_model.py              # Train GPT-2 model
│   ├── 7_validation.py               # Primary validation (PPL, MRR, AUC)
│   └── supplementary_validation/     # Extended evaluation suite
│       ├── README.md                 # Usage guide
│       ├── translation_quality.py    # chrF + COMET scoring
│       └── llm_scoring_colab.ipynb   # LLM-as-a-judge (Colab notebook)
│
├── utils/
│   ├── run_experiments.py            # Batch experiment launcher
│   ├── create_subset_dataset.py      # Create dataset subsets
│   ├── merge_results.ipynb           # Merge evaluation results
│   └── run_validations.sh            # Validation execution script
│
├── data/                             # Datasets and statistics
│   ├── data_stats/                   # Dataset dimensions and statistics
│   │   ├── full/                     # Full dataset stats
│   │   ├── full_bucketed_mono/       # Full monolingual buckets stats
│   │   └── medium_bucketed_mono/     # Subset monolingual buckets stats
│   └── test/                         # Tokenized test set split
│
├── results/
│   ├── logs/                         # Training histories and metrics
│   │   └── full_runs/                # Full dataset runs (r0v0–r0v5)
│   └── validation/                   # Evaluation results
│       ├── primary/                  # Core PPL + ACC + MRR metrics
│       └── supplementary/            # chrF, COMET, LLM results
│           ├── llm-as-judge/         # LLM eval scoring data
│           ├── retrieval_disc/       # Retrieval metrics data
│           ├── translation_quality/  # chrF and COMET scoring data
│           └── tsv/                  # Generated translation text files
│
└── docs/                             # Additional documentation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data preparation

```bash
# Download dataset
python scripts/1_download_dataset.py

# Preprocess and split
python scripts/2_preprocess_and_split.py

# Create monolingual pairs and bucket by length
python scripts/3_mono_and_bucket.py

# Pre-tokenize
python scripts/4_pretokenize.py

# Train BPE tokenizer
python scripts/5_train_tokenizer.py --vocab_size 50000
```

### 3. Training

```bash
# Train a single model using a config file
python utils/run_experiments.py \
  --jsonl configs/experiments_r0v0_full.jsonl \
  --device 0

# Or run training directly
python scripts/6_train_model.py \
  --data_path /path/to/tokenized_bucketed_mono \
  --tok_path /path/to/tokenizer \
  --output_dir /path/to/output \
  --device 0
```

### Pretrained Checkpoints

Pretrained model weights are available on Google Drive for reproducibility:

📦 [Download Checkpoints (PLM_Checkpoint)](https://drive.google.com/drive/u/1/folders/1TDlKE9MBgC7GUjfrFGSq67bvfbXoMRyS)

```
PLM_Checkpoint/
├── r0v0_gpt_base/model.pt
├── r0v1_gpt_lr/model.pt
├── r0v2_gpt_cosine1/model.pt
├── r0v3_gpt_cosine2/model.pt
├── r0v4_gpt_reg1/model.pt
├── r0v5_gpt_reg2/model.pt
└── tokenizer/
```

### 4. Validation

```bash
# Primary validation (PPL, MRR, AUC)
python scripts/7_validation.py \
  --model_path /path/to/model.pt \
  --data_path data \
  --split test \
  --output_dir results/validation/primary/ \
  --device 0

# Extended: chrF + COMET translation quality
python scripts/supplementary_validation/translation_quality.py \
  --model_path /path/to/model.pt \
  --data_path data \
  --split test \
  --output_dir results/validation/supplementary/ \
  --device 0
```

## Evaluation Metrics

### Primary
| Metric | Description |
|--------|-------------|
| **PPL FR\|EN** | Perplexity of French given English context |
| **PPL EN\|FR** | Perplexity of English given French context |
| **MRR** | Mean Reciprocal Rank on hard-negative retrieval |
| **AUC** | Area Under ROC Curve for discrimination |

### Supplementary
| Metric | Description |
|--------|-------------|
| **chrF** | Character-level F-score for translation quality |
| **COMET** | Neural MT evaluation (wmt22-comet-da) |
| **LLM-as-Judge** | External LLM scores accuracy, fluency, completeness, conciseness (1–5) |
