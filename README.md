# PLM_Thoth

Bilingual (English-French) GPT-2 pretraining project.

## Project Structure

```
PLM_Thoth/
├── model.py              # Core GPT-2 model definition
├── requirements.txt      # Python dependencies
│
├── scripts/              # Main pipeline (run in order)
│   ├── 0_bootstrap_tokenizer.py
│   ├── 1_download_dataset.py
│   ├── 2_preprocess_and_split.py
│   ├── 2+_mono_and_bucket.py
│   ├── 3_train_tokenizer.py
│   ├── 4_pretokenize.py
│   ├── 5_train_model_advanced.py
│   ├── 6_validation_only2.py
│   ├── 7_generation_eval.py
│   └── 8_llm_scoring.py
│
├── configs/              # Experiment configurations
│   ├── active/           # Current experiment configs
│   └── archive/          # Archived configs
│
├── utils/                # Utility scripts
│   ├── run_experiments.py
│   └── create_subset_dataset.py
│
├── results/              # Experiment outputs
│   ├── validation/       # Validation results
│   └── generation/       # Generated text samples
│
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
└── legacy/               # Deprecated scripts
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python utils/run_experiments.py \
  --jsonl configs/active/experiments_r0v0_full.jsonl \
  --device 0

# Run validation
python scripts/6_validation_only2.py \
  --config configs/active/experiments_r0v0_full.jsonl \
  --device 0

# Create subset dataset (e.g., 25%)
python utils/create_subset_dataset.py \
  --data_path /path/to/tokenized_bucketed_mono \
  --tok_path /path/to/tokenizer \
  --out_path /path/to/subset \
  --frac 0.25
```

## Model Architecture

- **Type**: GPT-2 (Pre-LayerNorm)
- **Parameters**: ~124M
- **Vocab Size**: 50,000
- **Max Sequence Length**: 128

## Evaluation

### Validation Metrics
- **PPL FR|EN**: Perplexity of French given English (EN→FR)
- **PPL EN|FR**: Perplexity of English given French (FR→EN)
- **Accuracy/F1**: Discrimination task performance

### Generation Evaluation
```bash
# Generate text with beam search
python scripts/7_generation_eval.py \
  --model_path /path/to/model.pt \
  --tokenizer_path /path/to/tokenizer \
  --output_path ./generations.json

# Score with LLM (mistral/croissant/qwen/llama/deepseek/phi)
python scripts/8_llm_scoring.py \
  --input_path ./generations.json \
  --scorer mistral \
  --use_4bit
```
