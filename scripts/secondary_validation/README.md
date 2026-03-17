# Supplementary Validation

Extended evaluation metrics for bilingual model quality assessment,
complementing the primary PPL + ACC validation in `scripts/6_validation.py`.

## Scripts

### `retrieval_discrimination.py` — Bitext Retrieval + Discrimination

Evaluates the model's ability to match and distinguish correct translation pairs.

| Metric | Description |
|--------|-------------|
| **Pass@1 / Pass@5** | Fraction of queries where the correct translation is ranked 1st / top-5 |
| **MRR** | Mean Reciprocal Rank across retrieval pools |
| **AUC (L1–L4)** | ROC-AUC for distinguishing real pairs from four difficulty levels of negatives |

```bash
python supplementary_validation/retrieval_discrimination.py \
    --model_path /path/to/model \
    --data_path /path/to/tokenized_dataset \
    --device 0
```

### `translation_quality.py` — chrF + COMET + LLM-as-Judge

Evaluates the quality of model-generated translations via three independent metrics,
designed to run sequentially:

| Mode | Metric | Description |
|------|--------|-------------|
| `chrf` | **chrF** | Character-level F-score (greedy decode EN→FR vs reference) |
| `comet` | **COMET** | Neural MT evaluation using `Unbabel/wmt22-comet-da` |
| `llm` | **LLM-as-Judge** | External LLM scores accuracy, fluency, completeness, conciseness (1–5) |

**Recommended usage (per model id, e.g. `r0v0_full`)**  
Assume:

- `MODEL_DIR=/path/to/models/experiments/r0/r0v0_full`  
- `DATA_PATH=/path/to/tokenized_bucketed_mono`  
- `RESULT_EXTENDED=results/validation/extended`

```bash
# 1) chrF + blocked decoding (no-repeat-3-gram, primary)
python supplementary_validation/translation_quality.py \
    --mode chrf \
    --model_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --split test \
    --max_n_chrf 1024 \
    --no_repeat_ngram_size 3 \
    --output_dir "$RESULT_EXTENDED" \
    --device 0

# 2) chrF + free decoding (ablation; no ngram blocking)
python supplementary_validation/translation_quality.py \
    --mode chrf \
    --model_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --split test \
    --max_n_chrf 1024 \
    --no_repeat_ngram_size 0 \
    --output_dir "$RESULT_EXTENDED" \
    --device 0

# 3) COMET scoring on both blocked + free TSVs
python supplementary_validation/translation_quality.py \
    --mode comet \
    --input_dir "$RESULT_EXTENDED" \
    --comet_batch_size 8 \
    --device 0

# 4) (Optional) LLM-as-judge scoring (paired ablation if both TSVs exist)
python supplementary_validation/translation_quality.py \
    --mode llm \
    --input_dir "$RESULT_EXTENDED" \
    --scorer phi \
    --top_n 10 \
    --device 0
```

With the current `results/validation/extended` layout:

- TSVs are saved under `tsv/blocked/` and `tsv/free/`.  
- Translation quality JSON/TXT go under `translation_quality/blocked/` and `translation_quality/free/`.  
- LLM ablation summaries are written to the top-level `extended/` directory.

### `llm_scoring_colab.ipynb` — Colab Notebook

Interactive notebook for running LLM-as-Judge scoring on Google Colab with free GPU.

## Dependencies

Beyond the base project requirements:

```
sacrebleu          # chrF
unbabel-comet      # COMET (optional)
transformers       # LLM scorer (optional)
scikit-learn       # AUC
```
