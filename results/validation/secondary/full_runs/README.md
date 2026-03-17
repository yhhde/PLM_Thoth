# Extended validation results

Layout: **type first** (tsv, translation_quality, retrieval_disc); **translation_quality** is split into blocked and free.

- **tsv/**
  - **blocked/** — `r0v*_blocked_translations.tsv` (no-repeat-3-gram)
  - **free/** — `r0v*_free_translations.tsv` (no ngram blocking)

- **translation_quality/**
  - **blocked/** — `r0v*_translation_quality.json`, `.txt`
  - **free/** — `r0v*_translation_quality_ablation.json`, `.txt`
  - 
When running `translation_quality.py` with `--input_dir results/validation/extended`, the script uses these paths automatically.
