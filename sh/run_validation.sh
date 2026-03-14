#!/bin/bash
# Corrected PPL + Discrimination (ACC) validation for 6 large models

DATA_PATH="/scratch/${USER}/thoth_project/datasets/unpc_en-fr/large/tokenized_bucketed_mono"
BASE_OUT="/scratch/${USER}/thoth_project/models/experiments/r0"

SCRIPT_PATH="/nethome/hyonghua/Thoth/6_validation.py"

OUT_DIR="/nethome/hyonghua/thoth_project/results/validation"
mkdir -p "$OUT_DIR"

DEVICE=${DEVICE:-3}

declare -A MODEL_IDS
MODEL_IDS=(
    ["r0v0_full"]="r0v0"
    ["r0v1_full_lr1e4_wd1e3_warmup01"]="r0v1"
    ["r0v2_lr5e5_cos_w05_wd001_d015_fp16"]="r0v2"
    ["r0v3_lr3e5_cos_w05_wd001_d015_fp16"]="r0v3"
    ["r0v4_lr5e5_cos_w05_wd01_d01_fp16"]="r0v4"
    ["r0v5_lr1e4_cos_w05_wd01_d015_fp16"]="r0v5"
)

EXPERIMENTS=(
    "r0v0_full"
    "r0v1_full_lr1e4_wd1e3_warmup01"
    "r0v2_lr5e5_cos_w05_wd001_d015_fp16"
    "r0v3_lr3e5_cos_w05_wd001_d015_fp16"
    "r0v4_lr5e5_cos_w05_wd01_d01_fp16"
    "r0v5_lr1e4_cos_w05_wd01_d015_fp16"
)

echo "PPL + Discrimination (ACC) validation — 6 models"
echo "Output: $OUT_DIR"
echo "GPU: $DEVICE"
echo "=================================================="

for EXP in "${EXPERIMENTS[@]}"; do
    MODEL_DIR="${BASE_OUT}/${EXP}"
    MID="${MODEL_IDS[$EXP]}"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "Skipping (not found): $MODEL_DIR"
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Validating: ${MID} (${EXP})"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="$DEVICE" python "$SCRIPT_PATH" \
        --model_path "$MODEL_DIR" \
        --data_path "$DATA_PATH" \
        --split "test" \
        --device 0 \
        --model_id "$MID" \
        --out_dir "$OUT_DIR"

    echo "Finished: ${MID}"
done

echo ""
echo "=================================================="
echo "All validations complete. Results in: $OUT_DIR"
echo "=================================================="
