#!/bin/bash
# PPL + Retrieval + Discrimination evaluation for r2 / r3 / r4 experiments

SCRIPT_PATH="/nethome/${USER}/Thoth/7_validation.py"
DEVICE=3

DATA_PATH="/scratch/${USER}/thoth_project/datasets/unpc_en-fr/large/tokenized_bucketed_mono"

run_eval() {
    local MODEL_DIR="$1"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Skipping (not found): $MODEL_DIR"
        return
    fi

    EXP=$(basename "$MODEL_DIR")
    echo "=================================================="
    echo "Validating: ${EXP}"
    echo "=================================================="

    python "$SCRIPT_PATH" \
        --model_path "$MODEL_DIR" \
        --data_path "$DATA_PATH" \
        --split "test" \
        --device "$DEVICE" \
        --max_n_ret 4096 \
        --pool_size 20 \
        --ret_negative_mode "hard" \
        --hard_neg_ratio 0.5

    echo "Finished: ${EXP}"
    echo ""
}

# ================================================================== #
#  Round 0 — full dataset                                           #
# ================================================================== #

echo ""
echo "========== Round 1 (medium) =========="
BASE_R0="/scratch/${USER}/thoth_project/models/full_runs"
R0_EXPERIMENTS=(
    "r0v0_baseline"
    "r0v1_lr1e-4_wd1e3_warmup01"
    "r0v2_lr5e-5_cos_w05_wd001_d015_fp16"
    "r0v3_lr3e-5_cos_w05_wd001_d015_fp16"
    "r0v4_lr5e-5_cos_w05_wd01_d01_fp16"
    "r0v5_lr1e-4_cos_w05_wd01_d015_fp16"
)
for EXP in "${R0_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R0}/${EXP}"
done

# ================================================================== #
#  Round 1 — medium dataset                                           #
# ================================================================== #
echo ""
echo "========== Round 1 (medium) =========="
BASE_R1="/scratch/${USER}/thoth_project/models/experiments/r1"
R1_EXPERIMENTS=(
    "r1_v0_baseline"
    "r1_v1_preproc"
    "r1_v2_lr"
    "r1_v3_lr_preproc"
    "r1_v4_drop02"
    "r1_v5_drop02_preproc"
    "r1_v6_drop02_lr"
    "r1_v7_drop02_lr_preproc"
    "r1_v8_drop005"
    "r1_v9_drop005_preproc"
    "r1_v10_drop005_lr"
    "r1_v11_drop005_lr_preproc"
    "r1_v12_wd01"
    "r1_v13_wd01_preproc"
    "r1_v14_wd01_lr"
    "r1_v15_wd01_lr_preproc"
    "r1_v16_wd0005"
    "r1_v17_wd0005_preproc"
    "r1_v18_wd0005_lr"
    "r1_v19_wd0005_lr_preproc"
)
for EXP in "${R1_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R1}/${EXP}"
done

# ================================================================== #
#  Round 2 — medium dataset                                           #
# ================================================================== #
echo ""
echo "========== Round 2 (medium) =========="
BASE_R2="/scratch/${USER}/thoth_project/models/experiments/r2"
R2_EXPERIMENTS=(
    "r2v0_baseline"
    "r2v1_wr01"
    "r2v2_wr001"
    "r2v3_lr5e-5"
    "r2v4_lr5e-5_wr01"
    "r2v5_lr5e-5_wr001"
    "r2v6_lr1e-4_wr001"
    "r2v7_lr1e-4_wr01"
    "r2r2v8_lr1e-4_wr001v8"
    "r2v9_lr5e-4_wr0"
    "r2v10_lr5e-4_wr01"
    "r2v11_lr5e-4_wr001"
    "r2v12_lr1e-3_wr0"
)
for EXP in "${R2_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R2}/${EXP}"
done

# ================================================================== #
#  Round 3 small — small dataset                                      #
# ================================================================== #
echo ""
echo "========== Round 3 Small =========="
BASE_R3S="/scratch/${USER}/thoth_project/models/experiments/r3/r3s"
R3S_EXPERIMENTS=(
    "r3v1_lr3e-4_cos_w03_fp16_b95"
    "r3v2_lr1e-4_cos_w03_fp16_b95"
    "r3v3_lr6e-4_cos_w03_fp16_b95"
    "r3v4_lr3e-4_cos_w01_fp16_b95"
    "r3v5_lr3e-4_cos_w05_fp16_b95"
    "r3v6_lr3e-4_lin_w03_fp16_b95"
    "r3v7_lr3e-4_cos_w03_fp32_b95"
    "r3v8_lr3e-4_cos_w03_fp16_b999"
    "r3v9_lr6e-4_cos_w05_fp16_b95"
)
for EXP in "${R3S_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R3S}/${EXP}"
done

# ================================================================== #
#  Round 3 medium — medium dataset                                    #
# ================================================================== #
echo ""
echo "========== Round 3 Medium =========="
BASE_R3M="/scratch/${USER}/thoth_project/models/experiments/r3/r3m"
R3M_EXPERIMENTS=(
    "r3v10_lr3e-4_cos_w03_fp16_b999"
    "r3v11_noacc_lr3e-4_cos_w03_fp16_b999"
    "r3v12_medium_lr2e-4_cos_w03_fp16_b999"
)
for EXP in "${R3M_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R3M}/${EXP}"
done

# ================================================================== #
#  Round 4 small — small dataset                                      #
# ================================================================== #
echo ""
echo "========== Round 4 Small =========="
BASE_R4S="/scratch/${USER}/thoth_project/models/experiments/r4/r4s"
R4S_EXPERIMENTS=(
    "r4v1_lr1e-4_lin_w10_wd001_d02_fp32"
    "r4v2_lr1e-4_lin_w10_wd01_d01_fp32"
    "r4v3_lr1e-4_lin_w10_wd1_d01_fp32"
    "r4v4_lr1e-4_lin_w10_wd01_d02_fp32"
    "r4v5_lr1e-4_lin_w10_wd001_d005_fp32"
    "r4v6_lr2e-4_cos_w05_wd005_d01_fp16"
    "r4v7_lr1e-4_cos_w05_wd005_d01_fp16"
    "r4v8_lr2e-4_cos_w10_wd005_d01_fp16"
    "r4v9_lr2e-4_cos_w05_wd01_d01_fp16"
    "r4v10_lr2e-4_cos_w05_wd005_d015_fp16"
)
for EXP in "${R4S_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R4S}/${EXP}"
done

# ================================================================== #
#  Round 4 medium — medium dataset                                    #
# ================================================================== #
echo ""
echo "========== Round 4 Medium =========="
BASE_R4M="/scratch/${USER}/thoth_project/models/experiments/r4/r4m"
R4M_EXPERIMENTS=(
    "r4v1m_lr1e-4_lin_w10_wd001_d02_fp32"
    "r4v2m_lr1e-4_lin_w10_wd01_d01_fp32"
    "r4v3m_lr1e-4_lin_w10_wd1_d01_fp32"
    "r4v4m_lr1e-4_lin_w10_wd01_d02_fp32"
    "r4v5m_lr1e-4_lin_w10_wd001_d005_fp32"
    "r4v6m_lr2e-4_cos_w05_wd005_d01_fp16"
    "r4v10m_lr2e-4_cos_w05_wd005_d015_fp16"
    "r4v11m_lr2e-4_cos_w10_wd005_d015_fp16"
    "r4v12m_lr3e-4_cos_w05_wd005_d015_fp16"
    "r4v13m_lr2e-4_cos_w05_wd01_d015_fp16"
    "r4v14m_lr2e-4_cos_w05_wd005_d02_fp16"
)
for EXP in "${R4M_EXPERIMENTS[@]}"; do
    run_eval "${BASE_R4M}/${EXP}"
done

echo ""
echo "All evaluations completed."
