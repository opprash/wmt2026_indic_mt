#!/bin/bash
# Master training script -- Hunyuan-MT-7B full-parameter SFT (augmented data)
# Order: Karbi -> targin -> Kokborok -> Nagamese -> Bodo
# Each language trains ONCE with the same hyperparameters (no LR sweep, no eval).
#   model:    Hunyuan-MT-7B  (full FT, DeepSpeed ZeRO-3, fa2)
#   dataset:  <lang>_train_add  (augmented corpus)
#   lr:       3e-5     (single, no sweep)
#   epochs:   4
#   batch:    per_device 2 × grad_accum 8 × 4 GPU = global 64
#   template: hunyuan
#
# Recommended launch:
#   nohup bash /base/rd1/hy_sft_add/run_all_hy_add.sh > /base/rd1/train_logs_hy_add/run_all_hy_add.log 2>&1 &

# NOTE: do NOT enable `set -e` -- let the loop continue past per-task failures.
set -u

HY_DIR=/base/rd1/hy_sft_add

LANGS=("karbi" "targin" "kokborok" "nagamese" "bodo")
LR="3e-5"

mkdir -p /base/rd1/train_logs_hy_add

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== Hunyuan-MT-7B full-SFT (augmented) pipeline started ====="

for lang in "${LANGS[@]}"; do
    SH="$HY_DIR/${lang}_yaml/${lang}_${LR}_sft.sh"

    echo "[$(ts)] >>>>> Task: ${lang} (lr=${LR}) <<<<<"
    if [[ ! -f "$SH" ]]; then
        echo "[$(ts)] [ERROR] launch script not found: $SH (skipped)"
        continue
    fi

    bash "$SH"
    RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "[$(ts)] [WARN] Training ${lang} exited non-zero (exit=${RC}); continuing..."
    else
        echo "[$(ts)] --- Finished ${lang} (exit=${RC}) ---"
    fi
done

echo "[$(ts)] ===== All Hunyuan-MT-7B full-SFT (augmented) tasks completed ====="
