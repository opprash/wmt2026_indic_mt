#!/bin/bash
# Master DPO training script -- Hunyuan-MT-7B full-parameter DPO
# Order: Karbi -> targin -> Kokborok -> Nagamese -> Bodo
# Each language trains ONCE with the same hyperparameters (no LR sweep, no eval).
#   base:     hy_sft/<lang>/3e-5  (the full-FT SFT checkpoint from hy_sft)
#   dataset:  <lang>_train_dpo
#   lr:       5e-6     (single, no sweep)
#   epochs:   3
#   batch:    per_device 2 × grad_accum 8 × 4 GPU = global 64
#   template: hunyuan
#   method:   full DPO + DeepSpeed ZeRO-3 + fa2
#
# Recommended launch:
#   nohup bash /base/rd1/dpo_all_hy/run_all_dpo_hy.sh > /base/rd1/dpo_logs_hy/run_all_dpo_hy.log 2>&1 &

# NOTE: do NOT enable `set -e` -- let the loop continue past per-task failures.
set -u

DPO_DIR=/base/rd1/dpo_all_hy

LANGS=("karbi" "targin" "kokborok" "nagamese" "bodo")
LR="5e-6"

mkdir -p /base/rd1/dpo_logs_hy

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== Hunyuan-MT-7B full-DPO pipeline started ====="

for lang in "${LANGS[@]}"; do
    SH="$DPO_DIR/${lang}_yaml/${lang}_${LR}_dpo.sh"

    echo "[$(ts)] >>>>> Task: ${lang} (lr=${LR}) <<<<<"
    if [[ ! -f "$SH" ]]; then
        echo "[$(ts)] [ERROR] launch script not found: $SH (skipped)"
        continue
    fi

    bash "$SH"
    RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "[$(ts)] [WARN] DPO ${lang} exited non-zero (exit=${RC}); continuing..."
    else
        echo "[$(ts)] --- Finished DPO ${lang} (exit=${RC}) ---"
    fi
done

echo "[$(ts)] ===== All Hunyuan-MT-7B full-DPO tasks completed ====="
