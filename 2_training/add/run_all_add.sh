#!/bin/bash
# Master training script -- add (extended-dataset) version
# Order: Karbi -> targin -> Kokborok -> Nagamese -> Bodo
# Each language trains ONE learning rate (no LR sweep, no eval comparison).
#   karbi    lr=5e-4  epochs=4
#   targin   lr=5e-4  epochs=4
#   kokborok lr=5e-4  epochs=3
#   nagamese lr=2e-4  epochs=3
#   bodo     lr=5e-4  epochs=3
#
# Recommended launch:
#   nohup bash /base/rd1/sft_add/run_all_add.sh > /base/rd1/train_logs_add/run_all_add.log 2>&1 &

# NOTE: do NOT enable `set -e` -- we want the loop to continue past per-task
# failures. `set -u` catches typos in variable names.
set -u

SFT_DIR=/base/rd1/sft_add

LANGS=("karbi"  "targin" "kokborok" "nagamese" "bodo")
LRS=(  "5e-4"   "5e-4"   "5e-4"     "2e-4"     "5e-4")

mkdir -p /base/rd1/train_logs_add

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== Pipeline (add) started ====="

for i in "${!LANGS[@]}"; do
    lang="${LANGS[$i]}"
    lr="${LRS[$i]}"
    SH="$SFT_DIR/${lang}_yaml/${lang}_${lr}_sft.sh"

    echo "[$(ts)] >>>>> Task: ${lang} (lr=${lr}) <<<<<"
    if [[ ! -f "$SH" ]]; then
        echo "[$(ts)] [ERROR] launch script not found: $SH (skipped)"
        continue
    fi

    bash "$SH"
    RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "[$(ts)] [WARN] Training ${lang} (lr=${lr}) exited non-zero (exit=${RC}); continuing..."
    else
        echo "[$(ts)] --- Finished ${lang} (exit=${RC}) ---"
    fi
done

echo "[$(ts)] ===== All training tasks completed ====="
