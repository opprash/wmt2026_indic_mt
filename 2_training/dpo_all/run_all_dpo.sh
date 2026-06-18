#!/bin/bash
# Master DPO training script
# Order: Karbi -> targin -> Kokborok -> Nagamese -> Bodo
# Each language sweeps 3 learning rates (1e-4, 2e-4, 5e-4), epochs=4.
# After all LRs of a language finish, evaluate_lr_dpo.py picks the best LR.
# An eval failure is logged but does NOT abort the pipeline.
#
# After ALL DPO tasks (and their evaluations) complete, this script kicks off
# an independent SFT-add run in the background via nohup. That extra task is
# NOT part of the DPO evaluation flow.
#
# Recommended launch:
#   nohup bash /base/rd1/dpo_all/run_all_dpo.sh > /base/rd1/dpo_logs/run_all_dpo.log 2>&1 &

# NOTE: do NOT enable `set -e` -- we want the loop to continue past per-task
# failures (both training and eval). `set -u` catches typos in variable names.
set -u

DPO_DIR=/base/rd1/dpo_all
EVAL_SCRIPT="$DPO_DIR/evaluate_lr_dpo.py"

LANGS=("karbi" "targin" "kokborok" "nagamese" "bodo")
LRS=("1e-4" "2e-4" "5e-4")

mkdir -p /base/rd1/dpo_logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== DPO Pipeline started ====="

for lang in "${LANGS[@]}"; do
    echo "[$(ts)] >>>>> DPO Task: ${lang} <<<<<"

    for lr in "${LRS[@]}"; do
        SH="$DPO_DIR/${lang}_yaml/${lang}_${lr}_dpo.sh"
        echo "[$(ts)] --- DPO train ${lang} lr=${lr} ---"
        if [[ ! -f "$SH" ]]; then
            echo "[$(ts)] [ERROR] launch script not found: $SH (skipped)"
            continue
        fi
        bash "$SH"
        TRAIN_RC=$?
        if [[ "$TRAIN_RC" -ne 0 ]]; then
            echo "[$(ts)] [WARN] DPO ${lang} lr=${lr} exited non-zero (exit=${TRAIN_RC}); continuing..."
        else
            echo "[$(ts)] --- Finished DPO ${lang} lr=${lr} (exit=${TRAIN_RC}) ---"
        fi
    done

    EVAL_LOG_DIR="/base/rd1/dpo_logs/${lang}"
    EVAL_LOG="${EVAL_LOG_DIR}/lr_comparison.log"
    mkdir -p "$EVAL_LOG_DIR"
    echo "[$(ts)] === Evaluating DPO ${lang} LR sweep -> ${EVAL_LOG} ==="

    if [[ ! -f "$EVAL_SCRIPT" ]]; then
        echo "[$(ts)] [WARN] evaluator script missing: ${EVAL_SCRIPT}; skipping eval for ${lang}"
    else
        # Evaluation must NEVER stop the pipeline. Run inside a subshell so any
        # unexpected failure is contained. stderr is merged into stdout so
        # errors also land in the per-task log; PIPESTATUS[0] gives python's
        # real exit code (tee otherwise hides it).
        (
            python3 "$EVAL_SCRIPT" --task "${lang}" 2>&1 | tee "$EVAL_LOG"
            exit "${PIPESTATUS[0]}"
        )
        EVAL_RC=$?
        if [[ "$EVAL_RC" -ne 0 ]]; then
            echo "[$(ts)] [WARN] DPO eval ${lang} failed (python exit=${EVAL_RC}); continuing to next task..."
        else
            echo "[$(ts)] === DPO eval ${lang} OK ==="
        fi
    fi
done

echo "[$(ts)] ===== All DPO training & per-task evaluations completed ====="

# ---------------------------------------------------------------------------
# Independent follow-up task: run the SFT-add pipeline in background via nohup.
# This is NOT part of the DPO evaluation and runs detached so it survives this
# script exiting.
# ---------------------------------------------------------------------------
ADD_SCRIPT=/base/rd1/run_all_add.sh
ADD_LOG=/base/rd1/train_log_add1.log

echo "[$(ts)] === Launching independent SFT-add task in background ==="
if [[ ! -f "$ADD_SCRIPT" ]]; then
    echo "[$(ts)] [WARN] follow-up script not found: $ADD_SCRIPT (not launched)"
else
    mkdir -p "$(dirname "$ADD_LOG")"
    nohup bash "$ADD_SCRIPT" > "$ADD_LOG" 2>&1 &
    ADD_PID=$!
    echo "[$(ts)] === Launched: $ADD_SCRIPT (PID=${ADD_PID}, log=$ADD_LOG) ==="
fi

echo "[$(ts)] ===== run_all_dpo.sh finished ====="
