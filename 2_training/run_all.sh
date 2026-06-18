#!/bin/bash
# Master training script
# Order: Karbi -> targin -> Kokborok -> Nagamese -> Bodo
# Each language trains 4 learning rates sequentially. After all LRs of a
# language finish, evaluate_lr.py is invoked to pick the best LR; an eval
# failure is logged but does NOT abort the pipeline for the remaining tasks.
#
# Recommended launch:
#   nohup bash /base/rd1/sft/run_all.sh > /base/rd1/train_logs/run_all.log 2>&1 &

# NOTE: do NOT enable `set -e` -- we want the loop to continue past per-task
# failures (both training and eval). `set -u` catches typos in variable names.
set -u

SFT_DIR=/base/rd1/sft
EVAL_SCRIPT="$SFT_DIR/evaluate_lr.py"

LANGS=("karbi" "targin" "kokborok" "nagamese" "bodo")
LRS=("5e-5" "1e-4" "2e-4" "5e-4")

mkdir -p /base/rd1/train_logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== Pipeline started ====="

for lang in "${LANGS[@]}"; do
    echo "[$(ts)] >>>>> Task: ${lang} <<<<<"

    for lr in "${LRS[@]}"; do
        SH="$SFT_DIR/${lang}_yaml/${lang}_${lr}_sft.sh"
        echo "[$(ts)] --- Train ${lang} lr=${lr} ---"
        if [[ ! -f "$SH" ]]; then
            echo "[$(ts)] [ERROR] launch script not found: $SH (skipped)"
            continue
        fi
        bash "$SH"
        TRAIN_RC=$?
        if [[ "$TRAIN_RC" -ne 0 ]]; then
            echo "[$(ts)] [WARN] Training ${lang} lr=${lr} exited non-zero (exit=${TRAIN_RC}); continuing..."
        else
            echo "[$(ts)] --- Finished ${lang} lr=${lr} (exit=${TRAIN_RC}) ---"
        fi
    done

    EVAL_LOG_DIR="/base/rd1/train_logs/${lang}"
    EVAL_LOG="${EVAL_LOG_DIR}/lr_comparison.log"
    mkdir -p "$EVAL_LOG_DIR"
    echo "[$(ts)] === Evaluating ${lang} LR sweep -> ${EVAL_LOG} ==="

    if [[ ! -f "$EVAL_SCRIPT" ]]; then
        echo "[$(ts)] [WARN] evaluator script missing: ${EVAL_SCRIPT}; skipping eval for ${lang}"
    else
        # Evaluation must NEVER stop the pipeline. Run inside a subshell so
        # any unexpected failure (script crash, missing python, etc.) is
        # contained. stderr is merged into stdout so errors also land in the
        # per-task log, and PIPESTATUS[0] gives python's real exit code (tee
        # otherwise hides it).
        (
            python3 "$EVAL_SCRIPT" --task "${lang}" 2>&1 | tee "$EVAL_LOG"
            exit "${PIPESTATUS[0]}"
        )
        EVAL_RC=$?
        if [[ "$EVAL_RC" -ne 0 ]]; then
            echo "[$(ts)] [WARN] Eval ${lang} failed (python exit=${EVAL_RC}); continuing to next task..."
        else
            echo "[$(ts)] === Eval ${lang} OK ==="
        fi
    fi
done

echo "[$(ts)] ===== All training & per-task evaluations completed ====="
