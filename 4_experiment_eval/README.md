# Step 4 — Experiment-result evaluation

Scores the model predictions from Step 3 to compare experiments (sft / sft_add /
dpo, Qwen vs Hunyuan) and choose the per-language submission system.

## Scripts

| Script | Reference? | Purpose |
|--------|-----------|---------|
| `evaluate_bodo.py` | uses WMT2025 gold | Official **5-metric suite**: BLEU (13a + indicnlp), METEOR, ROUGE-L, ChrF/ChrF++, TER. Reads LLaMA-Factory `generated_predictions.jsonl`. Exports metric fns reused by the others. |
| `fix_decode_and_eval.py` | uses gold | Repetition-fix post-processing (no-repeat-ngram + length cap) with **before/after** re-scoring — quantifies the decode_fix gain. |
| `qe_reference_free_eval.py` | **no reference** | Reference-free QE (surface/statistical, non-semantic): n-gram diversity, repetition, length ratio, source-copy/non-translation. Compares `hy_sft` vs `hy_sft_add` per language — the evidence behind the submission plan. |
| `evaluate_new_model.py`, `evaluate_wmt_methods.py` | mixed | Auxiliary scoring utilities used during experiments. |

## Usage

```bash
# reference-based (Bodo/Kokborok have public WMT2025 gold)
python evaluate_bodo.py --pred /.../hunyun_eval/bodo/sft/generated_predictions.jsonl
python fix_decode_and_eval.py --pred <generated_predictions.jsonl> --lang brx

# reference-free QE on the vLLM test predictions (predictions/<system>/<lang>_...)
python qe_reference_free_eval.py --pred-root ../5_final_submission/predictions
```

## METEOR data (once)

```bash
python -m nltk.downloader wordnet omw-1.4 punkt
```

These metrics inform the plan applied in Step 5: where `hy_sft_add` wins it
becomes contrastive 1; `hy_sft` is always contrastive 2; for Bodo the
take-shorter + decode_fix ensemble is contrastive 1.
