# Step 3 — Model prediction (post-training inference)

Two inference paths for the fine-tuned models, serving the two downstream stages:

| Script | Engine | Purpose | Feeds |
|--------|--------|---------|-------|
| `predict_llamafactory.yaml` | LLaMA-Factory `do_predict` | **experiment evaluation** — predict on the gold eval datasets, producing `generated_predictions.jsonl` that the 5-metric suite scores | → `../4_experiment_eval/` |
| `predict_final_test.py` | vLLM | **final test-set submission** — fast batch inference on the official test prompts, producing `<lang>_predicted.json` | → `../5_final_submission/` |

## A. LLaMA-Factory evaluation prediction (`predict_llamafactory.yaml`)

Runs in the LLaMA-Factory repo. Greedy decoding with repetition penalty, on a
registered `gold_sft_*` eval dataset:

```bash
llamafactory-cli train predict_llamafactory.yaml
# key fields:
#   model_name_or_path: <fine-tuned checkpoint>     # sft / sft_add / dpo variant
#   stage: sft   do_predict: true   finetuning_type: full
#   eval_dataset: gold_sft_<lang>                   # registered in dataset_info.json
#   template: hunyuan   do_sample: false   repetition_penalty: 1.2
#   output_dir: /.../hunyun_eval/<lang>/<variant>   # writes generated_predictions.jsonl
```

The `generated_predictions.jsonl` (fields `prompt`/`predict`/`label`) is consumed
by `../4_experiment_eval/evaluate_bodo.py` (BLEU/METEOR/ROUGE-L/ChrF/TER) and
`fix_decode_and_eval.py`. Uncomment the relevant `model_name_or_path` /
`eval_dataset` / `output_dir` lines for each experiment.

## B. vLLM final-test prediction (`predict_final_test.py`)

Multi-GPU sharded vLLM batch inference for the official test set. Auto-detects
LoRA adapters vs full-parameter checkpoints, greedy decoding (`temperature=0`)
with `repetition_penalty=1.2`, and retries empty outputs with rising temperature.
Preserves every input field (incl. `id`) and adds `predict`; shard order is
contiguous so output order == input order (keep the input id-sorted).

Edit the config block at the top, then run once **per system**, writing into the
committed Step-5 prediction folders:

```python
# ---- run for the hy_sft system (-> contrastive 2) ----
LANGUAGE_MODELS = { "<lang>": "/.../train_save/full/wmt_total/hy_<lang>/3e-5", ... }
INPUT_DIR  = "/.../sft_testset"                       # *_final_test.json prompts (from Step 1)
OUTPUT_DIR = "../5_final_submission/sft_predictions"
```
```python
# ---- run again for the hy_sft_add system (-> contrastive 1, non-bodo) ----
LANGUAGE_MODELS = { "<lang>": "/.../train_save/full/wmt_total/hy_<lang>_add/3e-5", ... }
OUTPUT_DIR = "../5_final_submission/sft_add_predictions"
```

This yields `<dir>/<lang>_predicted.json` (each item keeps its `id` from the Step-1
prompt + adds `predict`), exactly the layout that
`../5_final_submission/make_all_submissions.py` and
`../4_experiment_eval/qe_reference_free_eval.py` consume. The Bodo contrastive-1
input `sft_test_predictions/bodo_predicted.json` is then produced by
`../5_final_submission/short_and_fix.py` (take-shorter + decode_fix).

> Input prompts (`*_final_test.json`, with vocabulary-hint instructions) are built
> in Step 1 (`1_vocab_build/category2_augmented/build_final_testset.py`).
> The DPO rejected-sample generator lives in `../2_training/dpo_all/gen_dpo_rejected_vllm.py`.
