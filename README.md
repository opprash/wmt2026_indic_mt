# WMT26 Indic-MT — Team 星辰之力 (Star Power) System Code

Reproducible code for our submission to the **WMT26 Low-Resource Indic Languages
Machine Translation Shared Task (Category 2)**, covering five English→X directions:

| Language | ISO 639-3 | WMT pair code |
|----------|-----------|---------------|
| Bodo     | brx       | `en_to_bodo`  |
| Karbi    | mjw       | `en_to_mjw`   |
| Kokborok | trp       | `en_to_trp`   |
| Nagamese | nag       | `en_to_nag`   |
| Tagin    | tgj       | `en_to_tgj`   |

> **Submission category: Unconstrained.** All systems are built on **pretrained
> LLMs** (Qwen2.5-32B-Instruct, Hunyuan-MT-7B) plus official + public external
> data. Because our systems use pretrained models, we submit **two contrastive
> systems per language pair and no primary system** (see the system-description
> document for the rationale).

---

## Repository layout

The repo is organised as the five logical stages of the pipeline:

```
wmt2026_indic_mt/
├── 1_vocab_build/             # Step 1 — vocabulary building + fine-tuning data prep
│   ├── category2_official/    #   pipeline on the official Category-2 corpus
│   ├── category2_augmented/   #   same pipeline on official + external public corpus ("add")
│   └── tools/                 #   one-off Excel inspection helpers
├── 2_training/                # Step 2 — LLaMA-Factory SFT / DPO configs & launchers
│   ├── <lang>_yaml/           #   Qwen2.5-32B LoRA SFT, LR sweep {5e-5,1e-4,2e-4,5e-4}
│   ├── add/ hy_sft/ hy_sft_add/   Qwen/Hunyuan SFT (+ augmented variants)
│   ├── dpo_all/ dpo_all_hy/   #   Qwen / Hunyuan DPO (dpo_all/gen_dpo_rejected_vllm.py)
│   ├── run_all.sh             #   top-level SFT launcher
│   └── evaluate_lr.py         #   pick best LR from sweep
├── 3_prediction/             # Step 3 — post-training model prediction
│   ├── predict_llamafactory.yaml   LLaMA-Factory do_predict -> eval (feeds Step 4)
│   └── predict_final_test.py       vLLM batch inference -> test set (feeds Step 5)
├── 4_experiment_eval/        # Step 4 — experiment-result evaluation
│   ├── evaluate_bodo.py            5-metric suite on WMT2025 gold
│   ├── fix_decode_and_eval.py      decode-fix before/after re-scoring
│   └── qe_reference_free_eval.py   reference-free QE (hy_sft vs hy_sft_add)
├── 5_final_submission/       # Step 5 — final submission prediction & processing
│   ├── submission_plan.py          single source of truth for the plan
│   ├── short_and_fix.py            Bodo c1: take-shorter + decode_fix
│   ├── make_all_submissions.py     build official <TEAM>_<slot>_<pair>.txt
│   └── verify_order_vs_excel.py    verify line order vs official test Excel
├── docs/                      # pipeline overview & notes
├── requirements.txt
└── .gitignore
```

Each numbered directory has its own `README.md` with detailed usage.

---

## End-to-end pipeline

```
  [1] 1_vocab_build/         official Category-2 corpus (+ public external for "add")
                            build_vocab_alignment -> build_word_alignment (LLM)
                            -> build_finetune_dataset -> merge_train_test
                            => <lang>_train.json / <lang>_train_add.json (vocab-hint prompts)
                                         │
                                         ▼
  [2] 2_training/           LLaMA-Factory
                            Qwen2.5-32B LoRA SFT ── DPO   │  Hunyuan-MT-7B full SFT ── DPO
                            (+ _add variants on augmented data)
                            => fine-tuned checkpoints (hy_sft, hy_sft_add, ...)
                                         │
                                         ▼
  [3] 3_prediction/         predict_llamafactory.yaml  -> gold eval predictions ─┐
                            predict_final_test.py (vLLM) -> predictions/<sys>/<lang>_predicted.json ─┐
                                         │                                        │                  │
                  ┌──────────────────────┘                                        │                  │
                  ▼                                                               ▼                  │
  [4] 4_experiment_eval/    evaluate_bodo (5-metric) / fix_decode_and_eval /                          │
                            qe_reference_free_eval (hy_sft vs hy_sft_add)  -> pick per-lang system    │
                                                                                                      ▼
  [5] 5_final_submission/   submission_plan + short_and_fix (Bodo c1) ->
                            make_all_submissions -> <TEAM>_<slot>_<pair>.txt -> verify_order_vs_excel
```

### Two parallel modeling pipelines

1. **Qwen2.5-32B-Instruct + LoRA SFT** (→ optional LoRA DPO). General 32B
   instruction LLM, LoRA `r=8 / alpha=16`, DeepSpeed ZeRO-3, 4×GPU, bf16.
2. **Hunyuan-MT-7B + full-parameter SFT** (→ optional full DPO). 7B
   translation-specialized model, Flash-Attention-2, DeepSpeed ZeRO-3.

Each pipeline additionally has an **`_add`** variant trained on official + external
public corpus. The final submitted systems are Hunyuan-MT-7B full-parameter
variants (`sft`, `sft_add`, `sft→DPO`) plus the Bodo **take-shorter-merge +
decode_fix** ensemble.

---

## Reproduction quick-start

```bash
# 0. Environment
pip install -r requirements.txt
export DASHSCOPE_API_KEY=...        # required only for Step-1 LLM word alignment

# 1. Build vocab + fine-tuning data (per corpus variant)
cd 1_vocab_build/category2_official
python build_vocab_alignment.py
python build_word_alignment.py      # calls qwen3-max via DashScope
python build_finetune_dataset.py
python merge_train_test.py

# 2. Train (requires LLaMA-Factory + GPUs; edit absolute paths in the yaml first)
cd ../../2_training
bash run_all.sh                     # Qwen LoRA SFT sweep
bash hy_sft/run_all_hy.sh           # Hunyuan full SFT
#   ... DPO / add variants as needed

# 3. Predict with the trained models
cd ../3_prediction
#   (a) for experiment evaluation — LLaMA-Factory do_predict on the gold eval sets:
llamafactory-cli train predict_llamafactory.yaml
#   (b) for the final test set — vLLM, run once per system into predictions/<system>/:
#       edit LANGUAGE_MODELS + OUTPUT_DIR=../5_final_submission/predictions/hy_sft  (then hy_sft_add)
python predict_final_test.py

# 4. Evaluate experiments (pick per-language submission system)
cd ../4_experiment_eval
python evaluate_bodo.py --pred <generated_predictions.jsonl>        # 5-metric (gold)
python qe_reference_free_eval.py --pred-root ../5_final_submission/predictions  # reference-free QE

# 5. Build + verify the official submission files
cd ../5_final_submission
python make_all_submissions.py      # apply plan -> official .txt (2 contrastive, no primary)
python verify_order_vs_excel.py --excel-base "<.../Category 2>"
```

The final submission plan (see `5_final_submission/submission_plan.py`):
**contrastive 1** = Bodo take-shorter+decode_fix, others `hy_sft_add`;
**contrastive 2** = all `hy_sft`. No primary system.

---

## Important notes for reproduction

- **Secrets**: API keys are read from the `DASHSCOPE_API_KEY` environment variable.
  No keys are committed. Step-1 word alignment needs a DashScope account.
- **Absolute paths**: the training `*.yaml` / `*.sh` reference cluster-specific
  absolute paths (e.g. `/base/rd1/large_models/...`) and base-model locations.
  Adjust these to your environment before training.
- **Large artifacts are not committed**: intermediate corpora, vocab JSONs,
  datasets, model checkpoints and the raw Step-3 `predictions/` dumps are
  excluded via `.gitignore`. **Exception kept for reproducibility**: the three
  final model-prediction folders `5_final_submission/{sft_predictions,
  sft_add_predictions,sft_test_predictions}/` are committed, so the official
  submission `.txt` can be regenerated with `make_all_submissions.py`.
- **Base models** must be obtained separately (publicly available):
  `Qwen/Qwen2.5-32B-Instruct`, `tencent/Hunyuan-MT-7B`.






