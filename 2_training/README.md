# Step 2 — Training (LLaMA-Factory)

All training uses **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
(`llamafactory-cli train <config>.yaml`). Two parallel pipelines, each with an
`_add` (augmented-data) variant, optionally followed by DPO.

> ⚠️ The `*.yaml` / `*.sh` files contain **cluster-specific absolute paths**
> (base models under `/base/rd1/large_models/...`, output dirs, dataset names
> registered in `dataset_info.json`). Edit these for your environment before running.

## Base models (public, obtain separately)

- `Qwen/Qwen2.5-32B-Instruct` — LoRA SFT/DPO
- `tencent/Hunyuan-MT-7B` — full-parameter SFT/DPO

## Directory map

| Path | Pipeline | Stage | Tuning | LR |
|------|----------|-------|--------|-----|
| `<lang>_yaml/` + `run_all.sh` | Qwen2.5-32B | SFT | LoRA (r=8, α=16) | sweep {5e-5,1e-4,2e-4,5e-4} |
| `add/<lang>_yaml/` + `run_all_add.sh` | Qwen2.5-32B | SFT | LoRA | per-lang fixed best LR |
| `hy_sft/<lang>_yaml/` + `run_all_hy.sh` | Hunyuan-MT-7B | SFT | full + fa2 | 3e-5 |
| `hy_sft_add/<lang>_yaml/` + `run_all_hy_add.sh` | Hunyuan-MT-7B | SFT (augmented) | full | 3e-5 |
| `dpo_all/<lang>_yaml/` + `run_all_dpo.sh` | Qwen2.5-32B | DPO | LoRA, β=0.1 sigmoid | sweep {1e-4,2e-4,5e-4} |
| `dpo_all_hy/<lang>_yaml/` + `run_all_dpo_hy.sh` | Hunyuan-MT-7B | DPO | full | 5e-6 |

Languages: `bodo, karbi, kokborok, nagamese, targin`.
Shared config: 4×GPU, bf16, cosine LR, DeepSpeed ZeRO-3.

## Helpers

- `evaluate_lr.py` / `dpo_all/evaluate_lr_dpo.py` — pick the best LR from a sweep.
- `dpo_all/fix_dpo_data.py` / `dpo_all_hy/fix_dpo_data_hy.py` — normalize the
  ShareGPT-DPO `from` field to `gpt` so the LLaMA-Factory loader parses it.
- `dpo_all/dataset_info.json` — LLaMA-Factory dataset registration
  (`formatting: sharegpt`, `ranking: true` for DPO).

## Order

1. Qwen LoRA SFT sweep → `evaluate_lr.py` selects best LR.
2. (optional) Merge best LoRA into base weights (see system-description §3.4).
3. Qwen LoRA DPO sweep (base = best SFT) → `evaluate_lr_dpo.py`.
4. Hunyuan full SFT → Hunyuan full DPO.
5. `_add` variants reuse the same configs on `<lang>_train_add` datasets.

Inference (to produce `<lang>_predicted.json`) is run with LLaMA-Factory batch
prediction on the held-out / official test prompts built in Step 1; the outputs
feed Step 3.
