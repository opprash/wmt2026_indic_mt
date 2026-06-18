# Step 5 — Final submission prediction & processing

Assembles the official WMT26 submission `.txt` files directly from the original
model-prediction JSON folders, then verifies ordering before zipping.

## Submission plan (two contrastive systems, no primary)

Defined once in **`submission_plan.py`** and imported by the other scripts:

| Language | Contrastive system 1 → `submit_contrastive_1/` | Contrastive system 2 → `submit_contrastive_2/` |
|----------|----------------------|----------------------|
| bodo | `sft_test_predictions/bodo` (take-shorter + decode_fix) | `sft_predictions/bodo` (hy_sft) |
| karbi / kokborok / nagamese / targin | `sft_add_predictions/<lang>` (hy_sft_add) | `sft_predictions/<lang>` (hy_sft) |

All systems use the pretrained Hunyuan-MT-7B base -> none is a *primary*; we submit
only the two permitted contrastive systems per language pair.

## Inputs (kept in git for reproducibility)

The three original model-prediction JSON folders (`id` + `predict` per element),
produced by Step 3 (`../3_prediction/predict_final_test.py`). These are committed
so the exact submission can be regenerated; the assembled `submit_contrastive_*`
outputs are git-ignored (regenerable from these):

- `sft_predictions/<lang>_predicted.json`      — hy_sft (= contrastive 2, all langs)
- `sft_add_predictions/<lang>_predicted.json`  — hy_sft_add (= contrastive 1, non-bodo)
- `sft_test_predictions/bodo_predicted.json`   — Bodo take-shorter+decode_fix (= contrastive 1, bodo)

`sft_test_predictions` only contains Bodo. Build it with `short_and_fix.py` from
the Bodo `sft_add` and `sft` predictions.

## Flow

```bash
# (optional) build the Bodo take-shorter + decode_fix prediction:
python short_and_fix.py --file-add sft_add_predictions/bodo_predicted.json \
                        --file-sft sft_predictions/bodo_predicted.json \
                        --out sft_test_predictions/bodo_predicted.json

python make_all_submissions.py           # 3 JSON folders -> submit_contrastive_1/ , submit_contrastive_2/
python verify_order_vs_excel.py --excel-base "/path/to/Category 2"
```

## Scripts

| Script | Purpose |
|--------|---------|
| `submission_plan.py` | Single source of truth: team name, language->pair codes, and which JSON folder feeds each contrastive slot. |
| `short_and_fix.py` | `take_shorter_and_fix(...)` — per-id take-shorter merge + decode_fix; produces `sft_test_predictions/bodo` for Bodo contrastive 1. |
| `make_all_submissions.py` | Reads the three JSON folders -> writes `submit_contrastive_{1,2}/<TEAM>_<slot>_<pair>.txt`. Strict id order (line i == id i), one segment/line, whitespace collapsed, LF endings, official naming. |
| `prepare_wmt_submission.py` | Generic single-file JSON->txt helper (ad-hoc inspection). |
| `verify_order_vs_excel.py` | (A) prediction id <-> official `en-XX Test.xlsx` source alignment; (B) submission txt integrity vs the source JSON. Run before zipping. |

## Output & packaging

`submit_contrastive_1/*.txt` and `submit_contrastive_2/*.txt` have unique
filenames, so they go flat into one `<TEAM_NAME>.zip` together with the mandatory
abstract system-description file.
