# Step 1 — Data preparation & custom bilingual vocabulary

Builds the fine-tuning datasets with **Top-3 vocabulary-hint prompts** from the
official Category-2 corpus (`category2_official/`) and, separately, from the
official corpus merged with external public corpora (`category2_augmented/`,
the `_add` variant).

The two sub-folders run the **same pipeline** on different source corpora; their
core scripts are identical except for input/output paths. `category2_augmented/`
adds a few scripts for building the final WMT26 test-set prompts.

## Pipeline (run in order)

| # | Script | Input | Output | Notes |
|---|--------|-------|--------|-------|
| 1 | `build_vocab_alignment.py` | `words_dictionary.json` + 5 en–X Excel | `sentence_vocab/<lang>_vocab.json` | sentence-level dictionary matching (pandas/openpyxl) |
| 2 | `build_word_alignment.py` | `sentence_vocab/*` | `word_vocab/<lang>_vocab_word.json` | **calls qwen3-max** (DashScope); needs `DASHSCOPE_API_KEY`; multithreaded, resumable |
| 3 | `build_finetune_dataset.py` | `word_vocab/*` + Excel/JSON | `dataset/<lang>_{train,test}.json` | injects Top-3 vocab hints; train/test split |
| 4 | `merge_train_test.py` | `dataset/<lang>_{train,test}.json` | `dataset_all/<lang>.json` | merge full set |
| – | `build_gold_test.py` / `build_gold_dataset.py` | WMT2025 gold Excel + `word_vocab/*` | gold test JSON | only Bodo/Kokborok have public gold |
| – | `recover_blocked_words.py` | `sentence_vocab/*` + `word_vocab/*` | updates `word_vocab/*` | retries words blocked by content filter |
| – | `category2_augmented/build_final_testset*.py` | official WMT26 test Excel + `word_vocab/*` | test prompts JSON | builds the prompts actually translated for submission |

## Key constants (parameterize as needed)

- `MIN_PROB = 0.5` — minimum probability to keep a vocab candidate
- `TOP_K = 3` — candidates shown per source word
- `MAX_SAMPLES = 10` — max sentence pairs sent to the LLM per word

## Secrets & paths

- Set `DASHSCOPE_API_KEY` in the environment (no key is committed; the fallback
  default is empty and the script will fail fast if unset).
- Input Excel paths and output directories are defined near the top of each
  script — adjust to your local layout.

## `process_*_and_dataset_all.txt`

Verbatim run logs of the original interactive build sessions (kept as a
reproduction reference: exact commands, failures, and recovery steps). Secrets
in these logs have been redacted.
