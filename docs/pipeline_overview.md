# Pipeline overview & experiment matrix

## Experiment matrix (8 training experiments)

| ID | Base model | Data | Stage | Tuning |
|----|------------|------|-------|--------|
| A | Qwen2.5-32B-Instruct | official | SFT | LoRA |
| B | Qwen2.5-32B-Instruct | official | SFT→merge→DPO | LoRA |
| C | Qwen2.5-32B-Instruct | official+external (add) | SFT | LoRA |
| D | Qwen2.5-32B-Instruct | official | DPO (pure) | LoRA |
| E (hy_sft) | Hunyuan-MT-7B | official | SFT | full |
| F (hy_sft_add) | Hunyuan-MT-7B | official+external (add) | SFT | full |
| G (dpo_all_hy) | Hunyuan-MT-7B | official | SFT→DPO | full |
| Ensemble | Hunyuan-MT-7B | — | take-shorter merge + decode_fix | post-hoc |

Evaluated on the WMT2025 official gold test where available (Bodo, Kokborok;
2000 sentences/pair). Karbi/Nagamese/Tagin lack public official test sets and
were compared via internal held-out splits and training logs.

## Final submission plan (two contrastive systems, no primary)

Because every system relies on **pretrained LLMs** (Hunyuan-MT-7B; external
public data), none qualifies as a constrained primary; therefore we submit
**two contrastive systems per language pair and no primary**.

| Language | contrastive1 | contrastive2 |
|----------|--------------|--------------|
| Bodo (en_to_bodo) | take-shorter merge + decode_fix | sft_test |
| Kokborok (en_to_trp) | sft_add | sft (baseline) |
| Karbi (en_to_mjw) | sft_add | sft (baseline) |
| Nagamese (en_to_nag) | sft_add | sft (baseline) |
| Tagin (en_to_tgj) | sft_add | sft (baseline) |

File naming: `星辰之力_contrastive1_en_to_<code>.txt`,
`星辰之力_contrastive2_en_to_<code>.txt`.[targin_predicted.json](../5_final_submission/sft_add_predictions/targin_predicted.json)

## Key findings

- **Hunyuan-MT-7B full SFT > Qwen2.5-32B LoRA SFT** on the gold tests
  (Bodo ChrF 60.59 vs 48.72).
- **Decoder-side length/repetition control is the main bottleneck**: ~7% of
  predictions show repetition degeneration; decode-fix + take-shorter merge
  raised Bodo BLEU(13a) from 19.27 to 27.18.
- **Data augmentation is language-dependent**: `sft_add` helps Kokborok
  (+6.67 BLEU) but regresses Bodo — augment per language, not uniformly.
- **Custom vocabulary injection** gives net gains (Tagin 8:2 pre-experiment:
  +7.09 BLEU, +5.97 ChrF, −51.56 TER).
