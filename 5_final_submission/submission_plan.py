# -*- coding: utf-8 -*-
"""
Single source of truth for the WMT26 Indic-MT submission plan (Team 星辰之力).

We submit **two contrastive systems per language pair and no primary system**
(all systems use the pretrained Hunyuan-MT-7B base, so none qualifies as a
constrained primary — see the system-description document).

Final plan — assembled directly from the original model-prediction JSON folders:

  Contrastive system 1  -> submit_contrastive_1/
    - bodo : sft_test_predictions/bodo   (take-shorter merge + decode_fix, precomputed)
    - others: sft_add_predictions/<lang> (hy_sft_add)
  Contrastive system 2  -> submit_contrastive_2/
    - all  : sft_predictions/<lang>      (hy_sft)

make_all_submissions.py and verify_order_vs_excel.py import from here so the plan
is defined exactly once.
"""
import os

TEAM_NAME = "星辰之力"

# language file prefix -> official WMT language-pair token
LANG_TO_PAIR = {
    "bodo":     "en_to_bodo",
    "karbi":    "en_to_mjw",
    "kokborok": "en_to_trp",
    "nagamese": "en_to_nag",
    "targin":   "en_to_tgj",
}
LANGS = list(LANG_TO_PAIR)

# original model-prediction JSON folders (DATA; produced by Step-3 inference,
# git-ignored — not part of the committed codebase)
DIR_SFT      = "sft_predictions"        # hy_sft                       -> contrastive 2
DIR_SFT_ADD  = "sft_add_predictions"    # hy_sft_add                   -> contrastive 1 (non-bodo)
DIR_SFT_TEST = "sft_test_predictions"   # bodo take-shorter+decode_fix -> contrastive 1 (bodo)

# submission slot -> (official SUBMISSION_TYPE token, output directory)
SLOT_OUTDIR = {
    "contrastive1": "submit_contrastive_1",
    "contrastive2": "submit_contrastive_2",
}
SLOTS = list(SLOT_OUTDIR)


def source_path(slot, lang):
    """Path of the prediction JSON feeding a (slot, language)."""
    if slot == "contrastive2":
        return os.path.join(DIR_SFT, f"{lang}_predicted.json")
    if slot == "contrastive1":
        if lang == "bodo":
            return os.path.join(DIR_SFT_TEST, "bodo_predicted.json")
        return os.path.join(DIR_SFT_ADD, f"{lang}_predicted.json")
    raise ValueError(f"unknown slot: {slot}")
