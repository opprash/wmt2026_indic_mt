# -*- coding: utf-8 -*-
"""
Dedicated evaluation for wmt/new_model/predictions_result.jsonl.

This file uses a DIFFERENT schema from the other prediction files:
    {"english_input", "predicted_translation", "original_label", "prompt_truncated"}
(vs the LLaMA-Factory style {"prompt", "predict", "label"}).

It also outputs ROMANIZED (Latin-script) translations rather than Devanagari, so the
official-comparable BLEU here is the standard 13a tokenization; the IndicNLP-tokenized
BLEU/ROUGE-L variants degrade to whitespace splitting and are reported only for parity
with the other methods.

Metrics (same definitions as evaluate_bodo.py / the official WMT2025 suite):
    BLEU-4 (case-insensitive) | TER | ROUGE-L (F1) | ChrF / ChrF++ | METEOR

Usage:
    python evaluate_new_model.py [--pred wmt/new_model/predictions_result.jsonl]
"""
import os
import json
import argparse

# evaluate_bodo sets up UTF-8 stdout and nltk data on import; reuse its metric fns.
import sacrebleu
from evaluate_bodo import (
    bleu_score, rouge_l, meteor, IndicNLPTokenizer,
)

DEFAULT_PRED = os.path.join("wmt", "new_model", "predictions_result.jsonl")
OUT_FILE = "official_metrics_eval.json"

# Field names for THIS schema
PRED_KEY = "predicted_translation"
REF_KEY = "original_label"


def load_pairs_new_model(path):
    """Load (prediction, reference) pairs from the new_model schema."""
    preds, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            preds.append(str(o.get(PRED_KEY, "")).strip())
            refs.append(str(o.get(REF_KEY, "")).strip())
    return preds, refs


def evaluate(preds, refs):
    bleu_13a, sig_13a = bleu_score(preds, refs, "13a")
    bleu_indic, _ = bleu_score(preds, refs, "indic")
    ter = sacrebleu.TER().corpus_score(preds, [refs]).score
    rl_indic = rouge_l(preds, refs, IndicNLPTokenizer())
    rl_default = rouge_l(preds, refs, None)
    chrf = sacrebleu.CHRF().corpus_score(preds, [refs]).score
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    meteor_v = meteor(preds, refs)

    return {
        "num_samples": len(preds),
        "schema": "english_input / predicted_translation / original_label",
        "script": "romanized (Latin) -- 13a is the appropriate primary tokenization",
        "BLEU-4": {
            "official_comparable_13a": round(bleu_13a.score, 2),
            "indicnlp_tokenized": round(bleu_indic.score, 2),
            "signature_13a": sig_13a,
            "precisions_13a": [round(x, 2) for x in bleu_13a.precisions],
            "bp_13a": round(bleu_13a.bp, 4),
            "len_ratio_13a": round(bleu_13a.ratio, 4),
        },
        "TER": round(ter, 2),
        "ROUGE-L_F1": {
            "indicnlp_tokenized_x100": round(rl_indic.fmeasure * 100, 2),
            "indicnlp_P_x100": round(rl_indic.precision * 100, 2),
            "indicnlp_R_x100": round(rl_indic.recall * 100, 2),
            "default_ascii_0to1": round(rl_default.fmeasure, 4),
        },
        "ChrF": round(chrf, 2),
        "ChrF++": round(chrfpp, 2),
        "METEOR": round(meteor_v, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--out", default=None,
                    help="output path; default: <pred_dir>/official_metrics_eval.json")
    args = ap.parse_args()

    preds, refs = load_pairs_new_model(args.pred)
    res = evaluate(preds, refs)

    out_path = args.out or os.path.join(os.path.dirname(args.pred), OUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"new_model evaluation  |  samples = {res['num_samples']}  (romanized)")
    print("=" * 70)
    b = res["BLEU-4"]
    print(f"  BLEU-4   : {b['official_comparable_13a']:.2f}  (13a, case-insensitive)")
    print(f"             precisions={b['precisions_13a']} "
          f"BP={b['bp_13a']} len_ratio={b['len_ratio_13a']}")
    print(f"  METEOR   : {res['METEOR']:.4f}")
    print(f"  ROUGE-L  : {res['ROUGE-L_F1']['indicnlp_tokenized_x100']:.2f}  (F1)")
    print(f"  ChrF     : {res['ChrF']:.2f}   (chrF++ = {res['ChrF++']:.2f})")
    print(f"  TER      : {res['TER']:.2f}   (lower is better)")
    print("\nSaved ->", out_path)


if __name__ == "__main__":
    main()
