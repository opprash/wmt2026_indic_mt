# -*- coding: utf-8 -*-
"""
Evaluate WMT TarGIN fine-tuning methods with the official 5-metric suite.

No official baseline comparison -- just compute BLEU-4 / TER / ROUGE-L / ChrF / METEOR
for each fine-tuning method and save the result into that method's own directory.

Metric implementations are imported from evaluate_bodo.py (same definitions as the
official WMT2025 evaluation).

Usage:
    python evaluate_wmt_methods.py [--root wmt]
"""
import os
import json
import argparse

# evaluate_bodo sets up UTF-8 stdout and nltk data on import; reuse its metric fns.
import sacrebleu
from evaluate_bodo import (
    load_pairs, bleu_score, rouge_l, meteor, IndicNLPTokenizer,
)

PRED_FILE = "generated_predictions.jsonl"
OUT_FILE = "official_metrics_eval.json"


def evaluate_one(method_dir):
    preds, refs = load_pairs(os.path.join(method_dir, PRED_FILE))
    n = len(preds)

    bleu_13a, sig_13a = bleu_score(preds, refs, "13a")
    bleu_indic, _ = bleu_score(preds, refs, "indic")
    ter = sacrebleu.TER().corpus_score(preds, [refs]).score
    rl_indic = rouge_l(preds, refs, IndicNLPTokenizer())
    rl_default = rouge_l(preds, refs, None)
    chrf = sacrebleu.CHRF().corpus_score(preds, [refs]).score
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    meteor_v = meteor(preds, refs)

    return {
        "num_samples": n,
        "direction": "en->bodo (brx)",
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
    ap.add_argument("--root", default="wmt", help="dir containing method subdirs")
    args = ap.parse_args()

    methods = sorted(
        d for d in os.listdir(args.root)
        if os.path.isfile(os.path.join(args.root, d, PRED_FILE))
    )
    if not methods:
        print("No method dirs with", PRED_FILE, "under", args.root)
        return

    summary = []
    for m in methods:
        mdir = os.path.join(args.root, m)
        print(f"Evaluating {m} ...", flush=True)
        res = evaluate_one(mdir)
        out_path = os.path.join(mdir, OUT_FILE)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("  saved ->", out_path)
        summary.append((m, res))

    # Cross-method summary table
    print("\n" + "=" * 96)
    print("WMT TarGIN fine-tuning methods  |  English -> Bodo (brx)  |  official 5-metric suite")
    print("=" * 96)
    hdr = (f"{'Method':<26}{'BLEU(13a)':>10}{'BLEU(idx)':>10}"
           f"{'METEOR':>8}{'ROUGE-L':>9}{'ChrF':>7}{'ChrF++':>8}{'TER':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m, r in summary:
        print(f"{m:<26}"
              f"{r['BLEU-4']['official_comparable_13a']:>10.2f}"
              f"{r['BLEU-4']['indicnlp_tokenized']:>10.2f}"
              f"{r['METEOR']:>8.3f}"
              f"{r['ROUGE-L_F1']['indicnlp_tokenized_x100']:>9.2f}"
              f"{r['ChrF']:>7.2f}"
              f"{r['ChrF++']:>8.2f}"
              f"{r['TER']:>8.2f}")
    print("-" * len(hdr))
    print("ROUGE-L shown = IndicNLP-tokenized F1 (meaningful). BLEU(idx)=IndicNLP tok. "
          "TER lower is better.")
    print(f"\nSamples per method: {summary[0][1]['num_samples']}")


if __name__ == "__main__":
    main()
