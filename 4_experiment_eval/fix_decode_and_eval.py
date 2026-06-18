# -*- coding: utf-8 -*-
"""
Repetition-fix post-processing + re-evaluation for Devanagari (Bodo) predictions.

Many degenerate predictions "run away" -- the model loops / never stops, producing
outputs many times longer than the reference. This destroys corpus BLEU/ChrF/TER even
though most translations are fine. This script approximates the effect of decoding fixes
(no_repeat_ngram_size, repetition_penalty, max_new_tokens) by cleaning each prediction:

  1. no-repeat-ngram truncation : stop at the first n-gram that already appeared
                                  (what no_repeat_ngram_size would forbid -> the loop
                                  could not continue).
  2. collapse immediate token repeats (a a a -> a), optional.
  3. length cap                 : cap to max(ref_len * mult, ref_len + pad), bounded by
                                  an absolute cap (approximates max_new_tokens).

Then it re-scores BEFORE vs AFTER with the official 5-metric suite (BLEU/TER/ROUGE-L/
ChrF/METEOR), writes the cleaned predictions, and saves a comparison JSON.

NOTE: truncation is a *lower-bound* proxy for a real re-decode (which might emit a proper
sentence ending). Treat the AFTER numbers as a conservative estimate of the achievable gain.

Usage
-----
    python fix_decode_and_eval.py --pred gold_test/bodo/sft/generated_predictions.jsonl
    python fix_decode_and_eval.py --pred <file> --lang brx --no-repeat-n 3 \
                                  --len-cap-mult 2.0 --abs-cap 64 --collapse-immediate
"""
import os
import json
import argparse

# evaluate_bodo sets up UTF-8 stdout + nltk data on import; reuse its metric fns.
import sacrebleu
from indicnlp.tokenize import indic_tokenize
from evaluate_bodo import bleu_score, rouge_l, meteor, IndicNLPTokenizer, _iter_json_objects


def tok(text, lang):
    return indic_tokenize.trivial_tokenize(text.strip(), lang)


def fix_prediction(text, ref_len, lang, no_repeat_n, len_cap_mult, abs_cap,
                   collapse_immediate, pad=8):
    """Apply repetition-fix + length cap to a single prediction.

    Only degenerate predictions are altered. If nothing is truncated/collapsed, the
    ORIGINAL text is returned byte-identical -- this avoids a re-spacing artifact
    (re-joining tokens separates attached punctuation and would unfairly inflate
    TER/ChrF on already-clean outputs).
    """
    t = tok(text, lang)
    cap = int(min(abs_cap, max(ref_len * len_cap_mult, ref_len + pad)))
    seen = set()
    out = []
    truncated = False
    collapsed = False
    for i, w in enumerate(t):
        if len(out) >= cap:
            truncated = True
            break
        if collapse_immediate and out and w == out[-1]:
            collapsed = True
            continue  # drop immediate duplicate token
        if no_repeat_n and i >= no_repeat_n - 1:
            ng = tuple(t[i - no_repeat_n + 1:i + 1])
            if ng in seen:           # forbidden repeated n-gram -> stop here
                truncated = True
                break
            seen.add(ng)
        out.append(w)
    if not truncated and not collapsed:
        return text                  # clean prediction -> leave untouched
    return " ".join(out)


def load_pairs_with_extra(path):
    """Return (objs, preds, refs) keeping the original objects for re-writing."""
    with open(path, encoding="utf-8") as f:
        objs = list(_iter_json_objects(f.read()))
    preds = [str(o.get("predict", "")).strip() for o in objs]
    refs = [str(o.get("label", "")).strip() for o in objs]
    return objs, preds, refs


def score(preds, refs):
    bi, _ = bleu_score(preds, refs, "indic")
    b13, _ = bleu_score(preds, refs, "13a")
    ter = sacrebleu.TER().corpus_score(preds, [refs]).score
    rl = rouge_l(preds, refs, IndicNLPTokenizer())
    chrf = sacrebleu.CHRF().corpus_score(preds, [refs]).score
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    met = meteor(preds, refs)
    return {
        "BLEU_indicnlp": round(bi.score, 2),
        "BLEU_13a": round(b13.score, 2),
        "METEOR": round(met, 4),
        "ROUGE_L_F1": round(float(rl.fmeasure) * 100, 2),
        "ChrF": round(chrf, 2),
        "ChrF++": round(chrfpp, 2),
        "TER": round(ter, 2),
        "len_ratio": round(bi.ratio, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions jsonl (predict/label fields)")
    ap.add_argument("--lang", default="brx", help="IndicNLP language code (default brx=Bodo)")
    ap.add_argument("--no-repeat-n", type=int, default=3,
                    help="stop at first repeated n-gram (0 to disable)")
    ap.add_argument("--len-cap-mult", type=float, default=2.0,
                    help="length cap = ref_len * this")
    ap.add_argument("--abs-cap", type=int, default=64, help="absolute token cap")
    ap.add_argument("--collapse-immediate", action="store_true",
                    help="also collapse immediate duplicate tokens")
    ap.add_argument("--out-pred", default=None,
                    help="cleaned predictions jsonl (default: <pred>.fixed.jsonl)")
    ap.add_argument("--out-report", default=None,
                    help="comparison json (default: <dir>/fix_decode_eval.json)")
    args = ap.parse_args()

    objs, preds, refs = load_pairs_with_extra(args.pred)
    n = len(preds)

    fixed = [
        fix_prediction(p, len(tok(r, args.lang)), args.lang,
                       args.no_repeat_n, args.len_cap_mult, args.abs_cap,
                       args.collapse_immediate)
        for p, r in zip(preds, refs)
    ]

    n_changed = sum(1 for p, fp in zip(preds, fixed) if p.strip() != fp.strip())

    before = score(preds, refs)
    after = score(fixed, refs)

    # write cleaned predictions (preserve other fields)
    out_pred = args.out_pred or (os.path.splitext(args.pred)[0] + ".fixed.jsonl")
    with open(out_pred, "w", encoding="utf-8") as f:
        for o, fp in zip(objs, fixed):
            o2 = dict(o)
            o2["predict"] = fp
            f.write(json.dumps(o2, ensure_ascii=False) + "\n")

    report = {
        "pred_file": args.pred,
        "num_samples": n,
        "predictions_modified": n_changed,
        "params": {
            "lang": args.lang, "no_repeat_n": args.no_repeat_n,
            "len_cap_mult": args.len_cap_mult, "abs_cap": args.abs_cap,
            "collapse_immediate": args.collapse_immediate,
        },
        "before": before,
        "after_estimate": after,
        "delta": {k: round(after[k] - before[k], 2) for k in before},
        "note": "AFTER is a conservative (lower-bound) estimate via truncation; a real "
                "re-decode with repetition_penalty may score slightly higher.",
    }
    out_report = args.out_report or os.path.join(os.path.dirname(args.pred), "fix_decode_eval.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- print ----
    print("=" * 74)
    print(f"Repetition-fix re-eval  |  {args.pred}")
    print(f"samples={n}  modified={n_changed} ({100*n_changed/n:.1f}%)  "
          f"params: no_repeat_n={args.no_repeat_n} len_cap_mult={args.len_cap_mult} "
          f"abs_cap={args.abs_cap} collapse={args.collapse_immediate}")
    print("=" * 74)
    cols = ["BLEU_indicnlp", "BLEU_13a", "METEOR", "ROUGE_L_F1", "ChrF", "ChrF++", "TER", "len_ratio"]
    print(f"  {'metric':<14}{'BEFORE':>10}{'AFTER(est)':>12}{'delta':>10}")
    for c in cols:
        print(f"  {c:<14}{before[c]:>10}{after[c]:>12}{report['delta'][c]:>+10}")
    print(f"\ncleaned predictions -> {out_pred}")
    print(f"report              -> {out_report}")


if __name__ == "__main__":
    main()
