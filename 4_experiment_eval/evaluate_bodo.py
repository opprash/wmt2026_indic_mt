# -*- coding: utf-8 -*-
"""
WMT2025 Low-Resource Indic  English -> Bodo (en->brx)  re-evaluation.

Implements the 5 official metrics, each faithful to its source paper:

  1. BLEU-4   (Papineni et al. 2002) : geometric mean of 1-4gram precision x BP,
                                       case-insensitive. sacreBLEU corpus_bleu.
                                       Reported under standard 13a tokenization
                                       (official-comparable) AND IndicNLP tokenization.
  2. TER      (Snover et al. 2006)   : min edit ops (ins/del/sub/shift) / ref words.
                                       sacreBLEU TER. lower = better.
  3. ROUGE-L  (Lin 2004)             : LCS-based precision/recall -> F1.
                                       Reported with IndicNLP tokenization (meaningful)
                                       AND rouge_score default ASCII tokenizer
                                       (replicates the official 0.168 artifact).
  4. ChrF     (Popovic 2015)         : char 1-6 gram F-score, tokenization-free.
                                       sacreBLEU CHRF (beta=2). Also chrF++ (word_order=2).
  5. METEOR   (Banerjee & Lavie 2005): unigram P/R + stem/synonym match + fragmentation
                                       penalty. nltk meteor_score on IndicNLP tokens.

Compares against WMT2025 official Table 15 (English -> Bodo) baselines.
"""
import io
import json
import sys
import argparse

# Force UTF-8 stdout (Windows console is GBK and chokes on Devanagari)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sacrebleu
import nltk
from indicnlp.tokenize import indic_tokenize
from rouge_score import rouge_scorer, scoring
from nltk.translate.meteor_score import meteor_score

for _pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

LANG = "brx"  # Bodo, Devanagari script

# ---- Official WMT2025 Table 15 (English -> Bodo) baselines -----------------
# BLEU/ChrF/TER on 0-100 scale; METEOR/ROUGE-L on 0-1 scale (as published).
OFFICIAL = [
    # team,                         BLEU,  METEOR, ROUGE-L, ChrF,  TER
    ("DoDS-IITPKD (contrastive)",   24.97, 0.519,  0.169,   67.81, 51.50),
    ("DoDS-IITPKD (primary)",       24.45, 0.513,  0.168,   67.71, 51.84),
    ("JU-NLP (primary)",            19.71, 0.455,  0.169,   62.47, 64.97),
    ("Transformers (contrastive)",  19.30, 0.452,  0.168,   67.29, 72.92),
    ("BilbaoMT (contrastive)",      10.18, 0.283,  0.160,   46.87, 71.09),
    ("DPKM (primary)",               4.38, 0.132,  0.009,   35.50, 92.56),
    ("BVSLP (primary)",              1.35, 0.040,  0.168,   17.05, 106.11),
    ("CITK_MT (primary)",            0.31, 0.019,  0.003,    7.24, 808.91),
    ("RBG-AI (contrastive)",         0.20, 0.006,  0.027,    0.81, 131.96),
]


def indic_tok(text):
    return indic_tokenize.trivial_tokenize(text.strip(), LANG)


class IndicNLPTokenizer:
    """rouge_score tokenizer adapter using IndicNLP segmentation."""
    def tokenize(self, text):
        return indic_tok(text)


def _iter_json_objects(text):
    """Yield JSON objects from JSONL, a JSON array, or concatenated/pretty-printed JSON."""
    text = text.strip()
    if not text:
        return
    if text[0] == "[":                      # JSON array
        for o in json.loads(text):
            yield o
        return
    dec = json.JSONDecoder()
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        o, end = dec.raw_decode(text, idx)  # one object, ignores trailing whitespace
        yield o
        idx = end


def load_pairs(path):
    """Robust to JSONL, JSON array, and concatenated pretty-printed JSON."""
    preds, refs = [], []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for o in _iter_json_objects(text):
        preds.append(str(o.get("predict", "")).strip())
        refs.append(str(o.get("label", "")).strip())
    return preds, refs


def bleu_score(preds, refs, tokenize):
    """Original Papineni et al. (2002) corpus BLEU-4, case-insensitive.

    Faithful to the 2002 definition: modified clipped n-gram precision for n=1..4
    (uniform weights), geometric mean, brevity penalty BP=exp(1-r/c) when c<=r,
    aggregated at the corpus level. smooth_method='none' => NO smoothing (the paper
    has none); verified to match a from-scratch Papineni implementation.
    tokenize='13a' (standard) or 'indic' (IndicNLP pre-tokenized, BLEU itself unchanged).
    """
    if tokenize == "indic":
        tp = [" ".join(indic_tok(p)) for p in preds]
        tr = [" ".join(indic_tok(r)) for r in refs]
        m = sacrebleu.BLEU(tokenize="none", lowercase=True, smooth_method="none")
        return m.corpus_score(tp, [tr]), str(m.get_signature())
    m = sacrebleu.BLEU(tokenize=tokenize, lowercase=True, smooth_method="none")
    return m.corpus_score(preds, [refs]), str(m.get_signature())


def rouge_l(preds, refs, tokenizer):
    """Sentence-avg ROUGE-L F1 (Lin 2004). tokenizer=IndicNLP adapter or None(default)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=tokenizer)
    agg = scoring.BootstrapAggregator()
    for p, r in zip(preds, refs):
        agg.add_scores(scorer.score(r, p))  # (target/ref, prediction)
    return agg.aggregate()["rougeL"].mid  # .precision/.recall/.fmeasure (0-1)


def meteor(preds, refs):
    """Corpus METEOR = mean of sentence METEOR (Banerjee & Lavie 2005), IndicNLP tokens."""
    total = 0.0
    for p, r in zip(preds, refs):
        total += meteor_score([indic_tok(r)], indic_tok(p))
    return total / len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="generated_predictions.jsonl")
    ap.add_argument("--existing", default="predict_results.json")
    ap.add_argument("--out", default="new_eval_results.json")
    args = ap.parse_args()

    preds, refs = load_pairs(args.pred)
    n = len(preds)

    # --- 1. BLEU-4 ---
    bleu_13a, sig_13a = bleu_score(preds, refs, "13a")
    bleu_indic, sig_indic = bleu_score(preds, refs, "indic")

    # --- 2. TER ---
    ter = sacrebleu.TER().corpus_score(preds, [refs]).score

    # --- 3. ROUGE-L ---
    rl_indic = rouge_l(preds, refs, IndicNLPTokenizer())   # meaningful (0-1)
    rl_default = rouge_l(preds, refs, None)                # official replica (0-1)

    # --- 4. ChrF / ChrF++ ---
    chrf = sacrebleu.CHRF().corpus_score(preds, [refs]).score
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score

    # --- 5. METEOR ---
    meteor_v = meteor(preds, refs)

    results = {
        "num_samples": n,
        "direction": "en->bodo (brx)",
        "BLEU-4": {
            "official_comparable_13a": round(bleu_13a.score, 2),
            "indicnlp_tokenized": round(bleu_indic.score, 2),
            "signature_13a": sig_13a,
            "precisions_13a": [round(x, 2) for x in bleu_13a.precisions],
            "bp_13a": round(bleu_13a.bp, 4),
            "len_ratio_13a": round(bleu_13a.ratio, 4),
            "note": "case-insensitive; 13a = official-comparable, indicnlp = Indic-proper",
        },
        "TER": {"score": round(ter, 2), "note": "lower is better; sacreBLEU TER"},
        "ROUGE-L_F1": {
            "indicnlp_tokenized_x100": round(rl_indic.fmeasure * 100, 2),
            "indicnlp_P_x100": round(rl_indic.precision * 100, 2),
            "indicnlp_R_x100": round(rl_indic.recall * 100, 2),
            "default_ascii_0to1": round(rl_default.fmeasure, 4),
            "note": "default_ascii replicates official artifact (strips Devanagari); "
                    "indicnlp is the meaningful value",
        },
        "ChrF": {"chrF": round(chrf, 2), "chrF++": round(chrfpp, 2),
                 "note": "tokenization-free char 1-6 grams, beta=2"},
        "METEOR": {"score": round(meteor_v, 4), "note": "IndicNLP tokens, 0-1 scale"},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- Report ----------------
    print("=" * 78)
    print(f"WMT2025  English -> Bodo (brx)  re-evaluation   |   samples = {n}")
    print("=" * 78)

    print("\n[Our system - official-metric suite]")
    print(f"  BLEU-4   : {results['BLEU-4']['official_comparable_13a']:.2f}  (13a, case-insensitive)"
          f"   | {results['BLEU-4']['indicnlp_tokenized']:.2f} (IndicNLP tok)")
    print(f"             precisions(13a)={results['BLEU-4']['precisions_13a']} "
          f"BP={results['BLEU-4']['bp_13a']} len_ratio={results['BLEU-4']['len_ratio_13a']}")
    print(f"  METEOR   : {results['METEOR']['score']:.4f}")
    print(f"  ROUGE-L  : {results['ROUGE-L_F1']['indicnlp_tokenized_x100']:.2f} (IndicNLP, meaningful)"
          f"   | {results['ROUGE-L_F1']['default_ascii_0to1']:.4f} (default-ASCII = official replica)")
    print(f"  ChrF     : {results['ChrF']['chrF']:.2f}   (chrF++ = {results['ChrF']['chrF++']:.2f})")
    print(f"  TER      : {results['TER']['score']:.2f}   (lower is better)")

    # Comparison table vs official Table 15
    print("\n[vs WMT2025 official Table 15  (English -> Bodo)]")
    hdr = f"  {'System':<28}{'BLEU':>7}{'METEOR':>8}{'ROUGE-L':>9}{'ChrF':>7}{'TER':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ours = (
        "** OUR SYSTEM **",
        results["BLEU-4"]["official_comparable_13a"],
        results["METEOR"]["score"],
        results["ROUGE-L_F1"]["default_ascii_0to1"],   # ASCII replica to match their column
        results["ChrF"]["chrF"],
        results["TER"]["score"],
    )
    rows = [ours] + OFFICIAL
    # sort by ChrF desc (most reliable, tokenization-free) for ranking context
    rows_sorted = sorted(rows, key=lambda r: r[4], reverse=True)
    for team, bleu, met, rl, chrf_v, ter_v in rows_sorted:
        mark = " <--" if team.startswith("**") else ""
        print(f"  {team:<28}{bleu:>7.2f}{met:>8.3f}{rl:>9.3f}{chrf_v:>7.2f}{ter_v:>8.2f}{mark}")

    # explicit gap vs last-year winner (DoDS-IITPKD contrastive)
    win = OFFICIAL[0]
    w_bleu, w_meteor, w_chrf, w_ter = win[1], win[2], win[4], win[5]
    print(f"\n[Gap vs last-year best ({win[0]})]")
    print(f"  ChrF  : ours {results['ChrF']['chrF']:.2f}  vs  {w_chrf:.2f}   "
          f"=> {results['ChrF']['chrF'] - w_chrf:+.2f}   (higher better)")
    print(f"  TER   : ours {results['TER']['score']:.2f}  vs  {w_ter:.2f}   "
          f"=> {results['TER']['score'] - w_ter:+.2f}   (lower better)")
    print(f"  METEOR: ours {results['METEOR']['score']:.3f}  vs  {w_meteor:.3f}   "
          f"=> {results['METEOR']['score'] - w_meteor:+.3f}   (higher better)")
    print(f"  BLEU  : ours {results['BLEU-4']['official_comparable_13a']:.2f}  vs  {w_bleu:.2f}   "
          f"=> {results['BLEU-4']['official_comparable_13a'] - w_bleu:+.2f}   (higher better)")

    print("\nNote: ROUGE-L official column is a tokenization artifact (default ASCII tokenizer")
    print("      strips all Devanagari). Use IndicNLP ROUGE-L "
          f"({results['ROUGE-L_F1']['indicnlp_tokenized_x100']:.2f}) for real overlap.")
    print("\nSaved ->", args.out)


if __name__ == "__main__":
    main()
