# -*- coding: utf-8 -*-
"""
Reference-free (Quality-Estimation) evaluation of the trained-model predictions —
NO reference translations and NO semantic models (no embeddings / COMET / LLM).

Scores each system's predictions with purely surface/statistical signals that
correlate with degenerate or low-quality MT, then aggregates per system and per
language. This is the evidence behind the submission plan (hy_sft vs hy_sft_add).

Systems evaluated  : hy_sft, hy_sft_add   (predictions/<system>/<lang>_predicted.json)
Languages          : bodo, karbi, kokborok, nagamese, targin

Metrics (all reference-free, all surface-level)
  distinct_1/2/3 / rep3_rate / immediate_dup_rate / ttr / char_runaway  -> fluency
  length_ratio / pct_too_long / pct_too_short / empty_rate              -> length adequacy
  src_copy_rate / pct_untranslated                                      -> non-translation
  QUALITY_SCORE (0-100) = 100*(0.40*fluency + 0.30*len_adequacy
                                + 0.20*translation + 0.10*completeness)
Weights are constants below so the score can be re-weighted without touching the
metric code.
"""
import os
import re
import json
import argparse

# self-contained (no cross-stage import). The two trained systems map to the
# committed prediction folders under Step 5; default root points there so this
# runs out-of-the-box against the repo data.
SYSTEMS = ["hy_sft", "hy_sft_add"]
# Default to the romanized (Latin-script) languages: the whitespace/\w tokenizer
# below is valid for them. Bodo is Devanagari — this tokenizer over-splits its
# combining marks, so its length/diversity metrics are unreliable here (use an
# Indic tokenizer for Bodo). Bodo's submission decision uses take-shorter, not
# this hy_sft-vs-hy_sft_add comparison, so it is excluded by default; add it
# explicitly with `--langs bodo ...` if you supply Indic tokenization.
LANGS = ["karbi", "kokborok", "nagamese", "targin"]
PRED_ROOT = os.path.join("..", "5_final_submission")
SYSTEM_DIR = {"hy_sft": "sft_predictions", "hy_sft_add": "sft_add_predictions"}


def pred_path(system, lang, root=PRED_ROOT):
    return os.path.join(root, SYSTEM_DIR[system], f"{lang}_predicted.json")


W_FLUENCY, W_LENADEQ, W_TRANSL, W_COMPLETE = 0.40, 0.30, 0.20, 0.10
LR_LO, LR_HI = 0.6, 1.6
LR_MIN, LR_MAX = 0.2, 3.5
COPY_FREE, COPY_ZERO = 0.15, 0.65

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'/][^\W_]+)*|[^\s\w]", re.UNICODE)


def tok(text):
    return _TOKEN_RE.findall(str(text).strip())


def get_hyp(obj):
    for fld in ("predict", "tagin", "prediction"):
        if fld in obj:
            return str(obj[fld]).strip()
    raise KeyError(f"no prediction field in keys={list(obj.keys())}")


def get_src(obj):
    if obj.get("english"):
        return str(obj["english"]).strip()
    ins = str(obj.get("instruction", ""))
    return ins.split("\n\n")[-1].strip() if ins else ""


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def char_runaway(tokens):
    if not tokens:
        return 0.0
    w = max(tokens, key=len)
    if len(w) <= 25:
        return 0.0
    c3 = [w[i:i + 3] for i in range(len(w) - 2)]
    if not c3:
        return 0.0
    comp = 1.0 - len(set(c3)) / len(c3)
    return min(1.0, comp * (len(w) / 60.0))


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def len_adequacy(ratio):
    if LR_LO <= ratio <= LR_HI:
        return 1.0
    if ratio < LR_LO:
        return clamp01((ratio - LR_MIN) / (LR_LO - LR_MIN))
    return clamp01((LR_MAX - ratio) / (LR_MAX - LR_HI))


def score_segment(hyp, src):
    h, s = tok(hyp), tok(src)
    N, src_tokens = len(h), len(s)
    if N == 0:
        return {"length_ratio": 0.0, "distinct_1": 0.0, "distinct_2": 0.0,
                "distinct_3": 0.0, "rep3_rate": 1.0, "immediate_dup_rate": 0.0,
                "ttr": 0.0, "char_runaway": 0.0, "src_copy_rate": 0.0,
                "empty": 1, "too_long": 0, "too_short": 1, "untranslated": 0,
                "quality": 0.0}
    d1, d2, d3 = distinct_n(h, 1), distinct_n(h, 2), distinct_n(h, 3)
    dup = sum(1 for i in range(1, N) if h[i] == h[i - 1]) / N
    ttr = len(set(h)) / N
    crun = char_runaway(h)
    ratio = N / src_tokens if src_tokens else 0.0
    src_set = set(w.lower() for w in s)
    copy = (sum(1 for w in h if w.lower() in src_set) / N) if src_set else 0.0
    fluency = clamp01(0.5 * d2 + 0.5 * d3 - 0.5 * dup - crun)
    ladeq = len_adequacy(ratio) if src_tokens else 0.5
    transl = 1.0 - clamp01((copy - COPY_FREE) / (COPY_ZERO - COPY_FREE))
    quality = 100.0 * (W_FLUENCY * fluency + W_LENADEQ * ladeq +
                       W_TRANSL * transl + W_COMPLETE * 1.0)
    return {"length_ratio": ratio, "distinct_1": d1, "distinct_2": d2,
            "distinct_3": d3, "rep3_rate": 1.0 - d3, "immediate_dup_rate": dup,
            "ttr": ttr, "char_runaway": crun, "src_copy_rate": copy,
            "empty": 0, "too_long": int(ratio > 3.0), "too_short": int(ratio < 0.3),
            "untranslated": int(copy > 0.6), "quality": quality}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(seg):
    n = len(seg)
    if n == 0:
        return {}
    km = ["length_ratio", "distinct_1", "distinct_2", "distinct_3", "rep3_rate",
          "immediate_dup_rate", "ttr", "src_copy_rate", "quality"]
    a = {f"mean_{k}": round(mean([m[k] for m in seg]), 4) for k in km}
    a["n"] = n
    a["empty_rate"] = round(mean([m["empty"] for m in seg]), 4)
    a["pct_too_long"] = round(mean([m["too_long"] for m in seg]), 4)
    a["pct_too_short"] = round(mean([m["too_short"] for m in seg]), 4)
    a["pct_untranslated"] = round(mean([m["untranslated"] for m in seg]), 4)
    a["char_runaway_rate"] = round(mean([1 if m["char_runaway"] > 0 else 0 for m in seg]), 4)
    a["QUALITY_SCORE"] = a.pop("mean_quality")
    return a


def eval_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return aggregate([score_segment(get_hyp(o), get_src(o)) for o in data])


COLS = ["QUALITY_SCORE", "mean_distinct_2", "mean_distinct_3", "mean_rep3_rate",
        "mean_immediate_dup_rate", "mean_ttr", "mean_length_ratio",
        "mean_src_copy_rate", "empty_rate", "pct_too_long", "pct_too_short",
        "pct_untranslated", "char_runaway_rate"]
SHORT = {"QUALITY_SCORE": "QUALITY", "mean_distinct_2": "dist2",
         "mean_distinct_3": "dist3", "mean_rep3_rate": "rep3",
         "mean_immediate_dup_rate": "imdup", "mean_ttr": "ttr",
         "mean_length_ratio": "lenR", "mean_src_copy_rate": "copy",
         "empty_rate": "empty", "pct_too_long": "long", "pct_too_short": "short",
         "pct_untranslated": "untr", "char_runaway_rate": "crun"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--pred-root", default=PRED_ROOT,
                    help="root holding <system>/<lang>_predicted.json")
    ap.add_argument("--out", default="qe_eval_report.json")
    args = ap.parse_args()

    report = {"metric_type": "reference-free, surface/statistical (non-semantic)",
              "systems": {}}
    for sysname in args.systems:
        per_lang = {}
        for lang in args.langs:
            path = pred_path(sysname, lang, args.pred_root)
            if not os.path.exists(path):
                print(f"  [skip] missing {path}")
                continue
            per_lang[lang] = eval_file(path)
        if per_lang:
            keys = [k for k in next(iter(per_lang.values())) if k != "n"]
            agg = {k: round(mean([per_lang[l][k] for l in per_lang]), 4) for k in keys}
            agg["n_total"] = sum(per_lang[l]["n"] for l in per_lang)
        else:
            agg = {}
        report["systems"][sysname] = {"per_language": per_lang, "system_aggregate": agg}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print("Reference-free QE evaluation (no references, non-semantic)")
    print("=" * 92)
    for sysname in args.systems:
        block = report["systems"].get(sysname, {})
        per_lang = block.get("per_language", {})
        if not per_lang:
            continue
        print(f"\n### system: {sysname}")
        hdr = f"  {'lang':<10}" + "".join(f"{SHORT[c]:>9}" for c in COLS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for lang in args.langs:
            if lang in per_lang:
                a = per_lang[lang]
                print(f"  {lang:<10}" + "".join(f"{a.get(c, 0):>9}" for c in COLS))
        sa = block["system_aggregate"]
        print("  " + "-" * (len(hdr) - 2))
        print(f"  {'SYS_AVG':<10}" + "".join(f"{sa.get(c, 0):>9}" for c in COLS))

    print("\n  per-language QUALITY_SCORE  (hy_sft_add vs hy_sft):")
    sub = f"  {'lang':<10}" + "".join(f"{s:>14}" for s in args.systems) + f"{'better':>12}"
    print(sub)
    print("  " + "-" * (len(sub) - 2))
    for lang in args.langs:
        cells, scores = "", {}
        for s in args.systems:
            v = report["systems"][s]["per_language"].get(lang, {}).get("QUALITY_SCORE", None)
            scores[s] = v
            cells += f"{v if v is not None else '-':>14}"
        valid = {s: v for s, v in scores.items() if v is not None}
        best = max(valid, key=valid.get) if valid else "-"
        print(f"  {lang:<10}{cells}{best:>12}")
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
