# -*- coding: utf-8 -*-
"""
Take-shorter merge + decode-fix.

Used ONLY for the Bodo contrastive-system-1: for each id, pick the shorter of the
two decoding versions (hy_sft_add vs hy_sft), then apply a decode-fix
(no-repeat-ngram truncation + length cap) that approximates
`no_repeat_ngram_size` / `max_new_tokens` decoding.

Exposes `take_shorter_and_fix(...)` for import (used by make_all_submissions.py)
and a CLI for standalone use / inspection.

Bodo is Devanagari, so the IndicNLP `brx` tokenizer is used.
"""
import os
import json
import argparse

from indicnlp.tokenize import indic_tokenize


def tok(text, lang="brx"):
    return indic_tokenize.trivial_tokenize(str(text).strip(), lang)


def fix_prediction(text, ref_len, lang="brx", no_repeat_n=3, len_cap_mult=2.0,
                   abs_cap=64, collapse_immediate=False, pad=8):
    """Decode-fix a single prediction. Returns the ORIGINAL text byte-identical
    when nothing is truncated/collapsed (avoids a re-spacing artifact)."""
    t = tok(text, lang)
    cap = int(min(abs_cap, max(ref_len * len_cap_mult, ref_len + pad)))
    seen, out, truncated, collapsed = set(), [], False, False
    for i, w in enumerate(t):
        if len(out) >= cap:
            truncated = True
            break
        if collapse_immediate and out and w == out[-1]:
            collapsed = True
            continue
        if no_repeat_n and i >= no_repeat_n - 1:
            ng = tuple(t[i - no_repeat_n + 1:i + 1])
            if ng in seen:
                truncated = True
                break
            seen.add(ng)
        out.append(w)
    if not truncated and not collapsed:
        return text
    return " ".join(out)


def take_shorter_and_fix(items_a, items_b, lang="brx", no_repeat_n=3,
                         len_cap_mult=2.0, abs_cap=64):
    """Merge two prediction lists by taking the shorter prediction per id, then
    decode-fix the chosen text.

    items_a, items_b : lists of dicts with 'id' and 'predict' (e.g. hy_sft_add, hy_sft)
    Returns the FULL records from items_a (id/instruction/input/output/...) with only
    `predict` replaced by the take-shorter + decode-fixed text, sorted by id.
    Preserving the other fields keeps the output identical in shape to the raw
    prediction files, so downstream verify/source-alignment keeps working. Ties
    prefer items_a. The length cap uses the chosen prediction's own length (no gold
    reference), so only repetition triggers truncation.
    """
    base = {o["id"]: o for o in items_a}                # full records (instruction, ...)
    a = {o["id"]: str(o.get("predict", "")).strip() for o in items_a}
    b = {o["id"]: str(o.get("predict", "")).strip() for o in items_b}
    ids = sorted(set(a) & set(b))
    merged = []
    for _id in ids:
        ta, tb = a[_id], b[_id]
        chosen = ta if len(tok(ta, lang)) <= len(tok(tb, lang)) else tb
        ref_len = len(tok(chosen, lang))
        fixed = fix_prediction(chosen, ref_len, lang, no_repeat_n,
                               len_cap_mult, abs_cap)
        rec = dict(base[_id])                           # keep all original fields
        rec["predict"] = fixed                          # replace only predict
        merged.append(rec)
    return merged


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-add", required=True, help="hy_sft_add predictions json")
    ap.add_argument("--file-sft", required=True, help="hy_sft predictions json")
    ap.add_argument("--lang", default="brx")
    ap.add_argument("--out", required=True, help="output merged+fixed json")
    args = ap.parse_args()

    merged = take_shorter_and_fix(_load(args.file_add), _load(args.file_sft), args.lang)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"take-shorter + decode_fix: {len(merged)} segments -> {args.out}")


if __name__ == "__main__":
    main()
