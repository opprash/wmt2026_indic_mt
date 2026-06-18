# -*- coding: utf-8 -*-
"""
Assemble the official WMT26 Indic-MT submission .txt files directly from the
original model-prediction JSON folders, following submission_plan.py:

  Contrastive system 1  -> submit_contrastive_1/
    bodo  : sft_test_predictions/bodo   (take-shorter + decode_fix, precomputed)
    others: sft_add_predictions/<lang>  (hy_sft_add)
  Contrastive system 2  -> submit_contrastive_2/
    all   : sft_predictions/<lang>      (hy_sft)

No primary system. Output filenames follow the official template
  <TEAM_NAME>_<SUBMISSION_TYPE>_<LANGUAGE_PAIR>.txt
with one segment per line, strict id order (txt line i == element id i), no empty
lines, internal whitespace collapsed to single spaces, LF endings.
"""
import os
import re
import json
import argparse

from submission_plan import (TEAM_NAME, LANG_TO_PAIR, LANGS, SLOTS,
                             SLOT_OUTDIR, source_path)

_WS = re.compile(r"\s+", re.UNICODE)


def sanitize(text):
    """Collapse all whitespace (internal newlines/tabs/multi-space) to a single
    space so each prediction occupies exactly one line."""
    return _WS.sub(" ", str(text)).strip()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_txt(items, out_path):
    """Write predictions ordered strictly by id, one sanitized segment per line."""
    rows = sorted(((int(o["id"]), sanitize(o.get("predict", ""))) for o in items),
                  key=lambda x: x[0])
    ids = [r[0] for r in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError(f"{out_path}: ids are not a contiguous 1..N sequence")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for _id, txt in rows:
            if not txt:
                raise ValueError(f"{out_path}: empty prediction at id {_id}")
            f.write(txt + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=TEAM_NAME)
    args = ap.parse_args()

    print("=" * 76)
    print(f"TEAM = {args.team}   (two contrastive systems, no primary)")
    print("=" * 76)
    total = 0
    for slot in SLOTS:
        out_dir = SLOT_OUTDIR[slot]
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n### {slot} -> {out_dir}/")
        for lang in LANGS:
            src = source_path(slot, lang)
            if not os.path.exists(src):
                print(f"  [skip] {lang}: missing source {src}")
                continue
            fname = f"{args.team}_{slot}_{LANG_TO_PAIR[lang]}.txt"
            n = write_txt(load_json(src), os.path.join(out_dir, fname))
            print(f"  {lang:<9} [{src:<34}] -> {fname:<40} ({n} lines)")
            total += 1
    print("\n" + "=" * 76)
    print(f"wrote {total} submission files")
    print("next: python verify_order_vs_excel.py  (check line order vs official test set)")


if __name__ == "__main__":
    main()
