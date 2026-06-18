# -*- coding: utf-8 -*-
"""
Verify submission ordering is correct before zipping.

Two independent checks per language and slot:

  (A) Source alignment — the prediction source JSON (source_path(slot, lang)) has
      ids 1..N and its attached English source matches the official
      `en-XX Test.xlsx` row with the same S No. Proves our id numbering equals
      the official test-set order.

  (B) Submission txt integrity — the submission file
      (submit_contrastive_{1,2}/<TEAM>_<slot>_<pair>.txt) has exactly N lines,
      no empty lines, no CR, and line i equals the sanitized prediction for id i
      from the source JSON.

Requires the official test Excel directory (Category 2); point --excel-base at it.
"""
import os
import re
import json
import argparse

import openpyxl

from submission_plan import (TEAM_NAME, LANG_TO_PAIR, LANGS, SLOTS,
                             SLOT_OUTDIR, source_path)
from make_all_submissions import sanitize

EXCEL = {
    "bodo":     ("English - Bodo",     "en-bodo Test.xlsx"),
    "karbi":    ("English - Karbi",    "en-mjw Test.xlsx"),
    "kokborok": ("English - Kokborok", "en-trp Test.xlsx"),
    "nagamese": ("English - Nagamese", "en-nag Test.xlsx"),
    "targin":   ("English - Tagin",    "en-tgj Test.xlsx"),
}

_WS = re.compile(r"\s+", re.UNICODE)


def norm(s):
    return _WS.sub(" ", str(s)).strip()


def get_src(obj):
    if obj.get("english"):
        return norm(obj["english"])
    ins = str(obj.get("instruction", ""))
    return norm(ins.split("\n\n")[-1]) if ins else ""


def load_excel(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    out = {}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[0] is not None:
            out[int(r[0])] = norm(r[1])
    wb.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel-base", default=os.path.join(
        "..", "..", "final_test", "WMT 2026 IndicMT Test Data", "Category 2"))
    ap.add_argument("--langs", nargs="+", default=LANGS)
    args = ap.parse_args()

    overall_ok = True
    for lang in args.langs:
        print(f"\n===== {lang} =====")
        subdir, xlsx = EXCEL[lang]
        xpath = os.path.join(args.excel_base, subdir, xlsx)
        excel = load_excel(xpath) if os.path.exists(xpath) else None
        if excel is None:
            print(f"  [warn] official Excel not found: {xpath} (skip source-alignment)")

        for slot in SLOTS:
            src = source_path(slot, lang)
            fname = f"{TEAM_NAME}_{slot}_{LANG_TO_PAIR[lang]}.txt"
            tpath = os.path.join(SLOT_OUTDIR[slot], fname)
            if not os.path.exists(src):
                print(f"  [skip] {slot}: missing source {src}")
                continue
            data = json.load(open(src, encoding="utf-8"))
            expect = {int(o["id"]): sanitize(o["predict"]) for o in data}

            # (A) source alignment
            ids = [int(o["id"]) for o in data]
            a_ok = ids == list(range(1, len(data) + 1))
            mism = 0
            if excel is not None:
                if len(excel) != len(data):
                    a_ok = False
                for o in data:
                    if get_src(o) != excel.get(int(o["id"])):
                        mism += 1
                a_ok = a_ok and mism == 0

            # (B) submission txt integrity
            b_ok, bnote = True, ""
            if not os.path.exists(tpath):
                b_ok, bnote = False, "txt missing"
            else:
                raw = open(tpath, "rb").read().decode("utf-8")
                lines = raw.split("\n")
                if lines and lines[-1] == "":
                    lines = lines[:-1]
                probs = []
                if "\r" in raw:
                    probs.append("CR")
                if len(lines) != len(expect):
                    probs.append(f"lines={len(lines)}!={len(expect)}")
                else:
                    bad = sum(1 for i, ln in enumerate(lines, 1)
                              if ln != expect.get(i) or ln.strip() == "")
                    if bad:
                        probs.append(f"{bad} mismatch/empty")
                b_ok = not probs
                bnote = ";".join(probs)

            overall_ok = overall_ok and a_ok and b_ok
            extra = f" src-mismatch={mism}" if excel is not None else " (no excel)"
            print(f"  {slot:<12} src=[{os.path.basename(src)}]  "
                  f"(A){'OK' if a_ok else 'FAIL'}{extra}  "
                  f"(B){'OK' if b_ok else 'FAIL:' + bnote}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if overall_ok else "SOME CHECKS FAILED — see above")


if __name__ == "__main__":
    main()
