# -*- coding: utf-8 -*-
"""
Generic single-file helper: convert ONE prediction JSON to a plain-text file,
one segment per line, strict id order (1..N), whitespace collapsed to single
spaces, no empty lines, LF endings.

For the full submission use make_all_submissions.py (it applies the plan and the
official file naming). This script is a thin utility for ad-hoc conversion /
inspection of a single prediction file.

    python prepare_wmt_submission.py --in predictions/hy_sft/karbi_predicted.json \
                                     --out karbi.txt
"""
import os
import re
import json
import argparse

_WS = re.compile(r"\s+", re.UNICODE)


def sanitize(text):
    return _WS.sub(" ", str(text)).strip()


def convert(in_path, out_path):
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = sorted(((int(o["id"]), sanitize(o.get("predict", ""))) for o in data),
                  key=lambda x: x[0])
    ids = [r[0] for r in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError(f"{in_path}: ids are not a contiguous 1..N sequence")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for _id, txt in rows:
            if not txt:
                raise ValueError(f"{in_path}: empty prediction at id {_id}")
            f.write(txt + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    n = convert(args.in_path, args.out_path)
    print(f"{n} lines -> {args.out_path}")


if __name__ == "__main__":
    main()
