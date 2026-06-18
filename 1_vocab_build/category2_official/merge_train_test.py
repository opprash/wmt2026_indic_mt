# -*- coding: utf-8 -*-
"""
合并每个语种的 train + test → dataset_all/<lang>.json（全量数据）

输入：dataset/<lang>_train.json + dataset/<lang>_test.json
输出：dataset_all/<lang>.json
"""

import json
import os
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(SCRIPT_DIR, "dataset")
DST_DIR     = os.path.join(SCRIPT_DIR, "dataset_all")

LANGS = ["karbi", "targin", "bodo", "kokborok", "nagamese"]


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    total_all = 0
    for lang in LANGS:
        train_p = os.path.join(SRC_DIR, f"{lang}_train.json")
        test_p  = os.path.join(SRC_DIR, f"{lang}_test.json")
        if not (os.path.exists(train_p) and os.path.exists(test_p)):
            print(f"[skip] {lang}: missing source files")
            continue
        with open(train_p, "r", encoding="utf-8") as f:
            train = json.load(f)
        with open(test_p, "r", encoding="utf-8") as f:
            test = json.load(f)
        merged = train + test
        dst = os.path.join(DST_DIR, f"{lang}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        total_all += len(merged)
        print(f"{lang:10s}  train={len(train):6d}  test={len(test):5d}  total={len(merged):6d}  -> {dst}")
    print(f"\nGrand total across all languages: {total_all}")


if __name__ == "__main__":
    main()
