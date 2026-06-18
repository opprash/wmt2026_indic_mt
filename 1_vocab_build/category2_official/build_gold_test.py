# -*- coding: utf-8 -*-
"""
构造黄金测试集（gold test set）

输入：C:\\...\\gold_eval\\English-<Lang> WMT 2025 Test Set Gold.xlsx
  - 2 列：col0 = 英文源句，col1 = 目标语种翻译
  - 首行是表头（不一定含 "English" 字样，按固定首行 = header 跳过）

输出：gold_ecal_test/gold_test_<lang>.json
  - 格式与 dataset_all/<lang>.json 完全一致
  - 复用 word_vocab/<Lang>_vocab_word.json 注入词汇提示
  - 同样的 MIN_PROB / TOP_K 阈值（与 build_finetune_dataset.py 保持一致）

用法：
  python build_gold_test.py                  # 处理已配置的所有 gold 文件
  python build_gold_test.py --lang Bodo      # 只处理某一种
"""

import argparse
import json
import os
import sys

# 复用 build_finetune_dataset.py 里的 XLSX 解析器 + instruction 构造逻辑
from build_finetune_dataset import (
    _iter_xlsx_rows,
    build_instruction,
    MIN_PROB,
    TOP_K,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR   = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\gold_eval"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gold_ecal_test")

# 语种 → (Excel 文件名, word_vocab 文件名, 输出文件后缀)
GOLD_FILES = {
    "Bodo":     {
        "xlsx":  "English-Bodo WMT 2025 Test Set Gold.xlsx",
        "vocab": "Bodo_vocab_word.json",
        "out":   "gold_test_bodo.json",
    },
    "Kokborok": {
        # 文件名拼写是 "Kokbork"（少一个 o），按原样匹配
        "xlsx":  "English-Kokbork WMT 2025 Test Set Gold.xlsx",
        "vocab": "Kokborok_vocab_word.json",
        "out":   "gold_test_kokborok.json",
    },
}


def load_gold_pairs(xlsx_path: str):
    """读 gold xlsx，跳过首行表头，返回 [(english, target), ...]。"""
    rows = list(_iter_xlsx_rows(xlsx_path))
    pairs = []
    for r in rows[1:]:  # 跳过表头行
        e, t = r.get(0), r.get(1)
        if not isinstance(e, str) or not isinstance(t, str):
            continue
        e = e.strip(); t = t.strip()
        if e and t:
            pairs.append((e, t))
    return pairs


def process_lang(lang: str, vocab: dict):
    cfg = GOLD_FILES[lang]
    xlsx = os.path.join(GOLD_DIR, cfg["xlsx"])
    if not os.path.exists(xlsx):
        print(f"[skip] {lang}: missing xlsx {xlsx}")
        return

    pairs = load_gold_pairs(xlsx)
    if not pairs:
        print(f"[skip] {lang}: empty after parse")
        return

    out = []
    n_with_hint = 0
    n_total_hints = 0
    for eng, tgt in pairs:
        inst = build_instruction(eng, vocab, lang)
        if "Helpful" in inst:
            n_with_hint += 1
            n_total_hints += inst.count("\n- ")
        out.append({"instruction": inst, "input": "", "output": tgt})

    dst = os.path.join(OUTPUT_DIR, cfg["out"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{lang:9s}  total={len(out)}  with_hint={n_with_hint} "
          f"({n_with_hint*100/len(out):.1f}%)  avg_hints={n_total_hints/max(n_with_hint,1):.2f}")
    print(f"           saved: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None, choices=list(GOLD_FILES.keys()),
                    help="只处理某一种；不传则全跑")
    args = ap.parse_args()

    print(f"Filter: probability >= {MIN_PROB}, top-{TOP_K} candidates per word")

    langs = [args.lang] if args.lang else list(GOLD_FILES.keys())
    for lang in langs:
        vocab_path = os.path.join(SCRIPT_DIR, "word_vocab", GOLD_FILES[lang]["vocab"])
        if not os.path.exists(vocab_path):
            print(f"[skip] {lang}: missing vocab {vocab_path}")
            continue
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"\n=== {lang} (vocab={len(vocab)} words) ===")
        process_lang(lang, vocab)


if __name__ == "__main__":
    main()
