# -*- coding: utf-8 -*-
"""
将 gold_eval/ 下的两个 WMT 2025 Test Set Gold xlsx 整理为黄金测试集。

格式与 dataset_all/<lang>.json 一致（单 list，每条 instruction/input/output）；
词汇提示走同一份 word_vocab/<Lang>_vocab_word.json。

输入:
  gold_eval/English-Bodo WMT 2025 Test Set Gold.xlsx
  gold_eval/English-Kokbork WMT 2025 Test Set Gold.xlsx
输出:
  gold_eval_test/gold_test_bodo_add.json
  gold_eval_test/gold_test_kokborok_add.json
"""

import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR    = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\gold_eval"
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "gold_eval_test")
WORD_VOCAB_DIR = os.path.join(SCRIPT_DIR, "word_vocab")

MIN_PROB = 0.5
TOP_K    = 3

# (filename, vocab_lang, out_lang_slug)
LANG_FILES = [
    ("English-Bodo WMT 2025 Test Set Gold.xlsx",    "Bodo",     "bodo"),
    ("English-Kokbork WMT 2025 Test Set Gold.xlsx", "Kokborok", "kokborok"),
]

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PUNCT_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


# ---------- token + vocab hint (与 build_finetune_dataset.py 同) ----------
def tokenize_unique(sentence: str):
    if not isinstance(sentence, str):
        return []
    seen = set()
    out = []
    for raw in sentence.split():
        t = _PUNCT_STRIP.sub("", raw).lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_vocab_hint(eng: str, vocab: dict, target_lang: str) -> str:
    lines = []
    for token in tokenize_unique(eng):
        entry = vocab.get(token)
        if not entry:
            continue
        cands = entry.get("candidates") or []
        cands = [c for c in cands if isinstance(c, dict)
                 and c.get("target")
                 and float(c.get("probability", 0)) >= MIN_PROB]
        if not cands:
            continue
        cands = cands[:TOP_K]
        targets = " / ".join(
            f"{c['target']} ({float(c['probability']):.2f})" for c in cands
        )
        lines.append(f"- {token}: {targets}")
    if not lines:
        return ""
    return (
        f"Helpful {target_lang} vocabulary for words in this sentence:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def build_instruction(eng: str, vocab: dict, target_lang: str) -> str:
    hint = build_vocab_hint(eng, vocab, target_lang)
    head = f"Translate the following segment into {target_lang}, without additional explanation."
    return f"{head}\n\n{hint}{eng}"


# ---------- stdlib xlsx 读取 ----------
def _col_idx(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def iter_xlsx_rows(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        try:
            with z.open("xl/sharedStrings.xml") as f:
                for _, e in ET.iterparse(f, events=("end",)):
                    if e.tag == NS + "si":
                        parts = [t.text for t in e.iter(NS + "t") if t.text]
                        shared.append("".join(parts))
                        e.clear()
        except KeyError:
            pass
        sheet_files = sorted(
            [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")],
            key=lambda s: int(s[len("xl/worksheets/sheet"):-len(".xml")]) if s[len("xl/worksheets/sheet"):-len(".xml")].isdigit() else 1 << 30,
        )
        if not sheet_files:
            return
        with z.open(sheet_files[0]) as f:
            cur = None
            for _, e in ET.iterparse(f, events=("end",)):
                if e.tag == NS + "row":
                    if cur is not None:
                        yield cur
                    cur = None
                    e.clear()
                elif e.tag == NS + "c":
                    ref = e.attrib.get("r", "")
                    ct = e.attrib.get("t", "n")
                    v_node = e.find(NS + "v")
                    is_node = e.find(NS + "is")
                    val = None
                    if ct == "s" and v_node is not None and v_node.text is not None:
                        try:
                            val = shared[int(v_node.text)]
                        except Exception:
                            val = None
                    elif ct == "inlineStr" and is_node is not None:
                        val = "".join(t.text or "" for t in is_node.iter(NS + "t"))
                    elif v_node is not None and v_node.text is not None:
                        val = v_node.text
                    if cur is None:
                        cur = {}
                    cur[_col_idx(ref) if ref else 0] = val
                    e.clear()
            if cur is not None:
                yield cur


def detect_english_col(first_row):
    """gold 文件首行是表头：Bodo 是 ['Source Sentence', 'Target Sentence'], Kokbork 是 ['Source Sentence in English', 'Kokborok Translation']。
    简单按表头里有没有 'english' 或 'source'/'target' 判断。"""
    v0 = (first_row.get(0) or "").strip().lower() if isinstance(first_row.get(0), str) else ""
    v1 = (first_row.get(1) or "").strip().lower() if isinstance(first_row.get(1), str) else ""
    # 优先看 'english'
    if "english" in v0 and "english" not in v1:
        return 0
    if "english" in v1 and "english" not in v0:
        return 1
    # 退回看 source/target
    if "source" in v0:
        return 0
    if "source" in v1:
        return 1
    # 默认 col 0
    return 0


def load_pairs_from_excel(path):
    rows = list(iter_xlsx_rows(path))
    if not rows:
        return []
    eng_idx = detect_english_col(rows[0])
    tgt_idx = 1 - eng_idx
    pairs = []
    for r in rows[1:]:
        e = r.get(eng_idx)
        t = r.get(tgt_idx)
        if not isinstance(e, str) or not isinstance(t, str):
            continue
        e = e.strip(); t = t.strip()
        if e and t:
            pairs.append((e, t))
    return pairs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname, vocab_lang, slug in LANG_FILES:
        in_path = os.path.join(GOLD_DIR, fname)
        vocab_path = os.path.join(WORD_VOCAB_DIR, f"{vocab_lang}_vocab_word.json")
        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found"); continue
        if not os.path.exists(vocab_path):
            print(f"[skip] {vocab_path} not found"); continue

        print(f"\n=== {vocab_lang} ===")
        print(f"  source: {fname}")
        print(f"  vocab : {os.path.basename(vocab_path)}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  vocab loaded: {len(vocab)} words")

        pairs = load_pairs_from_excel(in_path)
        print(f"  pairs read: {len(pairs)}")

        out = []
        n_hint = 0
        n_hints_total = 0
        for eng, tgt in pairs:
            inst = build_instruction(eng, vocab, vocab_lang)
            if "Helpful" in inst:
                n_hint += 1
                n_hints_total += inst.count("\n- ")
            out.append({"instruction": inst, "input": "", "output": tgt})

        out_path = os.path.join(OUTPUT_DIR, f"gold_test_{slug}_add.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  total: {len(out)}")
        print(f"  with vocab hint: {n_hint} ({n_hint*100/max(len(out),1):.1f}%)")
        print(f"  avg hints / hinted sample: {n_hints_total / max(n_hint,1):.2f}")
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
