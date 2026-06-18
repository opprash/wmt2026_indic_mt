# -*- coding: utf-8 -*-
"""
将 final_test 下 CATEGORY 2 的 en-xx Test 文件转换为带词汇提示的提交格式。

每个语种文件夹下有两个文件（en-xx / xx-en），只需 en-xx（英文→目标语种）。
测试集只有英文列（S No + English Sentences），output 留空字符串。

输出：sft_add_testset/<slug>_final_test.json
格式：每条 {"id": <S No>, "instruction": "...", "input": "", "output": ""}
（id 取官方 S No，保证后续预测/提交时 txt 第 i 行对齐测试集第 i 行）

词汇提示：与 dataset_all/<lang>.json 完全一致，走 word_vocab/<Lang>_vocab_word.json
"""

import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR  = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\final_test\WMT 2026 IndicMT Test Data\Category 2"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sft_add_testset")
WORD_VOCAB_DIR = os.path.join(SCRIPT_DIR, "word_vocab")

MIN_PROB = 0.5
TOP_K    = 3

# (folder_name, en_file, vocab_lang, slug)
LANG_FILES = [
    ("English - Bodo",    "en-bodo Test.xlsx", "Bodo",     "bodo"),
    ("English - Karbi",   "en-mjw Test.xlsx",  "Karbi",    "karbi"),
    ("English - Kokborok","en-trp Test.xlsx",  "Kokborok", "kokborok"),
    ("English - Nagamese","en-nag Test.xlsx",  "Nagamese", "nagamese"),
    ("English - Tagin",   "en-tgj Test.xlsx",  "Targin",   "targin"),
]

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PUNCT_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


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
        sheets = sorted(
            [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")],
            key=lambda s: int(s[len("xl/worksheets/sheet"):-len(".xml")]) if s[len("xl/worksheets/sheet"):-len(".xml")].isdigit() else 1 << 30,
        )
        if not sheets:
            return
        with z.open(sheets[0]) as f:
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for folder, en_file, vocab_lang, slug in LANG_FILES:
        xlsx_path = os.path.join(FINAL_DIR, folder, en_file)
        if not os.path.exists(xlsx_path):
            print(f"[skip] {xlsx_path} not found"); continue

        vocab_path = os.path.join(WORD_VOCAB_DIR, f"{vocab_lang}_vocab_word.json")
        if not os.path.exists(vocab_path):
            print(f"[skip] {vocab_path} not found"); continue

        print(f"\n=== {slug} ===")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  vocab: {len(vocab)} words")

        rows = list(iter_xlsx_rows(xlsx_path))
        print(f"  xlsx rows: {len(rows)}")

        if not rows:
            print("  [skip] no data"); continue

        # 找英文列（列名含 "english" 或 "sentence"）；S No 列默认在第 1 列(idx 0)
        header = rows[0]
        eng_idx = None
        sno_idx = 0
        for k, v in header.items():
            if isinstance(v, str):
                vl = v.strip().lower()
                if "english" in vl or "sentence" in vl:
                    eng_idx = k
                elif "s no" in vl or "sno" in vl or vl in ("id", "s.no", "no"):
                    sno_idx = k
        if eng_idx is None:
            eng_idx = 1  # fallback: 第 2 列
        print(f"  english column: {eng_idx}  (S No column: {sno_idx})")

        out = []
        n_hint = 0
        n_hints_total = 0
        running = 0
        for r in rows[1:]:
            eng = r.get(eng_idx)
            if not isinstance(eng, str):
                continue
            eng = eng.strip()
            if not eng:
                continue
            running += 1
            # id == official S No so txt line i aligns to the test-set row i;
            # fall back to a running counter if the S No cell is missing/unparseable.
            try:
                _id = int(float(r.get(sno_idx)))
            except (TypeError, ValueError):
                _id = running
            inst = build_instruction(eng, vocab, vocab_lang)
            if "Helpful" in inst:
                n_hint += 1
                n_hints_total += inst.count("\n- ")
            out.append({"id": _id, "instruction": inst, "input": "", "output": ""})

        out_path = os.path.join(OUTPUT_DIR, f"{slug}_final_test.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  total: {len(out)}")
        print(f"  with vocab hint: {n_hint} ({n_hint*100/max(len(out),1):.1f}%)")
        print(f"  avg hints / hinted: {n_hints_total / max(n_hint,1):.2f}")
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
