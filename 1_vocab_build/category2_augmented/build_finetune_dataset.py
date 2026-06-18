# -*- coding: utf-8 -*-
"""
构造大模型微调数据集（注入词汇提示）

支持的语种：Tagin / Bodo / Karbi / Kokborok / Nagamese

来源策略：
  - Tagin：若存在 code\tagin\targin_train.json + targin_test.json，则沿用其原始 80/20 切分
  - 其它语种：直接从 Category II 下对应的 Excel 读取，按 --ratio (默认 0.8) 随机切分（--seed 固定种子可复现）

输出：
  - dataset/<lang>_train.json
  - dataset/<lang>_test.json

instruction 格式：
  Translate the following segment into <Lang>, without additional explanation.

  Helpful <Lang> vocabulary for words in this sentence:
  - word1: target1 (0.90) / target2 (0.65)
  - word2: target1 (0.95)

  <English sentence>

匹配逻辑（与 build_vocab_alignment.py 一致）：
  - 英文句子按空格 split
  - 每个 token 去首尾标点 + 小写化
  - 在 <Lang>_vocab_word.json 里查找完全匹配
  - 命中后取概率 >= MIN_PROB 的前 TOP_K 个候选作为提示
  - 全句无命中则不加 vocabulary 区块

用法：
  python build_finetune_dataset.py --lang Tagin
  python build_finetune_dataset.py --lang Bodo
  python build_finetune_dataset.py --lang Karbi --ratio 0.9 --seed 123
"""

import argparse
import json
import os
import random
import re
import sys


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR    = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\Category2_add_all"
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "dataset")

MIN_PROB     = 0.5   # 概率阈值：低于此值的候选不进 instruction
TOP_K        = 3     # 每个英文词最多展示几个候选

# 想要"塞全部候选不过滤"时取消下面注释：
# MIN_PROB     = 0.0
# TOP_K        = 100


_PUNCT_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")

# 语种 → Excel 文件名（Category2_add_all 下的扩充数据）
EXCEL_FILES = {
    "Karbi":    "karbi.xlsx",
    "Bodo":     "bodo.xlsx",
    "Kokborok": "kokborok.xlsx",
    "Nagamese": "nagamese.xlsx",
    "Targin":   "targin.xlsx",
}


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


def extract_english(instruction: str) -> str:
    marker = "\n\n"
    if marker in instruction:
        return instruction.split(marker, 1)[1].strip()
    return instruction.strip()


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


# ---------- 数据源加载 ----------
def load_pairs_from_existing_json(train_path: str, test_path: str):
    """从已有的 train/test JSON 中读取 (english, output) 对。"""
    def _load(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(extract_english(d.get("instruction", "")), d.get("output", "")) for d in data]
    return _load(train_path), _load(test_path)


def _iter_xlsx_rows(path: str):
    """纯标准库流式读取 xlsx 行（不依赖 pandas/numpy/openpyxl）。"""
    import zipfile
    import xml.etree.ElementTree as ET
    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    with zipfile.ZipFile(path) as z:
        # 共享字符串
        shared = []
        try:
            with z.open("xl/sharedStrings.xml") as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag == NS + "si":
                        # 累积所有 t 文本（处理 rich text）
                        parts = []
                        for t in elem.iter(NS + "t"):
                            if t.text:
                                parts.append(t.text)
                        shared.append("".join(parts))
                        elem.clear()
        except KeyError:
            pass  # 无共享字符串

        # 取 sheet 编号最小的 worksheet（按 sheetN.xml 中的 N 排序，避免 namelist 顺序错配）
        sheet_files = [
            n for n in z.namelist()
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        ]
        def _sheet_no(name):
            base = name[len("xl/worksheets/sheet"):-len(".xml")]
            try:
                return int(base)
            except ValueError:
                return 1 << 30
        sheet_files.sort(key=_sheet_no)
        if not sheet_files:
            return
        sheet_name = sheet_files[0]

        def col_idx(ref):
            """A1 -> 0, B2 -> 1, ..."""
            letters = "".join(ch for ch in ref if ch.isalpha())
            n = 0
            for ch in letters:
                n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
            return n - 1

        with z.open(sheet_name) as f:
            cur_row = None
            for _, elem in ET.iterparse(f, events=("end",)):
                if elem.tag == NS + "row":
                    if cur_row is not None:
                        yield cur_row
                    cur_row = None
                    elem.clear()
                elif elem.tag == NS + "c":
                    ref = elem.attrib.get("r", "")
                    cidx = col_idx(ref) if ref else 0
                    ctype = elem.attrib.get("t", "n")
                    v_node = elem.find(NS + "v")
                    is_node = elem.find(NS + "is")
                    val = None
                    if ctype == "s" and v_node is not None and v_node.text is not None:
                        try:
                            val = shared[int(v_node.text)]
                        except (ValueError, IndexError):
                            val = None
                    elif ctype == "inlineStr" and is_node is not None:
                        parts = []
                        for t in is_node.iter(NS + "t"):
                            if t.text:
                                parts.append(t.text)
                        val = "".join(parts)
                    elif v_node is not None and v_node.text is not None:
                        val = v_node.text
                    if cur_row is None:
                        cur_row = {}
                    cur_row[cidx] = val
                    elem.clear()
            if cur_row is not None:
                yield cur_row


def load_pairs_from_excel(lang: str, ratio: float, seed: int):
    """从 Excel 读取所有 (english, target) 对并随机切分（stdlib only，不依赖 numpy）。"""
    fname = EXCEL_FILES[lang]
    path = os.path.join(EXCEL_DIR, fname)

    # dict-row → (col0, col1) 元组
    def to_tuple(r):
        return (r.get(0), r.get(1))

    rows_iter = (to_tuple(r) for r in _iter_xlsx_rows(path) if r)

    # 拿首行
    try:
        first = next(rows_iter)
    except StopIteration:
        return [], []

    def has_eng(v):
        return isinstance(v, str) and "english" in v.strip().lower() and len(v.strip()) < 40

    if has_eng(first[0]) or has_eng(first[1]):
        eng_idx = 0 if has_eng(first[0]) else 1
        data_iter = rows_iter  # 首行被丢弃（是表头）
    else:
        # 无表头：先 buffer 30 行判定英文列
        buf = [first]
        for _ in range(29):
            try:
                buf.append(next(rows_iter))
            except StopIteration:
                break
        def asc_ratio(col_idx):
            tot = eng = 0
            for r in buf:
                v = r[col_idx]
                if isinstance(v, str):
                    for ch in v:
                        tot += 1
                        if ord(ch) < 128:
                            eng += 1
            return eng / tot if tot else 0
        eng_idx = 0 if asc_ratio(0) >= asc_ratio(1) else 1
        def chained():
            for r in buf: yield r
            for r in rows_iter: yield r
        data_iter = chained()
    tgt_idx = 1 - eng_idx

    pairs = []
    for r in data_iter:
        e, t = r[eng_idx], r[tgt_idx]
        if not isinstance(e, str) or not isinstance(t, str):
            continue
        e = e.strip(); t = t.strip()
        if e and t:
            pairs.append((e, t))

    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_train = int(round(len(pairs) * ratio))
    train = [pairs[i] for i in idx[:n_train]]
    test  = [pairs[i] for i in idx[n_train:]]
    return train, test


# ---------- 写文件 ----------
def write_split(pairs, dst_path: str, vocab: dict, target_lang: str) -> dict:
    out = []
    n_with_hint = 0
    n_total_hints = 0
    for eng, tgt in pairs:
        inst = build_instruction(eng, vocab, target_lang)
        if "Helpful" in inst:
            n_with_hint += 1
            n_total_hints += inst.count("\n- ")
        out.append({"instruction": inst, "input": "", "output": tgt})

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return {
        "total": len(out),
        "with_hint": n_with_hint,
        "avg_hints_per_hit": round(n_total_hints / max(n_with_hint, 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(EXCEL_FILES.keys()),
                    help="target language")
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="train ratio when splitting from Excel (default 0.8)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for split (default 42)")
    args = ap.parse_args()

    lang = args.lang
    vocab_path = os.path.join(SCRIPT_DIR, "word_vocab", f"{lang}_vocab_word.json")
    if not os.path.exists(vocab_path):
        sys.exit(f"vocab file not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocab loaded: {len(vocab)} words ({lang})")
    print(f"Filter: probability >= {MIN_PROB}, top-{TOP_K} candidates per word")

    print(f"Source: Excel ({EXCEL_FILES[lang]}); split ratio={args.ratio}, seed={args.seed}")
    train_pairs, test_pairs = load_pairs_from_excel(lang, args.ratio, args.seed)

    train_dst = os.path.join(OUTPUT_DIR, f"{lang.lower()}_train.json")
    test_dst  = os.path.join(OUTPUT_DIR, f"{lang.lower()}_test.json")

    print(f"\n=== Train ({lang}) ===")
    info = write_split(train_pairs, train_dst, vocab, lang)
    print(f"  total: {info['total']}")
    print(f"  with vocab hint: {info['with_hint']} ({info['with_hint']*100/info['total']:.1f}%)")
    print(f"  avg hints / hinted sample: {info['avg_hints_per_hit']}")
    print(f"  saved: {train_dst}")

    print(f"\n=== Test ({lang}) ===")
    info = write_split(test_pairs, test_dst, vocab, lang)
    print(f"  total: {info['total']}")
    print(f"  with vocab hint: {info['with_hint']} ({info['with_hint']*100/info['total']:.1f}%)")
    print(f"  avg hints / hinted sample: {info['avg_hints_per_hit']}")
    print(f"  saved: {test_dst}")


if __name__ == "__main__":
    main()
