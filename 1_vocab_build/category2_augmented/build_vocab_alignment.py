# -*- coding: utf-8 -*-
"""
词汇对照表生成脚本

逻辑：
  1. 加载 words_dictionary.json 中的英文词汇表（key 为单词）
  2. 遍历 EXCEL_DIR 下每个 Excel 文件（每个文件 2 列：英文 + 目标语种；karbi 含一列 reference 会被忽略）
  3. 对每条英文句子按空格切分成 token，再去标点 + 小写化，与词汇表做绝对匹配
  4. 命中即把 {english_sentence: target_translation} 追加进该词对应的 list
  5. 输出结构：[{word1: [{eng: trans}, ...]}, {word2: [...]}, ...]
  6. 文件名：<target_language>_vocab.json，保存到 OUTPUT_DIR

新数据 (Category2_add_all) 的 Excel 表头为 [<lang>, english]（语种列在前，英文列在后），
体量可达 16w+ 行，因此采用 stdlib `zipfile + xml.etree` 流式解析，避免 pandas/openpyxl 的内存负担。
"""

import os
import re
import json
import glob
import zipfile
import xml.etree.ElementTree as ET


# ============== 路径配置（按需修改） ==============
EXCEL_DIR  = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\Category2_add_all"
VOCAB_JSON = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\code\words_dictionary.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentence_vocab")
# ===================================================


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PUNCT_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(w).strip().lower() for w in data.keys() if str(w).strip()}


def tokenize(sentence):
    if not isinstance(sentence, str):
        return []
    out = []
    for raw in sentence.split():
        cleaned = _PUNCT_STRIP.sub("", raw).lower()
        if cleaned:
            out.append(cleaned)
    return out


def _col_idx(ref):
    """A1 -> 0, B2 -> 1, AA1 -> 26, ..."""
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def iter_xlsx_rows(path):
    """流式产出 xlsx 第一个 sheet（按 sheet<N>.xml 中 N 最小者）的每行，row = {col_idx: value}。"""
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


def derive_target_language(excel_path):
    """
    从文件名推导目标语种名，兼容两种命名：
      'English - Karbi Training Data 2026.xlsx' -> 'Karbi'
      'karbi.xlsx' -> 'Karbi'  (新数据：小写文件名，首字母大写规范化)
    """
    base = os.path.splitext(os.path.basename(excel_path))[0]
    s = re.sub(r"^English\s*-\s*", "", base, flags=re.IGNORECASE)
    s = re.sub(r"\s*Training\s*Data.*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    if s and s.islower():
        s = s[0].upper() + s[1:]
    return s


def detect_english_col(first_row, buffer_rows):
    """
    返回 (eng_idx, has_header)。
    - 若首行某列文本是 'english'（不论大小写），则视为有表头
    - 否则按前若干行 ASCII 占比判定哪一列是英文
    仅看前两列。
    """
    def is_eng_header(v):
        return isinstance(v, str) and v.strip().lower() == "english"

    v0 = first_row.get(0)
    v1 = first_row.get(1)
    if is_eng_header(v0) or is_eng_header(v1):
        return (0 if is_eng_header(v0) else 1, True)

    # 无表头：按 ASCII 占比判定
    def asc_ratio(col):
        tot = eng = 0
        for r in buffer_rows:
            v = r.get(col)
            if isinstance(v, str):
                for ch in v:
                    tot += 1
                    if ord(ch) < 128:
                        eng += 1
        return eng / tot if tot else 0.0

    return (0 if asc_ratio(0) >= asc_ratio(1) else 1, False)


def process_excel(excel_path, vocab_set, output_dir):
    fname = os.path.basename(excel_path)
    print(f"\n=== 处理: {fname} ===")

    rows_iter = iter_xlsx_rows(excel_path)
    try:
        first = next(rows_iter)
    except StopIteration:
        print("  [跳过] 空文件")
        return

    # 缓冲首 30 行用于英文列判定（避免一遍读完）
    buf = [first]
    for _ in range(29):
        try:
            buf.append(next(rows_iter))
        except StopIteration:
            break

    eng_idx, has_header = detect_english_col(first, buf)
    tgt_idx = 1 - eng_idx

    target_lang = derive_target_language(excel_path)
    print(f"  英文列索引: {eng_idx}  目标列索引: {tgt_idx}  有表头: {has_header}")
    print(f"  目标语种: {target_lang}")

    matches = {}
    skipped = 0
    total = 0

    def chained():
        for r in buf:
            yield r
        for r in rows_iter:
            yield r

    skipped_header = False
    for r in chained():
        if has_header and not skipped_header:
            skipped_header = True
            continue
        total += 1
        eng = r.get(eng_idx)
        tgt = r.get(tgt_idx)
        if not isinstance(eng, str) or not isinstance(tgt, str):
            skipped += 1
            continue
        eng = eng.strip()
        tgt = tgt.strip()
        if not eng or not tgt:
            skipped += 1
            continue

        tokens = set(tokenize(eng))
        if not tokens:
            continue
        hits = tokens & vocab_set
        if not hits:
            continue
        for w in hits:
            matches.setdefault(w, []).append({eng: tgt})

    output = [{w: matches[w]} for w in sorted(matches.keys())]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{target_lang}_vocab.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_pairs = sum(len(v) for v in matches.values())
    print(f"  数据行: {total}（不含表头）")
    print(f"  命中词汇数: {len(matches)}")
    print(f"  句对总数: {total_pairs}")
    print(f"  跳过(空/非文本)行: {skipped}")
    print(f"  输出: {out_path}")


def main():
    print(f"加载词汇表: {VOCAB_JSON}")
    vocab_set = load_vocabulary(VOCAB_JSON)
    print(f"词汇表大小: {len(vocab_set)}")

    excel_files = sorted(glob.glob(os.path.join(EXCEL_DIR, "*.xlsx")))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith("~$")]
    print(f"待处理 Excel 文件数: {len(excel_files)}")
    for p in excel_files:
        print(f"  - {os.path.basename(p)}")

    for path in excel_files:
        process_excel(path, vocab_set, OUTPUT_DIR)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
