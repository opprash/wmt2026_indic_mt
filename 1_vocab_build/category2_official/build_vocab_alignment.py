# -*- coding: utf-8 -*-
"""
词汇对照表生成脚本

逻辑：
  1. 加载 words_dictionary.json 中的英文词汇表（key 为单词）
  2. 遍历 Category II 下每个 Excel 文件（每个文件 2 列：英文 + 目标语种）
  3. 对每条英文句子按空格切分成 token，再去标点 + 小写化，与词汇表做绝对匹配
  4. 命中即把 {english_sentence: target_translation} 追加进该词对应的 list
  5. 输出结构：[{word1: [{eng: trans}, ...]}, {word2: [...]}, ...]
  6. 文件名：<target_language>_vocab.json，保存到 OUTPUT_DIR
"""

import os
import re
import json
import glob
import pandas as pd


# ============== 路径配置（按需修改） ==============
EXCEL_DIR  = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\Category II"
VOCAB_JSON = r"C:\Users\Administrator\Desktop\智能2026\个人\wmt2026\code\words_dictionary.json"
# 输出到 <脚本所在目录>/sentence_vocab/
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentence_vocab")
# ===================================================


# token 清洗：去掉首尾非字母数字字符
_PUNCT_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def load_vocabulary(path):
    """加载词汇表，返回小写的 set。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(w).strip().lower() for w in data.keys() if str(w).strip()}


def tokenize(sentence):
    """按空格切分英文句子，每个 token 去首尾标点并小写。"""
    if not isinstance(sentence, str):
        return []
    tokens = []
    for raw in sentence.split():
        cleaned = _PUNCT_STRIP.sub("", raw).lower()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def english_ratio(series, sample=30):
    """估算一列中 ASCII 字符占比，用来判断哪一列是英文列。"""
    eng, total = 0, 0
    for s in series.dropna().head(sample):
        if not isinstance(s, str):
            continue
        for ch in s:
            total += 1
            if ord(ch) < 128:
                eng += 1
    return (eng / total) if total else 0.0


def smart_read(path):
    """
    智能读取 Excel：
      - 先尝试 header=0；若列名里出现 'english' 则视为有正常表头
      - 否则按 header=None 重新读，把首行也当作数据
    返回 (DataFrame, eng_col, target_col)
    """
    df_h = pd.read_excel(path, header=0)
    cols = list(df_h.columns)

    has_english_header = any("english" in str(c).strip().lower() for c in cols)

    if has_english_header and len(cols) >= 2:
        df = df_h
    else:
        # 没有正常表头：首行其实是数据
        df = pd.read_excel(path, header=None)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    if df.shape[1] < 2:
        raise ValueError(f"列数不足 2：{path}")

    # 仅保留前两列
    df = df.iloc[:, :2]

    # 判定英文列：表头优先，否则按 ASCII 占比
    eng_col = None
    for c in df.columns:
        if "english" in str(c).strip().lower():
            eng_col = c
            break
    if eng_col is None:
        ratios = {c: english_ratio(df[c]) for c in df.columns}
        eng_col = max(ratios, key=ratios.get)

    target_col = [c for c in df.columns if c != eng_col][0]
    return df, eng_col, target_col


def derive_target_language(excel_path, target_col_name):
    """
    从文件名推导目标语种名：
      'English - Karbi Training Data 2026.xlsx' -> 'Karbi'
      'English-Nagamese  Training Data 2026.xlsx' -> 'Nagamese'
    若文件名无法解析，则退回使用列名。
    """
    base = os.path.splitext(os.path.basename(excel_path))[0]
    s = re.sub(r"^English\s*-\s*", "", base, flags=re.IGNORECASE)
    s = re.sub(r"\s*Training\s*Data.*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if not s:
        s = str(target_col_name).strip()
    # 文件名安全
    s = re.sub(r"\s+", "_", s)
    return s


def process_excel(excel_path, vocab_set, output_dir):
    fname = os.path.basename(excel_path)
    print(f"\n=== 处理: {fname} ===")

    try:
        df, eng_col, target_col = smart_read(excel_path)
    except Exception as e:
        print(f"  [跳过] 读取失败: {e}")
        return

    target_lang = derive_target_language(excel_path, target_col)
    print(f"  英文列: {eng_col!r}")
    print(f"  目标列: {target_col!r}")
    print(f"  目标语种: {target_lang}")
    print(f"  数据行数: {len(df)}")

    # word -> list of {english_sentence: target_translation}
    matches = {}
    skipped = 0

    for _, row in df.iterrows():
        eng_sent = row[eng_col]
        tgt_sent = row[target_col]

        if not isinstance(eng_sent, str) or not isinstance(tgt_sent, str):
            skipped += 1
            continue

        eng_sent_s = eng_sent.strip()
        tgt_sent_s = tgt_sent.strip()
        if not eng_sent_s or not tgt_sent_s:
            skipped += 1
            continue

        # 句子内去重（同一个 token 在句中重复时只算一次匹配）
        tokens = set(tokenize(eng_sent_s))
        if not tokens:
            continue

        hits = tokens & vocab_set
        if not hits:
            continue

        for w in hits:
            matches.setdefault(w, []).append({eng_sent_s: tgt_sent_s})

    # 按用户要求：最终是一个大 list，元素是 {word: [...]} 单键字典
    output = [{w: matches[w]} for w in sorted(matches.keys())]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{target_lang}_vocab.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_pairs = sum(len(v) for v in matches.values())
    print(f"  命中词汇数: {len(matches)}")
    print(f"  句对总数: {total_pairs}")
    print(f"  跳过(空/非文本)行: {skipped}")
    print(f"  输出: {out_path}")


def main():
    print(f"加载词汇表: {VOCAB_JSON}")
    vocab_set = load_vocabulary(VOCAB_JSON)
    print(f"词汇表大小: {len(vocab_set)}")

    excel_files = sorted(glob.glob(os.path.join(EXCEL_DIR, "*.xlsx")))
    # 过滤 Office 临时锁文件
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith("~$")]
    print(f"待处理 Excel 文件数: {len(excel_files)}")

    for path in excel_files:
        process_excel(path, vocab_set, OUTPUT_DIR)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
