# -*- coding: utf-8 -*-
"""
词语级别对照表生成脚本（调用 qwen3-max）

输入：sentence_vocab/<lang>_vocab.json   —— 句对级别的词汇表（list of {word: [{eng: trans}, ...]}）
输出：word_vocab/<lang>_vocab_word.json  —— 词语级别的对照表 {word: {candidates: [{target, probability}], ...}}

逻辑：
  1. 加载 vocab.json
  2. 跳过句对数 < 2 的词
  3. 句对数 > 10 的随机抽 10 个
  4. 用 requests + ThreadPool 并发调用 qwen3-max
  5. 解析 JSON 输出 → 合并 → 写文件（支持断点续跑）

用法（脚本与 sentence_vocab/、word_vocab/ 同级）：
  python build_word_alignment.py --input sentence_vocab/Karbi_vocab.json
  python build_word_alignment.py --input sentence_vocab/Karbi_vocab.json --limit 10        # 小规模测试
  python build_word_alignment.py --input sentence_vocab/Karbi_vocab.json --workers 16      # 调整并发
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests


# ============== 配置 ==============
# 默认走阿里云 qwen3-max；运行时可通过 --api-url / --api-key / --model 覆盖
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL   = "qwen3-max"

# 运行时由 main() 写入；call_llm 使用这些全局值（避免到处传参）
API_KEY: str = DEFAULT_API_KEY
API_URL: str = DEFAULT_API_URL
MODEL: str   = DEFAULT_MODEL

MAX_SAMPLES   = 10     # 每个词最多采样多少个句对喂给模型
MIN_PAIRS     = 2      # 句对数小于此值的词直接跳过
MAX_WORKERS   = 10     # 默认并发数
REQUEST_TIMEOUT = 90
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0
SAVE_EVERY    = 25     # 每 N 条增量保存一次
RANDOM_SEED   = 42     # 抽样可复现
# ==================================


# ---------- 数据加载 ----------
def load_vocab_file(path: str) -> Dict[str, List[Dict[str, str]]]:
    """Vocab json 是 list of single-key dicts，扁平化成 {word: [pairs]}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            for word, pairs in item.items():
                if isinstance(pairs, list):
                    out[word] = pairs
    elif isinstance(data, dict):
        # 兼容字典格式
        for word, pairs in data.items():
            if isinstance(pairs, list):
                out[word] = pairs
    return out


def derive_lang_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("_vocab.json", "").strip()


# ---------- 采样 ----------
def sample_pairs(pairs: List[Dict[str, str]], k: int, rng: random.Random) -> List[Dict[str, str]]:
    if len(pairs) <= k:
        return list(pairs)
    return rng.sample(pairs, k)


# ---------- 提示词 ----------
def build_prompt(word: str, target_lang: str, pairs: List[Dict[str, str]]) -> str:
    lines = []
    for i, p in enumerate(pairs, 1):
        if not isinstance(p, dict) or not p:
            continue
        eng, tgt = next(iter(p.items()))
        lines.append(f"{i}. EN: {eng}\n   {target_lang}: {tgt}")
    pairs_text = "\n".join(lines)

    return f"""You are an expert bilingual linguist building a word-level dictionary from parallel sentences.

ENGLISH WORD: "{word}"
TARGET LANGUAGE: {target_lang}

PARALLEL SENTENCES (English -> {target_lang}):
{pairs_text}

TASK:
1. For EACH sentence above, identify the word (or short token sequence) in the {target_lang} side that corresponds to the English word "{word}".
2. Aggregate evidence across all sentences. One English word may map to several distinct {target_lang} surface forms (inflections, synonyms, contextual variants); list each distinct candidate separately.
3. For every candidate, give:
   - "target": the {target_lang} word/phrase exactly as it appears (lowercase preferred unless it is a proper noun).
   - "probability": a float in [0.0, 1.0] indicating your confidence it is a valid translation of "{word}". Higher = stronger and more consistent evidence across the sentences provided.
4. If "{word}" is a function word (article, preposition, auxiliary) and has no overt counterpart in {target_lang}, return an empty candidates list.

STRICT OUTPUT RULES:
- Return EXACTLY ONE JSON object. No markdown fences, no prose, no explanations.
- Schema:
  {{"word": "{word}", "candidates": [{{"target": "<string>", "probability": <number 0..1>}}, ...]}}
- Sort candidates by probability descending.
- Use ONLY characters that appear in the {target_lang} sentences provided; do not invent translations.
"""


# ---------- JSON 抽取 ----------
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    # 去掉 ```json ... ``` 围栏
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # 取从第一个 { 到最后一个 } 的子串
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    chunk = s[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


# ---------- LLM 调用 ----------
def call_llm(prompt: str) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                # 速率限制：稍微多等一会
                time.sleep(RETRY_BACKOFF * attempt * (2 if r.status_code == 429 else 1))
                continue
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            parsed = extract_json(content)
            if parsed is None:
                last_err = f"unparseable JSON: {content[:300]}"
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return parsed
        except requests.RequestException as e:
            last_err = f"network: {e}"
            time.sleep(RETRY_BACKOFF * attempt)
        except Exception as e:
            last_err = f"unexpected: {e}"
            time.sleep(RETRY_BACKOFF * attempt)
    sys.stderr.write(f"  [API失败-{MAX_RETRIES}次] {last_err}\n")
    return None


# ---------- 单词处理 ----------
def process_one(word: str, pairs: List[Dict[str, str]], target_lang: str, rng: random.Random) -> Optional[dict]:
    sampled = sample_pairs(pairs, MAX_SAMPLES, rng)
    prompt = build_prompt(word, target_lang, sampled)
    result = call_llm(prompt)
    if result is None:
        return None

    raw_cands = result.get("candidates", [])
    if not isinstance(raw_cands, list):
        raw_cands = []

    cleaned = []
    for c in raw_cands:
        if not isinstance(c, dict):
            continue
        t = c.get("target")
        p = c.get("probability")
        if t is None or p is None:
            continue
        try:
            p = float(p)
        except Exception:
            continue
        # clip 到 [0,1]
        p = max(0.0, min(1.0, p))
        ts = str(t).strip()
        if not ts:
            continue
        cleaned.append({"target": ts, "probability": round(p, 4)})

    cleaned.sort(key=lambda x: x["probability"], reverse=True)
    return {
        "samples_used": len(sampled),
        "total_pairs": len(pairs),
        "candidates": cleaned,
    }


# ---------- 主流程 ----------
def atomic_save(data: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to <lang>_vocab.json")
    ap.add_argument("--output", default=None, help="output path (default: <lang>_vocab_word.json next to input)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N words (0 = all)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent API calls")
    ap.add_argument("--api-url", default=DEFAULT_API_URL, help="LLM endpoint")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM api key")
    ap.add_argument("--model",   default=DEFAULT_MODEL,   help="LLM model name")
    args = ap.parse_args()

    # 写入模块级全局，call_llm 内部使用
    global API_URL, API_KEY, MODEL
    API_URL = args.api_url
    API_KEY = args.api_key
    MODEL   = args.model

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")

    target_lang = derive_lang_from_filename(args.input)
    # 默认输出到 <脚本所在目录>/word_vocab/<lang>_vocab_word.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    word_vocab_dir = os.path.join(script_dir, "word_vocab")
    os.makedirs(word_vocab_dir, exist_ok=True)
    out_path = args.output or os.path.join(word_vocab_dir, f"{target_lang}_vocab_word.json")

    print(f"Input : {args.input}")
    print(f"Output: {out_path}")
    print(f"Target language: {target_lang}")
    print(f"Workers: {args.workers}  Model: {MODEL}  URL: {API_URL}")

    vocab = load_vocab_file(args.input)
    print(f"Total words in input: {len(vocab)}")

    # 过滤：跳过句对数 < MIN_PAIRS 的词
    eligible = {w: p for w, p in vocab.items() if len(p) >= MIN_PAIRS}
    print(f"Eligible (>={MIN_PAIRS} pairs): {len(eligible)}")

    # 断点续跑：加载已有结果
    existing: Dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
            print(f"Resume: {len(existing)} words already done, will skip them")
        except Exception:
            existing = {}

    todo_words = [w for w in eligible.keys() if w not in existing]
    todo_words.sort()  # 顺序稳定
    if args.limit > 0:
        todo_words = todo_words[: args.limit]
    print(f"To process this run: {len(todo_words)}")

    if not todo_words:
        print("Nothing to do.")
        return

    results: Dict[str, dict] = dict(existing)
    rng = random.Random(RANDOM_SEED)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}
    start_ts = time.time()
    total = len(todo_words)

    def worker(w: str):
        pairs = eligible[w]
        local_rng = random.Random(f"{RANDOM_SEED}-{w}")
        res = process_one(w, pairs, target_lang, local_rng)
        with lock:
            if res is not None:
                results[w] = res
                counters["ok"] += 1
            else:
                counters["fail"] += 1
            done = counters["ok"] + counters["fail"]
            if done % SAVE_EVERY == 0 or done == total:
                atomic_save(results, out_path)
            if done % 10 == 0 or done == total:
                elapsed = time.time() - start_ts
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] ok={counters['ok']} fail={counters['fail']} "
                      f"rate={rate:.2f}/s eta={eta:.0f}s",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, w) for w in todo_words]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                sys.stderr.write(f"[worker exception] {e}\n")

    atomic_save(results, out_path)
    elapsed = time.time() - start_ts
    print(f"\nDone. ok={counters['ok']} fail={counters['fail']} "
          f"total_saved={len(results)} elapsed={elapsed:.1f}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
