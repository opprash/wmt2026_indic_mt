# -*- coding: utf-8 -*-
"""
补救脚本：处理 build_word_alignment.py 跑剩的、被内容安全过滤拦截的词。

策略：
  1. 找出 sentence_vocab/<lang>_vocab.json 中所有 eligible（>=2 pairs）但不在 word_vocab/<lang>_vocab_word.json 里的词
  2. 对每个缺失词：
     a. 尝试 8 个不同随机种子，每次抽 min(10, len(pairs)) 个句对喂模型
     b. 若多句对仍被拦，逐对单独喂（单句对的内容暴露面更小，可能通过）
     c. 把所有 OK 的输出合并；按 target 聚合，取最高 probability，按概率降序
  3. 仍全部被拦的词：写空 candidates + note 标记
  4. 全程 atomic save，可重复运行

用法：
  python recover_blocked_words.py --lang Bodo
  python recover_blocked_words.py --lang Tagin
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests


API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL   = "qwen3-max"

MAX_SAMPLES_PER_TRY = 10
NUM_RANDOM_TRIES    = 8     # 多句对模式的不同采样次数上限
WORKERS             = 5
REQUEST_TIMEOUT     = 60
RETRY_NET_ERR       = 2     # 网络错误的重试次数（不针对内容过滤）
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))


_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def build_prompt(word: str, target_lang: str, pairs: List[Dict[str, str]]) -> str:
    lines = []
    for i, p in enumerate(pairs, 1):
        eng, tgt = next(iter(p.items()))
        lines.append(f"{i}. EN: {eng}\n   {target_lang}: {tgt}")
    pairs_text = "\n".join(lines)
    return (
        f"You are an expert bilingual linguist building a word-level dictionary from parallel sentences.\n\n"
        f"ENGLISH WORD: \"{word}\"\n"
        f"TARGET LANGUAGE: {target_lang}\n\n"
        f"PARALLEL SENTENCES (English -> {target_lang}):\n{pairs_text}\n\n"
        "TASK: identify the target word(s) corresponding to the English word; allow multiple candidates with probabilities (0-1).\n"
        f"OUTPUT ONLY one JSON object: {{\"word\":\"{word}\",\"candidates\":[{{\"target\":\"...\",\"probability\":<num>}}]}}\n"
    )


def call_once(prompt: str) -> (Optional[dict], str):
    """返回 (parsed_or_None, status_tag)。status_tag in {ok, blocked, neterr, parse_err}"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    last_err_tag = "neterr"
    for attempt in range(RETRY_NET_ERR + 1):
        try:
            r = requests.post(API_URL, headers=_HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content), "ok"
                except Exception:
                    last_err_tag = "parse_err"
                    continue
            elif r.status_code == 400 and "data_inspection_failed" in r.text:
                return None, "blocked"
            else:
                last_err_tag = "neterr"
                time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            last_err_tag = "neterr"
            time.sleep(1.0 * (attempt + 1))
    return None, last_err_tag


def aggregate_candidates(parsed_list: List[dict]) -> List[Dict]:
    """合并多次调用的 candidates；按 target 取最高 probability。"""
    best: Dict[str, float] = {}
    for p in parsed_list:
        for c in p.get("candidates", []) or []:
            if not isinstance(c, dict):
                continue
            t = c.get("target")
            pr = c.get("probability")
            if t is None or pr is None:
                continue
            try:
                pr = float(pr)
            except Exception:
                continue
            pr = max(0.0, min(1.0, pr))
            t = str(t).strip()
            if not t:
                continue
            best[t] = max(best.get(t, 0.0), pr)
    out = [{"target": t, "probability": round(p, 4)} for t, p in best.items()]
    out.sort(key=lambda x: x["probability"], reverse=True)
    return out


def recover_word(word: str, pairs: List[Dict[str, str]], target_lang: str) -> Dict:
    parsed_oks = []
    blocked_count = 0
    neterr_count = 0

    # === 阶段 1: 多句对，不同随机种子 ===
    k = min(MAX_SAMPLES_PER_TRY, len(pairs))
    seen_sigs = set()
    for t in range(NUM_RANDOM_TRIES):
        rng = random.Random(f"recover-{word}-{t}")
        if len(pairs) <= k:
            sample = list(pairs)
        else:
            sample = rng.sample(pairs, k)
        sig = tuple(sorted(id(p) for p in sample))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        parsed, tag = call_once(build_prompt(word, target_lang, sample))
        if tag == "ok":
            parsed_oks.append(parsed)
        elif tag == "blocked":
            blocked_count += 1
        else:
            neterr_count += 1
        if parsed_oks:
            # 成功一次就够，节省调用
            break

    # === 阶段 2: 全失败 → 逐对单独喂 ===
    if not parsed_oks:
        for p in pairs[:MAX_SAMPLES_PER_TRY]:
            parsed, tag = call_once(build_prompt(word, target_lang, [p]))
            if tag == "ok":
                parsed_oks.append(parsed)
            elif tag == "blocked":
                blocked_count += 1
            else:
                neterr_count += 1

    candidates = aggregate_candidates(parsed_oks)
    result = {
        "samples_used": min(MAX_SAMPLES_PER_TRY, len(pairs)),
        "total_pairs": len(pairs),
        "candidates": candidates,
    }
    if not parsed_oks:
        result["note"] = f"all calls blocked by content filter ({blocked_count}) or failed ({neterr_count})"
    elif blocked_count or neterr_count:
        result["note"] = (
            f"recovered after {blocked_count} blocked / {neterr_count} net err; "
            f"{len(parsed_oks)} OK calls aggregated"
        )
    return result


def atomic_save(data: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="target language (e.g. Bodo, Tagin)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    sent_path = os.path.join(SCRIPT_DIR, "sentence_vocab", f"{args.lang}_vocab.json")
    word_path = os.path.join(SCRIPT_DIR, "word_vocab",     f"{args.lang}_vocab_word.json")
    if not os.path.exists(sent_path) or not os.path.exists(word_path):
        sys.exit(f"missing files: {sent_path} or {word_path}")

    with open(sent_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pairs_map = {}
    for item in raw:
        for w, ps in item.items():
            pairs_map[w] = ps

    eligible = {w for w, ps in pairs_map.items() if len(ps) >= 2}

    with open(word_path, "r", encoding="utf-8") as f:
        done = json.load(f)
    missing = sorted(eligible - set(done.keys()))
    print(f"{args.lang}: eligible={len(eligible)}, done={len(done)}, missing={len(missing)}")
    if not missing:
        print("Nothing to recover.")
        return

    results = dict(done)
    counters = {"ok": 0, "empty": 0}

    def worker(w):
        res = recover_word(w, pairs_map[w], args.lang)
        return w, res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, w): w for w in missing}
        for fut in as_completed(futures):
            try:
                w, res = fut.result()
            except Exception as e:
                sys.stderr.write(f"[exc] {futures[fut]}: {e}\n")
                continue
            results[w] = res
            if res["candidates"]:
                counters["ok"] += 1
                tag = "OK"
            else:
                counters["empty"] += 1
                tag = "EMPTY"
            print(f"  [{tag}] {w}: {len(res['candidates'])} candidates" +
                  (f"  ({res.get('note','')})" if res.get('note') else ""))
            # incremental save
            atomic_save(results, word_path)

    print(f"\nDone. recovered={counters['ok']}, still_empty={counters['empty']}, total={len(results)}")
    print(f"Saved to: {word_path}")


if __name__ == "__main__":
    main()
