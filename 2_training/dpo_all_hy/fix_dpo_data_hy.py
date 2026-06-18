#!/usr/bin/env python3
"""Fix Hunyuan-DPO data: rewrite chosen.from and rejected.from to "gpt" in place.

Targets 5 files at /base/rd1/data/dataset_all/hy_<lang>_dpo_all.json:
    hy_karbi_dpo_all.json
    hy_targin_dpo_all.json
    hy_kokborok_dpo_all.json
    hy_nagamese_dpo_all.json
    hy_bodo_dpo_all.json

LlamaFactory's ShareGPT-DPO loader requires `chosen.from` and `rejected.from`
to match the configured `assistant_tag` (default "gpt"). The raw data uses
"human"/"sft" which fails parsing with `KeyError: 'sft'`. This script rewrites
all 5 files in place; `conversations[].from = "human"` (the user prompt tag)
is left untouched.

Run:
    python3 fix_dpo_data_hy.py
    python3 fix_dpo_data_hy.py --data-dir /custom/path/to/dataset_all
"""

import argparse
import json
import sys
from pathlib import Path

LANGS = ["karbi", "targin", "kokborok", "nagamese", "bodo"]
DEFAULT_DATA_DIR = "/base/rd1/data/dataset_all"
ASSISTANT_TAG = "gpt"


def find_file(data_dir: Path, lang: str):
    for name in (f"hy_{lang}_dpo_all.json", f"hy_{lang}_dpo_all"):
        p = data_dir / name
        if p.is_file():
            return p
    return None


def load_data(path: Path):
    """Load JSON array, single object, or JSONL. Returns (records, is_jsonl)."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse failed at line {ln}: {e}")
        return records, True

    if isinstance(data, list):
        return data, False
    if isinstance(data, dict):
        return [data], False
    raise ValueError(f"unexpected top-level type {type(data).__name__}")


def save_data(path: Path, records, is_jsonl: bool):
    if is_jsonl:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)


def fix_record(rec, stats):
    for key in ("chosen", "rejected"):
        node = rec.get(key)
        if not isinstance(node, dict) or "from" not in node:
            stats[f"missing_{key}"] += 1
            continue
        old = node["from"]
        if old == ASSISTANT_TAG:
            stats["already_ok"] += 1
            continue
        node["from"] = ASSISTANT_TAG
        stats["fixed"] += 1
        stats["fixed_by_old"][old] = stats["fixed_by_old"].get(old, 0) + 1


def fix_file(path: Path):
    records, is_jsonl = load_data(path)
    stats = {
        "fixed": 0,
        "already_ok": 0,
        "missing_chosen": 0,
        "missing_rejected": 0,
        "fixed_by_old": {},
    }
    for rec in records:
        if not isinstance(rec, dict):
            continue
        fix_record(rec, stats)
    save_data(path, records, is_jsonl)

    fmt = "jsonl" if is_jsonl else "json"
    by_old = ", ".join(f"{k}={v}" for k, v in stats["fixed_by_old"].items()) or "-"
    print(
        f"  {path.name} [{fmt}]: records={len(records)} "
        f"fixed={stats['fixed']} (was: {by_old}) "
        f"already_gpt={stats['already_ok']} "
        f"missing_chosen={stats['missing_chosen']} "
        f"missing_rejected={stats['missing_rejected']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"directory containing the hy_<lang>_dpo_all.json files (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] data dir not found: {data_dir}")
        sys.exit(1)

    print(f"Fixing Hunyuan-DPO files under {data_dir.resolve()}\n")
    n_ok = 0
    for lang in LANGS:
        path = find_file(data_dir, lang)
        if path is None:
            print(f"  [WARN] {lang}: file not found (hy_{lang}_dpo_all[.json])")
            continue
        try:
            fix_file(path)
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] hy_{lang}_dpo_all.json: {e}")

    print(f"\nDone. {n_ok}/{len(LANGS)} files processed.")


if __name__ == "__main__":
    main()
