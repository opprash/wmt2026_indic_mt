#!/usr/bin/env python3
"""Compare LoRA training results across learning rates.

Two modes:
  1) Per-task evaluation (recommended, invoked by run_all.sh after each task):
       python3 evaluate_lr.py --task karbi
     Reads <OUTPUT_ROOT>/<task>/<lr>/eval_results.json and train_results.json
     for every learning rate, picks the best by eval_loss, and writes the
     report to /base/rd1/train_logs/<task>/lr_comparison_report.json.

  2) Full sweep (all 5 tasks):
       python3 evaluate_lr.py
     Prints all per-task tables plus a global summary, writes one combined
     report under <OUTPUT_ROOT>/lr_comparison_report.json AND a per-task
     report under each /base/rd1/train_logs/<task>/.
"""

import argparse
import json
from pathlib import Path

OUTPUT_ROOT = Path("/base/rd1/large_models/train_save/lora/wmt_total")
LOG_ROOT = Path("/base/rd1/train_logs")
TASKS = ["karbi", "targin", "kokborok", "nagamese", "bodo"]
LRS = ["5e-5", "1e-4", "2e-4", "5e-4"]


def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] failed to load {path}: {e}")
        return None


def collect_metrics(task: str):
    rows = []
    for lr in LRS:
        run_dir = OUTPUT_ROOT / task / lr
        eval_data = load_json(run_dir / "eval_results.json")
        train_data = load_json(run_dir / "train_results.json")

        row = {
            "task": task,
            "lr": lr,
            "output_dir": str(run_dir),
            "eval_loss": eval_data.get("eval_loss") if eval_data else None,
            "eval_runtime": eval_data.get("eval_runtime") if eval_data else None,
            "eval_samples_per_second": (
                eval_data.get("eval_samples_per_second") if eval_data else None
            ),
            "train_loss": train_data.get("train_loss") if train_data else None,
            "train_runtime": train_data.get("train_runtime") if train_data else None,
            "train_samples_per_second": (
                train_data.get("train_samples_per_second") if train_data else None
            ),
            "epoch": (
                eval_data.get("epoch")
                if eval_data
                else (train_data.get("epoch") if train_data else None)
            ),
            "has_eval": eval_data is not None,
            "has_train": train_data is not None,
        }
        rows.append(row)
    return rows


def fmt(v, spec=".4f"):
    if v is None:
        return "    -    "
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def print_task_table(task: str, rows):
    print(f"\n=== Task: {task} ===")
    print(
        f"{'lr':<8}{'eval_loss':>12}{'train_loss':>13}"
        f"{'train_runtime':>16}{'eval_samples/s':>17}"
    )
    print("-" * 66)
    for r in rows:
        print(
            f"{r['lr']:<8}"
            f"{fmt(r['eval_loss']):>12}"
            f"{fmt(r['train_loss']):>13}"
            f"{fmt(r['train_runtime'], '.2f'):>16}"
            f"{fmt(r['eval_samples_per_second'], '.3f'):>17}"
        )


def pick_best(rows):
    candidates = [r for r in rows if r["eval_loss"] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["eval_loss"])


def write_task_report(task: str, rows, best):
    report = {
        "task": task,
        "output_root": str(OUTPUT_ROOT),
        "learning_rates": LRS,
        "rows": rows,
        "best": (
            None
            if best is None
            else {
                "lr": best["lr"],
                "eval_loss": best["eval_loss"],
                "train_loss": best["train_loss"],
                "output_dir": best["output_dir"],
            }
        ),
    }
    report_path = LOG_ROOT / task / "lr_comparison_report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Report written: {report_path}")
    except Exception as e:
        print(f"  [WARN] failed to write report {report_path}: {e}")


def evaluate_task(task: str):
    print(f"\nReading results for task '{task}' from: {OUTPUT_ROOT / task}")
    rows = collect_metrics(task)
    print_task_table(task, rows)
    best = pick_best(rows)
    if best is None:
        print(f"  [WARN] no eval_results.json available for task '{task}'")
    else:
        print(
            f"  >>> BEST lr for {task}: {best['lr']} "
            f"(eval_loss={best['eval_loss']:.4f})"
        )
    write_task_report(task, rows, best)
    return rows, best


def run_all():
    print(f"Reading LoRA results from: {OUTPUT_ROOT}")
    full_report = {"tasks": {}, "best_per_task": {}}

    for task in TASKS:
        rows, best = evaluate_task(task)
        full_report["tasks"][task] = rows
        full_report["best_per_task"][task] = (
            None
            if best is None
            else {
                "lr": best["lr"],
                "eval_loss": best["eval_loss"],
                "train_loss": best["train_loss"],
                "output_dir": best["output_dir"],
            }
        )

    print("\n========== SUMMARY (best learning rate per task) ==========")
    print(f"{'task':<12}{'best_lr':<10}{'eval_loss':>12}{'train_loss':>13}")
    print("-" * 47)
    for task in TASKS:
        info = full_report["best_per_task"].get(task)
        if info is None:
            print(f"{task:<12}{'-':<10}{'-':>12}{'-':>13}")
        else:
            print(
                f"{task:<12}{info['lr']:<10}"
                f"{fmt(info['eval_loss']):>12}"
                f"{fmt(info['train_loss']):>13}"
            )

    combined_path = OUTPUT_ROOT / "lr_comparison_report.json"
    try:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"\nCombined report written to: {combined_path}")
    except Exception as e:
        print(f"[WARN] failed to write combined report: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=TASKS,
        help="Evaluate a single task. If omitted, evaluates all tasks.",
    )
    args = parser.parse_args()

    if args.task:
        evaluate_task(args.task)
    else:
        run_all()


if __name__ == "__main__":
    main()
