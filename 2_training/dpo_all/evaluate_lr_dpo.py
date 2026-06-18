#!/usr/bin/env python3
"""Compare DPO LoRA training results across learning rates.

Two modes:
  1) Per-task (invoked by run_all_dpo.sh after each task):
       python3 evaluate_lr_dpo.py --task karbi
     Reads <OUTPUT_ROOT>/dpo_<task>/<lr>/eval_results.json and
     train_results.json for every learning rate, picks the best LR by
     eval_loss when available, otherwise falls back to train_loss, and
     writes the report to /base/rd1/dpo_logs/<task>/lr_comparison_report.json.

  2) Full sweep (all 5 tasks):
       python3 evaluate_lr_dpo.py

Note: DPO yamls in this directory have no `### eval` section, so eval is
disabled and only train_results.json will be produced. The script then ranks
by train_loss -- treat that ranking as a sanity check, not a quality signal
(lower train_loss can mean overfitting).
"""

import argparse
import json
from pathlib import Path

OUTPUT_ROOT = Path("/base/rd1/large_models/train_save/lora/wmt_total")
LOG_ROOT = Path("/base/rd1/dpo_logs")
TASKS = ["karbi", "targin", "kokborok", "nagamese", "bodo"]
LRS = ["1e-4", "2e-4", "5e-4"]


def task_dir(task: str) -> Path:
    return OUTPUT_ROOT / f"dpo_{task}"


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
        run_dir = task_dir(task) / lr
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
    print(f"\n=== DPO Task: {task} ===")
    print(
        f"{'lr':<8}{'eval_loss':>12}{'train_loss':>13}"
        f"{'train_runtime':>16}{'train_samples/s':>17}"
    )
    print("-" * 66)
    for r in rows:
        print(
            f"{r['lr']:<8}"
            f"{fmt(r['eval_loss']):>12}"
            f"{fmt(r['train_loss']):>13}"
            f"{fmt(r['train_runtime'], '.2f'):>16}"
            f"{fmt(r['train_samples_per_second'], '.3f'):>17}"
        )


def pick_best(rows):
    """Prefer eval_loss; fall back to train_loss when eval is disabled.

    Returns (best_row, metric_used) or (None, None) when nothing is loadable.
    """
    eval_candidates = [r for r in rows if r["eval_loss"] is not None]
    if eval_candidates:
        best = min(eval_candidates, key=lambda r: r["eval_loss"])
        return best, "eval_loss"
    train_candidates = [r for r in rows if r["train_loss"] is not None]
    if train_candidates:
        best = min(train_candidates, key=lambda r: r["train_loss"])
        return best, "train_loss"
    return None, None


def write_task_report(task: str, rows, best, metric):
    report = {
        "task": task,
        "stage": "dpo",
        "output_root": str(OUTPUT_ROOT),
        "task_dir": str(task_dir(task)),
        "learning_rates": LRS,
        "metric_used": metric,
        "rows": rows,
        "best": (
            None
            if best is None
            else {
                "lr": best["lr"],
                "metric": metric,
                "metric_value": best[metric],
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
    print(f"\nReading DPO results for task '{task}' from: {task_dir(task)}")
    rows = collect_metrics(task)
    print_task_table(task, rows)
    best, metric = pick_best(rows)
    if best is None:
        print(f"  [WARN] no train/eval results.json available for DPO task '{task}'")
    else:
        note = "" if metric == "eval_loss" else "  (fallback: eval disabled)"
        print(
            f"  >>> BEST lr for DPO {task}: {best['lr']} "
            f"({metric}={best[metric]:.4f}){note}"
        )
    write_task_report(task, rows, best, metric)
    return rows, best, metric


def run_all():
    print(f"Reading DPO LoRA results from: {OUTPUT_ROOT}")
    full_report = {"stage": "dpo", "tasks": {}, "best_per_task": {}}

    for task in TASKS:
        rows, best, metric = evaluate_task(task)
        full_report["tasks"][task] = rows
        full_report["best_per_task"][task] = (
            None
            if best is None
            else {
                "lr": best["lr"],
                "metric": metric,
                "metric_value": best[metric],
                "eval_loss": best["eval_loss"],
                "train_loss": best["train_loss"],
                "output_dir": best["output_dir"],
            }
        )

    print("\n========== DPO SUMMARY (best learning rate per task) ==========")
    print(f"{'task':<12}{'best_lr':<10}{'metric':<12}{'value':>12}")
    print("-" * 46)
    for task in TASKS:
        info = full_report["best_per_task"].get(task)
        if info is None:
            print(f"{task:<12}{'-':<10}{'-':<12}{'-':>12}")
        else:
            print(
                f"{task:<12}{info['lr']:<10}{info['metric']:<12}"
                f"{fmt(info['metric_value']):>12}"
            )

    combined_path = OUTPUT_ROOT / "lr_comparison_report_dpo.json"
    try:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"\nCombined DPO report written to: {combined_path}")
    except Exception as e:
        print(f"[WARN] failed to write combined report: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=TASKS,
        help="Evaluate a single DPO task. If omitted, evaluates all tasks.",
    )
    args = parser.parse_args()

    if args.task:
        evaluate_task(args.task)
    else:
        run_all()


if __name__ == "__main__":
    main()
