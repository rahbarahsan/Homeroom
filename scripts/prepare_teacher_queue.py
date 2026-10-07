"""Prewarm fixed exam/lesson requests through TeacherSession using training data only.

Cached replies remain charged at their original accounted cost in the main run.
This helper never loads official test examples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from homeroom import prompts as P
from homeroom.budget import Budget
from homeroom.config import load_config
from homeroom.data import TASK_DESCRIPTIONS, sample_k_shot
from homeroom.teacher import Request, TeacherSession, make_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["data"]["dataset"] != "banking77" or cfg["teacher"]["provider"] != "queue":
        raise SystemExit("prewarming supports the Banking77 queue teacher only")
    path = Path(cfg["data"].get("cache_dir", "data/raw")) / "banking77" / "train.csv"
    df = pd.read_csv(path).rename(columns={"category": "intent"})
    labels = sorted(df.intent.unique())
    indices = {label: i for i, label in enumerate(labels)}
    df["label"] = df.intent.map(indices)
    exp = cfg["experiment"]["full_name"]
    t, cc = cfg["teacher"], cfg["classroom"]
    for seed in cfg["experiment"]["seeds"]:
        sampled, _ = sample_k_shot(df[["text", "label"]], cfg["data"]["k_shot"], seed)
        budget = Budget(cfg["budget"]["usd_per_arm"], t["price_input_per_mtok"], t["price_output_per_mtok"],
                        ledger_path=Path("data/generated") / exp / f"prewarm_seed{seed}.jsonl")
        session = TeacherSession(make_teacher(cfg), budget, Path(cfg["data"]["teacher_cache_dir"]) / exp)
        for task in ("exam", "lesson"):
            requests = []
            for li, intent in enumerate(labels):
                seeds = sampled.loc[sampled.label == li, "text"].tolist()
                n = cc["exam_items_per_intent" if task == "exam" else "lesson_examples_per_intent"]
                prompt = P.exam_prompt if task == "exam" else P.lesson_prompt
                requests.append(Request(P.SYSTEM, prompt(TASK_DESCRIPTIONS["banking77"], intent, labels, seeds, n),
                                        task, {"intent": intent, "seeds": seeds, "n": n},
                                        f"seed={seed}|{task}|{intent}"))
            print(f"Preparing seed={seed} {task}: {len(requests)} budgeted requests", flush=True)
            replies, stopped = session.ask_many(requests)
            print(f"Prepared {len(replies)} replies; accounted={chr(36)}{budget.spent:.6f}", flush=True)
            if stopped:
                break


if __name__ == "__main__":
    main()
