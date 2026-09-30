"""Milestone 2 debugging: evaluate the student on a DEV split carved from UNUSED train data.

Never touches the official test set (hard rule 2). The dev split is sampled from the train rows
left over after the k-shot seed sample, with its own seed, so it never overlaps the student's data.

For each seed: train the SetFit student exactly as `real_fewshot` does (optionally with config
overrides), then compare classifier heads fitted on the same trained embeddings.

Usage:
  python scripts/debug_student_dev.py --config configs/pilot_banking77.yaml --seeds 0,1,2 \
      [--set student.setfit.num_iterations=40] [--tag name]
Writes results/dev_debug/<tag>.json (dev metrics only; no teacher text).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from homeroom.config import config_snapshot, load_config
from homeroom.data import load_task, sample_k_shot
from homeroom.evaluate import compute_metrics
from homeroom.students import _full_proba, make_student

DEV_SEED_OFFSET = 10_000


def set_path(cfg: dict, dotted: str, value) -> None:
    d = cfg
    *parents, last = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[last] = value


def dev_split(train, k: int, seed: int, per_intent: int):
    seeds_df, remaining = sample_k_shot(train, k, seed)
    dev, _ = sample_k_shot(remaining, per_intent, seed + DEV_SEED_OFFSET)
    return seeds_df, dev


def head_sweep(x_tr, y_tr, x_dev, y_dev, n_classes: int) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    out = {}
    for c in [1.0, 10.0, 100.0, 1000.0]:
        for max_iter in [100, 5000]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = LogisticRegression(C=c, max_iter=max_iter).fit(x_tr, y_tr)
            p = _full_proba(clf.predict_proba(x_dev), clf.classes_, n_classes)
            m = compute_metrics(y_dev, p)
            m["converged"] = not any(issubclass(x.category, ConvergenceWarning) for x in w)
            out[f"lr_C{c:g}_it{max_iter}"] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dev-per-intent", type=int, default=20)
    ap.add_argument("--set", action="append", default=[], help="dotted.key=yaml_value override")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        set_path(cfg, key, yaml.safe_load(val))
    task = load_task(cfg["data"]["dataset"], cfg["data"].get("cache_dir", "data/raw"))
    k = int(cfg["data"]["k_shot"])

    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        seeds_df, dev = dev_split(task.train, k, seed, args.dev_per_intent)
        t0 = time.time()
        student = make_student(cfg, task.n_classes, seed).fit(seeds_df.text.tolist(), seeds_df.label.tolist())
        train_s = round(time.time() - t0, 1)
        y_dev = dev.label.to_numpy()
        default = compute_metrics(y_dev, student.predict_proba(dev.text.tolist()))
        body = student.model.model_body
        x_tr = body.encode(seeds_df.text.tolist(), normalize_embeddings=student.model.normalize_embeddings)
        x_dev = body.encode(dev.text.tolist(), normalize_embeddings=student.model.normalize_embeddings)
        runs.append({"seed": seed, "train_s": train_s, "n_train": len(seeds_df), "n_dev": len(dev),
                     "embedding_norm_mean": float(np.linalg.norm(x_tr, axis=1).mean()),
                     "default_head": default,
                     "heads": head_sweep(x_tr, seeds_df.label.to_numpy(), x_dev, y_dev, task.n_classes)})
        print(f"seed={seed} train={train_s}s default_head dev_acc={default['accuracy']:.4f} "
              f"ece={default['ece']:.3f}", flush=True)
        for name, m in runs[-1]["heads"].items():
            print(f"    {name:18s} acc={m['accuracy']:.4f} ece={m['ece']:.3f} conv={m['converged']}")

    out = Path("results/dev_debug")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "overrides": args.set, "dev_per_intent": args.dev_per_intent,
         "dev_seed_offset": DEV_SEED_OFFSET, "config": config_snapshot(cfg), "runs": runs}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
