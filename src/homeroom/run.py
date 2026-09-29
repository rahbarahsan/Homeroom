"""CLI: homeroom run | summarize | check-gpu"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .arms import ARM_ORDER, ARMS, ArmContext
from .config import load_config, results_dir
from .data import load_task, sample_k_shot


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def cmd_run(args) -> None:
    cfg = load_config(args.config, dry_run=args.dry_run)
    out = results_dir(cfg)
    arms = args.arms.split(",") if args.arms else cfg.get("arms", ["real_fewshot", "classroom", "bulk"])
    arms = [a for a in ARM_ORDER if a in arms]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["experiment"].get("seeds", [0])
    task = load_task(cfg["data"]["dataset"], cfg["data"].get("cache_dir", "data/raw"))
    k = int(cfg["data"]["k_shot"])
    print(f"[homeroom] experiment={cfg['experiment']['full_name']} task={task.name} "
          f"classes={task.n_classes} k={k} arms={arms} seeds={seeds} student={cfg['student']['kind']} "
          f"teacher={cfg['teacher']['provider']}:{cfg['teacher'].get('model')}")
    for seed in seeds:
        seeds_df, _ = sample_k_shot(task.train, k, seed)
        ctx = ArmContext(cfg, task, seed, seeds_df, out)
        for arm in arms:
            path = out / f"{arm}_seed{seed}.json"
            if path.exists() and not args.force:
                print(f"  skip {path.name} (exists; use --force)")
                continue
            kwargs = {}
            if arm == "bulk" and cfg.get("bulk", {}).get("match_classroom_spend", True):
                cpath = out / f"classroom_seed{seed}.json"
                if cpath.exists():
                    kwargs["spend_cap"] = json.loads(cpath.read_text())["cost"]["spent_usd"]
            print(f"  running {arm} seed={seed} ...", flush=True)
            res = ARMS[arm](ctx, **kwargs)
            res.update({"experiment": cfg["experiment"]["full_name"], "git_commit": _git_commit(),
                        "student": cfg["student"], "teacher": {k2: v for k2, v in cfg["teacher"].items()
                                                               if "key" not in k2.lower()}})
            path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            t = res["test"]
            print(f"    -> acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} ece={t['ece']:.3f} "
                  f"spent=${res['cost']['spent_usd']:.4f} n_train={res['n_train']}")
    cmd_summarize(args)


def cmd_summarize(args) -> None:
    cfg = load_config(args.config, dry_run=getattr(args, "dry_run", False))
    out = results_dir(cfg)
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*_seed*.json"))]
    if not rows:
        print("no results yet"); return
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    base = by_arm.get("real_fewshot")
    base_acc = np.mean([r["test"]["accuracy"] for r in base]) if base else None

    def ms(xs):
        return f"{np.mean(xs):.4f} ± {np.std(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"

    lines = [f"# Results: {cfg['experiment']['full_name']}", "",
             "| Arm | Seeds | Accuracy | Macro-F1 | ECE | Spend (USD) | n_train | Δacc vs real_fewshot per $ |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in [a for a in ARM_ORDER if a in by_arm]:
        rs = by_arm[arm]
        acc = [r["test"]["accuracy"] for r in rs]
        spend = [r["cost"]["spent_usd"] for r in rs]
        eff = "—"
        if base_acc is not None and np.mean(spend) > 0:
            eff = f"{(np.mean(acc) - base_acc) * 100 / np.mean(spend):.2f} pts/$"
        lines.append(f"| {arm} | {len(rs)} | {ms(acc)} | {ms([r['test']['macro_f1'] for r in rs])} | "
                     f"{ms([r['test']['ece'] for r in rs])} | {np.mean(spend):.4f} | "
                     f"{int(np.mean([r['n_train'] for r in rs]))} | {eff} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_check_gpu(_args) -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed. Install the CUDA wheel for your system from https://pytorch.org, "
              "then `pip install -e .[gpu]`."); return
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {p.name} | compute {cap[0]}.{cap[1]} | VRAM free {free/2**30:.1f} / {total/2**30:.1f} GiB")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    if cap[0] < 8:
        print("Advice: pre-Ampere GPU -> use fp16 (not bf16); do not install flash-attn; use SDPA.")
    if total / 2**30 < 8:
        print("Advice: <8 GB VRAM -> SetFit / ModernBERT-base full FT OK; large encoders via LoRA or 8-bit "
              "Adam; decoders ≤1.5B only with 4-bit QLoRA; rent a GPU for ≥3B.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="homeroom")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run experiment arms")
    r.add_argument("--config", required=True)
    r.add_argument("--arms", help="comma list: real_fewshot,full_data,classroom,bulk")
    r.add_argument("--seeds", help="comma list, e.g. 0,1,2")
    r.add_argument("--dry-run", action="store_true", help="mock teacher + tf-idf student, no cost")
    r.add_argument("--force", action="store_true", help="overwrite existing results")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize", help="aggregate results into summary.md")
    s.add_argument("--config", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_summarize)
    g = sub.add_parser("check-gpu", help="print GPU capabilities and advice")
    g.set_defaults(func=cmd_check_gpu)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
