"""Validate saved experiments and export a report without loading datasets or models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

ARM_ORDER = ("real_fewshot", "full_data", "classroom", "bulk")
COST_LABELS = {
    "simulated": "Simulated teacher cost",
    "estimated": "Estimated API-equivalent teacher cost",
    "token_priced": "Teacher cost from API token usage",
}


class ReportError(ValueError):
    """The saved runs cannot support a complete, comparable report."""


def experiment_settings(cfg: dict) -> dict:
    """Compare scientific settings, excluding output paths and runtime timeouts."""
    return {
        "data": {k: cfg.get("data", {}).get(k) for k in ("dataset", "k_shot")},
        "teacher": {k: cfg.get("teacher", {}).get(k) for k in (
            "provider", "model", "temperature", "max_tokens",
            "price_input_per_mtok", "price_output_per_mtok")},
        **{k: cfg.get(k, {}) for k in ("student", "classroom", "bulk", "budget")},
    }


def _number(value, label: str, minimum=0.0, maximum=math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} must be a number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ReportError(f"{label} is out of range")
    return float(value)


def load_results(cfg: dict) -> tuple[list[dict], dict[str, str]]:
    """Read exactly the configured arms/seeds; never discover extra runs by globbing."""
    exp = cfg["experiment"]
    out = Path(exp.get("results_dir", "results")) / exp["full_name"]
    seeds = exp.get("seeds", [0])
    arms = cfg.get("arms", ["real_fewshot", "classroom", "bulk"])
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ReportError("experiment.seeds must contain distinct integers")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARM_ORDER for a in arms):
        raise ReportError("arms must contain distinct supported arm names")
    missing = [f"{arm}_seed{seed}.json" for arm in arms for seed in seeds
               if not (out / f"{arm}_seed{seed}.json").exists()]
    if missing:
        raise ReportError("Incomplete experiment; missing " + ", ".join(missing))
    rows, hashes = [], {}
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        for seed in seeds:
            path = out / f"{arm}_seed{seed}.json"
            raw = path.read_bytes()
            try:
                r = json.loads(raw)
                if (r["arm"], r["seed"], r["experiment"]) != (arm, seed, exp["full_name"]):
                    raise ReportError(f"{path.name}: result identity does not match its file")
                if experiment_settings(r["config"]) != experiment_settings(cfg):
                    raise ReportError(f"{path.name}: experiment settings differ from the config")
                if r["k_shot"] != cfg["data"]["k_shot"]:
                    raise ReportError(f"{path.name}: k_shot differs from the config")
                for metric in ("accuracy", "macro_f1", "ece"):
                    _number(r["test"][metric], f"{path.name}: {metric}", maximum=1.0)
                _number(r["n_train"], f"{path.name}: n_train")
                spent = _number(r["cost"]["spent_usd"], f"{path.name}: spent_usd")
                limit = r["cost"].get("limit_usd", cfg.get("budget", {}).get("usd_per_arm", 0))
                if spent > _number(limit, f"{path.name}: limit_usd") + 1e-6:
                    raise ReportError(f"{path.name}: spend exceeds the cap")
                if arm in ("real_fewshot", "full_data") and spent != 0:
                    raise ReportError(f"{path.name}: real-data baseline has teacher spend")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ReportError(f"{path.name}: malformed result ({exc})") from exc
            rows.append(r)
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
    by_key = {(r["arm"], r["seed"]): r for r in rows}
    if "bulk" in arms and cfg.get("bulk", {}).get("match_classroom_spend", True):
        if "classroom" not in arms:
            raise ReportError("A matched bulk report also requires classroom results")
        for seed in seeds:
            bulk, classroom = by_key["bulk", seed], by_key["classroom", seed]
            cap = _number(bulk.get("spend_cap_usd"), f"bulk seed {seed}: spend_cap_usd")
            if abs(cap - classroom["cost"]["spent_usd"]) > 1e-6:
                raise ReportError(f"bulk seed {seed}: cap is not matched to classroom spend")
            if bulk["cost"]["spent_usd"] > cap + 1e-6:
                raise ReportError(f"bulk seed {seed}: spend exceeds matched cap")
    return rows, hashes


def _stats(values) -> dict:
    values = list(values)
    return {"mean": fmean(values), "std": pstdev(values), "n": len(values)}


def aggregate(cfg: dict, rows: list[dict], hashes: dict[str, str]) -> dict:
    provider = cfg["teacher"]["provider"]
    basis = "simulated" if provider == "mock" else "estimated" if provider == "queue" else "token_priced"
    groups = {}
    for arm in ARM_ORDER:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        groups[arm] = {
            **{m: _stats(r["test"][m] for r in rs) for m in ("accuracy", "macro_f1", "ece")},
            "spent_usd": _stats(r["cost"]["spent_usd"] for r in rs),
            "n_train": _stats(r["n_train"] for r in rs),
        }
    pairs = []
    by_key = {(r["arm"], r["seed"]): r for r in rows}
    if "classroom" in groups and "bulk" in groups:
        for seed in cfg["experiment"].get("seeds", [0]):
            cls, bulk = by_key["classroom", seed], by_key["bulk", seed]
            pairs.append({
                "seed": seed,
                "classroom_accuracy": cls["test"]["accuracy"],
                "bulk_accuracy": bulk["test"]["accuracy"],
                "accuracy_difference_pp": 100 * (cls["test"]["accuracy"] - bulk["test"]["accuracy"]),
                "bulk_underspend_usd": cls["cost"]["spent_usd"] - bulk["cost"]["spent_usd"],
            })
    return {
        "schema_version": 1, "experiment": cfg["experiment"]["full_name"],
        "dataset": cfg["data"]["dataset"], "student": cfg["student"]["kind"],
        "teacher": cfg["teacher"]["model"], "seeds": cfg["experiment"].get("seeds", [0]),
        "cost_basis": basis, "standard_deviation": "population (ddof=0)",
        "groups": groups, "pairs": pairs, "source_sha256": hashes,
        "accuracy_difference_pp": _stats(p["accuracy_difference_pp"] for p in pairs) if pairs else None,
    }


def _fmt(stat: dict, scale=1.0, digits=2) -> str:
    return f"{scale * stat['mean']:.{digits}f} ± {scale * stat['std']:.{digits}f}"


def markdown_report(report: dict, rows: list[dict]) -> str:
    basis = report["cost_basis"]
    lines = [
        f"# Experiment report: {report['experiment']}", "",
        f"Dataset: **{report['dataset']}** · Student: **{report['student']}** · Teacher: **{report['teacher']}**",
        f"Seeds: {', '.join(map(str, report['seeds']))}. All configured runs are present.", "",
        f"**Cost basis: {COST_LABELS[basis]}.**",
    ]
    if basis == "estimated":
        lines += ["Queue usage uses characters / 4 and excludes thinking tokens. These amounts",
                  "are API-equivalent estimates, not provider bills or measured subscription costs."]
    elif basis == "simulated":
        lines += ["Mock outputs and costs validate orchestration only. They provide no evidence",
                  "about Banking77 quality, teacher capability, or actual spending."]
    else:
        lines += ["Usage is priced with the saved token rates. An invoice reconciliation is required",
                  "before treating these values as verified provider billing."]
    lines += [
        "Student training, hardware, and subscription costs are excluded.", "",
        "## Results", "",
        "| Arm | Accuracy (%) | Macro-F1 | ECE | Teacher cost (USD) | Training examples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, g in report["groups"].items():
        lines.append(f"| {arm} | {_fmt(g['accuracy'], 100)} | {_fmt(g['macro_f1'], digits=4)} | "
                     f"{_fmt(g['ece'], digits=4)} | {_fmt(g['spent_usd'], digits=4)} | "
                     f"{_fmt(g['n_train'], digits=1)} |")
    lines += ["", "Values are mean ± population standard deviation across seeds (ddof=0).",
              "Error bars describe seed variation; they are not confidence intervals.", "",
              "![Accuracy versus teacher cost](accuracy_vs_spend.png)", "",
              "Faint points are individual seeds; large markers show means with one standard",
              "deviation in each axis. This is an arm comparison, not a budget sweep.", ""]
    if report["pairs"]:
        diff = report["accuracy_difference_pp"]
        wins = sum(p["accuracy_difference_pp"] > 0 for p in report["pairs"])
        lines += ["## Paired classroom versus bulk comparison", "",
                  f"Classroom minus bulk accuracy: **{_fmt(diff)} percentage points**.",
                  f"Classroom wins on {wins} of {len(report['pairs'])} seeds.", "",
                  "| Seed | Classroom accuracy (%) | Bulk accuracy (%) | Difference (pp) | Bulk underspend (USD) |",
                  "|---|---:|---:|---:|---:|"]
        for p in report["pairs"]:
            lines.append(f"| {p['seed']} | {100*p['classroom_accuracy']:.2f} | "
                         f"{100*p['bulk_accuracy']:.2f} | {p['accuracy_difference_pp']:+.2f} | "
                         f"{p['bulk_underspend_usd']:.4f} |")
        lines += ["", "Bulk receives each seed's classroom spend as its cap. Its actual spend may",
                  "be lower because a remaining request cannot fit under the cap. Report that gap",
                  "alongside accuracy. Three seeds alone do not establish statistical significance.", ""]
    lines += ["## Provenance", "",
              "| Result | Git commit | Dirty checkout |",
              "|---|---|---|"]
    for r in rows:
        name = f"{r['arm']}_seed{r['seed']}.json"
        dirty = str(r.get("git_dirty", "unknown")).lower()
        lines.append(f"| [{name}]({name}) | {r.get('git_commit') or 'unknown'} | {dirty} |")
    lines += ["", "Scientific settings are validated against the selected config before reporting.",
              "Dirty historical checkouts cannot be reconstructed from a commit alone.",
              "Source file SHA-256 hashes and unrounded aggregates are recorded in [report.json](report.json).",
              "Teacher text and official test examples are excluded from this report.", ""]
    return "\n".join(lines)


def plot_report(report: dict, rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ReportError('Install plotting support: python -m pip install -e ".[report]"') from exc
    colors = {"real_fewshot": "#667085", "full_data": "#8b5cf6",
              "classroom": "#137c66", "bulk": "#d57724"}
    labels = {"real_fewshot": "Real few-shot", "full_data": "Full real data",
              "classroom": "Adaptive classroom", "bulk": "Bulk generation"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "homeroom-report"}):
        fig, ax = plt.subplots(figsize=(8, 5.3))
        for arm, g in report["groups"].items():
            rs = [r for r in rows if r["arm"] == arm]
            ax.scatter([r["cost"]["spent_usd"] for r in rs],
                       [100*r["test"]["accuracy"] for r in rs],
                       color=colors[arm], alpha=0.35, s=30, zorder=3)
            ax.errorbar(g["spent_usd"]["mean"], 100*g["accuracy"]["mean"],
                        xerr=g["spent_usd"]["std"], yerr=100*g["accuracy"]["std"],
                        fmt="o", color=colors[arm], markersize=8, capsize=4,
                        label=labels[arm], zorder=4)
        ax.set(xlabel=COST_LABELS[report["cost_basis"]] + " (USD)",
               ylabel="Held-out test accuracy (%)",
               title=f"{report['dataset']} · {len(report['seeds'])} seeds")
        ax.grid(axis="y", alpha=0.18)
        ax.margins(x=0.12, y=0.2)
        ax.legend(loc="best", frameon=False)
        note = "Mock teacher · demonstration only" if report["cost_basis"] == "simulated" else (
            "Cost excludes student training · bars = seed standard deviation")
        fig.text(0.12, 0.025, note, fontsize=9, color="#667085")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        try:
            fig.savefig(out / "accuracy_vs_spend.png", dpi=180)
            fig.savefig(out / "accuracy_vs_spend.svg", metadata={"Date": None})
        finally:
            plt.close(fig)


def write_report(cfg: dict) -> Path:
    rows, hashes = load_results(cfg)
    report = aggregate(cfg, rows, hashes)
    out = Path(cfg["experiment"].get("results_dir", "results")) / cfg["experiment"]["full_name"]
    plot_report(report, rows, out)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown_report(report, rows), encoding="utf-8")
    return out / "report.md"
