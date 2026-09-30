"""Experiment arms. Each returns a result dict; run.py writes it to results/.

Final evaluation on the official test set happens ONLY in `final_eval`.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts as P
from .budget import Budget
from .config import generated_dir
from .data import TaskData
from .evaluate import compute_metrics, top_confusions
from .students import free_gpu, make_student
from .teacher import Request, TeacherSession, make_teacher


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


@dataclass
class ArmContext:
    cfg: dict
    task: TaskData
    seed: int
    seeds_df: pd.DataFrame      # k real examples per intent (the only real training data)
    out_dir: Path

    def session(self, arm: str, limit_usd: float) -> TeacherSession:
        t = self.cfg["teacher"]
        budget = Budget(limit_usd, float(t.get("price_input_per_mtok") or 0),
                        float(t.get("price_output_per_mtok") or 0),
                        ledger_path=self.out_dir / "ledgers" / f"{arm}_seed{self.seed}.jsonl")
        cache = Path(self.cfg["data"].get("teacher_cache_dir", "data/cache")) / self.cfg["experiment"]["full_name"]
        return TeacherSession(make_teacher(self.cfg), budget, cache)

    def seeds_for(self, label: int) -> list[str]:
        return self.seeds_df.loc[self.seeds_df.label == label, "text"].tolist()

    def train(self, texts, labels):
        return make_student(self.cfg, self.task.n_classes, self.seed).fit(texts, labels)


def final_eval(ctx: ArmContext, student) -> dict:
    test = ctx.task.test
    return compute_metrics(test.label.to_numpy(), student.predict_proba(test.text.tolist()))


def _result(ctx, arm, student, n_train, extra=None, budget=None, t0=None) -> dict:
    res = {"arm": arm, "seed": ctx.seed, "k_shot": int(ctx.cfg["data"]["k_shot"]),
           "n_train": int(n_train), "test": final_eval(ctx, student),
           "cost": budget.summary() if budget else {"spent_usd": 0.0, "real_spent_usd": 0.0},
           "runtime_s": round(time.time() - (t0 or time.time()), 1)}
    res.update(extra or {})
    return res


# ---------------------------------------------------------------- baselines (no teacher)

def run_real_fewshot(ctx: ArmContext) -> dict:
    t0 = time.time()
    s = ctx.train(ctx.seeds_df.text.tolist(), ctx.seeds_df.label.tolist())
    return _result(ctx, "real_fewshot", s, len(ctx.seeds_df), t0=t0)


def run_full_data(ctx: ArmContext) -> dict:
    t0 = time.time()
    tr = ctx.task.train
    s = ctx.train(tr.text.tolist(), tr.label.tolist())
    return _result(ctx, "full_data", s, len(tr), t0=t0)


# ---------------------------------------------------------------- bulk synthetic

def run_bulk(ctx: ArmContext, spend_cap: float | None = None) -> dict:
    t0 = time.time()
    bc = ctx.cfg.get("bulk", {})
    cap = spend_cap if spend_cap is not None else float(ctx.cfg["budget"]["usd_per_arm"])
    sess = ctx.session("bulk", cap)
    n_per_call = int(bc.get("examples_per_call", 20))
    max_per_intent = int(bc.get("max_examples_per_intent", 200))
    labels = ctx.task.labels
    texts, ys = ctx.seeds_df.text.tolist(), ctx.seeds_df.label.tolist()
    seen = {norm(t) for t in texts}
    counts = {i: 0 for i in range(len(labels))}
    stopped_by, call_i, empty_passes = "cap_per_intent", 0, 0
    while any(c < max_per_intent for c in counts.values()) and empty_passes < 3:
        added_this_pass = 0
        batch, batch_labels = [], []
        for li, intent in enumerate(labels):
            if counts[li] >= max_per_intent:
                continue
            seeds = ctx.seeds_for(li)
            batch.append(Request(P.SYSTEM, P.bulk_prompt(ctx.task.description, intent, seeds, n_per_call),
                                 "bulk", {"intent": intent, "seeds": seeds, "n": n_per_call},
                                 f"seed={ctx.seed}|bulk|{intent}|{call_i}"))
            batch_labels.append(li)
            call_i += 1
        replies, budget_stopped = sess.ask_many(batch)
        for li, r in zip(batch_labels, replies):
            try:
                new = P.clean_examples(P.parse_json(r.text).get("examples"))
            except ValueError:
                continue
            for t in new:
                if counts[li] < max_per_intent and norm(t) not in seen:
                    seen.add(norm(t)); texts.append(t); ys.append(li); counts[li] += 1; added_this_pass += 1
        if budget_stopped:
            stopped_by = "budget"
            break
        empty_passes = empty_passes + 1 if added_this_pass == 0 else 0
    s = ctx.train(texts, ys)
    return _result(ctx, "bulk", s, len(texts), budget=sess.budget, t0=t0,
                   extra={"stopped_by": stopped_by, "n_generated": len(texts) - len(ctx.seeds_df),
                          "spend_cap_usd": cap})


# ---------------------------------------------------------------- classroom

def run_classroom(ctx: ArmContext) -> dict:
    t0 = time.time()
    cc = ctx.cfg.get("classroom", {})
    sess = ctx.session("classroom", float(ctx.cfg["budget"]["usd_per_arm"]))
    task, labels = ctx.task, ctx.task.labels
    rng = np.random.default_rng(ctx.seed)
    stopped_by, history, rules = "max_rounds", [], []

    texts, ys = ctx.seeds_df.text.tolist(), ctx.seeds_df.label.tolist()
    seen = {norm(t) for t in texts}
    definitions: dict[int, str] = {}
    diag, score = [], []  # exam halves: (text, label)

    def add(new, label) -> int:
        n = 0
        for t in new:
            k = norm(t)
            if k and k not in seen:
                seen.add(k); texts.append(t); ys.append(label); n += 1
        return n

    best = {"acc": -1.0, "student": None, "n_train": 0, "round": -1}
    # 1) Exam, written BEFORE lessons; split into diagnostic (shown to teacher) and score halves.
    n_exam = int(cc.get("exam_items_per_intent", 6))
    replies, budget_stopped = sess.ask_many([
        Request(P.SYSTEM, P.exam_prompt(task.description, intent, labels, n_exam), "exam",
                {"intent": intent, "seeds": ctx.seeds_for(li), "n": n_exam}, f"seed={ctx.seed}|exam|{intent}")
        for li, intent in enumerate(labels)])
    for li, r in enumerate(replies):
        try:
            items = P.clean_examples(P.parse_json(r.text).get("items"))
        except ValueError:
            items = []
        items = [t for t in items if norm(t) not in seen]
        rng.shuffle(items)
        half = len(items) // 2
        diag += [(t, li) for t in items[:half]]
        score += [(t, li) for t in items[half:]]
    seen |= {norm(t) for t, _ in diag + score}  # never train on exam items

    # 2) Lessons: definition + confusables + examples per intent.
    if budget_stopped:
        stopped_by = "budget"
    else:
        n_lesson = int(cc.get("lesson_examples_per_intent", 10))
        replies, budget_stopped = sess.ask_many([
            Request(P.SYSTEM, P.lesson_prompt(task.description, intent, labels, ctx.seeds_for(li), n_lesson),
                    "lesson", {"intent": intent, "seeds": ctx.seeds_for(li), "n": n_lesson},
                    f"seed={ctx.seed}|lesson|{intent}")
            for li, intent in enumerate(labels)])
        for li, r in enumerate(replies):
            try:
                d = P.parse_json(r.text)
                definitions[li] = str(d.get("definition", ""))
                add(P.clean_examples(d.get("examples")), li)
            except ValueError:
                pass
        if budget_stopped:
            stopped_by = "budget"

    # 3) Rounds: train -> exam -> diagnose -> remediate.
    max_rounds = int(cc.get("max_rounds", 4))
    min_gain = float(cc.get("stop_if_score_gain_below", 0.002))
    n_conf = int(cc.get("confusions_per_round", 15))
    n_rem = int(cc.get("examples_per_confusion", 8))
    d_texts, d_y = [t for t, _ in diag], np.array([y for _, y in diag], dtype=int)
    s_texts, s_y = [t for t, _ in score], np.array([y for _, y in score], dtype=int)

    for rnd in range(max_rounds + 1):
        student = ctx.train(texts, ys)
        s_acc = float((student.predict_proba(s_texts).argmax(1) == s_y).mean()) if len(s_y) else 0.0
        d_pred = student.predict_proba(d_texts).argmax(1) if len(d_y) else np.array([], dtype=int)
        history.append({"round": rnd, "n_train": len(texts), "score_acc": round(s_acc, 4),
                        "diag_acc": round(float((d_pred == d_y).mean()), 4) if len(d_y) else None,
                        "spent_usd": round(sess.budget.spent, 6)})
        improved = s_acc > best["acc"] + (min_gain if rnd > 0 else -1)
        if improved:
            best.update(acc=s_acc, student=student, n_train=len(texts), round=rnd)
        else:
            del student; free_gpu()
            stopped_by = "no_improvement"
            break
        if stopped_by == "budget" or rnd == max_rounds:
            break
        confusions = top_confusions(d_y, d_pred, n_conf)
        if not confusions:
            stopped_by = "no_confusions"
            break
        reqs = []
        for a, b, _ in confusions:
            mistakes = [t for t, y, p in zip(d_texts, d_y, d_pred) if y == a and p == b]
            reqs.append(Request(P.SYSTEM, P.remedial_prompt(task.description, labels[a], labels[b],
                                                            definitions.get(a, ""), definitions.get(b, ""),
                                                            mistakes, n_rem),
                                "remedial", {"seeds_a": ctx.seeds_for(a), "seeds_b": ctx.seeds_for(b), "n": n_rem},
                                f"seed={ctx.seed}|remedial|r{rnd}|{a}|{b}"))
        replies, budget_stopped = sess.ask_many(reqs)
        for (a, b, _), r in zip(confusions, replies):
            try:
                d = P.parse_json(r.text)
            except ValueError:
                continue
            rules.append({"round": rnd, "a": labels[a], "b": labels[b], "rule": d.get("rule")})
            add(P.clean_examples(d.get("a_examples")), a)
            add(P.clean_examples(d.get("b_examples")), b)
        if budget_stopped:
            stopped_by = "budget"
            # loop continues once more: retrain on what we have, then stop

    # Rules are teacher-written text: keep them out of the committed result JSON (hard rules 5/6).
    rules_path = generated_dir(ctx.cfg) / f"classroom_rules_seed{ctx.seed}.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    return _result(ctx, "classroom", best["student"], best["n_train"], budget=sess.budget, t0=t0,
                   extra={"stopped_by": stopped_by, "best_round": best["round"], "history": history,
                          "exam_sizes": {"diagnostic": len(diag), "score": len(score)},
                          "n_generated": best["n_train"] - len(ctx.seeds_df), "n_rules": len(rules)})


ARMS = {"real_fewshot": run_real_fewshot, "full_data": run_full_data,
        "classroom": run_classroom, "bulk": run_bulk}
ARM_ORDER = ["real_fewshot", "full_data", "classroom", "bulk"]  # bulk after classroom (spend matching)
