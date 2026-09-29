import json

import numpy as np
import pytest

from homeroom.budget import Budget, BudgetExceeded, Usage
from homeroom.data import load_task, sample_k_shot
from homeroom.evaluate import compute_metrics, ece, top_confusions
from homeroom.prompts import clean_examples, parse_json
from homeroom.teacher import MockTeacher, TeacherSession, make_teacher


def test_budget_cap_and_ledger(tmp_path):
    b = Budget(0.01, 1.0, 5.0, ledger_path=tmp_path / "l.jsonl")
    b.charge(Usage(1000, 1000), tag="x", cached=False)          # 0.001 + 0.005
    assert abs(b.spent - 0.006) < 1e-9
    with pytest.raises(BudgetExceeded):
        b.check(1000, 1000)
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 1


def test_refuse_paid_teacher_without_prices():
    cfg = {"teacher": {"provider": "openai_compatible", "model": "m",
                       "price_input_per_mtok": 0, "price_output_per_mtok": 0}}
    with pytest.raises(ValueError):
        make_teacher(cfg)


def test_cache_hit_is_accounted_but_not_real(tmp_path):
    t = MockTeacher("mock")
    s1 = TeacherSession(t, Budget(1.0, 1.0, 5.0), tmp_path)
    r1 = s1.ask("sys", "prompt", task="bulk", meta={"n": 3, "seeds": ["a"]}, salt="s")
    s2 = TeacherSession(t, Budget(1.0, 1.0, 5.0), tmp_path)
    r2 = s2.ask("sys", "prompt", task="bulk", meta={"n": 3, "seeds": ["a"]}, salt="s")
    assert r2.cached and r1.text == r2.text
    assert s2.budget.real_spent == 0 and s2.budget.spent == s1.budget.spent


def test_k_shot_sampling_reproducible():
    task = load_task("toy")
    a, rest = sample_k_shot(task.train, 3, seed=1)
    b, _ = sample_k_shot(task.train, 3, seed=1)
    assert a.equals(b)
    assert (a.groupby("label").size() == 3).all()
    assert len(a) + len(rest) == len(task.train)


def test_metrics():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    m = compute_metrics(y, p)
    assert m["accuracy"] == 0.75
    assert 0 <= m["ece"] <= 1
    assert top_confusions(y, p.argmax(1), 5) == [(1, 0, 1)]
    perfect = np.eye(2)[[0, 1]]
    assert ece(np.array([0, 1]), perfect) == 0.0


def test_parse_json_tolerant():
    assert parse_json('```json\n{"a": [1]}\n```') == {"a": [1]}
    assert parse_json('Sure! {"x": "y"} hope this helps') == {"x": "y"}
    assert clean_examples(["  ok text ", 3, "", "x"]) == ["ok text"]


def test_toy_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    shutil.copy(root / "configs" / "toy_smoke.yaml", tmp_path / "toy.yaml")
    from homeroom.run import main
    main(["run", "--config", "toy.yaml"])
    out = tmp_path / "results" / "toy_smoke"
    names = {p.name for p in out.glob("*.json")}
    assert {"real_fewshot_seed0.json", "classroom_seed0.json", "bulk_seed0.json"} <= names
    cls = json.loads((out / "classroom_seed0.json").read_text())
    bulk = json.loads((out / "bulk_seed0.json").read_text())
    assert cls["cost"]["spent_usd"] <= 0.2 + 1e-9
    assert bulk["spend_cap_usd"] == cls["cost"]["spent_usd"]      # spend matching
    assert (out / "summary.md").exists()
