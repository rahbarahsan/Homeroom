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
    # teacher text stays out of committed results; it goes to gitignored data/generated/
    assert len(cls["history"]) >= 2 and cls["cost"]["by_tag_usd"].get("remedial", 0) > 0  # loop ran
    assert "rules" not in cls and "n_rules" in cls
    assert (tmp_path / "data" / "generated" / "toy_smoke" / "classroom_rules_seed0.json").exists()
    # full config snapshot is recorded
    assert cls["config"]["classroom"]["max_rounds"] == 2 and cls["config"]["budget"]["usd_per_arm"] == 0.2


def test_bulk_refuses_without_classroom_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    shutil.copy(root / "configs" / "toy_smoke.yaml", tmp_path / "toy.yaml")
    from homeroom.run import main
    with pytest.raises(SystemExit, match="classroom"):
        main(["run", "--config", "toy.yaml", "--arms", "bulk"])


def test_config_snapshot_scrubs_secrets():
    from homeroom.config import config_snapshot
    snap = config_snapshot({"teacher": {"model": "m", "api_key": "sk-x", "max_tokens": 5},
                            "other": [{"secret_thing": 1, "ok": 2}]})
    assert snap == {"teacher": {"model": "m", "max_tokens": 5}, "other": [{"ok": 2}]}


def _answer_queue(qdir, stop, text='{"examples": ["ok"]}'):
    """Fake subagent: answers every pending request via a batch reply file."""
    import time as _t
    from pathlib import Path as _P
    bdir = _P(qdir) / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    n = 0
    while not stop.is_set():
        ids = [p.name[:-len(".req.json")] for p in _P(qdir).glob("*.req.json")
               if not (_P(qdir) / p.name.replace(".req.", ".reply.")).exists()]
        if ids:
            n += 1
            (bdir / f"b{n:04d}.reply.json").write_text(json.dumps({i: text for i in ids}))
        _t.sleep(0.05)


def test_queue_teacher_batches_and_caches(tmp_path):
    import threading
    from homeroom.teacher import QueueTeacher, Request
    qdir = tmp_path / "q"
    t = QueueTeacher("m", qdir, timeout_s=10, poll_s=0.02, max_tokens=100)
    stop = threading.Event()
    th = threading.Thread(target=_answer_queue, args=(qdir, stop), daemon=True); th.start()
    try:
        s = TeacherSession(t, Budget(10.0, 1.0, 5.0), tmp_path / "cache")
        reqs = [Request("sys", f"prompt {i}", "bulk", salt=str(i)) for i in range(5)]
        replies, stopped = s.ask_many(reqs)
        assert not stopped and len(replies) == 5 and all(r.text == '{"examples": ["ok"]}' for r in replies)
        assert len(list(qdir.glob("*.req.json"))) == 5 and s.budget.calls == 5
        s2 = TeacherSession(t, Budget(10.0, 1.0, 5.0), tmp_path / "cache")   # rerun: all cache hits
        replies2, _ = s2.ask_many(reqs)
        assert all(r.cached for r in replies2) and s2.budget.real_spent == 0
        assert abs(s2.budget.spent - s.budget.spent) < 1e-12
    finally:
        stop.set(); th.join()


def test_queue_waves_respect_hard_cap(tmp_path):
    import threading
    from homeroom.teacher import QueueTeacher, Request
    qdir = tmp_path / "q"
    # worst case per request: 100 output tokens * $5/MTok = $0.0005 (+ tiny input) -> cap fits ~3
    t = QueueTeacher("m", qdir, timeout_s=10, poll_s=0.02, max_tokens=100)
    stop = threading.Event()
    th = threading.Thread(target=_answer_queue, args=(qdir, stop), daemon=True); th.start()
    try:
        s = TeacherSession(t, Budget(0.0016, 1.0, 5.0), tmp_path / "cache")
        replies, stopped = s.ask_many([Request("sys", f"p{i}", "bulk", salt=str(i)) for i in range(50)])
        assert stopped and 0 < len(replies) < 50
        assert s.budget.spent <= s.budget.limit_usd
    finally:
        stop.set(); th.join()


def test_null_temperature_is_not_sent():
    t = make_teacher({"teacher": {"provider": "mock", "temperature": None}, "experiment": {"full_name": "x"}})
    assert t.temperature is None
