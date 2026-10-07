import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homeroom.checkpoint import atomic_json, checkpointed_train, training_key
from homeroom.students import make_student

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("unattended_study", SCRIPTS / "unattended_study.py")
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)


def toy():
    cfg = {"student": {"kind": "tfidf", "tfidf": {"C": 1}}}
    texts = ["red apple fruit", "green apple fruit", "blue truck vehicle", "red car vehicle"]
    return cfg, texts, [0, 0, 1, 1]


def test_atomic_json_preserves_old_file_on_interrupted_replace(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_json(path, {"old": True})
    def interrupted(*args):
        raise OSError("simulated interruption")
    monkeypatch.setattr("homeroom.checkpoint.os.replace", interrupted)
    with pytest.raises(OSError):
        atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}


def test_completed_fit_roundtrip_avoids_retraining(tmp_path):
    cfg, texts, labels = toy()
    calls = []
    def train():
        calls.append(True)
        return make_student(cfg, 2, 0).fit(texts, labels)
    first = checkpointed_train(cfg, 0, 2, texts, labels, train, tmp_path)
    second = checkpointed_train(cfg, 0, 2, texts, labels, train, tmp_path)
    assert len(calls) == 1
    np.testing.assert_allclose(first.predict_proba(texts), second.predict_proba(texts), rtol=0, atol=0)


def test_checkpoint_key_tracks_training_inputs_and_settings():
    cfg, texts, labels = toy()
    base = training_key(cfg, 0, 2, texts, labels)
    assert base != training_key(cfg, 1, 2, texts, labels)
    assert base != training_key(cfg, 0, 2, texts + ["another apple"], labels + [0])
    assert base != training_key({"student": {"kind": "tfidf", "tfidf": {"C": 2}}}, 0, 2, texts, labels)


def test_corrupt_model_is_preserved_and_retrained(tmp_path):
    cfg, texts, labels = toy()
    calls = []
    def train():
        calls.append(True)
        return make_student(cfg, 2, 0).fit(texts, labels)
    checkpointed_train(cfg, 0, 2, texts, labels, train, tmp_path)
    folder = tmp_path / training_key(cfg, 0, 2, texts, labels)
    (folder / "model.joblib").write_bytes(b"interrupted model")
    checkpointed_train(cfg, 0, 2, texts, labels, train, tmp_path)
    assert len(calls) == 2
    assert len(list(tmp_path.glob("*.incomplete.*"))) == 1


def test_schema_and_answer_counts_match_study():
    cfg = sup.config_guard()
    key = "a" * 64
    batch = {"system": "teacher", "requests": [{"id": key, "task": "lesson", "prompt": "lesson"}]}
    schema = sup.schema_for(batch, cfg)
    assert schema["properties"][key]["properties"]["examples"]["minItems"] == 10
    value = {key: {"definition": "definition", "confusable_with": [], "examples": ["example"] * 10}}
    assert len(json.loads(sup.validated_answers(batch, value, cfg)[key])["examples"]) == 10
    value[key]["examples"].pop()
    with pytest.raises(ValueError, match="count"):
        sup.validated_answers(batch, value, cfg)


def test_quota_backoff_uses_reset_or_slow_retry():
    assert sup.retry_time(1000, 1, quota=True, reset=5000) == 5090
    assert sup.retry_time(1000, 1, quota=True) == 4600
    assert sup.retry_time(1000, 2, quota=True) == 8200
    assert sup.retry_time(1000, 8, quota=True) == 22600
    assert sup.retry_time(1000, 1) == 1060


def test_quota_detection_only_uses_error_events(tmp_path):
    events, stderr = tmp_path / "events.jsonl", tmp_path / "stderr.log"
    events.write_text(json.dumps({"type": "item.completed", "item": {"text": "usage limit"}}) + "\n")
    stderr.write_text("")
    assert not sup.error_info(events, stderr)["quota"]
    events.write_text(json.dumps({"type": "turn.failed", "error": {"message": "usage_limit_reached",
                                                                  "resets_at": 9999999999}}) + "\n")
    info = sup.error_info(events, stderr)
    assert info["quota"] and info["reset_at"] == 9999999999


def test_api_credentials_are_removed_from_child_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-real-key")
    monkeypatch.setenv("CODEX_API_KEY", "dummy-not-a-real-key")
    env = sup.environment()
    assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env
    assert env["HF_HUB_OFFLINE"] == "1"


def test_reused_pid_is_not_an_owned_process(monkeypatch):
    fake = SimpleNamespace(create_time=lambda: 100, is_running=lambda: True, status=lambda: "running")
    monkeypatch.setattr(sup.psutil, "Process", lambda pid: fake)
    assert not sup.alive({"pid": 123, "created": 90})


def test_pause_checkpoint_stops_owned_work_and_preserves_replies(tmp_path, monkeypatch):
    for name in ("PRIVATE", "QUEUE", "CACHE", "RESULTS"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(sup, name, path)
    marker = sup.PRIVATE / "pause.request"
    marker.write_text("pause")
    reply = sup.QUEUE / "saved.reply.json"
    reply.write_text('{"text": "saved"}')
    monkeypatch.setattr(sup, "codex_command", lambda: ["fake-codex"])
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ChatGPT", stderr=""))
    stopped = []
    monkeypatch.setattr(sup, "stop_owned", lambda job: stopped.append(job["pid"]))
    runner = sup.Supervisor()
    runner.state["jobs"]["experiment"] = {"pid": 123}
    runner.run()
    assert stopped == [123]
    assert runner.state["status"] == "paused_by_user"
    assert json.loads(reply.read_text())["text"] == "saved"


def test_single_supervisor_preserves_adopted_running_worker(tmp_path, monkeypatch):
    for name in ("PRIVATE", "QUEUE", "CACHE", "RESULTS"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(sup, name, path)
    monkeypatch.setattr(sup, "codex_command", lambda: ["fake-codex"])
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ChatGPT", stderr=""))
    monkeypatch.setattr(sup, "alive", lambda job: True)
    monkeypatch.setattr(sup, "stop_owned", lambda job: None)
    runner = sup.Supervisor()
    runner.state["jobs"]["experiment"] = {"pid": 123, "created": 10}
    monkeypatch.setattr(runner, "worker", lambda: pytest.fail("duplicate GPU worker"))
    monkeypatch.setattr(runner, "batches", lambda: [])
    def sleep(seconds):
        (sup.PRIVATE / "pause.request").write_text("pause")
    monkeypatch.setattr(sup.time, "sleep", sleep)
    runner.run()
    assert runner.state["status"] == "paused_by_user"
