import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("write_teacher_replies", Path(__file__).resolve().parents[1] / "scripts" / "write_teacher_replies.py")
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def _batch(tmp_path):
    folder = tmp_path / "data" / "teacher_queue" / "study" / "batches"
    folder.mkdir(parents=True)
    batch = folder / "b0001.json"
    key = "a" * 64
    batch.write_text(json.dumps({"system": "teacher", "requests": [{"id": key, "task": "bulk", "prompt": "write examples"}]}))
    return batch, key


def test_batch_reply_is_atomic_and_idempotent(tmp_path):
    batch, key = _batch(tmp_path)
    answers = {key: json.dumps({"examples": ["one example"]})}
    out = writer.write_replies(batch, answers, root=tmp_path)
    assert json.loads(out.read_text()) == answers
    assert writer.write_replies(batch, answers, root=tmp_path) == out
    with pytest.raises(ValueError, match="replace"):
        writer.write_replies(batch, {key: json.dumps({"examples": ["changed"]})}, root=tmp_path)


@pytest.mark.parametrize("kind", ["wrong_ids", "wrong_schema", "oversize"])
def test_batch_writer_rejects_invalid_replies(tmp_path, kind):
    batch, key = _batch(tmp_path)
    answer = json.dumps({"examples": ["an example"]})
    if kind == "wrong_ids":
        answers = {"b" * 64: answer}
    elif kind == "wrong_schema":
        answers = {key: json.dumps({"items": ["exam data"]})}
    else:
        answers = {key: json.dumps({"examples": ["x" * 9000]})}
    with pytest.raises(ValueError):
        writer.write_replies(batch, answers, root=tmp_path)
    assert not batch.with_name("b0001.reply.json").exists()


def test_batch_writer_stays_inside_teacher_queue(tmp_path):
    batch, key = _batch(tmp_path)
    outside = tmp_path / "b0001.json"
    outside.write_bytes(batch.read_bytes())
    with pytest.raises(ValueError, match="data/teacher_queue"):
        writer.write_replies(outside, {key: json.dumps({"examples": ["x"]})}, root=tmp_path)
