"""Validate and atomically save one queue batch reply. Never calls a model."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"exam": ("items",), "lesson": ("definition", "confusable_with", "examples"),
          "bulk": ("examples",), "remedial": ("rule", "a_examples", "b_examples")}


def write_replies(batch: Path, answers: dict, *, root: Path = ROOT) -> Path:
    batch = batch.resolve()
    queue = (root / "data" / "teacher_queue").resolve()
    if not batch.is_relative_to(queue) or batch.parent.name != "batches" or not re.fullmatch(r"b\d+\.json", batch.name):
        raise ValueError("batch must be an existing numbered file in data/teacher_queue/EXPERIMENT/batches")
    data = json.loads(batch.read_text(encoding="utf-8"))
    requests = {r["id"]: r for r in data["requests"]}
    if not isinstance(answers, dict) or set(answers) != set(requests):
        raise ValueError("reply IDs must exactly match this batch")
    for key, text in answers.items():
        if not re.fullmatch(r"[0-9a-f]{64}", key) or not isinstance(text, str):
            raise ValueError("each reply needs a valid request ID and a JSON string")
        if len(text) > 8192:
            raise ValueError("reply exceeds the configured 2048-token character estimate")
        value = json.loads(text)
        task = requests[key]["task"]
        if task not in FIELDS or not isinstance(value, dict) or not set(FIELDS[task]) <= value.keys():
            raise ValueError(f"invalid {task} reply schema")
        for field in FIELDS[task]:
            if field in ("definition", "rule"):
                if not isinstance(value[field], str):
                    raise ValueError(f"{field} must be text")
            elif not isinstance(value[field], list) or any(not isinstance(item, str) for item in value[field]):
                raise ValueError(f"{field} must contain strings")
    dest = batch.with_name(batch.stem + ".reply.json")
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8")) == answers:
            return dest
        raise ValueError("refusing to replace an existing batch answer")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=batch.parent,
                                     suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(answers, handle, ensure_ascii=True)
        handle.write("\n")
    os.replace(tmp, dest)
    return dest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--reply-b64", required=True, help="base64 UTF-8 JSON object: request ID -> JSON reply string")
    args = parser.parse_args(argv)
    answers = json.loads(base64.b64decode(args.reply_b64, validate=True).decode("utf-8"))
    dest = write_replies(args.batch, answers)
    print(f"Saved {dest} ({len(answers)} answers)")


if __name__ == "__main__":
    main()
