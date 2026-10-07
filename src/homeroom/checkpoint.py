"""Atomic artifacts and reusable completed student fits (private, local checkpoints)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from .students import make_student


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def training_key(cfg, seed, n_classes, texts, labels) -> str:
    source = Path(__file__).with_name("students.py").read_bytes()
    payload = {"student": cfg["student"], "seed": int(seed), "n_classes": int(n_classes),
               "texts": list(texts), "labels": [int(y) for y in labels],
               "student_source": hashlib.sha256(source).hexdigest(), "format": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hash_file(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def checkpointed_train(cfg, seed, n_classes, texts, labels, train, root: Path):
    texts, labels = list(texts), [int(y) for y in labels]
    kind = cfg["student"]["kind"]
    if kind not in {"setfit", "tfidf"}:
        raise ValueError("completed-fit checkpoints support setfit and tfidf")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = training_key(cfg, seed, n_classes, texts, labels)
    dest = root / key
    manifest = dest / "manifest.json"
    if dest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            valid = (data["training_key"] == key and data["kind"] == kind and
                     data["hashes"] == file_hashes(dest))
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            student = make_student(cfg, n_classes, seed)
            if kind == "tfidf":
                import joblib
                student.model = joblib.load(dest / "model.joblib")
            else:
                from setfit import SetFitModel
                student.model = SetFitModel.from_pretrained(str(dest / "model"), local_files_only=True)
            print(f"[checkpoint] restored seed={seed} n_train={len(texts)} key={key[:12]}", flush=True)
            return student
        # Preserve damaged/incomplete state inside the same private checkpoint root.
        quarantine = root / (key + ".incomplete." + uuid.uuid4().hex)
        if not dest.is_relative_to(root) or not quarantine.is_relative_to(root):
            raise ValueError("checkpoint quarantine escapes its root")
        dest.rename(quarantine)
    student = train()
    tmp = root / (key + ".partial." + uuid.uuid4().hex)
    tmp.mkdir()
    if kind == "tfidf":
        import joblib
        joblib.dump(student.model, tmp / "model.joblib")
    else:
        student.model.save_pretrained(str(tmp / "model"))
    atomic_json(tmp / "manifest.json",
                {"training_key": key, "kind": kind, "seed": int(seed),
                 "n_train": len(texts), "hashes": file_hashes(tmp)})
    tmp.rename(dest)
    print(f"[checkpoint] saved seed={seed} n_train={len(texts)} key={key[:12]}", flush=True)
    return student
