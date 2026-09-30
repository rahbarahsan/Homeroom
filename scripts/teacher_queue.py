"""Operator helper for the `queue` teacher (answers come from an external subagent).

  python scripts/teacher_queue.py status --exp <experiment full name>
  python scripts/teacher_queue.py batch  --exp <experiment full name> [--size 20]

`batch` groups pending requests that are not yet in a batch into
data/teacher_queue/<exp>/batches/bNNNN.json  ({"system": ..., "requests": [{id, task, prompt}]})
and prints the paths. The answerer writes bNNNN.reply.json = {id: reply_text}; the running
pipeline ingests it (QueueTeacher.wait). Teacher text stays under data/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pending(qdir: Path) -> list[Path]:
    return sorted(p for p in qdir.glob("*.req.json")
                  if not (qdir / p.name.replace(".req.json", ".reply.json")).exists())


def _batched_ids(bdir: Path) -> set[str]:
    ids: set[str] = set()
    for b in bdir.glob("b*.json"):
        if b.name.endswith(".reply.json"):
            continue
        ids |= {r["id"] for r in json.loads(b.read_text(encoding="utf-8"))["requests"]}
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "batch"])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--root", default="data/teacher_queue")
    ap.add_argument("--size", type=int, default=20)
    args = ap.parse_args()
    qdir = Path(args.root) / args.exp
    bdir = qdir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    pending = _pending(qdir)
    done = len(list(qdir.glob("*.reply.json")))

    if args.cmd == "status":
        open_batches = [b.name for b in sorted(bdir.glob("b*.json")) if not b.name.endswith(".reply.json")
                        and not (bdir / b.name.replace(".json", ".reply.json")).exists()]
        print(json.dumps({"pending": len(pending), "answered": done, "open_batches": open_batches}))
        return

    batched = _batched_ids(bdir)
    todo = [p for p in pending if p.name[:-len(".req.json")] not in batched]
    n = max((int(b.name[1:5]) for b in bdir.glob("b*.json")), default=0)  # next index after the highest
    made = []
    for i in range(0, len(todo), args.size):
        reqs = [json.loads(p.read_text(encoding="utf-8")) for p in todo[i:i + args.size]]
        systems = {r["system"] for r in reqs}
        assert len(systems) == 1, "one system prompt per batch"
        n += 1
        path = bdir / f"b{n:04d}.json"
        path.write_text(json.dumps({"system": systems.pop(),
                                    "requests": [{"id": r["id"], "task": r["task"], "prompt": r["prompt"]}
                                                 for r in reqs]}, indent=1), encoding="utf-8")
        made.append(str(path).replace("\\", "/"))
    print(json.dumps({"new_batches": made, "pending": len(pending)}))


if __name__ == "__main__":
    main()
