"""Durable, bounded Banking77 queue-study supervisor using saved ChatGPT auth.

Teacher sessions are fresh, read-only Codex executions. Only fixed experiment
commands and validated replies are executed/written by this supervisor.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import yaml

from homeroom.checkpoint import atomic_json
from write_teacher_replies import write_replies

ROOT = Path(__file__).resolve().parents[1]
EXP = "pilot_banking77_codex_luna"
PRIVATE = ROOT / "data" / "generated" / EXP / "unattended"
QUEUE = ROOT / "data" / "teacher_queue" / EXP
CACHE = ROOT / "data" / "cache" / EXP
RESULTS = ROOT / "results" / EXP
CONFIG = ROOT / "configs" / (EXP + ".yaml")
PYTHON = str(ROOT / ".venv/Scripts/python.exe") if os.name == "nt" else sys.executable
SECRET_ENV = {"OPENAI_API_KEY", "CODEX_API_KEY", "TEACHER_API_KEY", "ANTHROPIC_API_KEY",
              "AZURE_OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}


def environment():
    env = {k: v for k, v in os.environ.items() if k.upper() not in SECRET_ENV}
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def config_guard():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    t = cfg["teacher"]
    if (cfg["experiment"]["name"] != EXP or t["provider"] != "queue" or
            t["model"] != "gpt-6-luna" or t["max_tokens"] != 2048 or
            cfg["budget"]["usd_per_arm"] != 0.075 or
            t["price_input_per_mtok"] != 0.1 or t["price_output_per_mtok"] != 0.5 or
            cfg["experiment"]["seeds"] != [0, 1, 2] or
            cfg["arms"] != ["real_fewshot", "classroom", "bulk"]):
        raise ValueError("study identity/provider/model/prices/caps changed; review before resuming")
    reference = RESULTS / "real_fewshot_seed0.json"
    if reference.exists():
        expected = json.loads(reference.read_text(encoding="utf-8"))["config"]
        for field in ("data", "teacher", "budget", "arms", "bulk", "classroom", "student"):
            if cfg[field] != expected[field]:
                raise ValueError("scientific settings differ from completed baseline: " + field)
    return cfg


def codex_command():
    native = shutil.which("codex.exe")
    if native:
        return [native]
    node = shutil.which("node.exe") or shutil.which("node")
    js = Path(os.environ.get("APPDATA", "")) / "npm/node_modules/@openai/codex/bin/codex.js"
    if node and js.is_file():
        return [node, str(js)]
    raise RuntimeError("installed Codex executable could not be resolved")


def schema_for(batch, cfg):
    fields = {"exam": ["items"], "lesson": ["definition", "confusable_with", "examples"],
              "bulk": ["examples"], "remedial": ["rule", "a_examples", "b_examples"]}
    counts = {"exam": cfg["classroom"]["exam_items_per_intent"],
              "lesson": cfg["classroom"]["lesson_examples_per_intent"],
              "bulk": cfg["bulk"]["examples_per_call"],
              "remedial": cfg["classroom"]["examples_per_confusion"]}
    props = {}
    for r in batch["requests"]:
        fs = {}
        for field in fields[r["task"]]:
            if field in {"definition", "rule"}:
                fs[field] = {"type": "string"}
            else:
                fs[field] = {"type": "array", "items": {"type": "string"}}
                if field != "confusable_with":
                    fs[field].update(minItems=counts[r["task"]], maxItems=counts[r["task"]])
        props[r["id"]] = {"type": "object", "properties": fs,
                          "required": list(fs), "additionalProperties": False}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def teacher_prompt(batch):
    return ("You are an isolated synthetic-data teacher. Adopt the SYSTEM ROLE below. "
            "Answer every assigned prompt independently using fresh, diverse model-generated "
            "text. Do not copy the real seed examples, use tools, read files, call another model, "
            "or build text from programmatic templates. Return only the JSON object required "
            "by the supplied output schema: request ID maps to its reply object. "
            "Each reply object serialized as compact JSON must fit within 3500 characters. "
            "Satisfy the exact requested number of examples.\n\nSYSTEM ROLE:\n" +
            batch["system"] + "\n\nASSIGNED REQUESTS:\n" +
            json.dumps(batch["requests"], ensure_ascii=True))


def validated_answers(batch, values, cfg):
    if not isinstance(values, dict) or set(values) != {r["id"] for r in batch["requests"]}:
        raise ValueError("teacher output IDs do not match assigned batch")
    expected = {"exam": ("items", cfg["classroom"]["exam_items_per_intent"]),
                "lesson": ("examples", cfg["classroom"]["lesson_examples_per_intent"]),
                "bulk": ("examples", cfg["bulk"]["examples_per_call"]),
                "remedial": ("a_examples", cfg["classroom"]["examples_per_confusion"])}
    answers = {}
    for r in batch["requests"]:
        value = values[r["id"]]
        field, n = expected[r["task"]]
        if not isinstance(value, dict) or len(value.get(field, [])) != n:
            raise ValueError("teacher example count does not match request")
        if r["task"] == "remedial" and len(value.get("b_examples", [])) != n:
            raise ValueError("teacher remedial side count does not match request")
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(text) > 3500:
            raise ValueError("teacher reply exceeds the study's operator size bound")
        answers[r["id"]] = text
    return answers


def error_info(events_path, stderr_path):
    errors = []
    reset = None
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") in {"error", "turn.failed"}:
                errors.append(json.dumps(event))
                def visit(v):
                    nonlocal reset
                    if isinstance(v, dict):
                        for k, x in v.items():
                            if k in {"resets_at", "reset_at"} and isinstance(x, (int, float)) and x > time.time():
                                reset = float(x)
                            visit(x)
                    elif isinstance(v, list):
                        for x in v:
                            visit(x)
                visit(event)
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    if any(s in stderr.lower() for s in ("usage_limit_reached", "hit your usage limit", "quota exceeded")):
        errors.append(stderr[-3000:])
    message = "\n".join(errors)
    quota = any(s in message.lower() for s in
                ("usage_limit", "usage limit", "quota exceeded", "insufficient_quota"))
    return {"quota": quota, "reset_at": reset, "message": message[-3000:]}


def retry_time(now, failures, quota=False, reset=None):
    if quota:
        return max(now + 60, reset + 90) if reset else now + min(3600 * 2 ** (failures - 1), 21600)
    return now + min(60 * 3 ** (failures - 1), 1800)


def alive(job):
    try:
        p = psutil.Process(job["pid"])
        return abs(p.create_time() - job["created"]) < 0.1 and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop_owned(job):
    if not alive(job):
        return
    parent = psutil.Process(job["pid"])
    children = parent.children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    _, still = psutil.wait_procs(children + [parent], timeout=5)
    for p in still:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def launch(args, cwd, prefix, prompt=None):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".events.jsonl").open("ab") as out, prefix.with_suffix(".stderr.log").open("ab") as err:
        proc = subprocess.Popen(args, cwd=cwd, env=environment(),
                                stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
                                stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    job = {"pid": proc.pid, "created": psutil.Process(proc.pid).create_time(),
           "started_at": time.time(), "prefix": str(prefix), "args": args}
    if prompt is not None:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
    return job


def quarantine(path):
    path = path.resolve()
    allowed = [QUEUE.resolve(), CACHE.resolve(), RESULTS.resolve()]
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError("artifact quarantine is outside this study")
    dest = path.with_name(path.name + ".incomplete." + uuid.uuid4().hex)
    if not any(dest.is_relative_to(root) for root in allowed):
        raise ValueError("quarantine target is outside this study")
    path.rename(dest)


def repair_partial_json():
    # Only damaged JSON is preserved under another name; no completed artifacts are deleted.
    paths = list(CACHE.glob("*.json")) + list(QUEUE.glob("*.reply.json")) + list(RESULTS.glob("*_seed*.json"))
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("artifact is not an object")
        except (ValueError, OSError):
            quarantine(p)


class Supervisor:
    def __init__(self, thread_id=None):
        self.cfg = config_guard()
        PRIVATE.mkdir(parents=True, exist_ok=True)
        self.path = PRIVATE / "state.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if thread_id:
            uuid.UUID(thread_id)
            self.state["thread_id"] = thread_id
        self.state.setdefault("jobs", {})
        self.state.setdefault("attempts", {})
        self.state.setdefault("failures", {})
        self.state.setdefault("batch_not_before", {})
        self.state.setdefault("quota_failures", 0)
        self.state.setdefault("not_before", 0)
        self.state.setdefault("main_not_before", 0)
        self.state.setdefault("main_failures", 0)
        self.state.setdefault("awake_until", time.time() + 12 * 3600)
        self.codex = codex_command()
        self.main_id = "experiment"
        self.handles = []

    def event(self, what, **data):
        with (PRIVATE / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), "event": what, **data}) + "\n")
        print(json.dumps({"event": what, **data}), flush=True)

    def save(self):
        self.state.update(supervisor_pid=os.getpid(), heartbeat=time.time(),
                          completed_results=sorted(p.name for p in RESULTS.glob("*_seed*.json")),
                          completed_batches=len(list((QUEUE / "batches").glob("*.reply.json"))))
        atomic_json(self.path, self.state)

    def worker(self):
        attempt = int(self.state.get("main_attempt", 0)) + 1
        self.state["main_attempt"] = attempt
        job = launch([PYTHON, str(ROOT / "scripts/checkpoint_run.py"), "run",
                      "--config", str(CONFIG)], ROOT, PRIVATE / f"experiment_attempt{attempt:03d}")
        self.state["jobs"][self.main_id] = job
        self.event("experiment_started", pid=job["pid"], attempt=attempt)

    def batches(self):
        subprocess.run([PYTHON, str(ROOT / "scripts/teacher_queue.py"), "batch",
                        "--exp", EXP, "--size", "5"], cwd=ROOT, env=environment(),
                       stdout=subprocess.DEVNULL, check=True, timeout=30)
        return [p for p in sorted((QUEUE / "batches").glob("b*.json"))
                if not p.name.endswith(".reply.json") and not p.with_name(p.stem + ".reply.json").exists()]

    def teacher(self, path):
        name = path.stem
        batch = json.loads(path.read_text(encoding="utf-8"))
        attempt = self.state["attempts"].get(name, 0) + 1
        self.state["attempts"][name] = attempt
        prefix = PRIVATE / "teachers" / f"{name}_attempt{attempt:03d}"
        schema = prefix.with_suffix(".schema.json")
        answer = prefix.with_suffix(".answer.json")
        atomic_json(schema, schema_for(batch, self.cfg))
        isolated = PRIVATE / "teacher_workspace"
        isolated.mkdir(exist_ok=True)
        args = self.codex + ["-a", "never", "exec", "--ignore-user-config", "--skip-git-repo-check",
                            "-s", "read-only", "-m", "gpt-6-luna",
                            "-c", 'model_reasoning_effort="low"', "--json",
                            "--output-schema", str(schema), "-o", str(answer), "-"]
        job = launch(args, isolated, prefix, teacher_prompt(batch))
        job.update(batch=str(path), answer=str(answer))
        self.state["jobs"][name] = job
        self.event("teacher_started", batch=name, pid=job["pid"], attempt=attempt)

    def finish_teacher(self, name, job):
        prefix = Path(job["prefix"])
        info = error_info(prefix.with_suffix(".events.jsonl"), prefix.with_suffix(".stderr.log"))
        if info["quota"]:
            self.state["quota_failures"] += 1
            self.state["not_before"] = retry_time(time.time(), self.state["quota_failures"], True, info["reset_at"])
            self.state["status"] = "waiting_for_quota"
            self.event("quota_pause", batch=name, retry_at=self.state["not_before"], error=info["message"])
            # Release GPU/RAM; completed fits and atomic answers survive.
            main = self.state["jobs"].pop(self.main_id, None)
            if main:
                stop_owned(main)
            return False
        try:
            events = prefix.with_suffix(".events.jsonl")
            forbidden = {"command_execution", "mcp_tool_call", "web_search",
                         "file_change", "collab_agent_tool_call"}
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("item", {}).get("type") in forbidden:
                    raise ValueError("isolated teacher used a tool; refusing its output")
            values = json.loads(Path(job["answer"]).read_text(encoding="utf-8").lstrip("\ufeff"))
            batch_path = Path(job["batch"])
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            answers = validated_answers(batch, values, self.cfg)
            write_replies(batch_path, answers)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            n = self.state["failures"].get(name, 0) + 1
            self.state["failures"][name] = n
            self.state["batch_not_before"][name] = retry_time(time.time(), n)
            self.event("teacher_retry", batch=name, failures=n, error=str(exc), provider_error=info["message"])
            if n >= 3:
                self.state["status"] = "teacher_needs_review"
                self.nudge("Teacher batch repeatedly failed validation. Review private supervisor logs and repair the routine failure; preserve model and scientific settings.")
            return False
        if time.time() >= self.state["not_before"]:
            self.state["quota_failures"] = 0
        self.state["failures"].pop(name, None)
        self.state["batch_not_before"].pop(name, None)
        self.event("teacher_saved", batch=name, answers=len(answers))
        return True

    def nudge(self, text):
        thread_id = self.state.get("thread_id")
        if not thread_id:
            return
        now = time.time()
        if now - self.state.get("last_nudge", 0) < 3600:
            return
        prompt = ("Continue the authorized Homeroom Goal. " + text +
                  " Read data/generated/pilot_banking77_codex_luna/unattended/state.json. "
                  "Do not start another supervisor or expose teacher text. Preserve the user's unrelated edits.")
        result = subprocess.run(self.codex + ["queue", "--thread", thread_id, "--approve-for-me",
                                             "-C", str(ROOT), "--message", prompt],
                                env=environment(), capture_output=True, timeout=60)
        self.state["last_nudge"] = now
        self.event("session_continuation", accepted=result.returncode == 0)

    def done(self):
        return all((RESULTS / f"{arm}_seed{seed}.json").is_file()
                   for seed in (0, 1, 2) for arm in ("real_fewshot", "classroom", "bulk"))

    def run(self, one_batch=False):
        repair_partial_json()
        self.state["status"] = "verifying_teacher" if one_batch else "running"
        self.save()
        # Saved CLI authentication must be ChatGPT, never an API-key route.
        auth = subprocess.run(self.codex + ["login", "status"], env=environment(),
                              capture_output=True, text=True, timeout=30)
        if auth.returncode or "ChatGPT" not in auth.stdout + auth.stderr:
            raise RuntimeError("unattended inference requires existing ChatGPT login")
        selected = None
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            while True:
                if (PRIVATE / "pause.request").exists():
                    for job in list(self.state["jobs"].values()):
                        stop_owned(job)
                    self.state["jobs"] = {}
                    self.state["status"] = "paused_by_user"
                    self.save()
                    return
                now = time.time()
                if os.name == "nt" and now >= self.state["awake_until"]:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
                for name, job in list(self.state["jobs"].items()):
                    if alive(job):
                        if name != self.main_id and now - job["started_at"] > 1800:
                            stop_owned(job)
                        else:
                            continue
                    self.state["jobs"].pop(name, None)
                    if name == self.main_id:
                        if not self.done():
                            self.state["main_failures"] += 1
                            self.state["main_not_before"] = retry_time(now, self.state["main_failures"])
                            self.event("experiment_restart_pending", failures=self.state["main_failures"])
                            repair_partial_json()
                            if self.state["main_failures"] >= 3:
                                self.nudge("Experiment process failed repeatedly. Diagnose the saved stderr logs and checkpoint recovery before restarting.")
                    else:
                        ok = self.finish_teacher(name, job)
                        if one_batch:
                            self.state["status"] = "teacher_verified" if ok else self.state["status"]
                            self.save()
                            return
                if self.done() and not one_batch:
                    for job in list(self.state["jobs"].values()):
                        if alive(job):
                            stop_owned(job)
                    self.state["jobs"] = {}
                    report = subprocess.run([PYTHON, "-m", "homeroom.run", "report",
                                             "--config", str(CONFIG)], cwd=ROOT, env=environment(),
                                            capture_output=True, text=True, timeout=180)
                    self.state["status"] = "results_complete" if report.returncode == 0 else "report_needs_review"
                    self.event(self.state["status"], report_message=(report.stdout + report.stderr)[-2000:])
                    self.nudge("All nine real experiment results are saved. Finish validated reporting, chart, documentation, authorized commit/push and the final user report. Do not mark the Goal complete until that work is done.")
                    self.save()
                    return
                if now < self.state["not_before"]:
                    self.state["status"] = "waiting_for_quota"
                    self.save()
                    time.sleep(30)
                    continue
                if not one_batch and self.main_id not in self.state["jobs"] and now >= self.state["main_not_before"]:
                    self.worker()
                candidates = self.batches()
                teacher_count = sum(k != self.main_id for k in self.state["jobs"])
                for path in candidates:
                    if one_batch and selected is not None:
                        break
                    if path.stem in self.state["jobs"] or now < self.state["batch_not_before"].get(path.stem, 0):
                        continue
                    if teacher_count >= (1 if one_batch else 3):
                        break
                    self.teacher(path)
                    selected = path.stem
                    teacher_count += 1
                self.state["status"] = "running" if not one_batch else "verifying_teacher"
                self.save()
                if one_batch and selected is None:
                    raise RuntimeError("no pending teacher batch available for live verification")
                time.sleep(5)
        finally:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def verify_process_recovery():
    prefix = PRIVATE / "verification" / "process"
    code = "import subprocess,sys,time; subprocess.Popen([PYTHON,'-c','import time; time.sleep(60)']); time.sleep(60)"
    job = launch([PYTHON, "-c", code], ROOT, prefix)
    time.sleep(2)
    descendants = [{"pid": p.pid, "created": p.create_time()}
                   for p in psutil.Process(job["pid"]).children(recursive=True)]
    stop_owned(job)
    if alive(job) or any(alive(child) for child in descendants):
        raise RuntimeError("owned process-tree shutdown did not complete")
    atomic_json(PRIVATE / "verification" / "process-recovery.json",
                {"process_tree_recovery": "passed", "descendants_stopped": len(descendants)})
    print("Owned process-tree recovery passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "resume", "verify-teacher", "verify-process", "status", "pause"])
    parser.add_argument("--thread-id", help="optional existing Codex task to continue")
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if args.action == "verify-process":
        verify_process_recovery()
        return
    if args.action == "status":
        p = PRIVATE / "state.json"
        print(p.read_text(encoding="utf-8") if p.exists() else "not started")
        return
    if args.action == "pause":
        (PRIVATE / "pause.request").write_text("pause requested\n", encoding="utf-8")
        return
    if args.action == "resume":
        marker = PRIVATE / "pause.request"
        if marker.exists():
            marker.rename(PRIVATE / ("pause.handled." + uuid.uuid4().hex))
    # Only one supervisor may own this study. Locks release automatically on process loss.
    with (PRIVATE / "supervisor.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print("study supervisor already running")
                return
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                print("study supervisor already running")
                return
        supervisor = Supervisor(args.thread_id)
        try:
            supervisor.run(one_batch=args.action == "verify-teacher")
        except Exception as exc:
            supervisor.state["status"] = "supervisor_needs_review"
            supervisor.event("supervisor_error", error=str(exc))
            supervisor.save()
            raise


if __name__ == "__main__":
    main()
