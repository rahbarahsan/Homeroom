"""Teacher backends + TeacherSession (the ONLY entry point for teacher calls).

TeacherSession.ask = budget pre-check -> disk cache -> provider call (with retries) -> charge.
TeacherSession.ask_many = the same for a list of requests; providers that support batching
(QueueTeacher) get whole waves of requests at once, everyone else is called sequentially.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .budget import Budget, BudgetExceeded, Usage


@dataclass
class Reply:
    text: str
    usage: Usage
    cached: bool = False


@dataclass
class Request:
    system: str
    prompt: str
    task: str
    meta: dict = field(default_factory=dict)
    salt: str = ""


class BaseTeacher:
    provider = "base"
    supports_batch = False

    def __init__(self, model: str, temperature: float | None = 0.9, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature  # None = don't send (models that reject sampling params)
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str, *, task: str, meta: dict) -> Reply:
        raise NotImplementedError


class OpenAICompatTeacher(BaseTeacher):
    """Any OpenAI-compatible endpoint: OpenAI, DeepSeek, OpenRouter, Together, vLLM, Ollama..."""
    provider = "openai_compatible"

    def __init__(self, model, base_url=None, api_key_env="TEACHER_API_KEY", **kw):
        super().__init__(model, **kw)
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or None, api_key=os.environ.get(api_key_env))

    def complete(self, system, prompt, *, task, meta):
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, **kw,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        u = r.usage
        return Reply(r.choices[0].message.content or "",
                     Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)))


class AnthropicTeacher(BaseTeacher):
    provider = "anthropic"

    def __init__(self, model, api_key_env="TEACHER_API_KEY", **kw):
        super().__init__(model, **kw)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get(api_key_env))

    def complete(self, system, prompt, *, task, meta):
        # Claude Opus 5.5 / Fable 5.x / Sonnet 5.x reject `temperature` (400): set it to null in config.
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        r = self.client.messages.create(
            model=self.model, system=system, max_tokens=self.max_tokens, **kw,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(b, "text", "") for b in r.content)
        return Reply(text, Usage(r.usage.input_tokens, r.usage.output_tokens))


class MockTeacher(BaseTeacher):
    """Offline teacher for --dry-run and tests: perturbs seed examples, returns valid JSON."""
    provider = "mock"
    _PRE = ["", "hi, ", "hello - ", "quick question: ", "please help, ", "urgent: "]
    _SUF = ["", " thanks", "?", " please", " asap", " - any idea?"]

    def _variants(self, seeds: list[str], n: int, rng: random.Random) -> list[str]:
        seeds = seeds or ["help me"]
        return [f"{rng.choice(self._PRE)}{rng.choice(seeds).lower()}{rng.choice(self._SUF)}" for _ in range(n)]

    def complete(self, system, prompt, *, task, meta):
        rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
        n = meta.get("n", 5)
        if task == "exam":
            # Short fragments of seeds: hard enough that dry runs exercise diagnosis + remediation.
            frags = [" ".join(s.split()[i:i + 2]) for s in meta.get("seeds", [])
                     for i in range(0, max(1, len(s.split()) - 1), 2)]
            out = {"items": [f"{rng.choice(self._PRE)}{rng.choice(frags or ['help me'])}" for _ in range(n)]}
        elif task == "lesson":
            out = {"definition": f"messages about {meta['intent']}", "confusable_with": [],
                   "examples": self._variants(meta.get("seeds", []), n, rng)}
        elif task == "bulk":
            out = {"examples": self._variants(meta.get("seeds", []), n, rng)}
        elif task == "remedial":
            out = {"rule": "mock rule", "a_examples": self._variants(meta.get("seeds_a", []), n, rng),
                   "b_examples": self._variants(meta.get("seeds_b", []), n, rng)}
        elif task == "verify":
            out = {"labels": meta.get("expected", [])}
        else:
            raise ValueError(task)
        text = json.dumps(out)
        return Reply(text, Usage(len(system + prompt) // 4, len(text) // 4))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 chars/token) for providers that report no usage."""
    return max(1, len(text) // 4)


class QueueTeacher(BaseTeacher):
    """File-based handoff to an external answerer (e.g. a subagent run by the operator).

    Each request is written to <queue_dir>/<key>.req.json; the teacher waits for <key>.reply.json.
    Answerers may instead reply per batch (see scripts/teacher_queue.py); `ingest_batch_replies`
    splits those into per-request reply files. Usage is ESTIMATED (chars/4, no thinking tokens),
    so spend is an API-equivalent estimate, not a bill.
    """
    provider = "queue"
    supports_batch = True

    def __init__(self, model, queue_dir, timeout_s: float = 4 * 3600, poll_s: float = 2.0, **kw):
        super().__init__(model, **kw)
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.poll_s = float(timeout_s), float(poll_s)

    def submit(self, key: str, system: str, prompt: str, task: str) -> None:
        req = self.queue_dir / f"{key}.req.json"
        if not req.exists():
            req.write_text(json.dumps({"id": key, "task": task, "system": system, "prompt": prompt}),
                           encoding="utf-8")

    def wait(self, keys: list[str]) -> dict[str, str]:
        deadline = time.time() + self.timeout_s
        out: dict[str, str] = {}
        while True:
            ingest_batch_replies(self.queue_dir)
            for k in keys:
                p = self.queue_dir / f"{k}.reply.json"
                if k not in out and p.exists():
                    out[k] = json.loads(p.read_text(encoding="utf-8"))["text"]
            if len(out) == len(keys):
                return out
            if time.time() > deadline:
                raise TimeoutError(f"queue teacher: {len(keys) - len(out)} replies missing after "
                                   f"{self.timeout_s:.0f}s in {self.queue_dir}")
            time.sleep(self.poll_s)

    def complete(self, system, prompt, *, task, meta):
        key = meta["_cache_key"]
        self.submit(key, system, prompt, task)
        text = self.wait([key])[key]
        return Reply(text, Usage(estimate_tokens(system + prompt), estimate_tokens(text)))


def ingest_batch_replies(queue_dir: Path) -> int:
    """Split batches/*.reply.json ({id: text}) into per-request <id>.reply.json files."""
    n = 0
    for bp in sorted((Path(queue_dir) / "batches").glob("*.reply.json")):
        try:
            replies = json.loads(bp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # answerer may still be writing it
        for k, text in replies.items():
            p = Path(queue_dir) / f"{k}.reply.json"
            if not p.exists() and (Path(queue_dir) / f"{k}.req.json").exists():
                if not isinstance(text, str):
                    text = json.dumps(text)
                p.write_text(json.dumps({"text": text}), encoding="utf-8")
                n += 1
    return n


def make_teacher(cfg: dict) -> BaseTeacher:
    t = cfg["teacher"]
    temp = t.get("temperature", 0.9)
    kw = {"temperature": None if temp in (None, "", "null") else float(temp),
          "max_tokens": int(t.get("max_tokens", 2048))}
    provider = t.get("provider", "mock")
    if provider == "mock":
        return MockTeacher("mock", **kw)
    if not t.get("model"):
        raise ValueError("teacher.model is empty: set TEACHER_MODEL in .env")
    if float(t.get("price_input_per_mtok") or 0) <= 0 or float(t.get("price_output_per_mtok") or 0) <= 0:
        raise ValueError("Refusing to run a paid teacher with unset prices. Set TEACHER_PRICE_IN / "
                         "TEACHER_PRICE_OUT (USD per million tokens) in .env.")
    if provider == "openai_compatible":
        return OpenAICompatTeacher(t["model"], base_url=t.get("base_url"), **kw)
    if provider == "anthropic":
        return AnthropicTeacher(t["model"], **kw)
    if provider == "queue":
        qdir = Path(t.get("queue_dir", "data/teacher_queue")) / cfg["experiment"]["full_name"]
        return QueueTeacher(t["model"], qdir, timeout_s=float(t.get("queue_timeout_s", 4 * 3600)), **kw)
    raise ValueError(f"unknown teacher provider: {provider}")


class TeacherSession:
    def __init__(self, teacher: BaseTeacher, budget: Budget, cache_dir: Path, retries: int = 3):
        self.teacher, self.budget, self.retries = teacher, budget, retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, prompt: str, salt: str) -> str:
        blob = json.dumps([self.teacher.provider, self.teacher.model, self.teacher.temperature,
                           system, prompt, salt])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _worst_case(self, system: str, prompt: str) -> tuple[int, int]:
        return len(system + prompt) // 3, self.teacher.max_tokens

    def _cached(self, key: str) -> Reply | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return Reply(d["text"], Usage(d["in"], d["out"]), cached=True)

    def _store(self, key: str, task: str, reply: Reply) -> None:
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"task": task, "text": reply.text, "in": reply.usage.input_tokens,
                        "out": reply.usage.output_tokens}), encoding="utf-8")

    def ask(self, system: str, prompt: str, *, task: str, meta: dict | None = None, salt: str = "") -> Reply:
        key = self._key(system, prompt, salt)
        meta = {**(meta or {}), "_cache_key": key}
        reply = self._cached(key)
        if reply is not None:
            self.budget.check(reply.usage.input_tokens, reply.usage.output_tokens)  # same cap on reruns
            self.budget.charge(reply.usage, tag=task, cached=True)
            return reply
        self.budget.check(*self._worst_case(system, prompt))
        last_err = None
        for attempt in range(self.retries):
            try:
                reply = self.teacher.complete(system, prompt, task=task, meta=meta)
                break
            except Exception as e:  # provider/network errors
                last_err = e
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"teacher call failed after {self.retries} attempts: {last_err}")
        self._store(key, task, reply)
        self.budget.charge(reply.usage, tag=task, cached=False)
        return reply

    def ask_many(self, requests: list[Request]) -> tuple[list[Reply], bool]:
        """Answer requests in order. Returns (replies for a prefix of `requests`, budget_stopped).

        Batch-capable teachers get waves of requests at once; a wave only contains as many
        requests as fit the remaining budget at their WORST-CASE cost, so the hard cap holds
        exactly as with sequential calls. Charges are applied in request order.
        """
        if not self.teacher.supports_batch:
            out = []
            for r in requests:
                try:
                    out.append(self.ask(r.system, r.prompt, task=r.task, meta=r.meta, salt=r.salt))
                except BudgetExceeded:
                    return out, True
            return out, False

        out: list[Reply] = []
        i = 0
        while i < len(requests):
            # Cache hits are charged immediately (in order), like `ask`.
            r = requests[i]
            key = self._key(r.system, r.prompt, r.salt)
            hit = self._cached(key)
            if hit is not None:
                try:
                    self.budget.check(hit.usage.input_tokens, hit.usage.output_tokens)
                except BudgetExceeded:
                    return out, True
                self.budget.charge(hit.usage, tag=r.task, cached=True)
                out.append(hit)
                i += 1
                continue
            # Build a wave of consecutive cache misses that fits the remaining budget at worst case.
            wave, reserved = [], 0.0
            while i + len(wave) < len(requests):
                rr = requests[i + len(wave)]
                kk = self._key(rr.system, rr.prompt, rr.salt)
                if self._cached(kk) is not None:
                    break
                est = self.budget.cost(Usage(*self._worst_case(rr.system, rr.prompt)))
                if self.budget.spent + reserved + est > self.budget.limit_usd:
                    break
                wave.append((rr, kk))
                reserved += est
            if not wave:
                return out, True
            for rr, kk in wave:
                self.teacher.submit(kk, rr.system, rr.prompt, rr.task)
            texts = self.teacher.wait([kk for _, kk in wave])
            for rr, kk in wave:
                reply = Reply(texts[kk], Usage(estimate_tokens(rr.system + rr.prompt),
                                               estimate_tokens(texts[kk])))
                self._store(kk, rr.task, reply)
                self.budget.charge(reply.usage, tag=rr.task, cached=False)
                out.append(reply)
            i += len(wave)
        return out, False
