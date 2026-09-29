"""Teacher backends + TeacherSession (the ONLY entry point for teacher calls).

TeacherSession.ask = budget pre-check -> disk cache -> provider call (with retries) -> charge.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .budget import Budget, Usage


@dataclass
class Reply:
    text: str
    usage: Usage
    cached: bool = False


class BaseTeacher:
    provider = "base"

    def __init__(self, model: str, temperature: float = 0.9, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
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
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
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
        r = self.client.messages.create(
            model=self.model, system=system, max_tokens=self.max_tokens, temperature=self.temperature,
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
            out = {"items": self._variants(meta.get("seeds", []), n, rng)}
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


def make_teacher(cfg: dict) -> BaseTeacher:
    t = cfg["teacher"]
    kw = {"temperature": float(t.get("temperature", 0.9)), "max_tokens": int(t.get("max_tokens", 2048))}
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

    def ask(self, system: str, prompt: str, *, task: str, meta: dict | None = None, salt: str = "") -> Reply:
        meta = meta or {}
        path = self.cache_dir / f"{self._key(system, prompt, salt)}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            reply = Reply(d["text"], Usage(d["in"], d["out"]), cached=True)
            self.budget.check(reply.usage.input_tokens, reply.usage.output_tokens)  # same cap on reruns
            self.budget.charge(reply.usage, tag=task, cached=True)
            return reply
        self.budget.check(len(system + prompt) // 3, self.teacher.max_tokens)
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
        path.write_text(json.dumps({"task": task, "text": reply.text, "in": reply.usage.input_tokens,
                                    "out": reply.usage.output_tokens}), encoding="utf-8")
        self.budget.charge(reply.usage, tag=task, cached=False)
        return reply
