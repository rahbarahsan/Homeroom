"""USD budget with a hard cap and an append-only JSONL ledger.

`spent` counts *accounted* cost (cache hits are charged at their original cost so reruns are
comparable); `real_spent` counts only money actually sent to the provider in this process.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Budget:
    limit_usd: float
    price_input_per_mtok: float
    price_output_per_mtok: float
    ledger_path: Path | None = None
    spent: float = 0.0
    real_spent: float = 0.0
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_tag: dict = field(default_factory=dict)

    def cost(self, usage: Usage) -> float:
        return (usage.input_tokens * self.price_input_per_mtok
                + usage.output_tokens * self.price_output_per_mtok) / 1_000_000

    @property
    def remaining(self) -> float:
        return self.limit_usd - self.spent

    def check(self, est_input_tokens: int, est_output_tokens: int) -> None:
        """Raise before a call if its worst-case cost could break the cap."""
        est = self.cost(Usage(est_input_tokens, est_output_tokens))
        if self.spent + est > self.limit_usd:
            raise BudgetExceeded(
                f"next call (≤${est:.4f}) would exceed cap ${self.limit_usd:.2f} (spent ${self.spent:.4f})")

    def charge(self, usage: Usage, *, tag: str, cached: bool) -> float:
        c = self.cost(usage)
        self.spent += c
        if not cached:
            self.real_spent += c
        self.calls += 1
        self.cached_calls += int(cached)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.by_tag[tag] = self.by_tag.get(tag, 0.0) + c
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "tag": tag, "cached": cached,
                                    "in": usage.input_tokens, "out": usage.output_tokens,
                                    "usd": round(c, 6), "spent": round(self.spent, 6)}) + "\n")
        return c

    def summary(self) -> dict:
        return {"limit_usd": self.limit_usd, "spent_usd": round(self.spent, 6),
                "real_spent_usd": round(self.real_spent, 6), "calls": self.calls,
                "cached_calls": self.cached_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "by_tag_usd": {k: round(v, 6) for k, v in self.by_tag.items()}}
