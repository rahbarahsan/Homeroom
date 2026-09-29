"""All teacher prompts live here. Changing a prompt => use a new experiment name.

Every prompt asks for JSON only; `parse_json` is tolerant of code fences / chatter.
"""
from __future__ import annotations

import json
import re

SYSTEM = (
    "You are an expert teacher preparing training material for a SMALL text classifier. "
    "The student model is weak: it learns only from the examples you write, so examples must be "
    "realistic, diverse, and unambiguous. Always answer with a single JSON object and nothing else."
)


def _bullets(items: list[str], limit: int = 10) -> str:
    return "\n".join(f"- {t}" for t in items[:limit]) or "- (none)"


def exam_prompt(task_desc: str, intent: str, all_intents: list[str], n: int) -> str:
    return f"""TASK: {task_desc}
ALL INTENT LABELS: {", ".join(all_intents)}

Write an EXAM for the intent "{intent}": {n} realistic user messages that a careful human would
label as "{intent}" and not as any other label. Include some harder, borderline-but-still-correct
phrasings. Vary length, tone and vocabulary. No numbering.

Return JSON: {{"items": ["...", "..."]}}"""


def lesson_prompt(task_desc: str, intent: str, all_intents: list[str], seeds: list[str], n: int) -> str:
    return f"""TASK: {task_desc}
ALL INTENT LABELS: {", ".join(all_intents)}

TARGET INTENT: "{intent}"
REAL EXAMPLES (style reference, do not copy):
{_bullets(seeds)}

Write a short LESSON for the student about this intent:
1. "definition": one or two sentences defining the intent precisely.
2. "confusable_with": up to 3 labels from the list that are most easily confused with it.
3. "examples": {n} new realistic user messages that clearly belong to "{intent}", diverse in
   length, tone, vocabulary and situation. Do not copy or lightly paraphrase the real examples.

Return JSON: {{"definition": "...", "confusable_with": ["..."], "examples": ["..."]}}"""


def bulk_prompt(task_desc: str, intent: str, seeds: list[str], n: int) -> str:
    return f"""TASK: {task_desc}

INTENT: "{intent}"
REAL EXAMPLES (style reference, do not copy):
{_bullets(seeds)}

Write {n} new, diverse, realistic user messages with the intent "{intent}".

Return JSON: {{"examples": ["...", "..."]}}"""


def remedial_prompt(task_desc: str, a: str, b: str, def_a: str, def_b: str,
                    mistakes: list[str], n: int) -> str:
    return f"""TASK: {task_desc}

The student keeps confusing these two intents:
- "{a}": {def_a or "(no definition yet)"}
- "{b}": {def_b or "(no definition yet)"}

Messages whose correct label is "{a}" but the student predicted "{b}":
{_bullets(mistakes, 5)}

Write remedial material that teaches the boundary:
1. "rule": one sentence a student could use to tell "{a}" from "{b}".
2. "a_examples": {n} NEW messages that are clearly "{a}" and not "{b}" (include near-boundary cases).
3. "b_examples": {n} NEW messages that are clearly "{b}" and not "{a}".
Do not copy or paraphrase the listed messages.

Return JSON: {{"rule": "...", "a_examples": ["..."], "b_examples": ["..."]}}"""


def verify_prompt(task_desc: str, candidates: list[str], messages: list[str]) -> str:
    listing = "\n".join(f"{i}. {m}" for i, m in enumerate(messages))
    return f"""TASK: {task_desc}

For each message choose exactly one label from: {", ".join(candidates)}.
{listing}

Return JSON: {{"labels": ["<label for message 0>", "<label for message 1>", "..."]}}"""


_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply. Raises ValueError if none."""
    cleaned = _FENCE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(cleaned[start:end + 1])


def clean_examples(items, max_len: int = 400) -> list[str]:
    out = []
    for x in items or []:
        if isinstance(x, str):
            s = x.strip().strip('"').strip()
            if 2 <= len(s) <= max_len:
                out.append(s)
    return out
