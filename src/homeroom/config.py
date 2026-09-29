"""Config loading: YAML with ${VAR} / ${VAR:-default} environment expansion."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# Settings used by --dry-run: mock teacher, tf-idf student, fake prices, tiny budget.
DRY_RUN_OVERRIDES = {
    "experiment": {"name_suffix": "_dryrun"},
    "teacher": {"provider": "mock", "model": "mock", "price_input_per_mtok": 1.0,
                "price_output_per_mtok": 5.0},
    "budget": {"usd_per_arm": 0.5},
    "student": {"kind": "tfidf"},
    "bulk": {"max_examples_per_intent": 30},
}


def load_config(path: str | Path, dry_run: bool = False) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    cfg = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if dry_run:
        cfg = deep_update(cfg, DRY_RUN_OVERRIDES)
    exp = cfg.setdefault("experiment", {})
    exp["full_name"] = exp.get("name", "experiment") + exp.get("name_suffix", "")
    return cfg


def results_dir(cfg: dict) -> Path:
    d = Path(cfg["experiment"].get("results_dir", "results")) / cfg["experiment"]["full_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def generated_dir(cfg: dict) -> Path:
    """Teacher-written artifacts (gitignored; not published until provider terms are checked)."""
    d = Path(cfg["data"].get("generated_dir", "data/generated")) / cfg["experiment"]["full_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_snapshot(cfg: dict) -> dict:
    """Full resolved config for the result JSON, with anything key/secret-like removed."""
    def scrub(v):
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()
                    if not any(s in k.lower() for s in ("key", "secret", "password"))}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v
    return scrub(copy.deepcopy(cfg))
