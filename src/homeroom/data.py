"""Datasets and few-shot sampling.

Banking77 is downloaded from the original PolyAI GitHub CSVs (no HF loading script needed).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}

TASK_DESCRIPTIONS = {
    "banking77": (
        "Classify a single customer-service message sent to a digital bank / card app into exactly "
        "one of 77 fine-grained intents. Many intents overlap (e.g. card_arrival vs "
        "card_delivery_estimate), so boundaries matter."
    ),
    "toy": "Classify a short customer message into one of a few banking intents.",
}


@dataclass
class TaskData:
    name: str
    labels: list[str]
    train: pd.DataFrame  # columns: text (str), label (int)
    test: pd.DataFrame
    description: str

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed trusted URL)
    return dest


def _load_banking77(cache_dir: Path) -> TaskData:
    frames = {}
    for split, url in BANKING77_URLS.items():
        df = pd.read_csv(_download(url, cache_dir / "banking77" / f"{split}.csv"))
        frames[split] = df.rename(columns={"category": "intent"})
    labels = sorted(frames["train"]["intent"].unique())
    idx = {l: i for i, l in enumerate(labels)}
    for split in frames:
        frames[split]["label"] = frames[split]["intent"].map(idx).astype(int)
        frames[split] = frames[split][["text", "label"]].reset_index(drop=True)
    return TaskData("banking77", labels, frames["train"], frames["test"], TASK_DESCRIPTIONS["banking77"])


def _load_toy() -> TaskData:
    """Tiny synthetic dataset for tests / offline smoke runs."""
    templates = {
        "card_arrival": ["where is my card", "my card has not arrived", "still waiting for my new card",
                         "card not delivered yet", "when will my card come", "has my card been sent"],
        "exchange_rate": ["what is the exchange rate", "how do you set exchange rates",
                          "rate for euro to dollar", "is the fx rate fair", "current exchange rate please",
                          "why is the rate different"],
        "lost_or_stolen_card": ["i lost my card", "my card was stolen", "someone took my card",
                                "cannot find my card anywhere", "report a stolen card", "card missing from wallet"],
        "top_up_failed": ["my top up failed", "top-up did not work", "could not add money",
                          "top up was declined", "adding funds keeps failing", "why did my top up fail"],
    }
    labels = sorted(templates)
    rows = []
    for li, lab in enumerate(labels):
        for t in templates[lab]:
            for p in ["", "hi, ", "hello "]:
                rows.append((p + t, li))
    df = pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    test = df.groupby("label", group_keys=False).head(4).reset_index(drop=True)
    train = df.drop(df.groupby("label", group_keys=False).head(4).index).reset_index(drop=True)
    return TaskData("toy", labels, train, test, TASK_DESCRIPTIONS["toy"])


def load_task(name: str, cache_dir: str | Path = "data/raw") -> TaskData:
    if name == "banking77":
        return _load_banking77(Path(cache_dir))
    if name == "toy":
        return _load_toy()
    raise ValueError(f"unknown dataset: {name}")


def sample_k_shot(df: pd.DataFrame, k: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample k examples per label (or all if fewer). Returns (sampled, remaining)."""
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in df.groupby("label"):
        n = min(k, len(g))
        picked.extend(rng.choice(g.index.to_numpy(), size=n, replace=False).tolist())
    sampled = df.loc[sorted(picked)].reset_index(drop=True)
    remaining = df.drop(index=picked).reset_index(drop=True)
    return sampled, remaining
