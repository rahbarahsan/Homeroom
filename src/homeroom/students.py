"""Student models. All expose fit(texts, labels) and predict_proba(texts) -> (n, n_classes).

- tfidf:         sklearn baseline, CPU, seconds. For smoke tests / dry runs only.
- setfit:        primary pilot student (matches the published Banking77 baselines).
- hf_classifier: plain sequence classifier (ModernBERT-base/large, Laya-style encoders).
Heavy imports are lazy so the core package works without torch.
"""
from __future__ import annotations

import dataclasses
import tempfile

import numpy as np


def _full_proba(partial: np.ndarray, seen_classes, n_classes: int) -> np.ndarray:
    out = np.zeros((partial.shape[0], n_classes), dtype=float)
    out[:, np.asarray(seen_classes, dtype=int)] = partial
    return out


def _filter_kwargs(dc_cls, kwargs: dict) -> dict:
    valid = {f.name for f in dataclasses.fields(dc_cls)}
    return {k: v for k, v in kwargs.items() if k in valid}


class TfidfStudent:
    def __init__(self, n_classes: int, cfg: dict, seed: int):
        self.n_classes, self.cfg, self.seed = n_classes, cfg, seed

    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            LogisticRegression(max_iter=3000, C=float(self.cfg.get("C", 10.0)), random_state=self.seed))
        self.model.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts):
        p = self.model.predict_proba(list(texts))
        return _full_proba(p, self.model.classes_, self.n_classes)


class SetFitStudent:
    def __init__(self, n_classes: int, cfg: dict, seed: int):
        self.n_classes, self.cfg, self.seed = n_classes, cfg, seed

    def fit(self, texts, labels):
        import torch
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        self.model = SetFitModel.from_pretrained(self.cfg.get("model", "sentence-transformers/all-mpnet-base-v2"))
        args = TrainingArguments(**_filter_kwargs(TrainingArguments, {
            "batch_size": int(self.cfg.get("batch_size", 16)),
            "num_epochs": int(self.cfg.get("num_epochs", 1)),
            "num_iterations": int(self.cfg.get("num_iterations", 20)),
            "max_steps": int(self.cfg.get("max_steps", -1)),
            "seed": self.seed,
            "use_amp": torch.cuda.is_available(),  # fp16 on Turing; never bf16
            "report_to": "none",
        }))
        ds = Dataset.from_dict({"text": list(texts), "label": [int(x) for x in labels]})
        Trainer(model=self.model, args=args, train_dataset=ds).train()
        return self

    def predict_proba(self, texts):
        p = self.model.predict_proba(list(texts))
        p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
        return _full_proba(p, self.model.model_head.classes_, self.n_classes)


class HFClassifierStudent:
    def __init__(self, n_classes: int, cfg: dict, seed: int):
        self.n_classes, self.cfg, self.seed = n_classes, cfg, seed

    def fit(self, texts, labels):
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, Trainer, TrainingArguments)

        name = self.cfg.get("model", "answerdotai/ModernBERT-base")
        self.max_length = int(self.cfg.get("max_length", 64))
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=self.n_classes)
        ds = Dataset.from_dict({"text": list(texts), "label": [int(x) for x in labels]})
        ds = ds.map(lambda b: self.tok(b["text"], truncation=True, max_length=self.max_length), batched=True)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(**_filter_kwargs(TrainingArguments, {
                "output_dir": tmp, "per_device_train_batch_size": int(self.cfg.get("batch_size", 32)),
                "num_train_epochs": float(self.cfg.get("epochs", 5)),
                "learning_rate": float(self.cfg.get("lr", 5e-5)), "warmup_ratio": 0.1,
                "weight_decay": 0.01, "fp16": torch.cuda.is_available(), "bf16": False,
                "save_strategy": "no", "report_to": [], "logging_steps": 50, "seed": self.seed,
            }))
            Trainer(model=self.model, args=args, train_dataset=ds,
                    data_collator=DataCollatorWithPadding(self.tok)).train()
        return self

    def predict_proba(self, texts, batch_size: int = 128):
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        outs = []
        texts = list(texts)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(device)
                outs.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.n_classes))


STUDENTS = {"tfidf": TfidfStudent, "setfit": SetFitStudent, "hf_classifier": HFClassifierStudent}


def make_student(cfg: dict, n_classes: int, seed: int):
    s = cfg["student"]
    kind = s.get("kind", "setfit")
    return STUDENTS[kind](n_classes, s.get(kind, {}) or {}, seed)


def free_gpu():
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
