# homeroom 🏫

**Budget-aware teacher-in-the-loop fine-tuning.** A strong "teacher" model writes lessons, sets
exams, diagnoses a small "student" model's mistakes, and writes remedial material only where the
student fails. The goal: a small specialist you own, trained for a few dollars of teacher spend.

> Status: **Phase 1 pilot** (Banking77). No results yet. This README will report honest numbers,
> including negative ones.

## Why

Independent developers can fine-tune small open models, but lack the data big labs have, and
asking a frontier model to label every example is expensive. homeroom asks a narrow, measurable
question:

> At equal teacher spend, does closed-loop teaching beat (a) a handful of real examples and
> (b) bulk synthetic data?

The loop idea itself builds on prior work (LLM2LLM, Lion, DataEnvGym, Montessori-Instruct,
pedagogically-inspired distillation). homeroom's focus is **accuracy per dollar**, fair equal-spend
baselines, and a simple tool. See [`docs/related-work.md`](docs/related-work.md).

## How the classroom works

```
real seed examples (k per intent)
        │
teacher writes EXAM ──► split: diagnostic half │ score half
        │
teacher writes LESSONS (definition + examples per intent)
        │
   ┌─► train student ─► take exam ─► top confusions (A mistaken for B)
   │                                        │
   └── teacher writes REMEDIAL examples for A vs B ◄┘   (stop: budget / no gain / max rounds)
        │
final evaluation on the official human-labeled test set (only here)
```

Every teacher call passes through a hard USD budget, a disk cache and a cost ledger.

## Quick start

```bash
git clone https://github.com/<you>/homeroom && cd homeroom
uv venv && uv pip install -e ".[dev]"        # or: python -m venv .venv && pip install -e ".[dev]"
pytest -q
homeroom run --config configs/pilot_banking77.yaml --dry-run --seeds 0   # mock teacher, no cost
```

GPU students (SetFit, ModernBERT):

```bash
# 1) install torch with the CUDA wheel for your machine: https://pytorch.org/get-started/locally/
# 2) then:
uv pip install -e ".[gpu]"
homeroom check-gpu
homeroom run --config configs/pilot_banking77.yaml --arms real_fewshot --seeds 0   # ≈88% expected
```

Paid teacher: copy `.env.example` to `.env`, set provider, model, key and **current prices**
(runs refuse to start with unset prices), then:

```bash
homeroom run --config configs/pilot_banking77.yaml
homeroom summarize --config configs/pilot_banking77.yaml
```

Windows tip: WSL2 is usually smoother for the GPU stack. Pre-Ampere GPUs (e.g. RTX 20xx) use fp16.

## Arms

| Arm | Description |
|---|---|
| `real_fewshot` | Student on k real examples per intent only |
| `bulk` | Uniform teacher-generated examples, spend matched to `classroom` |
| `classroom` | Exam → lessons → diagnose → remediate loop |
| `full_data` | Oracle upper bound on all real training data |

Published Banking77 references (SetFit + MPNet, Loukas et al. 2023): 10-shot 88.0%, 20-shot 91.2%,
full data 94.0%.

## Repo layout

See [`CLAUDE.md`](CLAUDE.md) (also the working guide for Claude Code) and [`docs/`](docs/).

## Data and terms

Code, configs and result tables are public. Teacher-generated text is **not** committed until the
teacher provider's terms are confirmed to allow it. Check your provider's terms before training on
its outputs.

## License

Apache-2.0
