# CLAUDE.md — homeroom

This file is the working memory for Claude Code. Read it fully before doing anything.
Deeper context lives in `docs/` (read `docs/research-context.md` before any design change).

## What this project is

**homeroom** is a budget-aware, teacher-in-the-loop fine-tuning framework ("AI classroom").
A strong **teacher** model (frontier API or strong open model) does not label every example.
Like a real teacher it: writes lessons per concept, sets exams, diagnoses the **student's**
specific mistakes, and writes targeted remedial material only where the student fails.
**Students** are small open models (≈20M–1.5B) that we fine-tune locally.

The research question is economic, not conceptual:
> For a fixed teacher budget (USD), does closed-loop teaching produce a better small specialist
> than (a) a handful of real examples alone and (b) bulk synthetic data bought with the same money?

The concept itself (diagnose → targeted data → retest) is prior art (LLM2LLM, Lion, DataEnvGym,
Montessori-Instruct, IOA 2026). Our contribution is **honest accuracy-per-dollar evidence** and an
**easy open tool** for independent developers. Never claim conceptual novelty in docs or README.

## Current phase: Phase 1 pilot (go / no-go)

Domain: **Banking77** intent classification (77 intents, 10,003 train / 3,080 test, short texts).
Arms (same teacher budget cap, 3 seeds each):

| Arm | Meaning |
|---|---|
| `real_fewshot` | Student trained only on k real examples/intent (k=10). Cost $0. |
| `bulk` | Teacher generates examples uniformly per intent, no feedback. Spend matched to classroom. |
| `classroom` | Exam → lessons → train → diagnose confusions on exam → remedial material → retrain. |
| `full_data` | Oracle upper bound: all 10,003 real training examples. Cost $0. |

Published reference numbers (Loukas et al. 2023, SetFit + all-mpnet-base-v2):
10-shot 88.0 acc, 20-shot 91.2, full data 94.0. GPT-4 3-shot in-context: 83.1.

**Go** if `classroom` reaches ≥ 91% test accuracy (the 20-real-shot level) from k=10 for ≤ ~$10
AND beats `bulk` at matched spend across 3 seeds. **No-go** → write a short negative-results post.

### Milestones (do them in order)
1. `homeroom run --dry-run` passes (mock teacher, tf-idf student, no GPU, no API cost).
2. Reproduce the baseline: `real_fewshot` with SetFit+mpnet at k=10 ≈ 88% (±2). If far off, fix
   the student pipeline before touching the teacher. This validates our evaluation.
3. Fill in teacher provider/model/prices in `.env`; run `classroom` + `bulk` for seed 0 at a tiny
   budget ($1) to check cost accounting against the provider dashboard.
4. Full pilot: 3 seeds × all arms; `homeroom summarize`.
5. Decide go / no-go; record it in `docs/decisions.md`.

Phase 2 ideas (NOT now): Jev as cheap grader/TA, Laya vs its ModernBERT base, Qwen-1.5B QLoRA
student, LEDGAR legal domain, STP/LLM-JEPA loss on the student. See `docs/experiment-plan.md`.

## Hard rules

1. **Never commit secrets.** Keys live only in `.env` (gitignored). Pre-commit runs gitleaks.
2. **Never look at the official test set before final evaluation.** Test data is used only in
   `final_eval`. No tuning, prompt changes, or stopping decisions based on test scores.
   Dev/stopping signal comes from the teacher-written exam ("score" half) or a real holdout.
3. **Every teacher call goes through `TeacherSession.ask`** (budget check + cache + ledger). No
   direct SDK calls anywhere else. Budget is a hard cap; `BudgetExceeded` must stop generation.
4. **Refuse to spend with unset prices.** If `price_*_per_mtok` is 0 for a real provider, abort.
5. **Every run writes a result JSON** to `results/<experiment>/` (config snapshot, seed, spend,
   tokens, metrics, git commit). Results are small and committed; teacher text is NOT committed.
6. **Do not publish teacher-generated datasets** (`data/cache/`, `data/generated/`) until the teacher
   provider's terms are checked (see `docs/decisions.md` → open items).
7. Fixed seeds everywhere (sampling, training, teacher cache salt). Report mean ± std over seeds.
8. Keep the teacher prompts in `src/homeroom/prompts.py` only. Changing a prompt = new experiment
   name (so cached/old results are not mixed).
9. Exam leakage: generated training text must be de-duplicated against exam items, and the teacher
   only sees the **diagnostic** half of the exam; stopping uses the **score** half.

## Hardware (developer machine)

NVIDIA RTX 2060 laptop, **6 GB VRAM** (~5.4 GB free), Turing (compute 7.5), 90 W cap, Windows (WDDM).
- Use **fp16**, never bf16. No FlashAttention-2 (needs Ampere+); default SDPA attention.
- Fits: SetFit (mpnet-base), ModernBERT-base full FT, ModernBERT-large/Laya with LoRA or 8-bit Adam,
  Qwen ≤1.5B 4-bit QLoRA (seq ≤1k, batch 1–2 + grad accumulation).
- Does not fit: 4B/9B students → rent (Kaggle/Colab T4 free tiers, or Modal/RunPod/Vast).
- Laptop throttles: keep runs short, checkpoint often. WSL2 is usually smoother than native Windows.
- Check with `homeroom check-gpu`.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"                 # core (no torch)
# torch: install the CUDA wheel matching your setup from pytorch.org, then:
uv pip install -e ".[gpu]"                             # setfit, transformers, datasets
pre-commit install

homeroom check-gpu
homeroom run --config configs/pilot_banking77.yaml --dry-run          # milestone 1
homeroom run --config configs/pilot_banking77.yaml --arms real_fewshot --seeds 0
homeroom run --config configs/pilot_banking77.yaml                     # full pilot
homeroom summarize --config configs/pilot_banking77.yaml
pytest -q
```

## Repo map

```
src/homeroom/
  config.py     YAML + ${ENV} expansion
  data.py       Banking77 download (PolyAI GitHub CSV), toy dataset, k-shot sampling
  budget.py     USD budget, hard cap, JSONL ledger
  teacher.py    OpenAI-compatible / Anthropic / Mock teachers + TeacherSession (cache+budget)
  prompts.py    all teacher prompts + robust JSON parsing
  students.py   tfidf (smoke test), setfit (primary), hf_classifier (ModernBERT/Laya-style)
  evaluate.py   accuracy, macro-F1, ECE, NLL, top confusion pairs
  arms.py       real_fewshot, full_data, bulk, classroom
  run.py        CLI: run / summarize / check-gpu
configs/        experiment configs (one file per experiment)
docs/           research context, plan, related work, decisions, kickoff prompts
results/        per-run JSON + summary.md (committed)
data/           raw downloads, teacher cache, generated data (gitignored)
```

## Coding conventions

- Python ≥3.10, type hints, small pure functions, no notebooks in `src/`.
- Heavy imports (torch, setfit, transformers) are lazy, inside student classes, so the core and
  tests run without a GPU stack.
- Add a test for any change to budget, sampling, dedup, or metrics.
- Prefer editing configs over adding CLI flags.
- When unsure about a research choice, write the question into `docs/decisions.md` instead of
  silently picking; ask the human.
