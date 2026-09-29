# Research context (from the design conversation, Sept 2026)

This is the distilled background for homeroom. Read before changing the design.
Sources are listed in `related-work.md`. Numbers marked (reported) come from papers/blogs and
were not re-verified by us.

## 1. The idea

Independent developers can fine-tune or RAG small open models into specialists, but lack the data
that frontier labs and big institutions have. Hand-made synthetic data is slow and often poor;
calling a frontier model once per example is expensive.

**"AI classroom":** a frontier teacher does not teach everything. It teaches concepts, prepares
materials the student can learn from, gives exams, and **changes the materials when they don't
work**. Sometimes a student becomes more expert in a narrow subject than its teacher.

## 2. Validation verdict: partially novel

The mechanism is prior art:
- **LLM2LLM** (2024): train student → find failures → teacher generates similar-but-new examples →
  repeat. Llama-2-7B on GSM8K: 1.52% → 25.70%; targeted 471 examples → 23.73% vs bulk 500 → 18.12%
  (reported).
- **Lion** (2023): teacher / referee / generator loop on "hard" instructions. ~450k API calls,
  ≈$900 for 70K examples vs Alpaca's <$500 bulk 52K → feedback loops can COST MORE.
- **DataEnvGym** (ICLR 2025): teacher agents in environments that return student errors/skills.
- **Montessori-Instruct** (ICLR 2025): teacher optimized (DPO) on data influence on the student;
  a tailored 8B teacher beat GPT-4o data.
- **IOA** (arXiv 2602.12172, 2026): "pedagogically-inspired" distillation — diagnose deficiencies,
  prerequisite curriculum, mastery gating (Bloom), ZPD. Essentially this idea, published.
- **Sakana RL Teachers** (2025): teachers trained to explain; a 7B teacher trained a 32B student.
- **CBIT** (arXiv 2511.07932): teacher writes reusable problem-GENERATOR programs → $0.71 vs $30.03
  per 1k problems (42× cheaper) (reported). Biggest cost lever: teacher writes generators, not items.
- Textbook-style data (phi-1, TinyStories, Cosmopedia): curriculum/prompt design > teacher size.

**What is left (our angle):** budget-aware, reproducible **accuracy-per-dollar** evidence against
equal-spend baselines, plus an easy open tool. Exams independent of teaching material. Teacher
calls for syllabus/exams/diagnosis/remediation, never blind per-example labeling.

## 3. Known risks (design must address)

- **Imitation ≠ capability** (Gudibande et al. 2023): students copy style, not knowledge → use
  ground-truth exams, narrow tasks.
- **Small-student capacity** (arXiv 2502.12143): ≤3B students learn worse from long CoT; simplify.
- **Model collapse:** replacing real data with synthetic degrades; accumulating with real anchors
  is fine → always keep real seed examples.
- **Teacher errors** propagate; overlapping labels make teacher labels noisy.
- **Exam contamination:** teacher writing both lessons and exams can inflate scores → final
  evaluation only on the official human-labeled test set; exam split into diagnostic/score halves;
  training text de-duplicated against exam.
- **Facts vs skills:** RAG beats fine-tuning for injecting facts (Ovadia et al.: 0.875 vs ~0.50 on
  current events, reported); fine-tuning is for skills/formats/decisions.
- **Terms of service:** OpenAI and Anthropic terms restrict using outputs to build competing models;
  open models vary (DeepSeek-R1 MIT allows distillation; most Qwen2.5 Apache-2.0; Llama 3.1+ allows
  with attribution; Gemma treats distilled models as derivatives). Check before publishing data or
  models. Not legal advice.

## 4. New models considered (Sept 2026)

These are non-generative "System One" decision models: structured input → typed outputs (choice /
score / yes-no) with probabilities in one forward pass.

- **Jev** (TypeSafe AI, launched 2026-09-15, closed API, early access). Claims: ~two orders of
  magnitude faster/cheaper than LLMs on decision tasks; input $0.042 / MTok, output free; calibrated
  probabilities; cannot produce type errors; cannot generate text. Evals built by its own team and
  referenced against GPT-6 Astra + Fable 5.1 averages (may inherit their errors). Cardinality ≤255.
  → **Role in homeroom (Phase 2): cheap teaching assistant** — grading, filtering, uncertainty
  routing. Cannot write lessons. Check its terms before training on its outputs.
- **Laya** (Convai Innovations, 2026-09-18, Apache-2.0, local): 421M English checkpoint on
  ModernBERT-large, 322M multilingual on mmBERT-base, router; 512-token context; framed as a fast
  base for specialization, not zero-shot; no training pipeline released.
  → **Role: one student arm**, always compared with plain ModernBERT-large (its base).
- **Kev** (Jared Palmer): 0.8B/4B/9B frozen base + rank-16 LoRA + pointer head; recipe published;
  ~$95 H100 time (reported). → Recipe reference for a decoder-based decision student (needs rented GPU).
- **LLM-JEPA / STP / DLLM-JEPA** (JEPA-style training losses): orthogonal student-side data
  efficiency (STP reports matching baseline with 16× less data on NL-RX-SYNTH, negligible
  overhead). → Phase 3 ablation, not core.

## 5. Why Banking77, and the targets

Banking77: 77 fine-grained, overlapping banking intents; 10,003 train / 3,080 test (40 per intent);
short messages (avg ~12 words); real business use case; public; strong published baselines.

| Setup | Accuracy (reported) | Source |
|---|---|---|
| Full data, BERT | 93.7 | Casanueva et al. 2020 |
| Full data, MPNet | 94.0 | Loukas et al. 2023 |
| SetFit MPNet 3-shot | 76.3 | Loukas et al. 2023 |
| SetFit MPNet 10-shot | 88.0 | Loukas et al. 2023 |
| SetFit MPNet 20-shot | 91.2 | Loukas et al. 2023 |
| GPT-4 3-shot in-context (curated / random) | 83.1 / 74.2 | Loukas et al. 2023 (cost ≈ $1,480) |

Notes: ceiling ≈94% partly due to label errors (Ying & Thomas 2022). Real examples are strong, and
the frontier teacher is not perfect on overlapping intents, so the loop must correct its own errors.

**Targets (from k=10 real/intent, baseline ≈88%):**
- Minimum success: ≥91% (= 20 real shots) for ≤ ~$10 teacher spend.
- Strong: ≥93% (near full-data ceiling) using ~1% of the real labels.
- Stretch: from k=3 (≈76%) to >90%.
Also report macro-F1 and calibration (ECE).

Second domain (Phase 2): **LEDGAR** (LexGLUE, 100 contract-provision types) — the "specialist
field without data" story. Avoid medical/moderation for now (privacy, contested labels).

## 6. Hardware

RTX 2060 laptop 6 GB (Turing): fp16 only, no FlashAttention-2. Fits SetFit, ModernBERT-base,
ModernBERT-large/Laya via LoRA/8-bit Adam, Qwen ≤1.5B via 4-bit QLoRA. Rent for ≥3B (Kaggle/Colab
free T4s; Modal/RunPod/Vast). Teacher cost dominates GPU cost for this project.

## 7. Expected value (honest)

- Useful: accuracy-per-dollar numbers for "replace per-call LLM classification with an owned tiny
  model"; an open tool (few real examples + budget → trained model + report).
- Publication: realistic = arXiv tech report / blog / workshop (efficient or data-centric NLP).
  Main conference only with large, multi-domain wins. Negative results are still publishable as a post.
- Competing answers: Jev may make a student unnecessary for some users (our counter: privacy,
  offline, no lock-in, latency); 20 real examples may be cheap to collect.
- Hence the **go/no-go pilot**: 1–2 weeks, ~$10–20, before building the framework further.

## 8. Repo policy

Public from day 1 for code, configs and results (timestamps priority, credibility). Keys only in
`.env` + gitleaks. Teacher-generated data stays private until provider terms are checked.
