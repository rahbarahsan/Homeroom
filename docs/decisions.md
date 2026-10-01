# Decision log

Format: date — decision — why. Add open questions at the bottom; resolve them explicitly.

- 2026-09 — Project name `homeroom`; Apache-2.0; public repo from day 1 (code/configs/results only).
- 2026-09 — Phase 1 domain Banking77; second domain LEDGAR later. Avoid medical/moderation.
- 2026-09 — Primary student SetFit + all-mpnet-base-v2 to reproduce published baselines (10-shot ≈ 88%).
- 2026-09 — Bulk arm spend is matched to classroom's actual spend per seed (fair comparison).
- 2026-09 — Classroom exam is teacher-written, split diagnostic/score; official test used only at the end.
- 2026-09 — Jev, Laya, other students, JEPA-style losses deferred to Phase 2/3.
- 2026-09-29 — Classroom boundary rules (teacher text) moved out of result JSON into gitignored
  `data/generated/<exp>/`; result JSON keeps only `n_rules` (hard rules 5/6).
- 2026-09-29 — Result JSON stores the full resolved config (secrets scrubbed) + `git_dirty` flag.
- 2026-09-29 — `bulk` with `match_classroom_spend` now refuses to run if the classroom result for
  that seed is missing (previously fell back silently to the full cap → unmatched comparison).
- 2026-09-29 — MockTeacher exam items are now short seed fragments so dry runs exercise
  diagnosis + remediation (mock only; no real prompt changed).
- 2026-09-29 — Milestone 1 done on native Windows (Python 3.12 venv via uv); WSL2 not installed and
  not needed so far.

## Milestone 1 result (dry run, seed 0, mock teacher, tf-idf student — plumbing only, not evidence)
real_fewshot 0.660 · classroom 0.665 ($0.237, 2 rounds, stopped: no_improvement) · bulk 0.663
($0.227 of $0.237 cap, stopped: budget).

## Milestone 2 — baseline reproduction (in progress, 2026-09-29)
Setup: native Windows, torch 2.11+cu128, setfit 1.2.0, transformers 5.17, sentence-transformers 6.1.
- Official run `real_fewshot` seed 0 (test, evaluated once): **acc 0.8312**, macro-F1 0.8277,
  ECE 0.492 — below the 88 ± 2 target.
- Debugging uses a DEV split only (`scripts/debug_student_dev.py`): 20/intent (1,540) sampled from
  train rows NOT in the k-shot sample, dev seed = seed + 10000. Test set is never loaded there.
- Dev, default config, seeds 0/1/2: acc 0.825 / 0.818 / 0.812 (mean 0.818) → not seed luck.
- Head: setfit's default `LogisticRegression()` (C=1) on unit-norm embeddings is badly
  under-confident (ECE ≈ 0.48). C=100 → ECE ≈ 0.05 but only +0.6 pt acc. All heads converged.
- Reference check (Loukas et al. 2023, arXiv 2308.14634): SetFit trained with "developers'
  recommended practices and hyperparameters" (≈ ours); 10-shot sampling procedure and number of
  runs are not stated; single number, no std. Prior SOTA they cite for 10-shot: 85.8 (Mehri & Eric).
  → 88.0 may not be reproducible with random 10-shot draws; treat it as an approximate target.
- More training (num_iterations 40, no step cap, ≈3,850 steps), dev, seed 0: acc 0.807 (default
  head) / 0.813 (C=100) vs 0.825 / 0.828 → longer training hurts (overfits). Seed 1 run was
  stopped by the host (low memory) before finishing.
- 2026-09-30 — **Decision A** (chosen by the human over B: keep 91% absolute, C: test older
  library versions): adopt `head_C: 100` (chosen on dev; experiment renamed
  `pilot_banking77_v2` so v1 results don't mix) and use our *measured* random-draw baselines as the
  reference instead of the published 88.0 / 91.2.
- Proposed go target (pending confirmation): keep the original meaning, "classroom from k=10
  reaches the 20-real-shot level", but measure that level ourselves: `real_fewshot` at k=20, same
  student recipe (`configs/pilot_banking77_k20.yaml`), 3 seeds. The ≤ ~$10 and beats-bulk conditions
  are unchanged.
- Official baselines (test, evaluated once per seed, head C=100):
  - `pilot_banking77_v2` real_fewshot k=10: 0.8373 / 0.8477 / 0.8383 → **0.8411 ± 0.0047**
    (macro-F1 0.841, ECE 0.033). Seed 0 v1→v2: 0.8312 → 0.8373, as predicted on dev.
  - `pilot_banking77_v2_k20` real_fewshot k=20: 0.8844 / 0.8805 / 0.8815 → **0.8821 ± 0.0017**
    (macro-F1 0.882, ECE 0.024).
  - Both ≈ 4 pt below Loukas et al. (88.0 / 91.2); 10→20-shot gain +4.1 pt (paper +3.2).
  - Caveat: k=20 hits the `max_steps: 2000` cap (≈ half an epoch). Kept deliberately = same recipe
    the classroom student gets when its data grows.
  - → Proposed go threshold: classroom ≥ **88.2%** (mean of 3 seeds). Milestone 2 done pending
    confirmation of this target.

## Milestone 3 — teacher via a subagent (experimental, 2026-09-30)
- Decision (human): while experimenting, the teacher is a subagent (Opus 5.5, the
  model of the operator's assistant session) instead of API calls. A real-API-key run follows later.
- Mechanism: `queue` teacher provider. `TeacherSession` still does budget check → cache → ledger;
  on a cache miss the request is written to `data/teacher_queue/<exp>/` and the pipeline waits.
  The operator batches requests (`scripts/teacher_queue.py`) and the tool-limited
  `homeroom-teacher` subagent (local agent definition, not committed; Read+Write only) writes batch replies.
  Local (uncommitted) permission settings deny Read on `data/raw/`, `results/`, `data/generated/` (rule 2).
- Arms now send each phase through `TeacherSession.ask_many`: batch-capable teachers get waves
  sized so the worst-case cost of the wave fits the remaining cap (hard cap unchanged); other
  teachers are called sequentially exactly as before (dry run reproduces earlier numbers).
- **Spend is an estimate**: tokens = chars/4, thinking tokens not counted, priced at Opus 5.5 API
  rates ($4 / $20 per MTok). Subagent results are labeled `*_subagent_*` and must NOT be mixed with
  or reported as API-billed results. Sampling temperature can't be set (Opus 5.5 rejects it).
- `temperature: null` in config now omits the parameter (needed for Opus 5.5 via API too).
- Cost-check result, `pilot_banking77_v2_subagent_costcheck`, seed 0, $1 cap (ESTIMATED spend):

  | Arm | Test acc | Macro-F1 | Spend | n_train | Teacher calls |
  |---|---|---|---|---|---|
  | real_fewshot | 0.8370 | 0.836 | $0 | 770 | 0 |
  | classroom | 0.8562 | 0.855 | $0.9638 | 1,499 | 77 exam ($0.358) + 73 lesson ($0.606) |
  | bulk | **0.8705** | 0.870 | $0.9213 | 3,331 | 130 bulk ($0.921) |

  - Accounting consistent: per-call estimates sum exactly to each arm's spend; caps respected.
    Bulk underspent its matched cap by $0.043 (≈ one worst-case call reserve, as predicted).
  - At $1, classroom never reached remediation (stopped by budget in lessons: 73/77). It spent 37%
    on the exam (no training data) and lessons yield 10 examples/call vs bulk 20 → 729 vs 2,561
    generated examples. So this run compares "exam + lessons" vs bulk, not the closed loop.
  - One seed, estimated cost, subagent teacher → not evidence for go/no-go.
  - Teacher label interpretation: for `get_physical_card` the teacher consistently followed the
    real seed examples (PIN-related) — label semantics come from the seeds, as intended.
- 2026-09-30 — Go threshold confirmed by the human: classroom ≥ **88.2%** (measured k=20 reference,
  as-is with the step cap), ≤ ~$10, and beats bulk at matched spend. CLAUDE.md updated.
- 2026-09-30 — Option 1 chosen by the human: `pilot_banking77_v2_subagent_usd3`, $3 per arm,
  `max_tokens` 1024 (smaller worst-case reserve → bigger waves, smaller bulk underspend).
  Its teacher cache was seeded with the 280 replies of the $1 cost check: same teacher, prompts
  and salts (cache keys identical), largest reply ≈ 421 tokens so the lower `max_tokens` would not
  have changed any of them. Reused replies are charged at their original estimated cost.
- $3 result, `pilot_banking77_v2_subagent_usd3`, seed 0 (ESTIMATED spend):

  | Arm | Test acc | Macro-F1 | ECE | Spend | n_train | Teacher calls |
  |---|---|---|---|---|---|---|
  | real_fewshot | 0.8373 | 0.836 | 0.037 | $0 | 770 | 0 |
  | classroom | 0.8695 | 0.869 | **0.030** | $1.1134 | 1,539 | 77 exam $0.358 + 77 lesson $0.641 + 15 remedial $0.115 |
  | bulk | 0.8695 | 0.869 | 0.045 | $1.0990 | 3,825 | 155 bulk $1.099 |

  - Classroom used $1.11 of $3: one remedial round (15 confusion pairs) did not raise score-half
    exam accuracy by ≥ 0.2 pt → `no_improvement`, round-0 student kept (n_train 1,539 = round 0).
    So remediation was tried once and rejected by the stopping rule; its data never reached the
    final student.
  - Tie on accuracy and macro-F1 (to 4 decimals; ECE differs, so different models). Classroom
    matched bulk with 40% of the training examples and better calibration.
  - Neither reaches 88.2. One seed → no conclusion. Bulk underspend now $0.014 (max_tokens 1024).
  - Stopping signal is coarse: score half = 3 items/intent = 231 items, one item = 0.43 pt, larger
    than the 0.2 pt min gain → stopping is dominated by noise (see open item on exam size).
- 2026-09-30 — Stopping-signal fix chosen by the human (options a + c):
  `pilot_banking77_v2_subagent_usd3_exam10_min2` = $3 run + `exam_items_per_intent: 10`
  (5 score items/intent → 385 items, one item = 0.26 pt) + new `classroom.min_rounds: 2` (at least
  two remedial rounds run before "no improvement" can stop the loop; the best-scoring round's
  student is kept). `min_rounds` defaults to 0, so earlier configs behave as before. Cache seeded
  from the $3 run (lesson and bulk prompts identical; exam prompts change with n, so exams are new).
- exam10_min2 result, seed 0 (ESTIMATED spend):

  | Arm | Test acc | Macro-F1 | ECE | Spend | n_train |
  |---|---|---|---|---|---|
  | real_fewshot | 0.8380 | 0.837 | 0.036 | $0 | 770 |
  | classroom | **0.8721** | 0.871 | 0.043 | $1.4837 | 2,148 |
  | bulk | 0.8711 | 0.871 | 0.048 | $1.4659 | 4,820 |

  - The loop ran 4 remedial rounds (15 / 12 / 13 / 15 confusion pairs). It continued past the
    2 forced rounds, so later rounds improved the score half; the kept student (n_train 2,148)
    includes remedial data — the first run where remediation reached the final model.
  - Classroom +0.10 pt over bulk with 45% of the training examples: within single-seed noise.
    Neither reaches 88.2. Bulk underspend $0.018.
  - Caveat: the exam/seed mismatch for `get_physical_card` (open item below) fed into remedial
    material in this run (the teacher taught "how do I get a physical card" → get_physical_card).
- examseeds result, seed 0 (ESTIMATED spend):

  | Arm | Test acc | Macro-F1 | ECE | Spend | n_train |
  |---|---|---|---|---|---|
  | real_fewshot | 0.8373 | 0.836 | 0.037 | $0 | 770 |
  | classroom | 0.8636 | 0.862 | 0.042 | $1.4735 | 2,029 |
  | bulk | **0.8705** | 0.870 | 0.050 | $1.4659 | 4,820 |

  - The fix worked as intended (all 10 `get_physical_card` exam items are PIN questions) and the
    loop ran 4 remedial rounds (13 / 9 / 10 / 13 pairs), but classroom test accuracy fell
    0.8721 → 0.8636 vs the previous run, and bulk (identical cached data, 0.8711 → 0.8705) wins.
  - Candidate explanations (not separable with one seed): seed noise of the classroom pipeline;
    an exam closer to the seeds is easier and gives a weaker diagnosis/stopping signal; the old
    mismatched exam accidentally widened boundaries. Bulk run-to-run noise on identical data ≈ 0.1 pt.
  - Seed-0 summary across all classroom variants: 0.856 / 0.870 / 0.872 / 0.864 vs bulk
    0.871 / 0.870 / 0.871 / 0.871. No variant clearly beats bulk; seeds 1–2 are needed.
- Open (addressed by the $3 run): to test the actual loop, the budget must cover exam + lessons + ≥1 remedial round (≈ $1.2+
  at Opus 5.5 rates) — or make the exam cheaper (fewer items / smaller model) or lessons richer.

## Open items
- [ ] Choose teacher provider/model; confirm its terms allow training a (non-competing) classifier on
      outputs and whether generated data may be published. Record the decision here.
- [ ] Jev terms of use regarding training other models on its outputs (Phase 2).
- [ ] Should the exam be written by a *different* model than the teacher? (ablation candidate)
- [ ] Is 6 exam items/intent enough signal for stopping? (check variance of score-half accuracy)
- [ ] Exam dedup is exact match after lowercasing/stripping punctuation; near-paraphrases of exam
      items can still enter training. Add fuzzy/embedding dedup before the pilot?
- [ ] `bulk` underspends its matched cap by up to one worst-case call (pre-check reserves
      `max_tokens` output): dry run 0.227 vs 0.237 (−4%). Allow a final call sized to the remaining
      budget, or report the gap as-is?
- [x] 2026-10-01 — Resolved (human-approved): `exam_prompt` now includes the k real seed examples
      of the intent (from the per-seed k-shot sample, not hardcoded). New experiment
      `pilot_banking77_v2_subagent_examseeds` (= exam10_min2 + this prompt change); cache seeded
      from exam10_min2 (lesson/bulk prompts unchanged, exam re-asked). Trade-off: the exam is less
      independent of the training seeds (may lean on their phrasing) but uses the dataset's label
      meaning.
- [ ] (original note) **Exam prompt has no seed examples** (found in the exam10_min2 run, 2026-09-30): the exam
      writer sees only the intent *name*, so for `get_physical_card` it wrote "how do I get a
      physical card" items, while the real seeds (and therefore lessons/bulk/student) are about the
      card PIN. The student is then marked wrong on mislabeled exam items, which pollutes diagnosis
      and the stopping score. Fix candidate: include the k real seeds in `exam_prompt` (seeds are
      training data, so the test set stays untouched). Prompt change → new experiment name.
- [ ] "Gap > 1 std" go criterion: population std (np.std, current) or sample std (ddof=1)?
