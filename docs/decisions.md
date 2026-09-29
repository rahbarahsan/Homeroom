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
- [ ] "Gap > 1 std" go criterion: population std (np.std, current) or sample std (ddof=1)?
