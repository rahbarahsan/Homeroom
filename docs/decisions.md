# Decision log

Format: date — decision — why. Add open questions at the bottom; resolve them explicitly.

- 2026-09 — Project name `homeroom`; Apache-2.0; public repo from day 1 (code/configs/results only).
- 2026-09 — Phase 1 domain Banking77; second domain LEDGAR later. Avoid medical/moderation.
- 2026-09 — Primary student SetFit + all-mpnet-base-v2 to reproduce published baselines (10-shot ≈ 88%).
- 2026-09 — Bulk arm spend is matched to classroom's actual spend per seed (fair comparison).
- 2026-09 — Classroom exam is teacher-written, split diagnostic/score; official test used only at the end.
- 2026-09 — Jev, Laya, other students, JEPA-style losses deferred to Phase 2/3.

## Open items
- [ ] Choose teacher provider/model; confirm its terms allow training a (non-competing) classifier on
      outputs and whether generated data may be published. Record the decision here.
- [ ] Jev terms of use regarding training other models on its outputs (Phase 2).
- [ ] Should the exam be written by a *different* model than the teacher? (ablation candidate)
- [ ] Is 6 exam items/intent enough signal for stopping? (check variance of score-half accuracy)
