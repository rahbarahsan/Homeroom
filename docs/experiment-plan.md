# Experiment plan

## Phase 1 — Pilot (go / no-go), Banking77, ~1–2 weeks, ~$10–20

**Question:** at equal teacher spend, does the classroom loop beat (a) real few-shot and
(b) bulk synthetic data?

| Arm | Real data | Teacher use | Spend |
|---|---|---|---|
| `real_fewshot` | k=10/intent | none | $0 |
| `bulk` | k=10/intent | uniform generation per intent, no feedback | = classroom's actual spend |
| `classroom` | k=10/intent | exam → lessons → train → diagnose confusions → remediate → retrain | cap `usd_per_arm` |
| `full_data` (oracle) | 10,003 | none | $0 |

Student: SetFit + all-mpnet-base-v2 (same family as published baselines). Seeds: 0, 1, 2.

**Protocol**
1. Dry run (`--dry-run`): pipeline works, no cost.
2. Baseline check: `real_fewshot` k=10 ≈ 88% ± 2. If not, fix student/eval first.
3. Cost check: seed 0, `usd_per_arm: 1.0`; compare ledger with provider dashboard.
4. Full pilot at `usd_per_arm` ∈ {2, 5, 10} (use separate experiment names), 3 seeds.
5. `homeroom summarize`; plot accuracy vs spend per arm.

**Metrics:** test accuracy, macro-F1, ECE, NLL; spend (accounted and real); tokens; n_train;
Δaccuracy over `real_fewshot` per dollar; classroom history (score-exam accuracy per round).

**Go:** classroom ≥91% at ≤$10 AND > bulk at matched spend (mean over 3 seeds, gap > 1 std).
**No-go:** write a short negative-results post; stop or pivot (e.g., generator-program teaching).

**Threats to validity:** teacher exam distribution ≠ real test distribution (that's the point of
the real test, but stopping may be miscalibrated); teacher label noise on overlapping intents;
prompt quality differences between arms (bulk and lesson prompts must stay comparable).

## Phase 2 — Only if Phase 1 is "go"

1. **Jev as teaching assistant:** Jev grades exam items / filters generated examples (keep only
   items Jev assigns to the intended label with high confidence); frontier teacher only writes
   materials and handles uncertain cases. Compare frontier-only vs Jev-assisted at equal spend.
   Also report Jev used directly (no student) as a cost/accuracy reference.
2. **Students:** ModernBERT-base / ModernBERT-large (LoRA) / Laya (vs its ModernBERT-large base) /
   Qwen-1.5B QLoRA with a classification head. Measure latency on CPU and GPU.
3. **Second domain:** LEDGAR (100 classes, longer texts — check 512-token limit).
4. **k sweep:** k ∈ {3, 5, 10, 20}.

## Phase 3 — Write-up and framework

- arXiv tech report / blog / workshop paper with accuracy-vs-dollar curves and ablations
  (no diagnosis, no lessons, no remedial, exam written by a different model).
- Framework polish: `homeroom init` for a user CSV with a few labeled rows; model card export.
- Optional ablations: teacher-written generator programs (CBIT-style); STP/LLM-JEPA student loss.
