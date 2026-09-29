# Kickoff prompts for Claude Code

Paste these one at a time into Claude Code at the repo root.

**1. Orientation**
> Read CLAUDE.md and everything in docs/. Summarize the goal, the current milestone, and the hard
> rules in 10 bullet points. Then list anything in the code that contradicts the docs.

**2. Milestone 1 — dry run**
> Set up the environment for my machine (Windows / RTX 2060 6GB; tell me whether to use WSL2).
> Run `pytest -q` and `homeroom run --config configs/pilot_banking77.yaml --dry-run --seeds 0`.
> Fix anything that fails without changing the experiment design.

**3. Milestone 2 — baseline reproduction**
> Install torch (correct CUDA wheel) and the [gpu] extra. Run `homeroom check-gpu`, then
> `homeroom run --config configs/pilot_banking77.yaml --arms real_fewshot --seeds 0`.
> Target ≈ 88% accuracy. If far off, investigate SetFit settings (do not look at test data to tune;
> use a held-out split of the *unused* train data for debugging only, and document it).

**4. Milestone 3 — cost check**
> I filled .env with my teacher. Make a copy of the pilot config named pilot_banking77_costcheck
> with usd_per_arm 1.0 and seeds [0]; run classroom then bulk. Show the ledger totals by tag and
> explain the spend. Stop and ask me before spending more.

**5. Milestone 4 — pilot**
> Run the full pilot (3 seeds). Then summarize, plot accuracy vs spend per arm into
> results/<exp>/accuracy_vs_spend.png, and draft the go/no-go note in docs/decisions.md.
