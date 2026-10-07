# Development workflow

## Validate the core

Install the development dependencies, run `pytest -q`, then run:

```bash
homeroom run --config configs/toy_smoke.yaml --dry-run --seeds 0
```

This checks orchestration without an API key or GPU. Keep mock results separate
from model-quality measurements.

## Reproduce a baseline

Install the GPU extra and the appropriate PyTorch build. Run `homeroom check-gpu`
before training. Start from a committed configuration and record the environment.

Use a holdout from unused training data for debugging. Do not tune on the
official test set. Measured project baselines are recorded in
[the decision log](decisions.md).

## Check teacher accounting

Configure the teacher and token prices. Start with a small explicit budget.
Compare ledger totals with provider usage before extending the experiment.
Keep estimated costs distinct from API-billed costs.

## Run a comparison

Give each experiment a distinct name. Run classroom before a bulk arm that
matches its actual spend, then summarize the configured seeds.

Report accuracy, macro-F1, calibration, examples, actual spending, and the
spread across seeds. Record the conclusion and remaining limitations in the
decision log.

See [DEVELOPMENT.md](../DEVELOPMENT.md) for the full experiment constraints.
