# Homeroom

**Teach small models with a fixed budget.**

Homeroom compares adaptive teaching with ordinary few-shot training and bulk
synthetic data. A teacher model builds an exam, diagnoses a student's errors,
and generates targeted training examples. Every teacher request passes through
a budget check, cache, and cost ledger.

The question is practical: **at the same teacher spend, does feedback produce a
better specialist?**

## Current results

The Banking77 pilot is in progress. Committed SetFit + MPNet baselines use
randomly sampled examples and three seeds:

| Training data | Test accuracy, mean ± standard deviation | Examples |
|---|---:|---:|
| 10 real examples per intent | 84.11% ± 0.47 percentage points | 770 |
| 20 real examples per intent | 88.21% ± 0.17 percentage points | 1,540 |

Sources: [10-shot results](results/pilot_banking77_v2/summary.md) and
[20-shot results](results/pilot_banking77_v2_k20/summary.md).

These are measured project baselines, not the published reference scores.
Exploratory teacher runs use one seed and estimated teacher costs; they do not
yet establish a consistent advantage over bulk generation. See the
[experiment history](docs/decisions.md) for configurations, results, and caveats.

## How it works

1. Sample a small set of real examples per intent.
2. Generate an exam and split it into diagnostic and scoring subsets.
3. Generate lessons and train the student.
4. Diagnose confusions using the diagnostic subset.
5. Generate targeted examples, retrain, and retain the best scoring round.
6. Stop at the budget, round, or improvement limit.
7. Evaluate the final model on the held-out human-labeled test set.

Real examples remain in the training set. Generated training text is
deduplicated against exam items, and the official test set is excluded from
teaching and stopping decisions.

## Quick start

Python 3.10 or newer. The mock-teacher path needs no API key or GPU.

```bash
git clone https://github.com/rahbarahsan/Homeroom.git
cd Homeroom
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
homeroom run --config configs/toy_smoke.yaml --dry-run --seeds 0
```

The dry run exercises the pipeline with a mock teacher and a TF-IDF student.
Its outputs are smoke-test artifacts, not model-quality or spending evidence.

### Reproducible example and report

Run a complete offline example with three seeds:

~~~bash
python -m pip install -e ".[report]"
homeroom demo
~~~

Open [the reporting guide](docs/reporting.md) for an example chart and the
output files. The demo writes a report and PNG/SVG charts to
results/toy_demo_dryrun/. It uses a mock teacher and toy data; its numbers
demonstrate the workflow.

To regenerate a report from saved experiment results:

~~~bash
homeroom report --config configs/YOUR_EXPERIMENT.yaml
~~~

The report requires all configured seeds, compares classroom and bulk by seed,
and labels simulated, estimated, and API token costs separately.

For GPU students, install the PyTorch build appropriate for your hardware,
then:

```bash
python -m pip install -e ".[gpu]"
homeroom check-gpu
homeroom run --config configs/pilot_banking77.yaml --arms real_fewshot --seeds 0
```

To use a real teacher, copy `.env.example` to `.env` and configure the provider,
model, API key, and token prices. Unset prices are rejected.

```bash
homeroom run --config configs/pilot_banking77.yaml
homeroom summarize --config configs/pilot_banking77.yaml
```

## Comparisons

| Arm | Training strategy |
|---|---|
| `real_fewshot` | Real seed examples only |
| `bulk` | Uniform synthetic examples, matched to classroom spend |
| `classroom` | Lessons, diagnosis, and targeted remediation |
| `full_data` | All available real training examples |

Reports include accuracy, macro-F1, calibration error, training-set size,
teacher tokens, spending, and the resolved experiment configuration. Estimated
costs and API-billed costs must be kept separate.

## Next experiments

- Complete the teacher comparison across three seeds.
- Evaluate Laya as an additional decision-model candidate, with an appropriate
  ModernBERT baseline and explicit hardware measurements. This integration is
  planned; current results do not include Laya.
- Measure exam quality and stopping-signal variance before expanding domains.

[Laya](https://github.com/NandhaKishorM/laya) produces typed decisions rather
than free-form lessons, so its intended role is a student or assessor.

## Development

- [Development guide](DEVELOPMENT.md): setup, commands, and evaluation rules.
- [Experiment plan](docs/experiment-plan.md): pilot design and later experiments.
- [Decision log](docs/decisions.md): changes, measured findings, and open questions.
- [Reporting guide](docs/reporting.md): reproducible demo, charts, and cost interpretation.
- [Related work](docs/related-work.md): research context and attribution.

The teaching loop builds on existing research. Homeroom's focus is reproducible
quality-versus-cost comparisons and a usable experimental workflow.

## License

Apache-2.0. See [LICENSE](LICENSE). Generated training datasets are excluded
from the public repository pending review of the teacher provider's terms.
