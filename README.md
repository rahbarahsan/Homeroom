# Homeroom

**Teach small models with a fixed budget.**

Homeroom compares adaptive teaching with ordinary few-shot training and bulk
synthetic data. A teacher model builds an exam, diagnoses a student's errors,
and generates targeted training examples. Every teacher request passes through
a budget check, cache, and cost ledger.

The question is practical: **at the same teacher spend, does feedback produce a
better specialist?**

## Current results

The completed Banking77 comparison uses **GPT-6 Luna**, SetFit + MPNet, ten
real examples per intent and seeds 0, 1 and 2:

| Arm | Test accuracy, mean ± standard deviation | Estimated teacher cost (USD), mean | Retained examples, mean |
|---|---:|---:|---:|
| Real few-shot | 84.12% ± 0.48 percentage points | 0 | 770 |
| Adaptive classroom | 85.17% ± 0.54 percentage points | 0.0413 | 2,014 |
| Bulk generation | 85.53% ± 0.55 percentage points | 0.0404 | 4,640 |

**Classroom tied bulk on one seed and lost on two.** The paired accuracy
difference was −0.36 ± 0.27 percentage points. This recipe did not reach the
88.2% target or beat bulk. Classroom retained about 57% fewer training
examples; its additional fits also took longer.

![Banking77 accuracy versus estimated teacher cost](results/pilot_banking77_codex_luna/accuracy_vs_spend.png)

Standard deviations describe variation across three seeds. Bulk's cap equals
classroom's realized estimated spend for each seed; it underspent by about
2.3–2.5%. Costs use characters / 4 and configured token rates, excluding
thinking, subscription usage, training and hardware. **These are estimates,
not provider bills.**

Read [the study and limitations](docs/banking77-codex-luna-study.md) or
[the complete report](results/pilot_banking77_codex_luna/report.md).
Rebuild the report from committed results:

~~~bash
python -m pip install -e ".[report]"
homeroom report --config configs/pilot_banking77_codex_luna.yaml
~~~

Earlier real-data references measured
[84.11% ± 0.47 at 10 shots](results/pilot_banking77_v2/summary.md) and
[88.21% ± 0.17 at 20 shots](results/pilot_banking77_v2_k20/summary.md).
Historical teacher studies remain separate; see
[the experiment history](docs/decisions.md).

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

- Compare lessons only and remediation using a human-labeled training-only
  holdout for model selection; record the additional validation labels.
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
- [Banking77 study](docs/banking77-codex-luna-study.md): completed comparison and negative finding.
- [Related work](docs/related-work.md): research context and attribution.

The teaching loop builds on existing research. Homeroom's focus is reproducible
quality-versus-cost comparisons and a usable experimental workflow.

## License

Apache-2.0. See [LICENSE](LICENSE). Generated training datasets are excluded
from the public repository pending review of the teacher provider's terms.
