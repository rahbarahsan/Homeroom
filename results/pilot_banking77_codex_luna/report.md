# Experiment report: pilot_banking77_codex_luna

Dataset: **banking77** · Student: **setfit** · Teacher: **gpt-6-luna**
Seeds: 0, 1, 2. All configured runs are present.

**Cost basis: Estimated API-equivalent teacher cost.**
Queue usage uses characters / 4 and excludes thinking tokens. These amounts
are API-equivalent estimates, not provider bills or measured subscription costs.
Student training, hardware, and subscription costs are excluded.

## Results

| Arm | Accuracy (%) | Macro-F1 | ECE | Teacher cost (USD) | Training examples |
|---|---:|---:|---:|---:|---:|
| real_fewshot | 84.12 ± 0.48 | 0.8409 ± 0.0052 | 0.0315 ± 0.0046 | 0.0000 ± 0.0000 | 770.0 ± 0.0 |
| classroom | 85.17 ± 0.54 | 0.8509 ± 0.0057 | 0.0489 ± 0.0048 | 0.0413 ± 0.0040 | 2013.7 ± 331.9 |
| bulk | 85.53 ± 0.55 | 0.8539 ± 0.0061 | 0.0536 ± 0.0019 | 0.0404 ± 0.0040 | 4640.0 ± 346.1 |

Values are mean ± population standard deviation across seeds (ddof=0).
Error bars describe seed variation; they are not confidence intervals.

![Accuracy versus teacher cost](accuracy_vs_spend.png)

Faint points are individual seeds; large markers show means with one standard
deviation in each axis. This is an arm comparison, not a budget sweep.

## Paired classroom versus bulk comparison

Classroom minus bulk accuracy: **-0.36 ± 0.27 percentage points**.
Classroom wins on 0 of 3 seeds.

| Seed | Classroom accuracy (%) | Bulk accuracy (%) | Difference (pp) | Bulk underspend (USD) |
|---|---:|---:|---:|---:|
| 0 | 85.06 | 85.06 | +0.00 | 0.0009 |
| 1 | 85.88 | 86.30 | -0.42 | 0.0011 |
| 2 | 84.58 | 85.23 | -0.65 | 0.0010 |

Bulk receives each seed's classroom spend as its cap. Its actual spend may
be lower because a remaining request cannot fit under the cap. Report that gap
alongside accuracy. Three seeds alone do not establish statistical significance.

## Provenance

| Result | Git commit | Dirty checkout |
|---|---|---|
| [real_fewshot_seed0.json](real_fewshot_seed0.json) | 9387398 | true |
| [real_fewshot_seed1.json](real_fewshot_seed1.json) | 9387398 | true |
| [real_fewshot_seed2.json](real_fewshot_seed2.json) | 0e77533 | true |
| [classroom_seed0.json](classroom_seed0.json) | 9abd515 | true |
| [classroom_seed1.json](classroom_seed1.json) | 9abd515 | true |
| [classroom_seed2.json](classroom_seed2.json) | 9abd515 | true |
| [bulk_seed0.json](bulk_seed0.json) | 9abd515 | true |
| [bulk_seed1.json](bulk_seed1.json) | 9abd515 | true |
| [bulk_seed2.json](bulk_seed2.json) | 9abd515 | true |

Scientific settings are validated against the selected config before reporting.
Dirty historical checkouts cannot be reconstructed from a commit alone.
Source file SHA-256 hashes and unrounded aggregates are recorded in [report.json](report.json).
Teacher text and official test examples are excluded from this report.
