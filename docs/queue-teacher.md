# Isolated queue teacher workflow

The queue backend writes budgeted requests and waits for external answers. It
does not call a paid model API. The Codex Luna study uses isolated GPT-6 Luna
agents with low reasoning; subscription use and GPU time are not included in
the API-equivalent cost estimates.

## Prepare and run

Install the project with its GPU and report extras as described in
[DEVELOPMENT.md](../DEVELOPMENT.md). Activate the environment before these commands:

```powershell
homeroom run --config configs/pilot_banking77_codex_luna.yaml
```

Optional preparation in a separate terminal:

```powershell
python scripts/prepare_teacher_queue.py --config configs/pilot_banking77_codex_luna.yaml
```

Preparation reads only cached Banking77 training data. It uses the same sampled
seeds, prompts, cache salts, teacher settings, and budget checks as the main
pipeline. It prepares fixed exams and lessons. Remedial requests come from the
main pipeline's diagnostic exam errors. Cache hits in the main run are charged
at their original accounted cost.

## Answer batches

```powershell
python scripts/teacher_queue.py batch --exp pilot_banking77_codex_luna --size 5
python scripts/teacher_queue.py status --exp pilot_banking77_codex_luna
```

Give a fresh teacher agent only its assigned batch file. It must follow the
batch's system role and each prompt's schema and requested number of examples.
Do not give it result files, official test data, the score half of the exam,
other batches, or this repository's research discussion. Generate actual model
answers; do not substitute mock data or text assembled by code templates.

Encode the UTF-8 JSON mapping `{request_id: JSON_reply_string}` as base64 and
save it with the validator:

```powershell
python scripts/write_teacher_replies.py --batch data/teacher_queue/pilot_banking77_codex_luna/batches/b0001.json --reply-b64 BASE64
```

The validator checks the batch path, exact request IDs, field types, and the
configured output-size estimate. It saves atomically and refuses to replace
different existing answers. It does not assess semantic quality or example
counts; the teacher must satisfy the prompt. The output-size bound is currently
8,192 characters, matching this study's 2,048-token character estimate.

## Quota interruptions and reporting

If generation hits quota, preserve all replies, caches, ledgers, and result
files. Wait for quota to recover before generating more answers. Queue waits
have a six-hour timeout in this config. After a timeout, rerun the same command
to reuse completed results and cached replies. An unfinished arm may need
retraining. Ledgers are append-only and can contain abandoned attempts; final
result JSON records the completed attempt's accounting.

After all configured arms and seeds finish:

```powershell
homeroom report --config configs/pilot_banking77_codex_luna.yaml
```

Keep teacher text under gitignored `data/`. Publish aggregate result JSON,
environment provenance, reports, and charts. Preserve the historical Claude
study separately. Compare only complete, consistently configured seed sets.
