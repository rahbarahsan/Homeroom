# Unattended Banking77 study

The study-specific supervisor runs the fixed Codex Luna comparison without
requiring an interactive terminal. It uses the existing ChatGPT login. It
refuses API-key authentication, removes API credential variables from child
environments, and pins the queue provider, model, prices, caps, seeds, and the
scientific settings of completed baseline results.

## Commands

Install the existing GPU setup and `.[report,unattended]` extras. In the repository:

```powershell
python scripts/unattended_study.py run
python scripts/unattended_study.py status
python scripts/unattended_study.py pause
python scripts/unattended_study.py resume
```

An optional `--thread-id UUID` queues updates to an existing Codex task using
workspace-scoped automatic approval review. A persistent
[Codex Goal](https://learn.chatgpt.com/use-cases/follow-goals) tracks the final
report, documentation, and authorized GitHub update. Teacher sessions use
[non-interactive Codex](https://learn.chatgpt.com/docs/non-interactive-mode)
with a read-only sandbox, no interactive approvals, and structured output.

## Runtime and isolation

- One experiment worker owns GPU training. At most three fresh Luna sessions
  answer budgeted five-request batches.
- Teachers receive only their assigned system role and prompts. They receive
  no results, official test examples, or score-half exam content. Outputs
  involving tool calls are rejected. Answers must pass schema, count, ID,
  and size validation before atomic saving.
- Existing replies are retained. The supervisor creates no mock answers and
  does not substitute another teacher model.
- Cached datasets and model weights are used offline. Teacher text, raw Codex
  events, model checkpoints, and supervisor state stay under gitignored data/.

## Recovery and quota

State is saved atomically to
`data/generated/pilot_banking77_codex_luna/unattended/state.json`.
An append-only progress log, separate process logs, attempt numbers, process
IDs plus creation times, and original model outputs accompany it.

Completed student fits are saved with input/configuration fingerprints and
file hashes. Restarting an unfinished arm replays its original algorithm and
accounting, restoring matching completed fits. Damaged artifacts are preserved
under an incomplete name and rebuilt. A crash during an unfinished fit may
require repeating that fit; completed fits and arm results survive.

The supervisor holds an operating-system lock. On restart it observes any
already-running owned workers instead of creating a duplicate GPU job.
Process shutdown checks creation times to avoid targeting reused process IDs.

On a usage-quota error, generation stops and the experiment worker is stopped
to release memory. Saved results, replies, and completed fits remain intact.
The supervisor waits until a reported reset time plus a safety margin. If no
reset time is available, retries back off from one hour to a maximum of six
hours. It never purchases credits, performs a manual reset, changes billing,
or switches to paid API calls. Other failures retry with a slower progression
and notify the tracked Codex task after repeated failures.

A current-user Windows Task Scheduler entry named
`Homeroom-CodexLuna-Unattended` restarts the supervisor at logon and every
15 minutes if it exits. It runs with limited privileges and no stored password.
The lock prevents duplicate supervisors. A pause marker prevents the restart
task from overriding an explicit user pause.

The process temporarily prevents automatic system sleep for up to twelve
hours; it does not change system power settings or keep the display on.
Recovery requires the machine to be running and the user session available.
After twelve hours, the normal sleep policy applies and work resumes when
the machine is available again. Remove the restart task after the study and
publication work are complete.

## Validation before the long run

- 40 tests passed, including atomic writes, model checkpoint replay, corrupt
  checkpoint preservation, quota scheduling, credential-variable filtering,
  reused PID protection, explicit pause, and avoiding duplicate GPU workers.
- A live five-request Luna lesson batch passed the unattended route. Its
  event stream contained only an agent message, with no tool calls.
- A cached SetFit model with a head fitted to four toy sentences was saved
  and restored. Predicted probabilities were identical.
- A detached probe survived the launching shell's exit and correctly stopped
  its owned dummy process tree.

The historical Claude study remains separate. This study keeps the same Luna
model and low reasoning across both the initial isolated subagents and the
later isolated CLI sessions. The transport change is recorded in environment
provenance. Costs remain API-equivalent character estimates, excluding
thinking, subscription usage, and GPU time.
