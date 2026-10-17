# Paused run — Unseen letters erase outcome credit

> 2026-10-11T02:50:54.159Z · paper `Paper/main.tex` · Lean project `lean`
>
> Paused immediately, with 2 agent(s) mid-work.

## Done

| Milestone | Finished | Took |
| --- | --- | --- |
| Read the theorem statement | 22:29:28 | — |
| Find needed assumptions | 22:34:11 | — |
| Decide paper issues | 22:36:56 | — |
| Write the Lean model | 22:49:27 | — |
| Write the analysis interface | 22:49:30 | — |
| Plan the proof | 23:02:11 | — |
| scaffold | 23:13:24 | — |
| round:1 | 23:47:06 | — |
| round:2 | 02:50:52 | — |

## Lean files

- [ ] `Trace/Analysis/ZeroOrderCredit.lean`

## Interrupted by the cancel

2 agent(s) were stopped mid-work at 02:50:52. What each had written by then is below. None of it was reverted — a cancel stops the agents, it does not touch your files — but a file here may hold half of an edit, so it is not evidence of anything until it compiles.

- **Prove Trace.Proof.zeroOrderCredit** (`Trace/Analysis/ZeroOrderCredit.lean`) — `Trace/Analysis/ParamNorm.lean`, `Trace/Analysis/ZeroOrderPattern.lean`, `Trace/Analysis/ZeroOrderTrack.lean`, `Trace/Analysis/ZeroOrderCredit.lean`
- **Prove Trace.Proof.unseenLettersScoreCredit** (`Trace/Analysis/ScoreCredit.lean`) — nothing written yet

Picking this up reopens the goals in those files rather than repairing them, and falls back to repair if reopening does not get the project building.

## Picking it up

Next would be **round:3**.

In the editor: the side bar offers it when a paused run is found, or run the command **Tex2Lean: Resume a paused run**.

From a script, by opening the editor's URI for this run:

```sh
# macOS
open 'vscode://kuldeepmeel.tex2lean4/resume?slug=thm-main-14o4fct'
# Linux
xdg-open 'vscode://kuldeepmeel.tex2lean4/resume?slug=thm-main-14o4fct'
```

Or read `.blueprint/run/thm-main-14o4fct/checkpoint.json` directly — it is schema `tex2lean.checkpoint/1`, and `completed`, `next` and `briefs` are the fields that say what is left. A reader that does not recognise the schema string should stop rather than guess.
