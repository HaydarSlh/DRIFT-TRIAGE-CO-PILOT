# Prompt Changelog

Every prompt change goes in this file. Bump the `VERSION:` comment at the top
of the affected prompt module **in the same PR**, and re-record any trajectory
snapshots that drift (`uv run pytest tests/snapshot/ --snapshot-update`).

Format:

```
## [<prompt name>] <new version> — <date>

- What changed (one sentence).
- Why it changed (one sentence — link a trace, an issue, or a regression test).
- Trajectory impact: which snapshots needed updating, or "none".
```

Roll back a prompt change the same way you'd revert any other code change:
identify the commit, `git revert <sha>`, and let CI re-validate the snapshots.

---

## [supervisor] 0.2.0 — 2026-05-07

- Forked from CP4 research-report; rewritten for Week 5 drift triage.
- Node names changed from researcher/critic/writer to triage/action/comms.
- Trajectory impact: all snapshots need re-recording.

## [triage] 0.2.0 — 2026-05-07

- Forked from CP4 researcher; rewritten for drift triage. Gathers context about
  drift events and classifies severity.
- Trajectory impact: replaces researcher snapshots entirely.

## [action] 0.2.0 — 2026-05-07

- Forked from CP4 critic; rewritten for remediation action decisions. Proposes
  action_type ∈ {none, replay_test, retrain_shadow, rollback}.
- Trajectory impact: replaces critic snapshots entirely.

## [comms] 0.2.0 — 2026-05-07

- Forked from CP4 writer; rewritten for notification drafting. Pauses for HIL
  approval before Production-touching actions.
- Trajectory impact: replaces writer snapshots entirely.