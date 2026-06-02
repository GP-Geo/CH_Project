# AGENTS.md — instructions for Codex

This repository is in an active **package-first refactor**. Before doing any
work, read these four files in order:

1. `AGENT_RULES.md` — binding rules (do-not-touch paths, git rules, behavior
   preservation).
2. `AGENT_STATE.md` — current branch, canonical ownership, shims, next task.
3. `AGENT_BACKLOG.md` — the slice you should pick up.
4. `AGENT_RUN_LOG.md` — what was done last.

## Workflow

- **One bounded slice at a time.** Pick the next slice from `AGENT_BACKLOG.md`.
- **Preserve scientific behavior exactly** — no change to thresholds, feature
  order, prediction schema, class counts, model artifact paths, or numeric
  output.
- **Merge-and-consolidate, not blind replacement.** Move the better
  implementation to the canonical location; reduce the old path to a shim so old
  imports keep working.
- **Do not touch forbidden paths:** `data/`, root `/models/`, `notebooks/`
  (unless explicitly requested), raw DEMs/shapefiles/GeoPackages, generated
  outputs, trained model artifacts.
- **Do not `git push`. Do not merge branches. Do not merge into `main`.**
- **Do not delete files** unless a task explicitly allows it.
- **Run tests:** targeted first
  (`conda run -n ch-heads python -m pytest <files>`), then full
  (`conda run -n ch-heads python -m pytest`). Commit only if green.
- **After a completed slice:** update `AGENT_STATE.md` and append to
  `AGENT_RUN_LOG.md`.
- **Stop if uncertain about scientific behavior** and ask.
