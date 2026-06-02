# AGENT_RULES.md — strict rules for refactor agents

These rules are binding for any agent (Claude Code, Codex, or otherwise)
continuing the package-first refactor of this project. Read this file, plus
`AGENT_STATE.md`, `AGENT_BACKLOG.md`, and `AGENT_RUN_LOG.md`, **before** making
any change.

## Workflow rules

1. **Work one bounded slice at a time.** Pick a single slice from
   `AGENT_BACKLOG.md`. Do not bundle unrelated changes.
2. **Preserve scientific behavior exactly.** No change to thresholds, feature
   order, prediction schema, model artifact paths, class counts, or numeric
   results. This is a structural refactor, not a science change.
3. **Merge-and-consolidate, never blind-replace.** When two modules overlap,
   keep the better real implementation, move it to the canonical location, and
   reduce the other to a shim. Do not delete behavior to "clean up".
4. **Keep compatibility shims.** Old import paths (notebooks, scripts, user
   code) must keep working. A shim re-exports from the new canonical module.
5. **Run focused tests, then full pytest when practical.** Use
   `conda run -n ch-heads python -m pytest <targeted files>` while iterating,
   then `conda run -n ch-heads python -m pytest` before committing.
6. **Commit only if tests pass.** If tests fail, fix or revert — do not commit
   red.
7. **Update `AGENT_STATE.md` and `AGENT_RUN_LOG.md` after every completed
   slice.** State reflects the new reality; the run log appends one entry.
8. **Stop if uncertain about scientific behavior.** When a change might alter
   numeric output or model semantics, stop and ask rather than guess.

## Do-not-touch rules

- **Do not touch `data/`** (raw or generated).
- **Do not touch root `/models/`** — trained model artifacts are preserved
  as-is and are git-ignored.
- **Do not touch `notebooks/`** unless the task explicitly requests it.
- **Do not touch** raw DEMs, shapefiles, GeoPackages, generated outputs, or any
  trained model artifact.
- **Do not delete files** unless a task explicitly allows it. Reduce to a shim
  or mark as archive instead.

## Git rules

- **Do not `git push`.**
- **Do not merge into `main`.**
- **Do not merge branches.**
- Work on the active refactor branch (see `AGENT_STATE.md`); if the default
  branch is checked out, branch first.
- Commit messages end with the project's standard co-author trailer.

## Reference

- Project operating rules also live in `CLAUDE.md` and `docs/`. Where this file
  and `docs/` agree, follow them. Where a task contradicts these rules, stop
  and confirm.
