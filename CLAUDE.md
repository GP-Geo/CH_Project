# CLAUDE.md — developer hub

Detects **channel-head coupling** in drainage networks (Goren & Shelef 2024),
trained on Earth and applied to Mars. Full documentation lives in [`docs/`](docs/).

## Pipeline status

See [`STAGE_ASSET_MAP.md`](STAGE_ASSET_MAP.md) for the current per-stage
coverage. The package-first refactor is **complete**; all core logic lives in
`channel_heads/`; the CLI lives in the `channel_heads/cli/` package — run `python -m channel_heads <command>` (or the `channel-heads` console script).

## Start here

| If you need… | Read |
|---|---|
| High-level pipeline stages 0–14 | [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md) |
| Stage-by-stage inputs/outputs/deps | [docs/pipeline.md](docs/pipeline.md) |
| Package architecture + public API | [docs/architecture.md](docs/architecture.md) |
| Package API, ML pipeline, testing, conventions | [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) |
| Regime quick reference (the three frozen regimes) | [docs/regimes_summary.md](docs/regimes_summary.md) |
| Regime calibration rationale (frozen parameters) | [docs/REGIME_SELECTION.md](docs/REGIME_SELECTION.md) |
| Model variants + Mars threshold | [docs/modeling.md](docs/modeling.md) |
| Mars cross-planet pipeline | [docs/MARS_PIPELINE.md](docs/MARS_PIPELINE.md) |
| Repo / scripts / data layout | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |
| Data keep/regenerate/stale classification | [docs/DATA_STATUS.md](docs/DATA_STATUS.md) |
| Data categories + safe cleanup | [docs/data_management.md](docs/data_management.md) |
| End-to-end pipeline regeneration order | [docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md) |
| Notebook catalogue | [docs/notebooks.md](docs/notebooks.md) |
| Open work + scientific risk register | [docs/ROADMAP_AND_RISKS.md](docs/ROADMAP_AND_RISKS.md) |
| User quickstart | [README.md](README.md) |

## Operating rules

- **Never delete data.** Mark as legacy/archive instead (see `docs/data_management.md`).
- The production models `models/xgb_touching_classifier.json` (threshold 0.577406)
  and `models/cnn_outlet_final.pt` are **preserved as-is**; regime variants use
  explicit suffixes (`_geom_*`, `_reg{A,B,C}`).
- Mars CNN patches **must** stay 5-class to match the Earth-trained CNN.
- Unit conversions go through `channel_heads/units.py` (single source of truth).
- Validate with `conda run -n ch-heads pytest -q`; run `ruff` **targeted** on
  touched files only.
- `channel_heads/` public API is re-exported from `__init__.py`.
- **Do not push or merge into `main`** on the active development branch.
- **Stop if uncertain about scientific behavior** — numeric outputs, feature
  order, model thresholds, and prediction schemas must not change silently.

## Environment

```bash
conda env create -f env/environment.yml && conda activate ch-heads
pip install -e ".[dev,geo,viz,cnn,ml]"
```
