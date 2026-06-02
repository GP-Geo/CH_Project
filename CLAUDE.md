# CLAUDE.md — developer hub

This project detects **channel-head coupling** in drainage networks (Goren &
Shelef 2024), trained on Earth and applied to Mars. Full documentation now lives
in [`docs/`](docs/) — this file is a thin pointer.

## Start here

| If you need… | Read |
|--------------|------|
| **Package-first architecture + public API (start here)** | [docs/architecture.md](docs/architecture.md) |
| Stage-by-stage pipeline (inputs/outputs/deps) | [docs/pipeline.md](docs/pipeline.md) |
| Model variants + Mars threshold issue | [docs/modeling.md](docs/modeling.md) |
| Data categories + safe cleanup | [docs/data_management.md](docs/data_management.md) |
| Notebook catalogue | [docs/notebooks.md](docs/notebooks.md) |
| Package API, ML pipeline, testing, conventions | [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) |
| Repo / scripts / data layout & inventory | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) |
| Mars cross-planet pipeline & phase history | [docs/MARS_PIPELINE.md](docs/MARS_PIPELINE.md) |
| Open work + scientific risk register | [docs/ROADMAP_AND_RISKS.md](docs/ROADMAP_AND_RISKS.md) |
| Data keep/regenerate/stale classification | [docs/DATA_STATUS.md](docs/DATA_STATUS.md) |
| End-to-end pipeline regeneration order | [docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md) |
| User quickstart | [README.md](README.md) |

## Operating rules (always apply)

- **Never delete data.** Mark legacy / archive instead (see PROJECT_STRUCTURE.md).
- The production model `models/xgb_touching_classifier.json` (threshold 0.577406)
  and `models/cnn_outlet_final.pt` are **preserved as-is**; research/regime
  variants use explicit suffixes (`_geom_*`, `_reg{A,B,C}`).
- Mars CNN patches **must** stay 5-class to match the Earth-trained CNN.
- Unit conversions go through `channel_heads/units.py` (single source of truth).
- Validate with `conda run -n ch-heads pytest -q`; run `ruff` **targeted** on
  touched files only (repo-wide has known pre-existing errors).
- `channel_heads/` public API is re-exported from `__init__.py`.

## Environment

```bash
conda env create -f env/environment.yml && conda activate ch-heads
pip install -e ".[dev,geo,viz,cnn,ml]"
```
