#!/usr/bin/env python
"""Generate notebooks/pipeline/00..14 — one executable notebook per pipeline
stage. Each is lightweight: it imports the canonical channel_heads package and
loads the artifacts already on disk, verifying/visualizing that stage rather than
re-running heavy compute. Cells are defensive (skip gracefully if an artifact is
absent) so the set executes end-to-end via nbconvert.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path("/Users/guypi/Projects/channel-heads")
OUT = ROOT / "notebooks/pipeline"
OUT.mkdir(parents=True, exist_ok=True)

SETUP = (
    "import warnings; warnings.filterwarnings('ignore')\n"
    "from pathlib import Path\n"
    "import pandas as pd\n"
    "import channel_heads as ch\n"
    "from channel_heads.io.paths import PROJECT_ROOT, RESULTS_DIR, EXAMPLE_DEMS\n"
    "ROOT = PROJECT_ROOT\n"
    "MODELS = ROOT / 'models'\n"
    "MARS_OUT = ROOT / 'data/Mars/model_outputs'\n"
    "MARS_IN  = ROOT / 'data/Mars/model_inputs'\n"
    "print('channel_heads', ch.__version__, '| root', ROOT)\n"
)

# (num, title, intro_md, [code cells]) ----------------------------------------
STAGES: list[tuple[int, str, str, list[str]]] = [
    (0, "Project setup and assumptions",
     "Canonical paths, frozen regime presets, and unit contracts. Everything "
     "downstream resolves through `channel_heads.io.paths`, `channel_heads.regimes`, "
     "and `channel_heads.units`.",
     ["from channel_heads.regimes import REGIMES\n"
      "print('Earth DEMs:', len(EXAMPLE_DEMS))\n"
      "import pandas as pd\n"
      "pd.DataFrame([{ 'regime': r.name, 'threshold_km2': r.threshold_km2,\n"
      "  'pre_remove_max_order': r.pre_remove_max_order,\n"
      "  'order_gap_to_prune': r.order_gap_to_prune } for r in REGIMES.values()])"]),
    (1, "Earth source-data exploration",
     "The 17 cropped Earth DEMs (RAW_KEEP). Deep-dive: "
     "`notebooks/analysis/01_earth_source_data_qa.ipynb`.",
     ["import rasterio\n"
      "rows = []\n"
      "for name, p in list(EXAMPLE_DEMS.items()):\n"
      "    p = Path(p)\n"
      "    if not p.exists():\n"
      "        continue\n"
      "    with rasterio.open(p) as src:\n"
      "        rows.append({'basin': name, 'crs': str(src.crs), 'w': src.width, 'h': src.height})\n"
      "pd.DataFrame(rows).head(20)"]),
    (2, "Earth interactive network exploration",
     "Stream extraction + Strahler pruning per basin (regime presets). Deep-dive: "
     "`notebooks/analysis/02_earth_network_explorer.ipynb`. Here we load the Stage-5 QA gate.",
     ["qa = RESULTS_DIR / 'stage5_earth_network_qa_report.csv'\n"
      "print('QA report:', qa.exists())\n"
      "pd.read_csv(qa).head() if qa.exists() else 'run Stage 5 first'"]),
    (3, "Mars interactive network exploration",
     "Mars valley-network topology (391 networks). Deep-dive: "
     "`notebooks/mars/00_mars_network_explorer.ipynb`.",
     [CHANGED,
    (4, "Earth-Mars regime calibration",
     "The three frozen complexity regimes (regA/B/C). Rationale: `docs/REGIME_SELECTION.md`.",
     ["from channel_heads.regimes import REGIMES\n"
      "pd.DataFrame([r.__dict__ for r in REGIMES.values()])"]),
    (5, "Final Earth network generation and QA",
     "Per-basin network QA gate (0 hard flags). Build: "
     "`python -m channel_heads build-earth-features --regime <r>`.",
     ["qa = RESULTS_DIR / 'stage5_earth_network_qa_report.csv'\n"
      "df = pd.read_csv(qa) if qa.exists() else pd.DataFrame()\n"
      "print('rows', len(df)); df.head()"]),
    (6, "Earth pair and label generation",
     "Touching/non-touching channel-head pairs per regime (`master_dataset_reg*.csv`).",
     ["for r in ['regA','regB','regC']:\n"
      "    p = RESULTS_DIR / f'master_dataset_{r}.csv'\n"
      "    if p.exists():\n"
      "        d = pd.read_csv(p)\n"
      "        print(f'{r}: {len(d)} pairs, {int(d.y.sum())} touching ({100*d.y.mean():.1f}%)')"]),
    (7, "Earth model-input construction",
     "5-class 128x128 CNN patches + manifests. Build: "
     "`python -m channel_heads build-cnn-patches --regime <r> [--workers N]`.",
     ["for r in ['regA','regB','regC']:\n"
      "    p = RESULTS_DIR / f'raster_manifest_{r}.csv'\n"
      "    if p.exists():\n"
      "        d = pd.read_csv(p)\n"
      "        ok = (d.raster_status=='ok').sum() if 'raster_status' in d else len(d)\n"
      "        print(f'{r}: {len(d)} rows, {ok} ok')",
      "import numpy as np, matplotlib.pyplot as plt\n"
      "m = RESULTS_DIR / 'raster_manifest_regA.csv'\n"
      "if m.exists():\n"
      "    d = pd.read_csv(m); d = d[d.raster_path.notna()]\n"
      "    arr = np.load(d.raster_path.iloc[0])\n"
      "    lab = arr.argmax(0) if arr.ndim==3 else arr\n"
      "    plt.imshow(lab, cmap='tab10'); plt.title('example 5-class patch (argmax)'); plt.colorbar()"]),
    (8, "Model training",
     "Per-regime CNN + combined geom+CNN-emb XGBoost. Train: "
     "`python -m channel_heads train-cnn-regime` then `train-combined-xgb-regime`.",
     ["p = MODELS / 'ALL_MODELS_METRICS.csv'\n"
      "pd.read_csv(p) if p.exists() else 'train models first'"]),
    (9, "Earth model validation and tuning",
     "LOBO cross-validation + operating thresholds. Run: "
     "`python -m channel_heads eval-lobo-cv` / `retune-threshold-regime`.",
     ["p = MODELS / 'lobo_cv_metrics.csv'\n"
      "print('thresholds (operating, max-precision):')\n"
      "for r in ['regA','regB','regC']:\n"
      "    t = MODELS / f'optimal_threshold_geom_plus_cnn_emb_{r}.txt'\n"
      "    if t.exists(): print(' ', r, t.read_text().split()[0])\n"
      "pd.read_csv(p) if p.exists() else 'run eval-lobo-cv'"]),
    (10, "Final Mars model-input generation",
     "Mars 5-class patches + embeddings (frozen CNN). Build: "
     "`python -m channel_heads run-mars-pipeline --stage patches` / `--stage embeddings`.",
     ["p = MARS_IN / 'mars_cnn_patch_index.parquet'\n"
      "if p.exists():\n"
      "    d = pd.read_parquet(p)\n"
      "    print('patches:', len(d))\n"
      "    print(d['patch_status'].value_counts().to_dict() if 'patch_status' in d else '')"]),
    (11, "Mars inference",
     "Regime combined predictions on Mars. Run: "
     "`python -m channel_heads run-mars-combined-regime --regime <r>`.",
     ["rows=[]\n"
      "for r in ['regA','regB','regC']:\n"
      "    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'\n"
      "    if p.exists():\n"
      "        d = pd.read_parquet(p)\n"
      "        rows.append({'regime':r,'pairs':len(d),'touching_%':round(100*d.pred_touching.mean(),1),\n"
      "                     'mean_prob':round(d.prob_touching.mean(),3)})\n"
      "pd.DataFrame(rows)"]),
    (12, "Mars threshold and prediction analysis",
     "Coupling rate vs decision threshold per regime "
     "(`mars_threshold_sensitivity.csv/.png`).",
     ["import matplotlib.pyplot as plt\n"
      "p = MARS_OUT / 'mars_threshold_sensitivity.csv'\n"
      "if p.exists():\n"
      "    s = pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)\n"
      "    for r,g in s[s.threshold.between(0.3,0.95)].groupby('regime'):\n"
      "        g=g.sort_values('threshold'); plt.plot(g.threshold,g.touching_frac,marker='o',label=r)\n"
      "    plt.xlabel('threshold'); plt.ylabel('touching frac'); plt.legend(); plt.title('Mars coupling vs threshold')"]),
    (13, "Scientific interpretation",
     "Per-regime coupling, networks-with-coupling, and cross-regime consensus "
     "(`mars_regime_interpretation.csv`). Deep-dive: "
     "`notebooks/interpretation/00_scientific_summary.ipynb`.",
     ["p = MARS_OUT / 'mars_regime_interpretation.csv'\n"
      "pd.read_csv(p) if p.exists() else 'run Stage 12/13 analysis'"]),
    (14, "Figures, poster, and reporting",
     "Refreshed figures. Build: `python -m channel_heads make-result-figures` / "
     "`generate-poster-figures` + `scripts/rendering/*`.",
     ["from IPython.display import Image, display\n"
      "figs = sorted((MARS_OUT/'figures_combined').glob('*.png'))[:3]\n"
      "print('figures:', len(list((MARS_OUT/'figures_combined').glob('*.png'))))\n"
      "for f in figs:\n"
      "    print(f.name); display(Image(filename=str(f)))"]),
]


def build():
    for num, title, intro, cells in STAGES:
        nb = new_notebook()
        nb.cells.append(new_markdown_cell(
            f"# Stage {num} — {title}\n\n{intro}\n\n"
            f"See `docs/PIPELINE_DESIGN.md` (Stage {num}) and `STAGE_ASSET_MAP.md`. "
            "This notebook loads canonical package APIs + on-disk artifacts; it does "
            "not re-run heavy compute (use the `channel-heads` CLI for that)."))
        nb.cells.append(new_code_cell(SETUP))
        for c in cells:
            nb.cells.append(new_code_cell(c))
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = OUT / f"{num:02d}_{title.lower().replace(' ','_').replace('-','_').replace(',','')}.ipynb"
        nbf.write(nb, str(path))
        print("wrote", path.name)


if __name__ == "__main__":
    build()
