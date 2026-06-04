#!/usr/bin/env python
"""Generate notebooks/pipeline/00..14 — the canonical, **informative** per-stage
walkthrough of the channel-head coupling pipeline.

Each notebook is a real research notebook: scientific narrative + substantive
analysis/visualization using the canonical ``channel_heads`` package and the
artifacts on disk, plus interpretation. Recipes are distilled from the themed
deep-dive notebooks (analysis/, training/, mars/, regime/, diagnostics/,
interpretation/). Heavy topo/stream work is run on a single representative basin
and wrapped defensively so the whole set executes headless.

Regenerate, then execute:
    python notebooks/pipeline/_build_pipeline_notebooks.py
    MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace \\
      --ExecutePreprocessor.timeout=900 notebooks/pipeline/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Shared, defensive setup prepended to every notebook.
SETUP = """\
%matplotlib inline
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import channel_heads as ch
from channel_heads.io.paths import PROJECT_ROOT, RESULTS_DIR, EXAMPLE_DEMS
ROOT     = PROJECT_ROOT
MODELS   = ROOT / 'models'
MARS_OUT = ROOT / 'data/Mars/model_outputs'
MARS_IN  = ROOT / 'data/Mars/model_inputs'
REGIMES_ = ['regA', 'regB', 'regC']
# Frozen model features (5 dimensionless) + operating thresholds (Stage 9).
MODEL_FEATURES = ['orientation_diff_deg','headhead_dist_norm','apex_angle_deg',
                  'strahler_order_diff','proximity_profile_norm']
OP_THR = {'regA': 0.756326, 'regB': 0.779264, 'regC': 0.759369}
print('channel_heads', ch.__version__, '| root', ROOT)
"""


def C(s):
    return ("code", s)


def M(s):
    return ("md", s)


# Per-stage content: (num, title, [cells]) -----------------------------------
STAGES: list[tuple[int, str, list[tuple[str, str]]]] = []

# Stage 0
STAGES.append((0, "Project setup and assumptions", [
    M("""## What this project does

We detect **channel-head coupling** in drainage networks — pairs of channel heads
that meet at a confluence whose contributing areas are spatially *touching* — and
ask how common that coupling is on **Mars**, using a classifier **trained on Earth**
(method after *Goren & Shelef 2024*).

Coupling is a fingerprint of **drainage-divide mobility**: when two growing channel
heads share/contest a divide, their basins press together. The coupled fraction of
confluences is a proxy for how dynamic a landscape's divides are — and Earth vs Mars
probes whether martian valley networks froze in a fluvially-active or degraded state.

### The transfer-learning idea
Mars valley networks are larger and lower-resolution than Earth basins, so the model
uses **5 dimensionless features** describing the *geometry* of a confluence
(orientation contrast, normalized head–head distance, apex angle, Strahler-order
difference, a normalized proximity profile). Scale-free geometry lets an Earth-trained
model apply to Mars without rescaling. A small **CNN** adds raster *shape context* via
a 4-D embedding of a 5-class confluence patch.

### Frozen contracts (never change silently)
- Production `xgb_touching_classifier.json` (threshold **0.577406**) and `cnn_outlet_final.pt`.
- The **5 dimensionless features** (order matters) and the **5-class** CNN patch.
- Unit conversions go through `channel_heads.units`."""),
    C(SETUP),
    M("### Canonical paths, the 17 Earth basins, and the three frozen regimes"),
    C("""from channel_heads.regimes import REGIMES
print('Earth DEMs available:', len(EXAMPLE_DEMS))
print('Frozen model features:', MODEL_FEATURES)
pd.DataFrame([{'regime': r.name, 'threshold_km2': r.threshold_km2,
               'pre_remove_max_order': r.pre_remove_max_order,
               'order_gap_to_prune': r.order_gap_to_prune,
               'character': c} for r, c in zip(
                   REGIMES.values(),
                   ['dense base / aggressive tip removal',
                    'sparse base / minimal pruning',
                    'intermediate density'])])"""),
    M("""The three regimes (regA/B/C) re-run the whole pipeline at different network
*complexities* — the project's way of quantifying **calibration uncertainty**. No
single pruning is "correct"; we report the spread."""),
    C("""# Unit contract demo: the same arc-degree distance differs in metres by latitude.
from channel_heads.units import compute_meters_per_degree
for lat in (10, 36, 60):
    print(f'lat {lat:>2}deg:  1 deg longitude = {compute_meters_per_degree(lat):8.1f} m')"""),
    M("""**Why this matters:** a head–head distance in pixels/degrees must be converted with
the basin's latitude before it is physical — and Mars uses a different planetary
radius. Centralizing this in `units.py` keeps Earth and Mars numerically comparable.

Deep dives: `docs/PIPELINE_DESIGN.md`, `docs/REGIME_SELECTION.md`, `docs/modeling.md`."""),
]))

# Stage 1
STAGES.append((1, "Earth source-data exploration", [
    M("""## Earth source data — 17 SRTM basins

The training landscapes are 17 cropped SRTM DEMs (RAW_KEEP) from *Goren & Shelef
(2024)*, spanning climate, lithology and relief. Each basin carries an elevation
threshold `z_th` (mask below it) so the analysis runs on the upland, channelized part.

We audit every DEM (CRS, resolution, elevation range) and visualize one as a
hillshade — the raw material every downstream stage derives from."""),
    C(SETUP),
    C("""import rasterio
from channel_heads.basin_config import get_basin_config, BASIN_CONFIG
rows = []
for name, p in EXAMPLE_DEMS.items():
    p = Path(p)
    if not p.exists():
        continue
    with rasterio.open(p) as src:
        z = src.read(1, masked=True)
        cfg = get_basin_config(name) if name in BASIN_CONFIG else {}
        rows.append({'basin': name, 'crs': str(src.crs), 'cell_deg': round(abs(src.transform.a), 6),
                     'w': src.width, 'h': src.height,
                     'z_min': float(z.min()), 'z_max': float(z.max()), 'z_th': cfg.get('z_th')})
dems = pd.DataFrame(rows).sort_values('basin').reset_index(drop=True)
print(f'{len(dems)} DEMs; single CRS = {dems.crs.nunique()==1}')
dems"""),
    M("All basins share a geographic CRS and a ~0.000833 deg (~30 m) cell — consistent SRTM."),
    C("""# Hillshade of one basin to make the topography tangible.
from matplotlib.colors import LightSource
basin = 'inyo'
with rasterio.open(EXAMPLE_DEMS[basin]) as src:
    z = src.read(1).astype(float)
z[z <= 0] = np.nan
ls = LightSource(azdeg=315, altdeg=45)
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=2), cmap='gray')
im = ax.imshow(z, cmap='terrain', alpha=0.45)
ax.set_title(f'{basin}: SRTM elevation + hillshade'); ax.axis('off')
plt.colorbar(im, ax=ax, shrink=0.7, label='elevation (m)'); plt.show()"""),
    M("Deep dive: `notebooks/analysis/01_earth_source_data_qa.ipynb`."),
]))

# Stage 2
STAGES.append((2, "Earth interactive network exploration", [
    M("""## From DEM to drainage network

A DEM becomes a **channel network** via flow routing (`FlowObject`), thresholding
flow accumulation to initiate streams (`StreamObject(threshold)`), and **Strahler
pruning** of the finest tips. The stream-initiation threshold is the biggest control
on **drainage density** (Dd = stream length / area) — the quantity we later match
between Earth and Mars.

We extract one basin's network and sweep the threshold to see how Dd responds, with
the three frozen regime thresholds overlaid."""),
    C(SETUP),
    C("""from channel_heads.dd_calibration import collect_basin_metrics_for_dem
from channel_heads.basin_config import get_basin_config
from channel_heads.regimes import REGIMES
BASIN = 'inyo'
cfg = get_basin_config(BASIN); lat, z_th = cfg['lat'], cfg['z_th']
m = max(collect_basin_metrics_for_dem(dem_path=EXAMPLE_DEMS[BASIN], basin_name=BASIN,
        thresholds_km2=[0.10], z_th=z_th, lat_deg=lat), key=lambda x: x.n_stream_nodes)
print(f'{BASIN}: Dd={m.dd_true_km_km2:.3f} km/km2  length={m.stream_length_km:.0f} km  '
      f'area={m.basin_area_km2:.0f} km2  nodes={m.n_stream_nodes}')"""),
    M("### Drainage density vs stream-initiation threshold"),
    C("""SWEEP = [0.02, 0.05, 0.10, 0.25, 0.5, 1.0]
sweep = collect_basin_metrics_for_dem(dem_path=EXAMPLE_DEMS[BASIN], basin_name=BASIN,
        thresholds_km2=SWEEP, z_th=z_th, lat_deg=lat)
best = {}
for x in sweep:
    if x.threshold_km2 not in best or x.n_stream_nodes > best[x.threshold_km2].n_stream_nodes:
        best[x.threshold_km2] = x
sw = pd.DataFrame([{'threshold_km2': t, 'dd': best[t].dd_true_km_km2} for t in SWEEP if t in best])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sw.threshold_km2, sw.dd, 'o-', color='steelblue')
ax.set_xscale('log'); ax.set_xlabel('stream-initiation threshold (km2)')
ax.set_ylabel('drainage density (km/km2)'); ax.set_title(f'{BASIN}: Dd vs threshold')
for name, r, col in zip(REGIMES, REGIMES.values(), ['#e41a1c', '#377eb8', '#4daf4a']):
    ax.axvline(r.threshold_km2, ls='--', color=col, label=f'{name} ({r.threshold_km2})')
ax.legend(); ax.grid(alpha=0.3); plt.show()"""),
    M("""Lower thresholds -> denser networks. The regime thresholds (0.05 / 0.25 / 0.10 km2)
bracket a range of densities chosen so Earth Dd brackets martian Dd. Deep dives:
`notebooks/analysis/02_earth_network_explorer.ipynb`,
`notebooks/diagnostics/dd_threshold_calibration.ipynb`."""),
]))

# Stage 3
STAGES.append((3, "Mars interactive network exploration", [
    M("""## Mars valley networks — the target landscape

Mars networks are **already-mapped valley-network polylines** (391 networks), not
DEM-derived. Their geometry reflects whatever fluvial signal survived ~3.5 Gyr of
degradation. The Earth->Mars transfer is only valid if Earth training networks have
comparable **drainage density**, so we characterize the Mars Dd distribution here —
the target Stage 4 calibrates Earth against."""),
    C(SETUP),
    C("""from channel_heads.io.paths import MARS_DIR
from channel_heads.dd_calibration import mars_network_table
gpkg = MARS_DIR / 'topology' / 'mars_vn_topology_model_ready.gpkg'
net = mars_network_table(gpkg if gpkg.exists() else None)
print(f'Mars networks: {len(net)}')
net[['length_km','hull_area_km2','dd_hull_km_km2']].describe().round(2)"""),
    M("### Distribution of martian network drainage density"),
    C("""dd = net['dd_hull_km_km2'].dropna()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(dd, bins=40, color='indianred', edgecolor='white')
ax.axvline(dd.median(), color='k', ls='--', label=f'median {dd.median():.2f} km/km2')
ax.set_xlabel('hull drainage density (km/km2)'); ax.set_ylabel('network count')
ax.set_title('Mars valley-network Dd (n=%d)' % len(dd)); ax.legend(); plt.show()
print('Mars median Dd = %.3f km/km2 (Earth regimes bracket this)' % dd.median())"""),
    M("Deep dive: `notebooks/mars/00_mars_network_explorer.ipynb` (maps on MOLA hillshade)."),
]))

# Stage 4
STAGES.append((4, "Earth-Mars regime calibration", [
    M("""## Calibrating Earth complexity to Mars

Earth and Mars coupling rates are only comparable if the networks are matched in
**drainage density / complexity**. A sweep over thresholds and Strahler-pruning
levels (measuring Dd-hull vs the Mars reference) yields three frozen regimes that
**bracket** the martian Dd:

| Regime | threshold | pruning | character |
|---|---|---|---|
| regA | 0.05 km2 | drop <=2nd order, gap >=4 | densest |
| regB | 0.25 km2 | drop <=1st order, gap >=4 | sparsest |
| regC | 0.10 km2 | drop <=1st order, gap >=4 | intermediate |

Running all three and reporting the spread of Mars results **is** the calibration-
uncertainty estimate. Frozen upstream of every trained model."""),
    C(SETUP),
    C("""from channel_heads.regimes import REGIMES
pd.DataFrame([r.__dict__ for r in REGIMES.values()])"""),
    C("""from IPython.display import Image, display
cal = RESULTS_DIR / 'drainage_density_calibration/complexity_calibration'
csv = cal / 'dd_master_sweep_complexity.csv'
if csv.exists():
    s = pd.read_csv(csv); print('dd_master_sweep_complexity.csv:', s.shape); display(s.head())
for fig in ['fig_mars_vs_earth_at_matched_complexity.png', 'fig_dd_hull_matched_vs_full_vs_mars.png']:
    f = cal / fig
    if f.exists():
        print(fig); display(Image(filename=str(f)))"""),
    M("Rationale: `docs/REGIME_SELECTION.md`; deep dive: `notebooks/regime/00_calibration_overview.ipynb`."),
]))

# Stage 5
STAGES.append((5, "Final Earth network generation and QA", [
    M("""## Building & QA-gating the Earth feature tables

Per regime we build every basin's regime-pruned network, enumerate confluence pairs
per outlet, and compute features + the touching label — a **QA gate** that must pass
(0 hard flags) before training. Build:
`python -m channel_heads build-earth-features --regime <r>`. We load the QA report
and confirm it is clean."""),
    C(SETUP),
    C("""qa = RESULTS_DIR / 'stage5_earth_network_qa_report.csv'
if qa.exists():
    q = pd.read_csv(qa)
    print('QA rows:', len(q), '| columns:', list(q.columns))
    for c in [c for c in q.columns if 'flag' in c.lower()]:
        n = q[c].sum() if q[c].dtype != object else (q[c].astype(str).str.len() > 0).sum()
        print(f'  {c}: {int(n)} flagged')
    display(q.head(20))
else:
    print('QA report not found - run build-earth-features first.')"""),
    M("Deep dive: `notebooks/analysis/05_earth_network_qa.ipynb`."),
]))

# Stage 6
STAGES.append((6, "Earth pair and label generation", [
    M("""## Pairs, the touching label, and what separates the classes

For every outlet we take **first-meet pairs** of channel heads (the two heads that
first meet going downstream at a confluence). The label `y = touching` is a geometric
test: do the heads' contributing pixels touch (8-connectivity)? Trivial negatives are
removed (`filter_hard_negatives`) and negatives subsampled to ~3:1.

Below: per-regime class balance, then the **distributions of the 5 model features**
for touching vs non-touching — the signal the classifier exploits."""),
    C(SETUP),
    C("""rows = []
for r in REGIMES_:
    p = RESULTS_DIR / f'master_dataset_{r}.csv'
    if p.exists():
        d = pd.read_csv(p)
        rows.append({'regime': r, 'pairs': len(d), 'touching': int(d.y.sum()),
                     'touching_%': round(100*d.y.mean(), 1), 'basins': d.basin.nunique()})
pd.DataFrame(rows)"""),
    M("### Feature separation (regA): touching vs non-touching"),
    C("""d = pd.read_csv(RESULTS_DIR / 'master_dataset_regA.csv')
fig, axes = plt.subplots(1, len(MODEL_FEATURES), figsize=(16, 3))
for ax, feat in zip(axes, MODEL_FEATURES):
    for lbl, col, name in [(0, '#1f77b4', 'non-touch'), (1, '#d62728', 'touch')]:
        v = d.loc[d.y == lbl, feat].dropna()
        lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v.clip(lo, hi), bins=30, alpha=0.6, color=col, density=True, label=name)
    ax.set_title(feat, fontsize=9); ax.set_yticks([])
axes[0].legend(fontsize=8); plt.suptitle('regA - 5 model features by class', y=1.05); plt.show()"""),
    M("""Touching pairs trend toward smaller orientation contrast and shorter normalized
head–head distance — coupled heads sit closer and more aligned. No single feature
separates the classes, motivating a learned classifier + CNN shape context. Deep dive:
`notebooks/training/00_pair_sample_qa.ipynb` (renders actual pair geometries)."""),
]))

# Stage 7
STAGES.append((7, "Earth model-input construction", [
    M("""## 5-class confluence patches for the CNN

The CNN sees each confluence as a **128x128, 5-class** raster
(BACKGROUND / BRANCH_A / BRANCH_B / OTHER_STREAMS / CONFLUENCE_MARKER), drawn directly
into the final grid (frozen rasterizer contract). This encodes the *shape* of the
meeting the 5 scalar features cannot. Build:
`python -m channel_heads build-cnn-patches --regime <r> [--workers N]`."""),
    C(SETUP),
    C("""from channel_heads.rasterization.schema import (BACKGROUND, BRANCH_A, BRANCH_B,
                                                  OTHER_STREAMS, CONFLUENCE_MARKER, NUM_CLASSES)
print('NUM_CLASSES =', NUM_CLASSES, '->', dict(BACKGROUND=BACKGROUND, BRANCH_A=BRANCH_A,
      BRANCH_B=BRANCH_B, OTHER_STREAMS=OTHER_STREAMS, CONFLUENCE_MARKER=CONFLUENCE_MARKER))
for r in REGIMES_:
    p = RESULTS_DIR / f'raster_manifest_{r}.csv'
    if p.exists():
        d = pd.read_csv(p)
        ok = int((d.raster_status == 'ok').sum()) if 'raster_status' in d else len(d)
        print(f'{r}: {len(d)} rows, {ok} ok, {len(d)-ok} invalid')"""),
    M("### A gallery of real 5-class patches"),
    C("""from matplotlib.colors import ListedColormap
cmap = ListedColormap([[0.95,0.95,0.95],[0.85,0.33,0.10],[0.10,0.45,0.82],
                       [0.70,0.70,0.70],[0.90,0.80,0.0]])
man = pd.read_csv(RESULTS_DIR / 'raster_manifest_regA.csv')
man = man[man.raster_path.notna()]
if 'raster_status' in man: man = man[man.raster_status == 'ok']
def load_lab(p):
    p = Path(p); p = p if p.is_absolute() else ROOT / p
    a = np.load(p); return a.argmax(0) if a.ndim == 3 else a
samp = pd.concat([man[man.y==1].head(4), man[man.y==0].head(4)])
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (_, row) in zip(axes.ravel(), samp.iterrows()):
    ax.imshow(load_lab(row.raster_path), cmap=cmap, vmin=0, vmax=4)
    ax.set_title('touching' if row.y == 1 else 'non-touching', fontsize=9,
                 color='#d62728' if row.y==1 else '#1f77b4'); ax.axis('off')
plt.suptitle('regA 5-class patches (top: touching, bottom: non-touching)'); plt.show()"""),
    M("Deep dives: `notebooks/training/03_feature_engineering.ipynb`, `notebooks/diagnostics/rasterization_diagnostics.ipynb`."),
]))

# Stage 8
STAGES.append((8, "Model training", [
    M("""## Training the CNN + combined XGBoost (per regime)

Two models per regime: **OutletCNN** (5-class patch -> 4-D embedding, Taiwan held
out) and the **combined XGBoost** (5 features + 4 embeddings = 9, split by
`GroupShuffleSplit` on `basin__outlet` so no outlet leaks across train/test). Train:
`python -m channel_heads train-cnn-regime` then `train-combined-xgb-regime`."""),
    C(SETUP),
    M("### CNN convergence (per-regime loss curves)"),
    C("""fig, ax = plt.subplots(figsize=(8, 4))
for r in REGIMES_:
    h = MODELS / f'cnn_outlet_{r}_history.csv'
    if h.exists():
        hist = pd.read_csv(h)
        ax.plot(hist.epoch, hist.val_loss, label=f'{r} val', lw=2)
        ax.plot(hist.epoch, hist.train_loss, ls=':', alpha=0.6)
ax.set_xlabel('epoch'); ax.set_ylabel('BCE loss'); ax.set_title('CNN training (solid=val, dotted=train)')
ax.legend(); ax.grid(alpha=0.3); plt.show()"""),
    M("### Does the CNN embedding separate the classes?"),
    C("""d = pd.read_csv(RESULTS_DIR / 'master_dataset_regA_with_emb.csv')
fig, ax = plt.subplots(figsize=(6, 5))
for lbl, col, name in [(0, '#1f77b4', 'non-touch'), (1, '#d62728', 'touch')]:
    s = d[d.y == lbl].sample(min(1500, int((d.y==lbl).sum())), random_state=0)
    ax.scatter(s.emb_0, s.emb_1, s=6, alpha=0.3, c=col, label=name)
ax.set_xlabel('emb_0'); ax.set_ylabel('emb_1'); ax.set_title('regA CNN embedding (emb_0 vs emb_1)')
ax.legend(); plt.show()"""),
    M("### Model-comparison table (geom-only vs geom+CNN per regime)"),
    C("""p = MODELS / 'ALL_MODELS_METRICS.csv'
pd.read_csv(p) if p.exists() else 'train models first'"""),
    M("""The CNN embedding lifts ROC-AUC by ~0.09 to 0.17 over geometry alone per regime —
the patch shape carries real, complementary signal. Deep dives:
`notebooks/training/{02_train_classifier,04_cnn_embeddings,05_cnn_quick_eval}.ipynb`."""),
]))

# Stage 9
STAGES.append((9, "Earth model validation and tuning", [
    M("""## Honest validation + choosing an operating threshold

A single split can be lucky. **Leave-one-basin-out (LOBO)** CV holds out each basin in
turn (the analogue of "apply to a new landscape") and is our honest estimate. We then
pick a decision threshold: F1-optimal maximizes balanced accuracy on Earth, but for
label-free Mars we prefer a **precision-oriented** point (max precision at recall>=0.5).
Run: `python -m channel_heads eval-lobo-cv` / `retune-threshold-regime`."""),
    C(SETUP),
    M("### LOBO cross-validation (pooled + per-fold AUC)"),
    C("""p = MODELS / 'lobo_cv_metrics.csv'
lobo = pd.read_csv(p) if p.exists() else pd.DataFrame()
display(lobo)
if not lobo.empty:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(lobo.config, lobo.pooled_auc, color='steelblue', label='pooled AUC')
    ax.errorbar(lobo.config, lobo.fold_auc_mean, yerr=lobo.fold_auc_std, fmt='o', color='k',
                label='per-fold mean+/-std')
    ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.6, 1.0); ax.set_title('LOBO CV'); ax.legend(); plt.show()"""),
    M("### ROC / PR for the regA combined model (held-out outlets)"),
    C("""from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
d = pd.read_csv(RESULTS_DIR / 'master_dataset_regA_with_emb.csv')
feats = MODEL_FEATURES + [f'emb_{i}' for i in range(4)]
d['grp'] = d.basin.astype(str) + '__' + d.outlet.astype(str)
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(d, d.y, d.grp))
mdl = XGBClassifier(); mdl.load_model(str(MODELS / 'xgb_geom_plus_cnn_emb_regA.json'))
proba = mdl.predict_proba(d.iloc[te][feats].to_numpy())[:, 1]; y = d.iloc[te].y.to_numpy()
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
fpr, tpr, _ = roc_curve(y, proba); axs[0].plot(fpr, tpr); axs[0].plot([0,1],[0,1],'k:')
axs[0].set_title(f'ROC (AUC={roc_auc_score(y,proba):.3f})'); axs[0].set_xlabel('FPR'); axs[0].set_ylabel('TPR')
pr, rc, _ = precision_recall_curve(y, proba); axs[1].plot(rc, pr)
axs[1].axvline(0.5, color='r', ls='--', label='recall=0.5'); axs[1].legend()
axs[1].set_title(f'PR (AP={average_precision_score(y,proba):.3f})'); axs[1].set_xlabel('recall'); axs[1].set_ylabel('precision')
plt.show()"""),
    C("""print('Operating thresholds (max-precision @ recall>=0.5), used for Mars:')
for r in REGIMES_:
    t = MODELS / f'optimal_threshold_geom_plus_cnn_emb_{r}.txt'
    if t.exists(): print(f'  {r}: {t.read_text().split()[0]}')"""),
    M("""**Mars caveat:** the Earth F1 threshold (0.577 for production) is tuned to Earth's
base rate. Mars has no labels, so we do not copy it — we carry the precision-oriented
point and study sensitivity in Stage 12. Deep dives:
`notebooks/diagnostics/lobo_cv.ipynb`, `notebooks/regime/02_threshold_retune.ipynb`."""),
]))

# Stage 10
STAGES.append((10, "Final Mars model-input generation", [
    M("""## Mars model inputs — same patches, frozen CNN

Mars geometry (topology->pairs->features) is shared across regimes and raster-
independent. The raster-dependent part — **5-class patches** — is regenerated with the
*current* rasterizer (matching the Earth patches the CNN trained on), then embedded
with the **frozen** `cnn_outlet_final.pt`. Patches **must stay 5-class**. Build:
`python -m channel_heads run-mars-pipeline --stage patches` / `--stage embeddings`."""),
    C(SETUP),
    C("""idx = MARS_IN / 'mars_cnn_patch_index.parquet'
if idx.exists():
    d = pd.read_parquet(idx)
    print('Mars patches:', len(d), '|',
          d['patch_status'].value_counts().to_dict() if 'patch_status' in d else '')"""),
    M("### Example Mars 5-class patches"),
    C("""from matplotlib.colors import ListedColormap
cmap = ListedColormap([[0.95,0.95,0.95],[0.85,0.33,0.10],[0.10,0.45,0.82],
                       [0.70,0.70,0.70],[0.90,0.80,0.0]])
d = pd.read_parquet(MARS_IN / 'mars_cnn_patch_index.parquet')
col = 'patch_path' if 'patch_path' in d else 'raster_path'
ok = d[d.get('patch_status','ok') == 'ok'].head(6)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, (_, row) in zip(axes, ok.iterrows()):
    pp = Path(row[col]); pp = pp if pp.is_absolute() else ROOT / pp
    a = np.load(pp); lab = a.argmax(0) if a.ndim == 3 else a
    ax.imshow(lab, cmap=cmap, vmin=0, vmax=4); ax.axis('off')
plt.suptitle('Mars confluence patches (5-class, same contract as Earth)'); plt.show()"""),
    M("Deep dives: `notebooks/mars/{02_first_meet_pairs,03_pair_features}.ipynb`."),
]))

# Stage 11
STAGES.append((11, "Mars inference", [
    M("""## Applying the Earth-trained regime models to Mars

Per regime we recompute Mars embeddings with that regime's CNN, apply its combined
XGBoost, and threshold at the regime's operating point -> a touching probability +
decision per Mars confluence pair. Run:
`python -m channel_heads run-mars-combined-regime --regime <r>`. Below: per-regime
coupling rates, probability distributions, and a **map** of predicted coupling."""),
    C(SETUP),
    C("""rows = []
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if p.exists():
        d = pd.read_parquet(p)
        rows.append({'regime': r, 'pairs': len(d), 'networks': d.network_id.nunique(),
                     'touching_%': round(100*d.pred_touching.mean(), 1),
                     'high_conf>=0.8_%': round(100*(d.prob_touching>=0.8).mean(), 1),
                     'mean_prob': round(d.prob_touching.mean(), 3)})
pd.DataFrame(rows)"""),
    M("### Probability distributions per regime"),
    C("""fig, ax = plt.subplots(figsize=(8, 4))
for r, col in zip(REGIMES_, ['#e41a1c', '#377eb8', '#4daf4a']):
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if p.exists():
        d = pd.read_parquet(p)
        ax.hist(d.prob_touching, bins=40, histtype='step', lw=2, color=col, label=r)
        ax.axvline(OP_THR[r], color=col, ls='--', alpha=0.5)
ax.set_xlabel('P(touching)'); ax.set_ylabel('pairs')
ax.set_title('Mars coupling probability (dashed = operating threshold)'); ax.legend(); plt.show()"""),
    M("### Map: predicted coupling across Mars valley networks (regA)"),
    C("""import geopandas as gpd
g = MARS_OUT / 'mars_combined_regA_predictions.gpkg'
if g.exists():
    gdf = gpd.read_file(g, layer='pairs')
    fig, ax = plt.subplots(figsize=(11, 6))
    gdf.plot(ax=ax, column='prob_touching', cmap='RdYlBu_r', linewidth=0.6,
             legend=True, legend_kwds={'label': 'P(touching)', 'shrink': 0.6})
    ax.set_title('regA - predicted channel-head coupling on Mars'); ax.set_aspect('equal'); ax.axis('off')
    plt.show()"""),
    M("Deep dive: `notebooks/regime/01_mars_inference.ipynb`."),
]))

# Stage 12
STAGES.append((12, "Mars threshold and prediction analysis", [
    M("""## Threshold sensitivity — the scientific knob on label-free Mars

Mars has no ground truth, so the coupling *rate* depends on the chosen threshold. We
sweep it and report how the coupled fraction responds per regime. A trustworthy
conclusion is **stable** across a sensible band and **consistent** across regimes —
not an artifact of one cut."""),
    C(SETUP),
    C("""SWEEP = np.round(np.arange(0.30, 0.96, 0.05), 2)
fig, ax = plt.subplots(figsize=(8, 5))
for r, col in zip(REGIMES_, ['#e41a1c', '#377eb8', '#4daf4a']):
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if p.exists():
        pr = pd.read_parquet(p).prob_touching.to_numpy()
        ax.plot(SWEEP, [(pr >= t).mean() for t in SWEEP], 'o-', color=col, label=r)
        ax.axvline(OP_THR[r], color=col, ls='--', alpha=0.4)
ax.set_xlabel('decision threshold'); ax.set_ylabel('fraction predicted touching')
ax.set_title('Mars coupling rate vs threshold (dashed = operating point)')
ax.legend(); ax.grid(alpha=0.3); plt.show()"""),
    M("""Smooth curves near the operating points -> the precision-oriented choice is stable;
regB > regA > regC at every threshold -> the regime *ordering* of coupling is robust.
Deep dive: `notebooks/mars/05_mars_threshold_sensitivity.ipynb`."""),
]))

# Stage 13
STAGES.append((13, "Scientific interpretation", [
    M("""## What Mars is telling us

We read the result scientifically: how much of the martian valley network is
**coupled**, how that splits across networks, and how regime-robust it is. High
coupling indicates divides that were mobile while the valleys were active — a more
Earth-like, dynamic fluvial regime."""),
    C(SETUP),
    C("""p = MARS_OUT / 'mars_regime_interpretation.csv'
display(pd.read_csv(p) if p.exists() else pd.DataFrame())"""),
    M("### Cross-regime consensus (how regime-robust is each pair's call?)"),
    C("""base = None
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if not p.exists(): continue
    d = pd.read_parquet(p)[['pair_id', 'prob_touching']].copy()
    d[r] = (d['prob_touching'] >= OP_THR[r]).astype(int)
    d = d[['pair_id', r]]
    base = d if base is None else base.merge(d, on='pair_id')
if base is not None:
    s = base[REGIMES_].sum(axis=1); n = len(base)
    print(f'pairs={n}')
    print(f'  all 3 TOUCHING     : {int((s==3).sum())} ({100*(s==3).mean():.1f}%)')
    print(f'  all 3 NON-touching : {int((s==0).sum())} ({100*(s==0).mean():.1f}%)')
    print(f'  consensus          : {100*((s==0)|(s==3)).mean():.1f}%')
    print(f'  regime-sensitive   : {100*((s>0)&(s<3)).mean():.1f}%')"""),
    M("### Per-network coupled fraction (regA)"),
    C("""d = pd.read_parquet(MARS_OUT / 'mars_combined_regA_predictions.parquet')
per = d.groupby('network_id').pred_touching.mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(per, bins=25, color='seagreen', edgecolor='white')
ax.axvline(per.mean(), color='k', ls='--', label=f'mean {per.mean():.2f}')
ax.set_xlabel('coupled fraction within a network'); ax.set_ylabel('networks')
ax.set_title('regA - per-network coupled fraction'); ax.legend(); plt.show()"""),
    M("Deep dive: `notebooks/interpretation/00_scientific_summary.ipynb`."),
]))

# Stage 14
STAGES.append((14, "Figures, poster, and reporting", [
    M("""## Publication figures

Render scripts + presentation notebooks produce the result figures: ROC panels, the
Mars coupling map, **vector** contact sheets (high-confidence and disagreement cases),
and per-outlet drawings. Build: `python -m channel_heads make-result-figures` /
`generate-poster-figures` and `scripts/rendering/*`."""),
    C(SETUP),
    C("""figdir = MARS_OUT / 'figures_combined'
pngs = sorted(figdir.glob('*.png'))
print(f'{len(pngs)} figures in {figdir.relative_to(ROOT)}:')
for f in pngs: print('  ', f.name)"""),
    C("""from IPython.display import Image, display
for name in ['mars_networks_channels.png', 'contact_sheet_high_conf_both.png',
             'contact_sheet_disagreement.png']:
    f = figdir / name
    if f.exists():
        print(name); display(Image(filename=str(f)))"""),
    M("""Deep dives: `notebooks/presentation/{result_figures,mars_contact_sheets,
per_outlet_touching_pairs}.ipynb`. Per project convention, Mars contact sheets use
**vector polyline** drawings (not raster patches)."""),
]))


def build():
    for num, title, cells in STAGES:
        nb = new_notebook()
        nb.cells.append(new_markdown_cell(
            f"# Stage {num} — {title}\n\n"
            f"_Pipeline stage {num} of `docs/PIPELINE_DESIGN.md`. Uses the canonical "
            f"`channel_heads` package + on-disk artifacts; heavy rebuilds run via the "
            f"`channel-heads` CLI (see each stage's command)._"))
        for kind, src in cells:
            nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        slug = title.lower().replace(" ", "_").replace("-", "_").replace(",", "")
        nbf.write(nb, str(OUT / f"{num:02d}_{slug}.ipynb"))
        print("wrote", f"{num:02d}_{slug}.ipynb")


if __name__ == "__main__":
    build()
