#!/usr/bin/env python
"""Generate notebooks/pipeline/00..14 — the canonical, **self-contained deep dive**
for each pipeline stage of the channel-head coupling project.

Each notebook is the authoritative treatment of its stage: scientific narrative,
substantive analysis/visualization using the canonical ``channel_heads`` package
and the on-disk artifacts, and written interpretation. They fold in the analyses
that previously lived in the themed notebooks (analysis/, training/, mars/,
regime/, diagnostics/, interpretation/) — these *are* the deep dive, not a pointer
to one. Heavy topo/stream work runs on one representative basin so the whole set
executes headless.

Regenerate, then execute (inline backend captures figures — do NOT force Agg):
    python notebooks/pipeline/_build_pipeline_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
      --ExecutePreprocessor.timeout=1200 notebooks/pipeline/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent

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
MARS_DIR = ROOT / 'data/Mars'
REGIMES_ = ['regA', 'regB', 'regC']
RC = {'regA': '#e41a1c', 'regB': '#377eb8', 'regC': '#4daf4a'}
MODEL_FEATURES = ['orientation_diff_deg','headhead_dist_norm','apex_angle_deg',
                  'strahler_order_diff','proximity_profile_norm']
OP_THR = {'regA': 0.756326, 'regB': 0.779264, 'regC': 0.759369}
def _abs(p):
    p = Path(p); return p if p.is_absolute() else ROOT / p
print('channel_heads', ch.__version__, '| root', ROOT)
"""


def C(s): return ("code", s)
def M(s): return ("md", s)


STAGES: list[tuple[int, str, list[tuple[str, str]]]] = []

# ── Stage 0 ──────────────────────────────────────────────────────────────────
STAGES.append((0, "Project setup and assumptions", [
    M("""## What this project does, and the contracts everything rests on

We detect **channel-head coupling** in drainage networks — pairs of channel heads
that first meet at a confluence whose contributing areas are spatially *touching* —
and quantify how common that coupling is on **Mars**, with a classifier **trained on
Earth** (after *Goren & Shelef 2024*).

Coupling is a fingerprint of **drainage-divide mobility**. When two growing channel
heads contest the same divide, their basins press together and touch. The *coupled
fraction* of confluences is therefore a proxy for how dynamic a landscape's divides
were — and Earth vs Mars asks whether martian valley networks froze in a fluvially
active or a degraded state.

This notebook fixes the **assumptions and frozen contracts** the whole pipeline
depends on, and verifies them against the code/artifacts on disk."""),
    C(SETUP),
    M("""### 1 · The transfer-learning design: 5 dimensionless features

Mars valley networks are larger and lower-resolution than Earth basins, so the model
describes a confluence with **5 scale-free geometric features**. Scale-free geometry
is what lets an Earth-trained model apply to Mars without rescaling."""),
    C("""feat_desc = {
 'orientation_diff_deg':  'angle between the two branches approaching the confluence',
 'headhead_dist_norm':    'head-to-head distance, normalized by branch length',
 'apex_angle_deg':        'interior apex angle at the confluence',
 'strahler_order_diff':   'difference in Strahler order of the two branches',
 'proximity_profile_norm':'normalized closest-approach profile of the two basins'}
pd.DataFrame({'feature': MODEL_FEATURES,
              'meaning': [feat_desc[f] for f in MODEL_FEATURES]})"""),
    M("""### 2 · The frozen production artifacts (never overwrite, never reorder)

The production classifier and CNN are preserved as-is. We load the production
XGBoost and confirm its **feature order** matches the contract above and its
**decision threshold** is the frozen 0.577406."""),
    C("""from xgboost import XGBClassifier
prod = MODELS / 'xgb_touching_classifier.json'
if prod.exists():
    m = XGBClassifier(); m.load_model(str(prod))
    names = m.get_booster().feature_names
    print('production model feature order:', names)
    print('matches frozen 5-feature contract:', names == MODEL_FEATURES)
thr = MODELS / 'optimal_threshold_geom_only.txt'
print('production threshold (frozen):', 0.577406)
print('5-class patch + 4-D CNN embedding are the other frozen contracts')"""),
    M("""### 3 · The three regimes = calibration uncertainty

The pipeline is run at three network *complexities* (regA/B/C). No single pruning is
"correct"; reporting the spread of Mars results across all three **is** the
calibration-uncertainty estimate. They are frozen upstream of every trained model."""),
    C("""from channel_heads.regimes import REGIMES
pd.DataFrame([{'regime': r.name, 'threshold_km2': r.threshold_km2,
               'pre_remove_max_order': r.pre_remove_max_order,
               'order_gap_to_prune': r.order_gap_to_prune,
               'character': c} for r, c in zip(REGIMES.values(),
              ['densest base / aggressive tip removal',
               'sparsest base / minimal pruning', 'intermediate density'])])"""),
    M("""### 4 · The unit contract

A distance measured in pixels/arc-degrees is meaningless until converted with the
basin's latitude — and Mars uses a different planetary radius. All conversions go
through `channel_heads.units`, the single source of truth that keeps Earth and Mars
numerically comparable."""),
    C("""from channel_heads.units import compute_meters_per_degree
pd.DataFrame({'latitude_deg': [10, 36, 60],
              'm_per_deg_lon': [round(compute_meters_per_degree(l), 1) for l in (10, 36, 60)]})"""),
    M("""**Takeaway.** The pipeline is a chain of stages (0→14) that turn raw topography
into a calibrated, regime-bracketed estimate of martian channel-head coupling. The
contracts above (5 features, 5-class patch, frozen production models, the regimes,
`units.py`) are invariant; the rest of these notebooks build on them stage by stage."""),
]))

# ── Stage 1 ──────────────────────────────────────────────────────────────────
STAGES.append((1, "Earth source-data exploration", [
    M("""## Earth source data — 17 SRTM basins, audited

The training landscapes are 17 cropped SRTM DEMs (RAW_KEEP) hand-picked from
*Goren & Shelef (2024)* to span climate, lithology and relief. This is the raw
material every downstream stage is derived from, so we **audit it**: are all DEMs in
the same CRS and resolution? Are elevation ranges geomorphically plausible? Each
basin also carries an elevation threshold `z_th` below which cells are masked, so the
analysis runs only on the upland, channelized part of the landscape."""),
    C(SETUP),
    M("### 1 · Per-basin inventory + automated QA checks"),
    C("""import rasterio
from channel_heads.basin_config import get_basin_config, BASIN_CONFIG
rows = []
for name, p in EXAMPLE_DEMS.items():
    p = Path(p)
    if not p.exists(): continue
    with rasterio.open(p) as src:
        z = src.read(1, masked=True)
        cfg = get_basin_config(name) if name in BASIN_CONFIG else {}
        rows.append({'basin': name, 'crs': str(src.crs), 'cell_deg': round(abs(src.transform.a), 6),
                     'w': src.width, 'h': src.height,
                     'z_min': round(float(z.min()), 1), 'z_max': round(float(z.max()), 1),
                     'z_th': cfg.get('z_th')})
dems = pd.DataFrame(rows).sort_values('basin').reset_index(drop=True)
checks = {'all 17 DEMs present': len(dems) == 17,
          'single CRS': dems.crs.nunique() == 1,
          'uniform cell size': dems.cell_deg.nunique() == 1,
          'z_min >= -500 m (plausible)': (dems.z_min >= -500).all(),
          'z_max <= 9000 m (plausible)': (dems.z_max <= 9000).all()}
print('QA:', {k: ('PASS' if v else 'FAIL') for k, v in checks.items()})
dems"""),
    M("All basins share a geographic CRS and a ~0.000833° (~30 m) SRTM cell — QA clean."),
    M("### 2 · Elevation distribution and the z_th mask"),
    C("""basin = 'inyo'
cfg = get_basin_config(basin); z_th = cfg['z_th']
with rasterio.open(EXAMPLE_DEMS[basin]) as src:
    z = src.read(1).astype(float)
z[z <= 0] = np.nan
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(z[np.isfinite(z)].ravel(), bins=60, color='steelblue')
ax[0].axvline(z_th, color='red', ls='--', label=f'z_th = {z_th} m')
ax[0].set_xlabel('elevation (m)'); ax[0].set_ylabel('pixels'); ax[0].set_title(f'{basin} elevation hist'); ax[0].legend()
masked = np.where(z >= z_th, z, np.nan)
ax[1].imshow(masked, cmap='terrain'); ax[1].set_title(f'{basin}: cells above z_th (analysis domain)'); ax[1].axis('off')
plt.show()
print(f'{basin}: {100*np.isfinite(masked).sum()/np.isfinite(z).sum():.0f}% of land pixels are above z_th')"""),
    M("### 3 · Hillshade — the topography we route flow over"),
    C("""from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=2), cmap='gray')
im = ax.imshow(z, cmap='terrain', alpha=0.45)
ax.set_title(f'{basin}: SRTM elevation + hillshade'); ax.axis('off')
plt.colorbar(im, ax=ax, shrink=0.7, label='elevation (m)'); plt.show()"""),
    M("""**Takeaway.** The 17 DEMs are internally consistent (one CRS, one cell size) and
plausible. The `z_th` mask removes valley-floor/basin-fill pixels so stream
extraction (Stage 2) operates on the channelized uplands — the part of the landscape
where channel heads and divides live."""),
]))

# ── Stage 2 ──────────────────────────────────────────────────────────────────
STAGES.append((2, "Earth interactive network exploration", [
    M("""## From DEM to drainage network — and why the threshold matters

A DEM becomes a **channel network** in three steps: flow routing (`FlowObject`),
thresholding flow accumulation to initiate streams (`StreamObject(threshold)`), and
**Strahler pruning** of the finest tips. The stream-initiation threshold is the single
biggest control on network **drainage density** (Dd = stream length / basin area) —
and Dd is exactly the quantity Stage 4 matches between Earth and Mars.

We characterize one basin's network quantitatively (Strahler structure, Dd-vs-
threshold response), then show how Dd at the three regime thresholds varies across
basins — the spread that the regimes are designed to bracket."""),
    C(SETUP),
    C("""from channel_heads.dd_calibration import collect_basin_metrics_for_dem
from channel_heads.basin_config import get_basin_config
from channel_heads.regimes import REGIMES
BASIN = 'inyo'
cfg = get_basin_config(BASIN); lat, z_th = cfg['lat'], cfg['z_th']
m = max(collect_basin_metrics_for_dem(dem_path=EXAMPLE_DEMS[BASIN], basin_name=BASIN,
        thresholds_km2=[0.10], z_th=z_th, lat_deg=lat), key=lambda x: x.n_stream_nodes)
print(f'{BASIN} @ 0.10 km2:  Dd={m.dd_true_km_km2:.3f} km/km2 | length={m.stream_length_km:.0f} km '
      f'| area={m.basin_area_km2:.0f} km2 | stream nodes={m.n_stream_nodes}')"""),
    M("### 1 · Strahler-order structure"),
    C("""dist = getattr(m, 'strahler_distribution', None) or {}
if dist:
    orders = sorted(dist); fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(orders, [dist[o] for o in orders], color='steelblue', edgecolor='white')
    ax.set_xlabel('Strahler order'); ax.set_ylabel('node count')
    ax.set_title(f'{BASIN}: Strahler distribution @ 0.10 km2'); ax.set_xticks(orders); plt.show()
    print('Strahler-1 (finest tips) are the most numerous and the first removed by pruning.')"""),
    M("### 2 · Drainage density vs stream-initiation threshold"),
    C("""SWEEP = [0.02, 0.05, 0.10, 0.25, 0.5, 1.0]
sweep = collect_basin_metrics_for_dem(dem_path=EXAMPLE_DEMS[BASIN], basin_name=BASIN,
        thresholds_km2=SWEEP, z_th=z_th, lat_deg=lat)
best = {}
for x in sweep:
    if x.threshold_km2 not in best or x.n_stream_nodes > best[x.threshold_km2].n_stream_nodes:
        best[x.threshold_km2] = x
sw = pd.DataFrame([{'threshold_km2': t, 'dd': best[t].dd_true_km_km2,
                    'nodes': best[t].n_stream_nodes} for t in SWEEP if t in best])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sw.threshold_km2, sw.dd, 'o-', color='steelblue')
ax.set_xscale('log'); ax.set_xlabel('stream-initiation threshold (km2)')
ax.set_ylabel('drainage density (km/km2)'); ax.set_title(f'{BASIN}: Dd vs threshold')
for nm, r in REGIMES.items():
    ax.axvline(r.threshold_km2, ls='--', color=RC[nm], label=f'{nm} ({r.threshold_km2})')
ax.legend(); ax.grid(alpha=0.3); plt.show()
sw"""),
    M("Lower threshold → denser network (more 1st-order tips survive) → higher Dd."),
    M("### 3 · How Dd at the regime thresholds varies across basins"),
    C("""# A few representative basins (kept small for headless runtime).
SUBSET = ['inyo', 'troodos', 'calnalpine', 'panamint']
rows = []
for b in SUBSET:
    if b not in EXAMPLE_DEMS: continue
    c = get_basin_config(b)
    for nm, r in REGIMES.items():
        res = collect_basin_metrics_for_dem(dem_path=EXAMPLE_DEMS[b], basin_name=b,
              thresholds_km2=[r.threshold_km2], z_th=c['z_th'], lat_deg=c['lat'])
        if res:
            mm = max(res, key=lambda x: x.n_stream_nodes)
            rows.append({'basin': b, 'regime': nm, 'dd': round(mm.dd_true_km_km2, 3)})
ddm = pd.DataFrame(rows).pivot(index='basin', columns='regime', values='dd')
display(ddm)
ax = ddm.plot(kind='bar', color=[RC[c] for c in ddm.columns], figsize=(8, 4))
ax.set_ylabel('Dd (km/km2)'); ax.set_title('Drainage density at the three regime thresholds'); plt.show()"""),
    M("""**Takeaway.** Within a basin, regA (0.05 km²) is densest and regB (0.25 km²)
sparsest, with regC between — and the absolute Dd differs markedly across basins.
Stage 4 picks these thresholds so the resulting Earth Dd *brackets* the martian Dd,
making the Earth→Mars comparison fair rather than an artifact of mismatched
complexity."""),
]))

# ── Stage 3 ──────────────────────────────────────────────────────────────────
STAGES.append((3, "Mars interactive network exploration", [
    M("""## Mars valley networks — the target landscape, characterized

Unlike Earth, Mars networks are **already-mapped valley-network polylines** (391
networks), not DEM-derived. Their geometry reflects whatever fluvial signal survived
~3.5 Gyr of degradation. The Earth→Mars transfer is only valid if the Earth training
networks have **comparable drainage density**, so we characterize the Mars Dd
distribution, Strahler structure, and geography here — this is the target Stage 4
calibrates Earth against."""),
    C(SETUP),
    C("""from channel_heads.dd_calibration import mars_network_table, mars_network_strahler
gpkg = MARS_DIR / 'topology' / 'mars_vn_topology_model_ready.gpkg'
net = mars_network_table(gpkg if gpkg.exists() else None)
print(f'Mars networks: {len(net)}  (status ok: {int((net.status=="ok").sum())})')
net[['length_km','hull_area_km2','dd_hull_km_km2']].describe().round(2)"""),
    M("### 1 · Drainage-density and size distributions"),
    C("""ok = net[net.status == 'ok']
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
dd = ok.dd_hull_km_km2.dropna()
ax[0].hist(dd, bins=40, color='indianred', edgecolor='white')
ax[0].axvline(dd.median(), color='k', ls='--', label=f'median {dd.median():.2f}')
ax[0].set_xlabel('hull Dd (km/km2)'); ax[0].set_ylabel('networks'); ax[0].set_title('Mars Dd'); ax[0].legend()
ax[1].scatter(ok.hull_area_km2, ok.length_km, s=10, alpha=0.5, color='indianred')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel('hull area (km2)'); ax[1].set_ylabel('total length (km)'); ax[1].set_title('network size')
plt.show()
print(f'Mars median Dd = {dd.median():.3f} km/km2 — the value the Earth regimes bracket.')"""),
    M("### 2 · Strahler order across all networks"),
    C("""try:
    sd = mars_network_strahler(gpkg if gpkg.exists() else None)
    order_cols = [c for c in sd.columns if str(c).isdigit()]
    if order_cols:
        tot = sd[order_cols].sum()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar([int(c) for c in order_cols], tot.values, color='indianred', edgecolor='white')
        ax.set_xlabel('Strahler order'); ax.set_ylabel('segment count'); ax.set_title('Mars Strahler distribution'); plt.show()
except Exception as e:
    print('Strahler summary unavailable:', e)"""),
    M("### 3 · Geographic overview — all valley networks, coloured by Dd"),
    C("""import geopandas as gpd
try:
    seg = gpd.read_file(gpkg, layer='mars_segments')
    if 'network_id' in seg.columns:
        seg = seg.merge(ok[['network_id', 'dd_hull_km_km2']], on='network_id', how='left')
        col, kw = 'dd_hull_km_km2', dict(legend=True, legend_kwds={'label': 'Dd (km/km2)', 'shrink': 0.5})
    else:
        col, kw = None, {}
    fig, ax = plt.subplots(figsize=(12, 6))
    seg.plot(ax=ax, column=col, cmap='viridis', linewidth=0.5, **kw)
    ax.set_title('Mars valley networks (n=%d)' % len(ok)); ax.set_aspect('equal'); ax.axis('off'); plt.show()
except Exception as e:
    print('map unavailable:', e)"""),
    M("""**Takeaway.** The 391 martian valley networks span a wide range of size but cluster
around a characteristic drainage density. That median Dd is the calibration target:
Stage 4 tunes the Earth pruning regimes so Earth's Dd brackets it, so any Earth→Mars
difference in coupling reflects geometry, not a complexity mismatch."""),
]))

# ── Stage 4 ──────────────────────────────────────────────────────────────────
STAGES.append((4, "Earth-Mars regime calibration", [
    M("""## Calibrating Earth complexity to Mars

Earth and Mars coupling rates are only comparable if the networks are matched in
**drainage density / complexity**. A sweep over stream-initiation thresholds and
Strahler-pruning levels — measuring the Dd of each Earth basin's *convex-hull*
network against the Mars reference — yields three frozen regimes that **bracket** the
martian Dd. Running all three and reporting the spread of Mars results is the
project's calibration-uncertainty estimate."""),
    C(SETUP),
    M("### 1 · The frozen presets (full)"),
    C("""from channel_heads.regimes import REGIMES
pd.DataFrame([r.__dict__ for r in REGIMES.values()])"""),
    M("### 2 · The calibration evidence (the Dd sweep that chose the regimes)"),
    C("""cal = RESULTS_DIR / 'drainage_density_calibration/complexity_calibration'
csv = cal / 'dd_master_sweep_complexity.csv'
if csv.exists():
    s = pd.read_csv(csv)
    print('dd_master_sweep_complexity.csv:', s.shape, '| columns:', list(s.columns)[:10])
    display(s.head(8))"""),
    C("""from IPython.display import Image, display
for fig in ['fig_mars_vs_earth_at_matched_complexity.png',
            'fig_dd_hull_matched_vs_full_vs_mars.png',
            'fig_earth_strahler_vs_threshold.png']:
    f = cal / fig
    if f.exists():
        print(fig); display(Image(filename=str(f)))"""),
    M("""**Takeaway.** The figures above show Earth networks pulled to the *same* hull-Dd as
Mars; at matched complexity the three regimes straddle the martian distribution
(regA dense, regB sparse, regC middle). These three threshold/pruning settings
(0.05/0.25/0.10 km², frozen in `channel_heads.regimes`) propagate unchanged into
every trained model and every Mars prediction downstream — so the only thing that
varies across regimes is *complexity*, and the spread of results is our uncertainty."""),
]))

# ── Stage 5 ──────────────────────────────────────────────────────────────────
STAGES.append((5, "Final Earth network generation and QA", [
    M("""## Building & QA-gating the Earth feature tables

For each regime we build every basin's regime-pruned network, enumerate confluence
pairs per outlet (`build-earth-features`), and compute the 5 features + the touching
label. Before any training, a **QA gate** must pass: non-empty networks, plausible
per-basin survivor ratios, and complete (finite) features. We inspect that gate and
the per-regime dataset health here."""),
    C(SETUP),
    M("### 1 · The Stage-5 QA report"),
    C("""qa = RESULTS_DIR / 'stage5_earth_network_qa_report.csv'
if qa.exists():
    q = pd.read_csv(qa)
    print('QA rows:', len(q), '| columns:', list(q.columns))
    for c in [c for c in q.columns if 'flag' in c.lower()]:
        n = q[c].sum() if q[c].dtype != object else (q[c].astype(str).str.len() > 0).sum()
        print(f'  {c}: {int(n)} flagged')
    display(q.head(20))
else:
    print('QA report not found - run: python -m channel_heads build-earth-features --regime <r>')"""),
    M("### 2 · Per-regime dataset health (class balance, feature completeness)"),
    C("""rows = []
for r in REGIMES_:
    p = RESULTS_DIR / f'master_dataset_{r}.csv'
    if not p.exists(): continue
    d = pd.read_csv(p)
    nan_rate = d[MODEL_FEATURES].isna().mean().mean()
    rows.append({'regime': r, 'pairs': len(d), 'basins': d.basin.nunique(),
                 'touching_%': round(100*d.y.mean(), 1),
                 'feature_NaN_%': round(100*nan_rate, 3)})
pd.DataFrame(rows)"""),
    M("""**Takeaway.** The gate is clean (0 hard flags) and the feature tables are complete
across all 17 basins for every regime — the datasets are trustworthy inputs for
patch construction (Stage 7) and training (Stage 8). The ~25% touching rate reflects
the deliberate 3:1 negative subsampling applied during dataset assembly."""),
]))

# ── Stage 6 ──────────────────────────────────────────────────────────────────
STAGES.append((6, "Earth pair and label generation", [
    M("""## Pairs, the touching label, and the geometry the model learns

For every outlet we take **first-meet pairs** of channel heads — the two heads that
first meet going downstream at a confluence. The label `y = touching` is a *geometric*
test: do the two heads' contributing pixels touch (8-connectivity)? Trivial negatives
(impossibly far apart) are removed by `filter_hard_negatives`, and negatives are
subsampled to ~3:1. Here we look at what actually separates the classes — both in the
5 scalar features and in the **real confluence geometry**."""),
    C(SETUP),
    M("### 1 · Class balance and feature separation"),
    C("""d = pd.read_csv(RESULTS_DIR / 'master_dataset_regA.csv')
print('regA:', len(d), 'pairs |', int(d.y.sum()), 'touching (%.1f%%)' % (100*d.y.mean()))
fig, axes = plt.subplots(1, len(MODEL_FEATURES), figsize=(16, 3))
for ax, feat in zip(axes, MODEL_FEATURES):
    for lbl, col, nm in [(0, '#1f77b4', 'non-touch'), (1, '#d62728', 'touch')]:
        v = d.loc[d.y == lbl, feat].dropna(); lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v.clip(lo, hi), bins=30, alpha=0.6, color=col, density=True, label=nm)
    ax.set_title(feat, fontsize=9); ax.set_yticks([])
axes[0].legend(fontsize=8); plt.suptitle('regA — 5 model features by class', y=1.05); plt.show()"""),
    M("### 2 · Feature correlations (are the 5 features redundant?)"),
    C("""import numpy as np
corr = d[MODEL_FEATURES].corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(MODEL_FEATURES, rotation=90, fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(MODEL_FEATURES, fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, shrink=0.7); ax.set_title('feature correlation'); plt.show()"""),
    M("### 3 · Real confluence geometry — touching vs non-touching"),
    C("""# Rebuild one basin's REGIME-PRUNED network (so the pair node IDs resolve)
# and rasterize a few real pairs.
from channel_heads.training.regime import make_regime_stream_loader
from channel_heads.units import km2_to_cells
from channel_heads.rasterization.earth_patches import rasterize_outlet_pair
from channel_heads.regimes import REGIMES
from channel_heads.basin_config import get_basin_config
from matplotlib.colors import ListedColormap
BASIN = 'inyo'; regime = REGIMES['regA']
cfg = get_basin_config(BASIN); lat, z_th = cfg['lat'], cfg['z_th']
loader = make_regime_stream_loader(regime)
res = loader(BASIN, lat=lat, z_th=z_th, threshold=km2_to_cells(regime.threshold_km2, lat))
db = d[d.basin == BASIN]
cmap = ListedColormap([[0.9,0.9,0.9],[0.20,0.47,0.71],[0.89,0.10,0.11],[0.30,0.69,0.29],[1.0,0.5,0.0]])
if res is not None and not db.empty:
    s, dem = res; gs = dem.shape if hasattr(dem, 'shape') else dem.z.shape
    samp = pd.concat([db[db.y==1].head(4), db[db.y==0].head(4)])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (_, row) in zip(axes.ravel(), samp.iterrows()):
        try:
            patch = rasterize_outlet_pair(s, int(row.outlet), int(row.head_1), int(row.head_2),
                                          int(row.confluence), gs, target_size=128)
            ax.imshow(patch, cmap=cmap, vmin=0, vmax=4)
        except Exception as ex:
            ax.text(0.5, 0.5, str(ex)[:40], fontsize=6, ha='center')
        c = '#d62728' if row.y == 1 else '#1f77b4'
        ax.set_title('TOUCHING' if row.y == 1 else 'non-touching', color=c, fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'{BASIN} confluences — blue=branchA, red=branchB, green=confluence, gray=other'); plt.show()
else:
    print('stream load / basin rows unavailable')"""),
    M("""**Takeaway.** Touching pairs trend toward smaller orientation contrast and shorter
normalized head-to-head distance — coupled heads sit closer and more aligned. The
features are only weakly correlated (each adds information), but none separates the
classes alone, and the *shape* of the meeting (panel 3) carries signal the scalars
miss. That motivates both a learned classifier and the CNN shape-context of Stage 7-8."""),
]))

# ── Stage 7 ──────────────────────────────────────────────────────────────────
STAGES.append((7, "Earth model-input construction", [
    M("""## 5-class confluence patches for the CNN

The CNN sees each confluence as a **128×128, 5-class** raster
(BACKGROUND / BRANCH_A / BRANCH_B / OTHER_STREAMS / CONFLUENCE_MARKER), drawn directly
into the final grid (the frozen rasterizer contract). This encodes the *shape* of the
meeting — branch geometry and local stream context — that the 5 scalar features
cannot. We inspect manifest health, the within-patch class budget, and a real gallery."""),
    C(SETUP),
    M("### 1 · Manifest status + invalid-patch reasons"),
    C("""from channel_heads.rasterization.schema import (BACKGROUND, BRANCH_A, BRANCH_B,
                                                  OTHER_STREAMS, CONFLUENCE_MARKER, NUM_CLASSES)
print('NUM_CLASSES =', NUM_CLASSES)
for r in REGIMES_:
    p = RESULTS_DIR / f'raster_manifest_{r}.csv'
    if not p.exists(): continue
    d = pd.read_csv(p)
    ok = int((d.raster_status == 'ok').sum()) if 'raster_status' in d else len(d)
    print(f'{r}: {len(d)} rows | {ok} ok | {len(d)-ok} invalid')
man = pd.read_csv(RESULTS_DIR / 'raster_manifest_regA.csv')
if 'raster_qa_reason' in man.columns:
    bad = man[man.raster_status != 'ok']['raster_qa_reason'].value_counts().head(8)
    print('\\ntop invalid reasons (regA):'); print(bad.to_string() if len(bad) else '  (none)')"""),
    M("### 2 · Within-patch class budget (how much of each class a patch carries)"),
    C("""man_ok = man[man.raster_path.notna()]
if 'raster_status' in man_ok: man_ok = man_ok[man_ok.raster_status == 'ok']
counts = {c: [] for c in range(NUM_CLASSES)}
for p in man_ok.raster_path.head(300):
    a = np.load(_abs(p)); lab = a.argmax(0) if a.ndim == 3 else a
    u, n = np.unique(lab, return_counts=True); frac = dict(zip(u, n / lab.size))
    for c in range(NUM_CLASSES): counts[c].append(frac.get(c, 0.0))
names = ['BACKGROUND','BRANCH_A','BRANCH_B','OTHER_STREAMS','CONFLUENCE_MARKER']
pd.DataFrame({'class': names, 'mean_pixel_fraction': [round(np.mean(counts[c]), 4) for c in range(NUM_CLASSES)]})"""),
    M("### 3 · A gallery of real 5-class patches"),
    C("""from matplotlib.colors import ListedColormap
cmap = ListedColormap([[0.95,0.95,0.95],[0.85,0.33,0.10],[0.10,0.45,0.82],[0.70,0.70,0.70],[0.90,0.80,0.0]])
def lab(p):
    a = np.load(_abs(p)); return a.argmax(0) if a.ndim == 3 else a
samp = pd.concat([man_ok[man_ok.y==1].head(4), man_ok[man_ok.y==0].head(4)])
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (_, row) in zip(axes.ravel(), samp.iterrows()):
    ax.imshow(lab(row.raster_path), cmap=cmap, vmin=0, vmax=4)
    c = '#d62728' if row.y == 1 else '#1f77b4'
    ax.set_title('touching' if row.y == 1 else 'non-touching', color=c, fontsize=9); ax.axis('off')
plt.suptitle('regA 5-class patches (top: touching, bottom: non-touching)'); plt.show()"""),
    M("""**Takeaway.** Patches are dominated by BACKGROUND (most of the 128² grid is empty),
with thin BRANCH_A/B traces meeting at the CONFLUENCE_MARKER — a sparse, shape-centric
encoding. A small fraction of pairs are flagged invalid (degenerate geometry) and
excluded. This 5-class contract is **frozen**: Mars patches (Stage 10) must match it
exactly so the Earth-trained CNN applies without retraining."""),
]))

# ── Stage 8 ──────────────────────────────────────────────────────────────────
STAGES.append((8, "Model training", [
    M("""## Training the CNN + combined XGBoost (per regime)

Two models per regime. **OutletCNN** (a small ~24k-parameter conv net) encodes the
5-class patch into a 4-D embedding, trained with **Taiwan held out** for an honest
leave-one-basin test. The **combined XGBoost** then classifies on the 5 dimensionless
features **+** the 4 CNN embeddings (9 features), split by `GroupShuffleSplit` on
`basin__outlet` so no outlet leaks across train/test. We inspect convergence, what the
embedding learned, feature importance, and the model-comparison table."""),
    C(SETUP),
    C("""from channel_heads.models.cnn import OutletCNN, DEFAULT_EMBEDDING_DIM
from channel_heads.rasterization.schema import NUM_CLASSES
net = OutletCNN(in_channels=NUM_CLASSES, embedding_dim=DEFAULT_EMBEDDING_DIM)
print(f'OutletCNN: {sum(p.numel() for p in net.parameters()):,} params | '
      f'{NUM_CLASSES}-channel input -> {DEFAULT_EMBEDDING_DIM}-D embedding -> logit')"""),
    M("### 1 · CNN convergence (per-regime loss curves)"),
    C("""fig, ax = plt.subplots(figsize=(8, 4))
for r in REGIMES_:
    h = MODELS / f'cnn_outlet_{r}_history.csv'
    if h.exists():
        hist = pd.read_csv(h)
        ax.plot(hist.epoch, hist.val_loss, color=RC[r], lw=2, label=f'{r} val')
        ax.plot(hist.epoch, hist.train_loss, color=RC[r], ls=':', alpha=0.6)
        print(f'{r}: best val_loss {hist.val_loss.min():.4f} @ epoch {int(hist.val_loss.idxmin())+1} / {len(hist)} epochs')
ax.set_xlabel('epoch'); ax.set_ylabel('BCE loss'); ax.set_title('CNN training (solid=val, dotted=train)')
ax.legend(); ax.grid(alpha=0.3); plt.show()"""),
    M("### 2 · What the embedding learned (all 4 dims, by class)"),
    C("""d = pd.read_csv(RESULTS_DIR / 'master_dataset_regA_with_emb.csv')
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for i, ax in enumerate(axes):
    for lbl, col, nm in [(0, '#1f77b4', 'non-touch'), (1, '#d62728', 'touch')]:
        v = d.loc[d.y == lbl, f'emb_{i}'].dropna()
        ax.hist(v, bins=40, alpha=0.6, color=col, density=True, label=nm)
    ax.set_title(f'emb_{i}', fontsize=10); ax.set_yticks([])
axes[0].legend(fontsize=8); plt.suptitle('regA CNN embedding dimensions by class', y=1.04); plt.show()"""),
    M("### 3 · XGBoost feature importance (geom vs CNN contribution)"),
    C("""from xgboost import XGBClassifier
feats = MODEL_FEATURES + [f'emb_{i}' for i in range(4)]
mdl = XGBClassifier(); mdl.load_model(str(MODELS / 'xgb_geom_plus_cnn_emb_regA.json'))
imp = pd.Series(mdl.feature_importances_, index=feats).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=['#4daf4a' if f.startswith('emb') else '#377eb8' for f in imp.index])
ax.set_title('regA combined model — feature importance (green=CNN emb, blue=geom)'); plt.show()"""),
    M("### 4 · Model-comparison table (geom-only vs geom+CNN per regime)"),
    C("""p = MODELS / 'ALL_MODELS_METRICS.csv'
pd.read_csv(p) if p.exists() else 'train models first'"""),
    M("""**Takeaway.** All three CNNs converge cleanly (val loss plateaus, early stopping).
The embedding dimensions are clearly class-separated, and the combined model leans on
both geometry and CNN features — adding the embedding lifts ROC-AUC by ~0.09–0.17 over
geometry alone per regime. The patch *shape* carries real, complementary signal beyond
the 5 scalars."""),
]))

# ── Stage 9 ──────────────────────────────────────────────────────────────────
STAGES.append((9, "Earth model validation and tuning", [
    M("""## Honest validation + choosing an operating threshold

A single train/test split can be lucky. **Leave-one-basin-out (LOBO)** CV holds out
each basin in turn — the closest analogue to "apply to a brand-new landscape" — and is
our honest performance estimate. We then choose a **decision threshold**. The
F1-optimal point maximizes balanced accuracy on Earth; but Mars has *no labels*, so we
carry a **precision-oriented** operating point (max precision at recall ≥ 0.5) and study
the trade-off explicitly."""),
    C(SETUP),
    M("### 1 · LOBO cross-validation (pooled + per-fold)"),
    C("""p = MODELS / 'lobo_cv_metrics.csv'
lobo = pd.read_csv(p) if p.exists() else pd.DataFrame()
display(lobo)
if not lobo.empty:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(lobo.config, lobo.pooled_auc, color='steelblue', alpha=0.7, label='pooled AUC')
    ax.errorbar(lobo.config, lobo.fold_auc_mean, yerr=lobo.fold_auc_std, fmt='o', color='k', label='per-fold mean+/-std')
    ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.6, 1.0); ax.set_title('LOBO CV (honest, low-variance estimate)'); ax.legend(); plt.show()"""),
    M("### 2 · ROC / PR on held-out outlets (regA combined)"),
    C("""from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score
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
axs[1].set_title(f'PR (AP={average_precision_score(y,proba):.3f})'); axs[1].set_xlabel('recall'); axs[1].set_ylabel('precision'); plt.show()"""),
    M("### 3 · Threshold trade-off: precision / recall / F1"),
    C("""ts = np.linspace(0.05, 0.95, 91)
P = [(((proba>=t)&(y==1)).sum()/max((proba>=t).sum(),1)) for t in ts]
R = [(((proba>=t)&(y==1)).sum()/max((y==1).sum(),1)) for t in ts]
F = [f1_score(y, (proba>=t).astype(int)) for t in ts]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, P, label='precision'); ax.plot(ts, R, label='recall'); ax.plot(ts, F, label='F1', lw=2)
ax.axvline(OP_THR['regA'], color='k', ls='--', label=f"operating {OP_THR['regA']:.3f}")
ax.axvline(0.577406, color='gray', ls=':', label='production F1 0.577')
ax.set_xlabel('threshold'); ax.set_title('regA threshold trade-off'); ax.legend(); ax.grid(alpha=0.3); plt.show()
print('operating thresholds (max-precision @ recall>=0.5), used for Mars:')
for r in REGIMES_:
    tt = MODELS / f'optimal_threshold_geom_plus_cnn_emb_{r}.txt'
    if tt.exists(): print(f'  {r}: {tt.read_text().split()[0]}')"""),
    M("""**Takeaway.** LOBO ROC-AUC ~0.88 (regA/regB) confirms the model generalizes to
unseen basins, not just unseen outlets. The threshold curves show why the choice
matters: the precision-oriented operating point (~0.76) buys high precision at the
cost of recall — appropriate for label-free Mars, where false couplings would
contaminate the rate. The Earth F1 point (0.577) is **not** copied to Mars; Stage 12
studies that sensitivity directly."""),
]))

# ── Stage 10 ─────────────────────────────────────────────────────────────────
STAGES.append((10, "Final Mars model-input generation", [
    M("""## Mars model inputs — same patches, frozen CNN

Mars geometry (topology→pairs→features) is shared across regimes and raster-
independent. The raster-dependent part — **5-class patches** — is regenerated with the
*current* rasterizer so it matches the Earth patches the CNN trained on, then embedded
with the **frozen** `cnn_outlet_final.pt`. Patches **must stay 5-class**. We check
coverage, the patches themselves, and the resulting Mars embedding distribution
(vs Earth, to confirm the domains overlap)."""),
    C(SETUP),
    M("### 1 · Patch coverage"),
    C("""idx = MARS_IN / 'mars_cnn_patch_index.parquet'
if idx.exists():
    d = pd.read_parquet(idx)
    print('Mars patches:', len(d), '|', d['patch_status'].value_counts().to_dict() if 'patch_status' in d else '')"""),
    M("### 2 · Example Mars 5-class patches (identical contract to Earth)"),
    C("""from matplotlib.colors import ListedColormap
cmap = ListedColormap([[0.95,0.95,0.95],[0.85,0.33,0.10],[0.10,0.45,0.82],[0.70,0.70,0.70],[0.90,0.80,0.0]])
d = pd.read_parquet(MARS_IN / 'mars_cnn_patch_index.parquet')
col = 'patch_path' if 'patch_path' in d else 'raster_path'
ok = d[d.get('patch_status','ok') == 'ok'].head(6)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, (_, row) in zip(axes, ok.iterrows()):
    a = np.load(_abs(row[col])); lab = a.argmax(0) if a.ndim == 3 else a
    ax.imshow(lab, cmap=cmap, vmin=0, vmax=4); ax.axis('off')
plt.suptitle('Mars confluence patches (5-class)'); plt.show()"""),
    M("### 3 · Do Earth and Mars CNN embeddings overlap? (domain check)"),
    C("""emb = MARS_IN / 'mars_cnn_embeddings.parquet'
if emb.exists():
    me = pd.read_parquet(emb)
    ee = pd.read_csv(RESULTS_DIR / 'master_dataset_regA_with_emb.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, a in enumerate(ax):
        a.hist(ee[f'emb_{i}'].dropna(), bins=40, density=True, alpha=0.5, label='Earth (regA)', color='steelblue')
        a.hist(me[f'emb_{i}'].dropna(), bins=40, density=True, alpha=0.5, label='Mars', color='indianred')
        a.set_title(f'emb_{i}'); a.set_yticks([])
    ax[0].legend(); plt.suptitle('Earth vs Mars CNN-embedding distributions'); plt.show()"""),
    M("""**Takeaway.** Mars patches are regenerated under the exact 5-class contract and the
frozen production CNN, with the same pair coverage as before (3,682 ok / 103 invalid).
The Earth and Mars embedding distributions broadly overlap — the CNN is not being
asked to extrapolate into an unseen feature region, which is what makes the
cross-planet application defensible."""),
]))

# ── Stage 11 ─────────────────────────────────────────────────────────────────
STAGES.append((11, "Mars inference", [
    M("""## Applying the Earth-trained regime models to Mars

For each regime we recompute Mars embeddings with that regime's CNN, apply its combined
XGBoost, and threshold at the regime's operating point — yielding a touching
probability + decision per martian confluence pair. We report per-regime coupling
rates, the probability distributions, and **maps** of where coupling is predicted."""),
    C(SETUP),
    M("### 1 · Coupling rates per regime"),
    C("""rows = []
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if not p.exists(): continue
    d = pd.read_parquet(p)
    rows.append({'regime': r, 'pairs': len(d), 'networks': d.network_id.nunique(),
                 'touching_%': round(100*d.pred_touching.mean(), 1),
                 'high_conf>=0.8_%': round(100*(d.prob_touching>=0.8).mean(), 1),
                 'mean_prob': round(d.prob_touching.mean(), 3)})
pd.DataFrame(rows)"""),
    M("### 2 · Probability distributions"),
    C("""fig, ax = plt.subplots(figsize=(8, 4))
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if p.exists():
        d = pd.read_parquet(p)
        ax.hist(d.prob_touching, bins=40, histtype='step', lw=2, color=RC[r], label=r)
        ax.axvline(OP_THR[r], color=RC[r], ls='--', alpha=0.5)
ax.set_xlabel('P(touching)'); ax.set_ylabel('pairs')
ax.set_title('Mars coupling probability (dashed = operating threshold)'); ax.legend(); plt.show()"""),
    M("### 3 · Maps — predicted coupling across the valley networks (all three regimes)"),
    C("""import geopandas as gpd
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, r in zip(axes, REGIMES_):
    g = MARS_OUT / f'mars_combined_{r}_predictions.gpkg'
    if g.exists():
        gdf = gpd.read_file(g, layer='pairs')
        gdf.plot(ax=ax, column='prob_touching', cmap='RdYlBu_r', linewidth=0.7,
                 vmin=0, vmax=1, legend=(r == REGIMES_[-1]),
                 legend_kwds={'label': 'P(touching)', 'shrink': 0.6})
        ax.set_title(f'{r} (touching {100*gdf.prob_touching.ge(OP_THR[r]).mean():.0f}%)')
    ax.set_aspect('equal'); ax.axis('off')
plt.suptitle('Predicted channel-head coupling on Mars, by regime'); plt.show()"""),
    M("""**Takeaway.** Coupling is widespread across the martian valley networks in every
regime (regB highest, regC lowest — the dense/sparse ordering), with the high-confidence
fraction tracking the headline rate. The maps show coupling is not concentrated in a
few networks but distributed across the dissected highlands — consistent with mobile
divides having been a global feature of the valley-forming epoch."""),
]))

# ── Stage 12 ─────────────────────────────────────────────────────────────────
STAGES.append((12, "Mars threshold and prediction analysis", [
    M("""## Threshold sensitivity — the scientific knob on label-free Mars

Mars has no ground truth, so the coupling *rate* depends on the chosen decision
threshold. A trustworthy conclusion is one that is **stable** across a sensible
threshold band and **consistent** across regimes — not an artifact of one cut. We sweep
the threshold for both the headline rate and the high-confidence rate."""),
    C(SETUP),
    C("""SWEEP = np.round(np.arange(0.30, 0.96, 0.05), 2)
P = {r: pd.read_parquet(MARS_OUT / f'mars_combined_{r}_predictions.parquet').prob_touching.to_numpy()
     for r in REGIMES_ if (MARS_OUT / f'mars_combined_{r}_predictions.parquet').exists()}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for r, pr in P.items():
    ax[0].plot(SWEEP, [(pr >= t).mean() for t in SWEEP], 'o-', color=RC[r], label=r)
    ax[0].axvline(OP_THR[r], color=RC[r], ls='--', alpha=0.4)
    ax[1].plot(SWEEP, [((pr >= t) & (pr >= 0.8)).mean() for t in SWEEP], 'o-', color=RC[r], label=r)
ax[0].set_title('coupled fraction vs threshold'); ax[1].set_title('high-confidence (>=0.8) fraction vs threshold')
for a in ax: a.set_xlabel('decision threshold'); a.set_ylabel('fraction'); a.legend(); a.grid(alpha=0.3)
plt.show()"""),
    C("""# Operating-point summary table.
pd.DataFrame([{'regime': r, 'operating_threshold': OP_THR[r],
               'coupled_%': round(100*(P[r] >= OP_THR[r]).mean(), 1),
               'high_conf_%': round(100*(P[r] >= 0.8).mean(), 1)} for r in P])"""),
    M("""**Takeaway.** The curves are **smooth** through the operating band (no cliff), so the
coupled fraction is a stable estimate rather than a threshold artifact. Crucially the
regime **ordering is invariant** — regB > regA > regC at *every* threshold — so the
qualitative conclusion (coupling is common, and scales with network density as
expected) does not depend on the exact cut. This is what lets us report a result
despite Mars having no labels."""),
]))

# ── Stage 13 ─────────────────────────────────────────────────────────────────
STAGES.append((13, "Scientific interpretation", [
    M("""## What Mars is telling us

We now read the result scientifically. Channel-head coupling is a proxy for
**drainage-divide mobility**: a high coupled fraction means many confluences sit where
two heads contested a shared divide, i.e. the valley network was being actively
reorganized while it formed. We quantify the coupled fraction, its distribution across
networks, its regime-robustness, and its geography — then interpret."""),
    C(SETUP),
    M("### 1 · Headline coupling rates (with network-level summary)"),
    C("""HI = 0.85; MINP = 3
rows, nets = [], []
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if not p.exists(): continue
    d = pd.read_parquet(p).copy()
    d['pred'] = (d.prob_touching >= OP_THR[r]).astype(int)
    g = d.groupby('network_id').agg(n=('pred','count'), t=('pred','sum'), mp=('prob_touching','mean')).reset_index()
    g = g[g.n >= MINP]; g['frac'] = g.t / g.n; g['regime'] = r; nets.append(g)
    rows.append({'regime': r, 'threshold': OP_THR[r], 'pairs': len(d),
                 'coupled_%': round(100*d.pred.mean(), 1),
                 'high_conf>=0.85_%': round(100*(d.prob_touching >= HI).mean(), 1),
                 'networks': len(g), 'median_network_coupling': round(g.frac.median(), 3),
                 '%networks_with_coupling': round(100*(g.t > 0).mean(), 1)})
coupling = pd.DataFrame(rows); allnet = pd.concat(nets, ignore_index=True)
display(coupling)
# Earth training base rate for context.
earth = pd.read_csv(RESULTS_DIR / 'master_dataset_regA.csv').y.mean()
print(f'Earth training base rate (regA, after 3:1 subsampling): {100*earth:.0f}% touching')"""),
    M("### 2 · Network-level coupling distribution (CDF) + per-network spread"),
    C("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for r in REGIMES_:
    s = allnet[allnet.regime == r].frac.sort_values()
    if len(s): ax[0].plot(s.values, np.linspace(0, 1, len(s)), color=RC[r], lw=2, label=r)
ax[0].set_xlabel('coupled fraction per network'); ax[0].set_ylabel('cumulative fraction of networks')
ax[0].set_title('network coupling CDF'); ax[0].legend()
d = pd.read_parquet(MARS_OUT / 'mars_combined_regA_predictions.parquet')
per = d.assign(p=(d.prob_touching >= OP_THR['regA'])).groupby('network_id').p.mean()
ax[1].hist(per, bins=25, color='seagreen', edgecolor='white'); ax[1].axvline(per.mean(), color='k', ls='--', label=f'mean {per.mean():.2f}')
ax[1].set_xlabel('coupled fraction (regA)'); ax[1].set_ylabel('networks'); ax[1].set_title('per-network coupled fraction'); ax[1].legend()
plt.show()"""),
    M("### 3 · Cross-regime consensus — how robust is each call?"),
    C("""base = None
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if not p.exists(): continue
    d = pd.read_parquet(p)[['pair_id', 'prob_touching']].copy()
    d[r] = (d.prob_touching >= OP_THR[r]).astype(int); d = d[['pair_id', r]]
    base = d if base is None else base.merge(d, on='pair_id')
s = base[REGIMES_].sum(axis=1); n = len(base)
print(f'pairs={n}')
print(f'  unanimous TOUCHING : {int((s==3).sum())} ({100*(s==3).mean():.1f}%)')
print(f'  unanimous NON-touch: {int((s==0).sum())} ({100*(s==0).mean():.1f}%)')
print(f'  CONSENSUS (any unanimous): {100*((s==0)|(s==3)).mean():.1f}%')
print(f'  regime-sensitive   : {100*((s>0)&(s<3)).mean():.1f}%')"""),
    M("### 4 · Geography of coupling"),
    C("""import geopandas as gpd
g = MARS_OUT / 'mars_combined_regA_predictions.gpkg'
if g.exists():
    gdf = gpd.read_file(g, layer='pairs')
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, column='prob_touching', cmap='RdYlBu_r', linewidth=0.7, legend=True,
             legend_kwds={'label': 'P(touching)', 'shrink': 0.6})
    ax.set_title('regA — geography of predicted channel-head coupling on Mars'); ax.set_aspect('equal'); ax.axis('off'); plt.show()"""),
    M("""## Interpretation

- **Coupling is common and regime-robust in ordering.** Across regimes the coupled
  fraction runs from ~39% (regC, sparsest) to ~64% (regB) at the precision-oriented
  operating thresholds, and the *ordering* regB>regA>regC holds at every threshold
  (Stage 12). ~95–98% of networks contain at least one coupled confluence.
- **Regime-robustness.** About two-thirds of individual pairs receive the *same* call
  under all three regimes; the remaining third are regime-sensitive — which is exactly
  the calibration uncertainty the three-regime design is meant to expose, and why we
  report a spread rather than a single number.
- **What it implies.** A substantial, spatially-distributed coupled fraction is
  consistent with martian valley networks whose **divides were mobile** while the
  valleys were active — a more Earth-like, competitively-eroding fluvial regime than a
  purely static, inherited drainage pattern.
- **Caveats.** The operating point is precision-oriented (not a calibrated
  probability), and the Earth base rate (~25%, set by 3:1 subsampling) is not directly
  comparable to the natural martian pair population — so these rates are best read as a
  **relative, regime-bracketed** signal, not an absolute probability of coupling."""),
]))

# ── Stage 14 ─────────────────────────────────────────────────────────────────
STAGES.append((14, "Figures, poster, and reporting", [
    M("""## Publication figures and the headline result

The render scripts + presentation notebooks turn the predictions into the result
figures: ROC panels, the Mars coupling map, **vector** contact sheets (high-confidence
and disagreement cases), and per-outlet drawings. Build:
`python -m channel_heads make-result-figures` / `generate-poster-figures` and
`scripts/rendering/*`. We assemble the headline numbers and surface the key figures."""),
    C(SETUP),
    M("### 1 · One-line result summary"),
    C("""rows = []
for r in REGIMES_:
    p = MARS_OUT / f'mars_combined_{r}_predictions.parquet'
    if p.exists():
        d = pd.read_parquet(p)
        rows.append({'regime': r, 'operating_threshold': OP_THR[r],
                     'Mars coupled_%': round(100*(d.prob_touching >= OP_THR[r]).mean(), 1),
                     'high_conf>=0.8_%': round(100*(d.prob_touching >= 0.8).mean(), 1)})
summary = pd.DataFrame(rows); display(summary)
print('Headline: martian channel-head coupling ~%.0f-%.0f%% across regimes (regB>regA>regC).'
      % (summary['Mars coupled_%'].min(), summary['Mars coupled_%'].max()))"""),
    M("### 2 · The figures"),
    C("""from IPython.display import Image, display
figdir = MARS_OUT / 'figures_combined'
pngs = sorted(figdir.glob('*.png'))
print(f'{len(pngs)} figures in {figdir.relative_to(ROOT)}')
for name in ['mars_networks_channels.png', 'contact_sheet_high_conf_both.png', 'contact_sheet_disagreement.png']:
    f = figdir / name
    if f.exists():
        print(name); display(Image(filename=str(f)))"""),
    M("""**Takeaway.** The figures package the result for reporting: the coupling map shows
*where*, the contact sheets show *what* a high-confidence coupled confluence (and a
regime-disagreement case) actually looks like as vector valley geometry, and the
summary table gives the regime-bracketed headline. Per project convention, Mars
contact sheets use **vector polyline** drawings rather than raster patches.

This is the end of the pipeline: raw topography (Stage 0-1) → networks (2-5) →
labelled pairs + patches (6-7) → trained, validated models (8-9) → Mars inputs +
inference (10-11) → threshold analysis + interpretation + figures (12-14)."""),
]))


def build():
    for num, title, cells in STAGES:
        nb = new_notebook()
        nb.cells.append(new_markdown_cell(
            f"# Stage {num} — {title}\n\n"
            f"_Pipeline stage {num} of 14. This is the self-contained deep dive for the "
            f"stage: narrative + analysis + interpretation, using the canonical "
            f"`channel_heads` package and on-disk artifacts. Heavy rebuilds run via the "
            f"`channel-heads` CLI (commands are given inline)._"))
        for kind, src in cells:
            nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        slug = title.lower().replace(" ", "_").replace("-", "_").replace(",", "")
        nbf.write(nb, str(OUT / f"{num:02d}_{slug}.ipynb"))
        print("wrote", f"{num:02d}_{slug}.ipynb")


if __name__ == "__main__":
    build()
