# Notebooks

Notebooks are a primary interface: they explain the science, visualise, and
support decisions. Each lives at `notebooks/<role>/` (depth 2, required by the
in-notebook root-resolution cell) and calls `channel_heads.*` — no duplicated
logic. Types: **pipeline** · **decision-support** · **presentation** ·
**exploration** · **archive**.

## By role

### `mars/` — Mars cross-planet (pipeline)
| Notebook | Type | Purpose |
|----------|------|---------|
| `02_first_meet_pairs` | pipeline | First-meet pair extraction (`channel_heads.mars.pairs` / `pairing`). |
| `03_pair_features` | pipeline | 5-feature table build (`features`). |
| `04_xgb_inference_5feat` | pipeline | Production XGBoost inference (`models.xgboost`). |
| `dd_hull_mars_vs_earth_complexity` | decision-support | Earth vs Mars drainage-density / complexity. |

### `regime/` — drainage-density calibration (decision-support)
| Notebook | Type | Purpose |
|----------|------|---------|
| `00_calibration_overview` | decision-support | Regime/pruning calibration overview. |
| `01_mars_inference` | pipeline | Regime Mars inference (`models.embeddings`, `inference.regime`). |
| `02_threshold_retune` | decision-support | Re-tune a regime operating threshold (`models.thresholds`). |

### `diagnostics/` — QA / investigative
| Notebook | Type | Purpose |
|----------|------|---------|
| `dd_threshold_calibration` | decision-support | Stream-threshold vs Mars Dd calibration. |
| `earth_network_pruning_experiments` | exploration | Source of `channel_heads/pruning.py`. |
| `lobo_cv` | decision-support | Leave-one-basin-out CV (`eval`). |
| `regB_threshold` | decision-support | regB threshold diagnostic. |
| `rasterization_diagnostics` | QA | Patch rasterization sanity checks. |
| `stream_crossing_qa` | QA | Stream-crossing-filter QA (`pairing`, `viz`). |

### `presentation/` — poster / report figures
| Notebook | Type | Purpose |
|----------|------|---------|
| `result_figures` | presentation | Model-comparison / result figures. |
| `mars_contact_sheets` | presentation | Mars contact sheets (vector polylines). |
| `per_outlet_touching_pairs` | presentation | Per-outlet touching-pair figures. |
| `simple_mars_earth_dd_presentation` | presentation | Earth vs Mars Dd presentation. |

### `training/` — Earth training (pipeline)
`00_full_pipeline` → `05_cnn_quick_eval`: dataset build → train classifier →
feature engineering → CNN embeddings → quick eval. Back this with
`pipelines.train_earth_models()`.

### `analysis/` — Earth basin analysis (exploration)
`01_single_basin_test` → `04_all_basins_full` (use `channel_heads.plotting_utils`).

### `archive/` — superseded (do not maintain)
`experiment_250th/350th/500th`, `experiment_template`, `optimization_review.md`.

## Canonical notebooks (rebuild target)

The five that should be kept clean, explanatory, and decision-oriented (markdown
motivation → input/output paths → visual QA → decision points → exported
figures → conclusion):

1. **Mars full pipeline** — `mars/` end-to-end via `channel_heads.pipelines`.
2. **Model comparison / threshold decision** — `regime/02_threshold_retune` +
   `models.comparison` (see [modeling.md](modeling.md) on the Mars threshold).
3. **Drainage-density / pruning decision** — `diagnostics/dd_threshold_calibration`.
4. **Poster / report figures** — `presentation/result_figures` (+
   `pipelines.generate_poster_figures`).
5. **Diagnostics / QA** — `diagnostics/stream_crossing_qa`.

> Rebuilding these to the canonical template (and archiving the rest) is tracked
> as remaining work — see the report / `docs/architecture.md` backlog.
