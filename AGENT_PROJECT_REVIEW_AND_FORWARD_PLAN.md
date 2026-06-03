# AGENT_PROJECT_REVIEW_AND_FORWARD_PLAN.md

_Strategic project review + forward-wave planning audit._
_Date: 2026-06-04 · Branch: `refactor/package-first-architecture` · Working tree: clean (before this report)._
_Scope: review, planning, and targeted documentation correction only. No code, notebooks, data, or models were modified. No training, inference, or regeneration was run._

---

## 1. Executive summary

- **Is the repo structurally stable?** **Yes.** The package-first refactor is genuinely complete. `channel_heads/` owns all pipeline/model/raster/training logic; `scripts/cli/` are thin wrappers; notebooks are foldered by stage and call `channel_heads.*`. The public API is cleanly re-exported from `__init__.py`. This is the strongest part of the project.
- **Is the scientific pipeline clear?** **Yes, conceptually.** The 15-stage design (`docs/PIPELINE_DESIGN.md`) is coherent, and `STAGE_ASSET_MAP.md` maps each stage to assets. The Earth→Mars transfer logic and the regime-calibration rationale are well documented.
- **What is complete?** Stages 0, 4, 5, 6 (Earth network + pairs + labels), 12, 13 (notebooks). Package architecture, docs hub, test suite (~37 test files mapping 1:1 to modules), regime calibration (frozen presets + rationale).
- **What is partial?** Stages 1–3 (QA/explorer notebooks exist but are exploratory). Stage 14 (`data/results/final_figures/` is empty; presentation notebooks exist). Stages 8–11 carry **stale** artifacts (built on pre-rewrite data) that are usable for structural testing only.
- **Single biggest blocker:** **The Stage 7 raster chain is in an interrupted, internally-inconsistent state, and no final regime is chosen.** All three `raster_manifest_reg{A,B,C}.csv` files index `_rasters_reg*/` directories that are no longer on disk in `data/results/` (archived 2026-06-03), and a regA regen was started and stopped mid-run, leaving its output fragmented across three locations. Because every trained model and Mars prediction depends on this chain, nothing downstream can be trusted as "final" until Stage 7 is reconciled. **Per project policy this is deferred, not the immediate next action** — but it is the gate to any final scientific result.
- **What should the next broad wave focus on?** **Clarity and reconciliation before regeneration** (matches the stated project direction). Specifically: document/reconcile the interrupted Stage 7 state, finish the notebook QA pass, lock down "which artifacts are trustworthy," and stage a *clean* regA rebuild plan — **without** mass retraining yet. See §12.

---

## 2. Current repository structure review

| Area | Current role | Role clear? | Overlap? | Stable? | Touch next wave? |
|------|--------------|-------------|----------|---------|-------------------|
| `channel_heads/` | All importable logic (pipeline, models, rasterization, features, pairing, training, viz, io, eval, mars) | **Yes** | Minor curated-surface overlaps (see §3) | **Stable** | No (docs only) |
| `scripts/cli/` | Thin CLI wrappers over package functions | **Yes** | Wrappers intentionally mirror package API | Stable | Light: doc 2 unverified wrappers (§4) |
| `notebooks/` | Stage-foldered notebooks (analysis/mars/regime/training/diagnostics/presentation/interpretation/archive) | **Yes** | None | Stable | **Yes** — QA pass + order doc (§5) |
| `docs/` | Documentation hub (13 docs) | Mostly | Some redundancy (§6) | Stable, a few stale spots | **Yes** — targeted corrections (done; §14) |
| `tests/` | pytest suite, 1:1 to modules (~37 files, `.coverage` present) | **Yes** | None | Stable | Optional: run as a gate, don't expand |
| `data/` | Inputs + generated outputs (gitignored) | Partly | **Interrupted Stage 7 state** (§8) | **Not stable** in raster chain | **Yes** — reconcile/clarify only |
| `models/` | Trained artifacts (production frozen + regime variants) | **Yes** | None | Stable but several **stale** (§9) | No (do not retrain yet) |

**Bottom line:** code/test/doc/notebook structure is stable and does not need refactoring. The only unstable area is **generated data in the raster chain** (`data/results/_rasters_*`, `raster_manifest_*`), which needs *reconciliation*, not refactoring.

---

## 3. `channel_heads/` package ownership review

| Module / subpackage | Responsibility | Stages | Ownership clear? | Overlap / duplication | Recommended action |
|---|---|---|---|---|---|
| `io/` (`paths`, `tables`, `geopackage`, `cleanup`) | Canonical paths + read/write + safe cleanup | 0, all | **Clear** | `io.paths` mirrors legacy `config.py` (intentional shim) | **Keep** |
| `regimes.py` | `Regime` dataclass + frozen `regA/B/C` presets | 4 | **Clear** | — | **Keep** (frozen — do not touch) |
| `units.py` | Single source of truth for unit conversions | all | **Clear** | — | **Keep** |
| `dd_calibration.py` (64 KB) | Drainage-density / threshold calibration | 2, 4 | Clear | Large; some overlap with `viz/calibration.py` | **Keep**; clarify docs only |
| `pruning.py` | Strahler-strip + order-gap pruning | 4, 5 | Clear | — | **Keep** |
| `coupling_analysis.py` (22 KB) | `CouplingAnalyzer` (touch detection, mask cache) | 6 | Clear | — | **Keep** (perf item S-perf in roadmap) |
| `features/` (`asymmetry`, `geometry`, `paths`, `earth_*`, `mars_features`) | Dimensionless feature math, Earth + Mars | 5, 6, 7, 10 | **Clear** | `earth_geometry`/`geometry` + `earth_paths`/`paths` look parallel | Keep; **clarify docs** on earth_* vs shared split |
| `pairing/` (`dag`, `earth`, `filtering`, `mars_graph`) | Graph-agnostic first-meet core + Earth/Mars adapters | 6, 10 | **Clear** | Shared core is the de-dup target (done) | **Keep** |
| `rasterization/` (`patches`, `earth_patches`, `earth_batch`, `mars_patches`, `drawing`, `manifest`, `schema`) | 5-class patch generation + manifest + drawing | 7, 10 | **Clear** | Curated surface over old `rasterizer.py` | **Keep** |
| `models/` (`xgboost`, `cnn`, `cnn_features`, `embeddings`, `comparison`, `thresholds`, `regime`, `mars_combined`, `mars_inference`, `device`) | Model load/train-glue/inference/compare | 8, 9, 11 | Mostly | **`models` vs `training` vs `inference`** — see note | Keep; **clarify docs** on the 3-way split |
| `training/` (`cnn`, `xgboost`, `datasets`, `labeling`, `regime`) | Earth training entry points + dataset assembly + labeling | 6, 8 | Mostly | `training/xgboost.py` vs `models/xgboost.py`; `training/cnn.py` vs `models/cnn.py` | **Clarify docs** (real confusion risk — see note) |
| `inference/` | Thin re-export shim over `models.*` | 11 | Clear (declared compat shim) | Pure re-export | **Keep** as-is (documented compat surface) |
| `eval/` (`metrics`, `lobo`, `splitting`, `diagnostics`) | Thresholding, metrics, grouped/LOBO splits | 9 | **Clear** | — | **Keep** |
| `mars/` (`topology`, `pairs`) | Mars graph build + first-meet pairs | 10 | **Clear** | Shares `pairing` core (by design) | **Keep** |
| `pipelines/` (`earth`, `mars`, `poster`) | Readable top layer, one function per stage | all | **Clear** | Orchestrates the above; `scripts/cli` overlap is intentional | **Keep** (this is the front door) |
| `viz/` (`earth`, `contact_sheet`, `curves`, `per_outlet`, `stream_crossing`, `calibration`) | Vector figures + DEM/basin plotting | 1–3, 14 | **Clear** | `rasterization` makes pixels, `viz` makes vector figures — distinct | **Keep** |
| `basin_config.py` | Per-basin params (Goren & Shelef 2024) | 0, 1 | Clear | — | **Keep** |
| `geometric_analysis.py` | Re-export shim (asymmetry+geometry+labeling) | — | Clear (declared shim) | Pure re-export | **Keep**; roadmap W1 (split source) is *later/optional* |
| `cli.py`, `logging_config.py`, `stream_utils.py` | `ch-analyze` CLI / logging / helpers | 0 | Clear | — | **Keep** |

**Targeted overlap notes (document, do not move):**
- **`training/` vs `models/`** is the one place a new reader can get confused: both have `cnn.py` and `xgboost.py`. The intended split is *`training/*` = fit + persist; `models/*` = load + predict + compare*. This is correct but **not stated anywhere** — add one sentence to `docs/architecture.md` or `DEVELOPER_GUIDE.md`. **Do not merge or move** — risk is high, benefit low.
- **`models/` vs `inference/`:** `inference/` is explicitly a re-export shim (per `AGENT_STATE.md`). Fine — keep, leave the one-line note that already exists.
- **`features/earth_geometry.py` vs `features/geometry.py`** (and `earth_paths` vs `paths`): looks like duplication but is the Earth-specific-vs-shared split. Confirm in a docstring sentence; no move.
- **`pipelines/` vs `scripts/cli/`:** intentional (library vs command surface). No action.

**No file moves are recommended.** Every potential overlap is either an intentional curated surface or an Earth/shared split; the cost of moving (path-coupled scripts, test imports, notebook root-resolution) exceeds the clarity benefit. Prefer the documentation sentences above.

---

## 4. `scripts/cli/` command-surface review

| Script | What it does | Class | Package module | Stage | Safe now? | Recommendation |
|---|---|---|---|---|---|---|
| `run_mars_pipeline.py` | Mars stage/all runner | production | `pipelines.run_full_mars_pipeline` + per-stage | 10–11 | ⚠️ produces stale-chain outputs if rerun against stale rasters | Keep; primary Mars entry |
| `run_mars_inference.py` | Combined Mars inference | production | `pipelines.run_mars_combined_inference` | 11 | ⚠️ depends on embeddings | Keep |
| `run_mars_combined_regime.py` | Per-regime Mars inference | production | `models.regime` + `models.xgboost` | 11 | ⚠️ stale regime models | Keep |
| `build_earth_features_regime.py` | Per-basin Earth features (regime) | production | `training.regime.build_regime_feature_dataset` | 5/7 | ✅ tabular, raster-independent | Keep |
| `build_cnn_patches_regime.py` | Regime CNN patches | production | `training.regime.build_regime_patch_dataset` | 7 | ⚠️ **this is the interrupted regA step** | Keep; use for clean regen |
| `train_cnn_baseline.py` | Train production CNN | production | `training.cnn.train_cnn` | 8 | model-producing — do not run this wave | Keep |
| `train_cnn_regime.py` | Train per-regime CNN | production | `training.cnn` + regime helpers | 8 | model-producing | Keep |
| `train_cnn_multiseed.py` | Multi-seed CNN training | production/diagnostic | `training.cnn` + split helpers | 8/9 | model-producing | Keep |
| `train_combined_xgb_phase6b.py` | Train 3 Earth XGB variants | production | `training.xgboost` | 8 | model-producing | Keep |
| `train_combined_xgb_regime.py` | Train per-regime geom+emb XGB | production | `training.xgboost` + regime paths | 8 | model-producing | Keep |
| `eval_lobo_cv.py` | LOBO cross-validation | diagnostic | `eval.lobo` | 9 | ✅ read/eval | Keep |
| `retune_threshold_regime.py` | Re-tune regime threshold | diagnostic | `eval` + `models.xgboost` | 9/12 | ✅ | Keep |
| `make_result_figures.py` | Batch result figures | diagnostic | `viz` + `eval` | 14 | ✅ (writes REPORT figs) | Keep |
| `generate_poster_figures.py` | Poster figures | production | `pipelines.generate_poster_figures` | 14 | ✅ | Keep |
| `run_mars_pipeline.py`/wrappers | (see above) | — | — | — | — | — |

**Unverified surfaces (documentation says they exist; confirm before relying):** `scripts/README.md` references `cli/run_mars_inference.py` and `cli/run_mars_combined_regime.py` against `channel_heads.pipelines.run_mars_combined_inference` / `models.regime`. The files are present; **the exact package call paths were not executed in this review** — flag as *verify on next use*, not as broken.

**`scripts/*.sh`:**
- `run_full_rebuild.sh` — one-shot baseline+regime orchestrator. **Current but path-coupled.** Use only after Stage 7 readiness; **not this wave** (it triggers retraining).
- `run_regime_pipeline.sh` — runs Steps 2→6 for **one** regime per call, and **only accepts `regA`/`regB`** (exits on anything else). regC must be run via per-step CLI. (Doc corrected — see §14.) Use only after readiness.
- `clean-cache.sh`, `setup-hooks.sh` — maintenance; current; safe; path-stable. Fine to use now.

**Verdict:** the CLI surface is coherent and well-mapped to the package. No renames needed now. A future `ch-*` console-entry-point family (wrapping these) is a *nice-to-have*, not required.

---

## 5. Notebook system review

| Path | Stage | Role | Run now? | Future-facing / stale | Key I/O |
|---|---|---|---|---|---|
| `analysis/01_earth_source_data_qa.ipynb` | 1 | QA gate | ✅ safe (read) | future-facing | DEMs → CRS/z-range checks |
| `analysis/02_earth_network_explorer.ipynb` | 2 | exploration | ✅ | future-facing | DEM → threshold/pruning preview |
| `analysis/05_earth_network_qa.ipynb` | 5 | QA gate | ✅ | future-facing | networks → `stage5_earth_network_qa_report.csv` |
| `mars/00_mars_network_explorer.ipynb` | 3 | exploration | ✅ | future-facing | valleys+MOLA → maps |
| `mars/02_first_meet_pairs.ipynb` | 10 | pipeline | ✅ (read) | future-facing | topology → pairs |
| `mars/03_pair_features.ipynb` | 10 | pipeline | ✅ | future-facing | pairs → 5-feature table |
| `mars/04_xgb_inference_5feat.ipynb` | 11 | pipeline | ⚠️ uses stale-chain inputs | future-facing | features → tabular preds |
| `mars/05_mars_threshold_sensitivity.ipynb` | 12 | analysis | ✅ | future-facing | preds → threshold sweep |
| `mars/dd_hull_mars_vs_earth_complexity.ipynb` | 4 | decision (heavy) | review-only | future-facing | DD/complexity calibration |
| `regime/00_calibration_overview.ipynb` | 4 | decision (frozen record) | ✅ read-only | canonical record | calibration evidence |
| `regime/01_mars_inference.ipynb` | 11 | pipeline | ⚠️ stale regime models | future-facing | regime Mars inference |
| `regime/02_threshold_retune.ipynb` | 9/12 | decision | ✅ | future-facing | threshold retune |
| `training/00_pair_sample_qa.ipynb` | 6 | QA | ✅ | future-facing | pairs → visual samples |
| `training/01_prepare_dataset.ipynb` | 6 | pipeline | ✅ | future-facing | dataset prep |
| `training/02_train_classifier.ipynb` | 8 | pipeline (heavy) | model-producing | future-facing | XGB train |
| `training/03_feature_engineering.ipynb` | 7 | pipeline (heavy) | review-only | future-facing | features + patch preview |
| `training/04_cnn_embeddings.ipynb` | 8 | pipeline (heavy) | model-producing | future-facing | CNN train + embeddings |
| `training/05_cnn_quick_eval.ipynb` | 9 | QA | ✅ read | future-facing | CNN eval |
| `diagnostics/dd_threshold_calibration.ipynb` | 2/4 | decision | review-only | future-facing | threshold vs DD |
| `diagnostics/earth_network_pruning_experiments.ipynb` | 2 | exploration (823 KB) | review-only | source of `pruning.py` | pruning sweep |
| `diagnostics/lobo_cv.ipynb` | 9 | QA | ✅ | future-facing | LOBO CV |
| `diagnostics/regB_threshold.ipynb` | 9 | QA | ✅ | future-facing | regB threshold |
| `diagnostics/rasterization_diagnostics.ipynb` | 7 | QA | ✅ | **directly relevant to Stage-7 reconciliation** | patch sanity |
| `diagnostics/stream_crossing_qa.ipynb` | 6 | QA | ✅ | future-facing | crossing filter QA |
| `interpretation/00_scientific_summary.ipynb` | 13 | interpretation | ✅ | future-facing | coupling rates/geography |
| `presentation/result_figures.ipynb` | 14 | presentation | ⚠️ stale inputs | future-facing | result figures |
| `presentation/mars_contact_sheets.ipynb` | 14 | presentation | ⚠️ | future-facing | vector contact sheets |
| `presentation/per_outlet_touching_pairs.ipynb` | 14 | presentation | ⚠️ | future-facing | per-outlet figures |
| `presentation/simple_mars_earth_dd_presentation.ipynb` | 14 | presentation (2.1 MB) | review-only | future-facing | DD presentation |
| `archive/00_full_pipeline`, `01_single_basin_test`, `02_multi_basin`, `03/04_all_basins`, `experiment_*` | — | archive | ❌ do not run | **stale (provenance only)** | — |

**Recommended canonical notebook order** (stage-numbered):
`analysis/01 → analysis/02 → mars/00 → regime/00 (+ mars/dd_hull, diagnostics/dd_threshold_calibration) → analysis/05 → training/00 → training/01 → training/03 → training/02 → training/04 → training/05 / diagnostics/lobo_cv → mars/02 → mars/03 → mars/04 / regime/01 → mars/05 → interpretation/00 → presentation/*`.

- **Safe to ignore for normal work:** everything in `notebooks/archive/`.
- **Need manual review (heavy / decision):** `mars/dd_hull_mars_vs_earth_complexity`, `diagnostics/earth_network_pruning_experiments`, `presentation/simple_mars_earth_dd_presentation`.
- **Should eventually be archived:** none new — archive already holds the superseded set. Re-evaluate `presentation/simple_mars_earth_dd_presentation` only if `result_figures` supersedes it.
- **Missing / weak stages:** Stage 7 has only diagnostic/heavy notebooks (`training/03`, `rasterization_diagnostics`) — there is **no lightweight, runnable Stage-7 readiness notebook**, which is exactly why the interrupted raster state went unnoticed. Stage 14 has notebooks but **no output** (`final_figures/` empty).

---

## 6. Documentation review

**Canonical sources of truth (confirmed):**
| Question | Canonical doc |
|---|---|
| Project overview | `README.md` (user) + `CLAUDE.md` (developer hub) |
| Pipeline design | `docs/PIPELINE_DESIGN.md` (15-stage intent) + `docs/pipeline.md` (Mars I/O graph) |
| Current execution status | `AGENT_STATE.md` + `STAGE_ASSET_MAP.md` |
| Data status | `docs/DATA_STATUS.md` (+ `docs/data_management.md` for policy) |
| Modeling metrics | `docs/modeling.md` (narrative) + `models/ALL_MODELS_METRICS.csv` (numbers, **stale**) |
| Notebook order | `docs/notebooks.md` |
| Regimes | `docs/REGIME_SELECTION.md` |

**Redundant / stale / unclear (found this audit):**
- `STAGE_ASSET_MAP.md` and `docs/pipeline.md` both describe stages from different angles (15-stage scientific vs Mars-phase I/O). Not harmful — keep both, but they're a mild duplication.
- **Stale spots corrected this session** (see §14): `REGIME_SELECTION.md` threshold filename; `PIPELINE_RERUN.md` orchestrator claim; `DATA_STATUS.md`/`AGENT_STATE.md`/`STAGE_ASSET_MAP.md` Stage-7 state.
- **Flagged, NOT changed (scientific number — verify manually):** `README.md` says "XGBoost classifier (Earth held-out test AUC ≈ 0.92)". `models/ALL_MODELS_METRICS.csv` shows `baseline geom_only` (5 feat, the production feature set) at **ROC-AUC 0.747**, and the **0.91** figures belong to the geom+CNN variants. The README's "~0.92" most plausibly refers to a CNN-augmented model, not the 5-feature production model. This touches frozen scientific claims and a stale metrics file — **do not edit without confirming which model/metric the README means.**

**Recommended documentation mental model:**
- **Daily-use:** `CLAUDE.md`, `AGENT_STATE.md`, `STAGE_ASSET_MAP.md`, `docs/notebooks.md`.
- **Planning:** `docs/PIPELINE_DESIGN.md`, `docs/ROADMAP_AND_RISKS.md`, this file.
- **Reference (read when needed):** `docs/architecture.md`, `docs/DEVELOPER_GUIDE.md`, `docs/REGIME_SELECTION.md`, `docs/modeling.md`, `docs/MARS_PIPELINE.md`, `docs/DATA_STATUS.md`, `docs/PIPELINE_RERUN.md`, `docs/data_management.md`, `docs/PROJECT_STRUCTURE.md`, `docs/pipeline.md`.
- **Historical (do not read unless tracing provenance):** `notebooks/archive/README.md`, `notebooks/archive/optimization_review.md`.

---

## 7. Pipeline alignment review (stages 0–14)

| Stage | Status | Current assets | Missing | Next wave? | Needs full data regen? |
|---|---|---|---|---|---|
| 0 Setup/assumptions | ✅ complete | `io/paths`, `regimes`, `units`, `CLAUDE.md`, `architecture.md` | — | docs only | No |
| 1 Earth source QA | 🔶 partial | `analysis/01`, `basin_config`, 17 DEMs | formal "all DEMs verified" run record | optional run | No |
| 2 Earth net explore | 🔶 partial | `analysis/02`, `dd_calibration`, `pruning` | — | no | No |
| 3 Mars net explore | 🔶 partial | `mars/00`, `mars/topology` | — | no | No |
| 4 Regime calibration | ✅ complete | `regimes.py` (frozen), `REGIME_SELECTION.md`, `regime/00` | — | no (frozen) | No |
| 5 Final Earth net + QA | ✅ complete | `analysis/05`, `training/regime`, QA report (0 hard flags) | — | no | No |
| 6 Earth pairs + labels | ✅ complete | `pairing/earth`, `training/labeling`, `master_dataset_reg*.csv`, `training/00` | — | no | No |
| 7 Earth model-input | ⚠️ **blocked/interrupted** | `rasterization/*`, `build_cnn_patches_regime`, `training/03`, `rasterization_diagnostics` | **clean rasters + valid manifests; a runnable readiness notebook** | **reconcile/document** | **Yes (deferred)** |
| 8 Model training | ⚠️ stale | `training/{cnn,xgboost}`, regime + baseline models in `models/` | fresh inputs from Stage 7 | no (defer) | Yes (after 7) |
| 9 Earth validation | ⚠️ stale | `eval/lobo`, `lobo_cv_metrics.csv`, `ALL_MODELS_METRICS.csv` | fresh metrics post-retrain | partial (read-only review ok) | Yes (after 8) |
| 10 Mars model-input | ⚠️ stale (present) | `mars/{topology,pairs}`, `features/mars_features`, model_inputs parquet | fresh patches/embeddings | no (defer) | Yes (after 8) |
| 11 Mars inference | ⚠️ stale (present) | `models/mars_combined`, predictions parquet (regA/B/C + baseline) | fresh preds | no (defer) | Yes (after 10) |
| 12 Threshold analysis | ✅ complete | `mars/05`, `regime/02` | — | analyze on current preds OK | No (uses existing preds) |
| 13 Interpretation | ✅ complete | `interpretation/00` | refresh after final preds | optional | Eventually |
| 14 Figures/reporting | 🔶 partial | presentation notebooks + scripts; `final_figures/` **empty** | actual rendered final figures | optional | Eventually |

---

## 8. Data and artifact state review

**Source / keep (never touch):** `data/cropped_DEMs/` (17 Earth DEMs), `data/raw/` (SRTM), `data/final_valleys/` (Mars vectors), `data/Mars/` DEM + MOLA hillshade. Classified `RAW_KEEP`.

**Generated / current (tabular, unaffected by raster rewrite):** `master_dataset_reg{A,B,C}.csv`, `build_earth_features_reg*_stats.csv`, `master_dataset_v2.csv`, Mars 5-feature tables (`mars_pair_features_5feat_*`), geom-only XGBoost inputs. These are trustworthy.

**Generated / stale (`STALE_AFTER_RASTER_FIX`):** everything in the raster→CNN→embedding→combined chain — `master_dataset_*_with_emb.csv`, `*_v4_cnn_full.csv`, CNN models, geom+cnn XGBoost, Mars `cnn_patches_5class/`, `mars_cnn_*`, combined predictions. Usable for structural testing only.

**Archived:** `data/_stage7_archive_20260603_231506/results/` holds `_rasters_regA` (full 17, May 31, pre-rewrite), `_rasters_regA_partial_rerun` (12 basins, Jun 4), `_rasters_regB`, `_rasters_regC`, and the Jun 1–2 manifests. `data/outputs/` and `data/archive/` are `LEGACY`. `data/_rebuild_backup_20260531/` (~97 MB) is `BACKUP` (off-repo).

**⚠️ Ambiguous / interrupted state (the headline finding):**
- **Restored raster manifests do NOT match disk.** `data/results/raster_manifest_reg{A,B,C}.csv` each list **17 basins** but reference `data/results/_rasters_reg{A,B,C}/` paths. Verified by resolving sample rows: regB → `_rasters_regB/...` **MISSING**, regC → `_rasters_regC/...` **MISSING** (those dirs are archived, not in `data/results/`). regA is a **mix**: `raster_status` = 16,886 ok / 1,423 invalid / 7,607 failed, but on disk only 5 basins (toano, troodos, tsugaru, vallefertil, yoro) exist; the other 12 resolve MISSING.
- **regA regen was interrupted** (user-confirmed). Its output is split: `data/results/_rasters_regA/` (5 basins) + archive `_rasters_regA_partial_rerun/` (12 basins) = the 17, but fragmented across two roots, and the live manifest indexes only the `data/results` location.
- The docs previously described these manifests as benign "stale compatibility artifacts (pre-rewrite data)." That undersells the risk: **they index files that are largely absent.** Any Stage-8 code that reads `raster_path` will silently drop most basins or error.

**Are restored manifests clearly marked as stale?** Now **yes** — corrected this session in `AGENT_STATE.md`, `STAGE_ASSET_MAP.md`, `DATA_STATUS.md` to "NEEDS REVIEW / not trustworthy until clean regen" (§14).

**Should-not-touch:** all `RAW_KEEP`; production `models/xgb_touching_classifier.json` + `models/cnn_outlet_final.pt`; the backup directory; the Stage-7 archive (it's the only clean copy of regA's full set).

---

## 9. Model state review

**Models present (`models/`):**
- **Production (frozen — do not overwrite):** `xgb_touching_classifier.json` (threshold 0.577406), `cnn_outlet_final.pt`.
- **Baseline variants:** `xgb_geom_only.json`, `xgb_geom_plus_cnn_emb.json`, `xgb_geom_plus_cnn_logit.json` + `optimal_threshold_*`/`feature_columns_*`.
- **Regime (per regA/B/C):** `cnn_outlet_reg{A,B,C}.pt`, `xgb_geom_plus_cnn_emb_reg{A,B,C}.json`, `optimal_threshold_geom_plus_cnn_emb_reg{A,B,C}.txt`, `feature_columns_geom_plus_cnn_emb_reg{A,B,C}.txt`, per-regime metrics CSVs.
- **Metrics:** `ALL_MODELS_METRICS.csv`, `lobo_cv_metrics.csv`, `combined_models_comparison.csv`, `calibration_experiment.csv`.

**Production vs legacy vs regime vs Mars:** production = the two frozen artifacts; regime = `_reg{A,B,C}`; baseline = `_geom_*`; Mars-related = predictions live under `data/Mars/model_outputs/`, not `models/`.

**Current vs stale metrics:** `ALL_MODELS_METRICS.csv` and the regime metrics are **stale** (geom+cnn / regime models were trained on pre-rewrite rasters → `STALE_AFTER_RASTER_FIX`). The geom-only line is tabular and less affected. Treat all CNN-derived AUC/F1 numbers as provisional.

**Retrain now or defer?** **Defer.** Retraining before Stage 7 is reconciled would bake the broken/fragmented raster state into new models. Per project policy and the audit, retraining is **not** this wave.

**What to do before retraining:** (1) choose a final regime *or* commit to reporting all three; (2) clean regen of rasters + manifests with the current rasterizer; (3) verify manifest `raster_path` entries all resolve; (4) confirm `optimal_threshold_geom_plus_cnn_emb_regB.txt` is the intended value (a mid-session change was previously flagged in `PIPELINE_RERUN.md`).

**Missing model comparisons:** a single, current, side-by-side table of {geom-only, geom+emb, geom+logit} × {baseline, regA, regB, regC} on **fresh** data with LOBO + held-out, plus the explicit Mars operating-threshold justification. The pieces exist (`models/comparison.py`, `eval/`) but the consolidated current table does not.

---

## 10. Risk register

| # | Risk | Sev | Likelihood | Stage | Mitigation | Blocks next wave? |
|---|---|---|---|---|---|---|
| R1 | **Stale/broken raster manifests used accidentally** in training (point at missing files; 12/17 regA missing) | **High** | **High** if a rebuild is run blindly | 7→8 | Marked NEEDS REVIEW (done); clean regen + manifest-resolves check before any train | **No** (wave avoids retrain) but gates final results |
| R2 | Interrupted regA regen fragmented across 3 locations → wrong/partial inputs | High | Med | 7 | Documented in `AGENT_STATE.md`; reconcile by clean regen, not by stitching | No |
| R3 | Retraining before Stage-7 QA bakes in broken state | High | Med | 8 | Defer retrain; readiness gate first | Avoided by plan |
| R4 | Mars inference run with stale Earth models presented as final | Med | Med | 11 | Keep stale preds for structure only; label clearly; refresh after retrain | No |
| R5 | Stale modeling numbers (`ALL_MODELS_METRICS.csv`, README ~0.92) quoted as current | Med | Med | 9 | Flagged §6/§9; verify before citing | No |
| R6 | Deleting archives too early (the only clean regA full set is archived) | **High** | Low | 7 | "Never delete data" rule; archive is sole clean copy — protect | No |
| R7 | Package ownership confusion (`training` vs `models`) leads to edits in wrong layer | Low | Low | 8 | One-sentence doc note (recommended) | No |
| R8 | Unclear notebook status / no runnable Stage-7 readiness notebook | Med | Med | 7 | Add lightweight readiness/QA notebook (wave deliverable) | No |
| R9 | Over-refactoring instead of scientific progress | Med | Med | all | Structure is already stable — freeze refactor; prefer docs + reconciliation | This wave explicitly avoids it |
| R10 | S1 unit assumption (`upstream_distance()` ΔL ~3600×) | ~~Critical~~ **Resolved** | **Not a bug** | 6 | Verified 2026-06-04: `s.upstream_distance().max() = 0.2404` arc-degrees on CalnAlpine; `meters_per_unit = 97309 m/deg`; ΔL conversion correct | **Done** |
| R11 | `run_regime_pipeline.sh` silently can't run regC (exits) | Low | Low | 8 | Doc corrected (§14); use per-step CLI for regC | No |

---

## 11. Recommended next wave options

### A. Pipeline clarity + data-state reconciliation + notebook QA  *(recommended)*
- **Goal:** make the project's current state unambiguous and trustworthy *without* regenerating data or retraining.
- **Scope:** reconcile/document the interrupted Stage-7 raster state; mark stale artifacts; finish a read-only notebook QA pass; add the ownership-doc sentences; produce a clean regA *plan* (not execution); optionally run the read-only QA notebooks and pytest as a green-gate.
- **Why now:** matches stated direction (clarity before regeneration); the audit shows the only real instability is *clarity of data state*, not code.
- **Expected outputs:** corrected docs (this session) + a short "Stage 7 reconciliation plan" + a runnable Stage-7 readiness/QA notebook + green pytest.
- **Don't touch:** raw data, models, rasters, the Stage-7 archive, frozen artifacts.
- **Risk:** Low. **Stop when:** docs/notebooks reflect reality, pytest green, regA plan written.

### B. Earth model-input / full-rerun readiness wave
- **Goal:** get Stage 7 to a clean, verifiable state (clean rasters + manifests that resolve) for ONE regime.
- **Scope:** clean `build_cnn_patches_regime.py --regime regA` end-to-end; verify every `raster_path` resolves; rebuild regA manifest; **no training.**
- **Why now:** unblocks everything downstream; directly fixes R1/R2.
- **Outputs:** trustworthy regA rasters + manifest; updated `DATA_STATUS.md` tags.
- **Don't touch:** models, other regimes' archives, production artifacts.
- **Risk:** Medium (regeneration). **Stop when:** regA manifest 100% resolves and `rasterization_diagnostics` passes.

### C. Mars analysis + threshold interpretation wave
- **Goal:** extract maximum scientific signal from *existing* (clearly-labeled-stale) predictions.
- **Scope:** run `mars/05_mars_threshold_sensitivity`, `regime/02`, `interpretation/00` read-only; document the Mars operating-threshold decision.
- **Why now:** Stages 12–13 are complete and don't need regeneration; produces scientific value despite stale upstream.
- **Outputs:** threshold-sensitivity findings + interpretation notes (caveated as provisional).
- **Don't touch:** models, rasters. **Risk:** Low. **Stop when:** threshold decision documented.

### D. Final reproducibility / full rerun wave
- **Goal:** clean end-to-end regen + retrain for the chosen regime(s); refresh all stale artifacts and figures.
- **Scope:** `run_full_rebuild.sh` + per-regime steps; LOBO + held-out; refresh `final_figures/`.
- **Why now:** **only after A/B** — needs a finalized regime and a clean Stage 7.
- **Outputs:** fresh models, predictions, metrics table, figures; `DATA_STATUS` flipped to `CAN_REGENERATE`.
- **Don't touch:** frozen production artifacts (unless intended). **Risk:** High (heavy compute, scientific). **Stop when:** post-rebuild checklist green.

---

## 12. Recommended next wave

**Wave A — Pipeline clarity + data-state reconciliation + notebook QA.**

- **Objectives:**
  1. Reconcile and document the interrupted Stage-7 raster state (done in part this session; finalize a short reconciliation plan).
  2. Add a **runnable, lightweight Stage-7 readiness notebook** that checks every `raster_path` in a manifest resolves and reports per-basin coverage (this is the missing guard that let the interrupted state slip).
  3. Add the two ownership-clarifying doc sentences (`training/*` = fit+persist vs `models/*` = load+predict; Earth-specific vs shared `features/*`).
  4. Run the read-only QA notebooks (`analysis/01`, `analysis/05`, `diagnostics/lobo_cv`, `diagnostics/rasterization_diagnostics`) and `pytest -q` as a green-gate — **no output regeneration**.
  5. Write the **clean regA regen plan** (commands + verification), to be executed in Wave B.
- **Files likely to touch:** `docs/architecture.md` or `docs/DEVELOPER_GUIDE.md` (1–2 sentences); a new `notebooks/diagnostics/` or `notebooks/training/` readiness notebook; possibly `STAGE_ASSET_MAP.md`. This planning doc.
- **Files NOT to touch:** any `data/` raster/manifest content, `models/*`, the Stage-7 archive, frozen production artifacts, `regimes.py`.
- **Deliverables:** corrected/clarified docs; one new readiness notebook; green pytest; a written regA reconciliation+regen plan.
- **Validation checks:** `conda run -n ch-heads pytest -q` green; readiness notebook reports manifest coverage; `git status` shows only docs/notebook additions.
- **Stop conditions:** state is unambiguous, pytest green, regA plan written. **Do not** start regeneration or retraining.
- **What comes after:** Wave B (clean regA Stage-7 regen) → then a scoped retrain (Wave D), with Wave C (Mars threshold/interpretation on current preds) runnable in parallel any time.

---

## 13. One-page project mental model

- **Understand the project →** `README.md`, then `CLAUDE.md`, then `docs/architecture.md`.
- **Run the pipeline →** `docs/pipeline.md` + `docs/PIPELINE_RERUN.md`; entry points `channel_heads.pipelines` and `scripts/cli/run_mars_pipeline.py`.
- **Inspect notebooks →** `docs/notebooks.md` (canonical order in §5 here); start in `notebooks/analysis/` and `notebooks/mars/`.
- **Understand regimes →** `docs/REGIME_SELECTION.md` + `channel_heads/regimes.py` (frozen) + `notebooks/regime/00_calibration_overview.ipynb`.
- **Know what data is stale/current →** `docs/DATA_STATUS.md` + `AGENT_STATE.md` (Stage 7 caveat) + `STAGE_ASSET_MAP.md`.
- **Continue development →** start with **Wave A** above: read this file §12, then the Stage-7 caveat in `AGENT_STATE.md`. Do **not** retrain or regenerate rasters until Stage 7 is reconciled.

---

## 14. Documentation corrections made

| File | What was stale/incorrect | Correction | Why safe |
|---|---|---|---|
| `AGENT_STATE.md` | Stage-7 bullet described manifests as benign "stale compatibility artifacts (pre-rewrite data)" and said rasters were simply archived — omitted the interrupted regA regen and that live manifests point at missing files. | Rewrote the Stage-7 bullet to state the manifests reference missing on-disk rasters, document the fragmented regA output (5 in `data/results`, 12 in archive partial-rerun, full pre-rewrite set in archive), and instruct a clean regen before finalizing. | Purely factual; verified on disk by path resolution. No data changed. |
| `STAGE_ASSET_MAP.md` | Stage-7 status text undersold the inconsistency. | Replaced with a "NEEDS REVIEW — interrupted state" note matching `AGENT_STATE.md`. | Same verification; aligns two status docs. |
| `docs/REGIME_SELECTION.md` | Downstream-artifacts table listed `models/threshold_reg{A,B,C}.json` — **no such file exists**. | Corrected to `models/optimal_threshold_geom_plus_cnn_emb_reg{A,B,C}.txt` (the files that actually exist). | Verified by `ls models/`; matches `STAGE_ASSET_MAP.md` Stage 8. |
| `docs/PIPELINE_RERUN.md` | Comment claimed `run_regime_pipeline.sh regA` "also runs regB, regC" — the script runs ONE regime per call and **rejects regC** (exits). | Corrected the comment; added a note that the script accepts only regA/regB and regC must use the per-step CLI. | Verified by reading the script (`case … regA|regB`). |
| `docs/DATA_STATUS.md` | "Last classified 2026-06-02" predated the 2026-06-03 archive; manifests still described as present/benign. | Added a dated 2026-06-04 update note: rasters archived, manifests reference missing paths, regA interrupted, treat manifests as NEEDS REVIEW. | Factual; cross-references `AGENT_STATE.md`. No tags rewritten beyond the note. |

**Not changed (flagged for manual verification):** `README.md` "Earth held-out test AUC ≈ 0.92" vs `ALL_MODELS_METRICS.csv` (geom-only 0.747; geom+CNN ~0.91). This is a frozen scientific claim against a stale metrics file — per the "stop if uncertain about scientific behavior" rule, left untouched and flagged in §6/§9.

---

## 15. Final recommendation

- **Continue organizing? →** Only the small amount in Wave A (docs + one readiness notebook). The code/test/architecture is **already stable — stop refactoring.**
- **Regenerate data? →** **Not yet.** First reconcile the interrupted Stage-7 raster state (Wave A documents it; Wave B executes a clean regA regen). Regenerating now would propagate the fragmented/missing-raster state.
- **Train models? →** **No.** Retraining before Stage 7 is clean would bake in the broken inputs. Defer to Wave D.
- **Work on Mars threshold/interpretation? →** **Yes, opportunistically** (Wave C) — Stages 12–13 are complete and run on existing predictions; just label results provisional.
- **Next bounded execution task →** **Wave A:** finalize the Stage-7 reconciliation plan, add a runnable manifest-coverage / Stage-7 readiness notebook, add the two ownership-doc sentences, run read-only QA notebooks + `pytest -q` as a green-gate, and write the clean regA regen plan for Wave B. **Stop there** — do not regenerate rasters or retrain in this wave.

---

### Validation / commands run during this review
- `git status` — clean before edits (`refactor/package-first-architecture`).
- Directory listings of `channel_heads/`, `scripts/`, `notebooks/`, `tests/`, `models/`, `data/`, `data/results/`, `data/Mars/model_outputs|model_inputs/`, `data/_stage7_archive_*/`.
- Python (read-only) inspection of `raster_manifest_reg{A,B,C}.csv`: distinct-basin counts (17 each), `raster_status` tallies, and `os.path.exists()` resolution of sample `raster_path` rows (confirmed most resolve MISSING).
- Read `cat`/`Read` of all docs listed in the prompt + `run_regime_pipeline.sh`, `ALL_MODELS_METRICS.csv`, `channel_heads/__init__.py`.
- **No** training, inference, figure rendering, or `pytest` run (review task). pytest is recommended as the Wave-A green-gate.
