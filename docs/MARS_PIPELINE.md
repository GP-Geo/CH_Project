# Mars Cross-Planet Pipeline

Train on Earth, infer on Mars. Consolidated from the Earth→Mars compatibility
report (`data/Mars/topology/mars_earth_model_compatibility_report.md`, kept in
place as the long-form source) and the Phase 3B/5/6A/6C summaries (now under
`data/_rebuild_backup_20260531/`, read-only).

> See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for the package and
> [ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md) for open calibration work.

---

## 1. Design contract (dual track)

The pipeline has **two first-class tracks** — the CNN is required, not optional:

- **Track A — tabular:** 5 dimensionless / angular features → XGBoost.
  Features (exact order): `orientation_diff_deg`, `headhead_dist_norm`,
  `apex_angle_deg`, `strahler_order_diff`, `proximity_profile_norm`. These are
  unit-free by design so an Earth-trained model transfers to Mars.
- **Track B — CNN / embeddings:** 5-class 128×128 `uint8` patches →
  `models/cnn_outlet_final.pt` → 4-dim embedding (and 1-dim logit).

**Critical invariants:**
- Mars patches **must** be 5-class (BACKGROUND / BRANCH_A / BRANCH_B /
  OTHER_STREAMS / CONFLUENCE_MARKER) to match the Earth-trained CNN. A 3-class
  variant was explicitly rejected.
- CNN preprocessing is verbatim Earth: `encode_raster_onehot` one-hot, **no
  mean/std normalization**, `augment=False` at inference, `embed() =
  ReLU(fc_embed(features))`.
- Production `xgb_touching_classifier.json` is preserved unchanged; combined
  variants get explicit suffixes.

**First-meet pair definition:** for each confluence, the pair(s) of channel
heads whose branches first meet there (Kahn topological sort + head-set
propagation), computed on the Mars directed graph
(`upstream_node_id → downstream_node_id`) rather than a TopoToolbox StreamObject.

---

## 2. Phase history (Phases 1–6C complete)

| Phase | What | Key artifacts | Package entry point |
|------|------|---------------|--------|
| 1 | Topology GPKG (391 networks, 5,619 segments, 6,003 nodes, 2,999 heads, 2,612 confluences). Input valley-network vectors: Alemanno, Orofino & Mancarella 2018, *Global map of Martian fluvial systems*, Earth and Space Science 5, 560–577, doi:10.1029/2018EA000362 | `data/Mars/topology/mars_vn_topology_model_ready.gpkg` | `pipelines.build_mars_topology` |
| 2B | First-meet pairs (15,677 pairs + per-branch polylines) | `mars_vn_pairs.gpkg` | `pipelines.extract_mars_pairs` |
| 3A | 5-feature table + terrestrial filtering. Stream-crossing filter dropped 11,892/15,677 (76%) → **3,785** model-ready pairs | `mars_pair_features_5feat_{all,model_ready}.parquet` + audit CSV | `pipelines.build_mars_features` |
| 3B | Tabular inference (production XGBoost, threshold 0.577406) → **2,183 touching (57.7%)**, 898 high-conf (≥0.80) | `mars_xgb_predictions_5feat.{parquet,csv,gpkg}` | `pipelines.run_mars_xgb_inference` |
| 4 | 5-class CNN patches (3,785, Earth-compatible) | `cnn_patches_5class/{network}/{pair}.npy` | `pipelines.build_mars_cnn_patches` |
| 5 | CNN embeddings (3,785 × emb_0..3; no collapse; r≈0.4–0.5 vs tabular prob). **`cnn_logit` was NOT persisted** — filled transiently in 6C | `mars_cnn_embeddings.parquet`, `mars_model_input_tabular_plus_cnn.parquet` | `pipelines.extract_mars_cnn_embeddings` |
| 6A | Audit: no combined XGBoost artifact existed on disk (notebooks 04/05 trained but never saved) | `phase_6a_combined_model_audit.md` | — |
| 6B | Trained + persisted 3 Earth XGBoost variants (geom_only / geom+emb / geom+logit), identical hyperparams + GroupShuffleSplit. Earth test ROC AUC: geom_only **0.77**, geom+emb **0.93**, geom+logit **0.93** | `xgb_geom_*.json` + `feature_columns_*` + `optimal_threshold_*` + `combined_models_comparison.csv` | `train_combined_xgb_phase6b.py` |
| 6C | Combined Mars inference. emb: **1,390 touching (36.7%)**, logit: 1,415 (37.4%); high-conf ≥0.80 ≈ 1,680 (44%). emb↔logit agreement **95%**; combined↔tabular **70%** | `mars_combined_model_predictions.{parquet,csv,gpkg}` + comparison/by-network CSVs + figures | `pipelines.run_mars_combined_inference` |

The old Mars stage scripts are archived (`scripts/_archive/`); the runnable
entry points are the CLI commands in `channel_heads/cli/`
(`python -m channel_heads <command>`), and reusable Mars inference logic lives
in `channel_heads/`.

Phase-6C thresholds (Earth-tuned, "max precision at recall ≥ 0.5") are applied
to Mars unchanged. The authoritative values are the on-disk files
`models/optimal_threshold_geom_plus_cnn_emb.txt` and
`models/optimal_threshold_geom_plus_cnn_logit.txt` — always read those rather
than any number quoted in prose (values quoted in earlier revisions of this doc
predate the current artifacts).

---

## 3. Strategic decisions

1. Production tabular XGBoost preserved as-is; research models live alongside with `_geom_*` / `_reg{A,B,C}` suffixes.
2. Earth-PR-tuned thresholds applied to Mars unchanged in 6C — a documented caveat; a future phase should sweep alternatives or rebuild on labeled Mars data.
3. Notebooks 04/05 trained combined models only in memory; `train_combined_xgb_phase6b.py` is now the canonical training recipe.
4. CNN `cnn_logit` (1-dim) and the 4-dim embedding carry essentially the same signal (Δ ROC AUC = 0.004 Earth; 95% Mars agreement). Model C (geom + logit, 6 features) is the cheaper-equally-good choice.

---

## 4. Calibration & threshold policy (findings — deprioritized)

> **Decision (2026-06-01):** a dedicated `channel_heads/calibration.py` module is
> **dropped**. A per-basin probability-standardization test was neutral-to-negative,
> so calibration is not a current priority. Threshold-sensitivity analysis is kept
> as a **notebook** task (`notebooks/mars/05_mars_threshold_sensitivity.ipynb`;
> the regime notebooks are archived under `notebooks/archive/regime/`), not a
> core package module. The findings below remain useful context.

### Historical: within-basin CV (superseded)

> **Methodology caveat:** the "LOBO" numbers below came from the old
> within-basin, leakage-prone evaluation — they are **not** cross-basin
> generalization estimates. Kept for the historical record; see the true-LOBO
> subsection below for the honest numbers.

| Config | LOBO per-basin AUC | Pooled AUC | F1 |
|--------|-------------------|-----------:|---:|
| baseline | 0.888 ± 0.092 | 0.916 | 0.642 |
| regA | 0.868 ± 0.059 | 0.900 | 0.616 |
| regB | 0.845 ± 0.089 | 0.889 | 0.614 |
| regC | 0.908 ± 0.046 | **0.710** | 0.452 |

Interpretation at the time: models are **not weak**. regA's apparent weakness
was likely seed/split variance. **regC has strong per-basin discrimination but
poor pooled calibration** (notably a Taiwan probability-scale mismatch).

Current Mars touching rates (from the 2026-06-13 regime retrain,
`data/Mars/model_outputs/mars_combined_reg{A,B,C}_predictions.parquet`):
regA **49.6%**, regB **59.7%**, regC **46.6%**.

### True leave-one-basin-out (2026-06/07, leakage-audited)

A true cross-basin LOBO engine now exists (`channel_heads/eval/lobo.py`, CLI
`lobo-validate`) with a per-fold leakage audit. Honest numbers — always label
which statistic you quote:

| Regime | geom-only pooled AUC | geom-only per-basin mean AUC | precomputed_emb pooled AUC (leakage-flagged) |
|--------|---------------------:|-----------------------------:|---------------------------------------------:|
| regA | 0.778 | 0.714 | 0.885 |
| regB | 0.769 | 0.701 | 0.883 |
| regC | 0.783 | 0.739 | 0.898 |

- **Leakage finding:** in `precomputed_emb` mode the `emb_*` features come from
  a CNN trained on 16/17 basins, so every fold's held-out basin is contaminated
  — see `data/results/lobo/*/precomputed_emb/leakage_audit.md`. Treat those
  pooled ~0.88–0.90 scores as optimistic.
- The old headline **~0.91** AUC is a *within-basin* held-out-test statistic,
  not a cross-basin one.
- Regenerate with
  `python -m channel_heads lobo-validate --regime regC --mode geom_only`.
- `per_fold_cnn` mode (retrains the CNN inside each fold; leak-free) is
  implemented (`channel_heads/eval/lobo_cnn.py`) but has **not yet been run** —
  no outputs on disk.

**Conclusion:** since per-basin standardization did not help, a Mars-appropriate
**threshold policy** (rather than probability calibration) is the lever worth
exploring — deferred to a later notebook, not built as a package module now.

---

## 5. Regime calibration (regA/regB/…)

Parallel re-runs of the Earth→Mars workflow under alternative network-pruning
regimes (km² area thresholds + Strahler-strip / order-gap pruning), orchestrated
by `scripts/run_regime_pipeline.sh`. Regime presets live in
`channel_heads/regimes.py` (the `REGIMES` mapping), consumed by the
`build-earth-features` CLI command (`channel_heads/cli/build_earth_features_regime.py`)
and the regime CNN/training commands. Per-regime
artifacts use `_reg{A,B,C}` suffixes and never overwrite production models.
