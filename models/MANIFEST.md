# Model artifact manifest

Every trained artifact in this directory, its provenance, and integrity
checksums. **The two production artifacts are frozen — never overwrite or
retrain them in place** (see `CLAUDE.md` operating rules). Regime variants use
explicit suffixes (`_geom_*`, `_reg{A,B,C}`).

## Frozen production artifacts

| Artifact | Trained | Operating threshold | SHA256 |
|---|---|---|---|
| `xgb_touching_classifier.json` | 2026-06-03 (Earth pairs, pre-regime pipeline) | **0.577406** (constant in `channel_heads/io/paths.py`; no threshold file) | `41a31ace2906a26430db90eebc50d54c5c5917607658a5d4729ab8638784618f` |
| `cnn_outlet_final.pt` | 2026-06-01 (Earth outlet patches, 5-class) | n/a (feature extractor) | `d85805965024c20dddcce73708c6e61afd54de8df41328f9c470bbbedba9125d` |

Thresholds were calibrated against these exact files; retraining will not
reproduce them bit-identically. Verify integrity with:
`shasum -a 256 -c` against the hashes above.

## Regime variants (retrained 2026-06-13 on the redefined presets)

All `_reg{A,B,C}` artifacts below were retrained on 2026-06-13 after the
2026-06 data-driven regime redefinition (T = 0.20 / 0.25 / 0.15, trim-only;
commit `e5a1083`). Inputs: regime-specific Stage-7 rasters + Stage-8 feature
tables (`data/results/_rasters_reg*/`, regime master datasets). Training entry
points: `channel-heads train-cnn-regime` / `train-xgb-regime` (orchestrated by
`scripts/run_regime_pipeline.sh <regA|regB|regC> [full|retrain]`).

| Regime | CNN | XGBoost (geom+cnn_emb) | Threshold | SHA256 (CNN / XGB) |
|---|---|---|---|---|
| regA | `cnn_outlet_regA.pt` | `xgb_geom_plus_cnn_emb_regA.json` | 0.769133 | `8a60d3ca…5df412` / `04340c6c…f70abaa` |
| regB | `cnn_outlet_regB.pt` | `xgb_geom_plus_cnn_emb_regB.json` | 0.773238 | `c6a34158…d74cd72b` / `49b7a69e…cf9568a` |
| regC | `cnn_outlet_regC.pt` | `xgb_geom_plus_cnn_emb_regC.json` | 0.810635 | `4f535086…79b68a6` / `1ba66007…179294d2` |

Full hashes: run `shasum -a 256 models/*.pt models/*.json`.

## Non-regime Earth model variants (2026-06-01)

`xgb_geom_only.json` (thr 0.595252), `xgb_geom_plus_cnn_emb.json`
(thr 0.856902), `xgb_geom_plus_cnn_logit.json` (thr 0.854628) — Phase-6
comparison variants; see `docs/modeling.md`.

## Sidecar files

- `optimal_threshold_*.txt` — **authoritative operating thresholds**, written
  at tuning time next to each model.
- `feature_columns_*.txt` — exact feature order each XGBoost model expects.
  The prediction schema must not change silently.
- `*_history.csv`, `*_multiseed.csv`, `*_metrics.csv`,
  `combined_models_comparison.csv`, `calibration_experiment.csv` — training
  curves and evaluation summaries for the artifact of the same name.
- `ALL_MODELS_METRICS.csv` (2026-06-04) — **predates the 2026-06-13 regime
  retrain**; where it disagrees with `optimal_threshold_*.txt`, the txt files
  win.
- `lobo_cv_metrics.csv` — within-basin CV summary, regenerated whenever
  `channel-heads eval-lobo-cv` runs (last: 2026-07-02). Within-basin numbers;
  for honest cross-basin validation see `channel-heads lobo-validate` and
  `data/results/lobo/`.
