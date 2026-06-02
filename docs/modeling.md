# Modeling

How the project decides whether two channel heads are **coupled** (touching),
trained on Earth and applied to Mars.

## Feature design (why it transfers)

The production tabular model uses **5 dimensionless features** so it is portable
across planets without rescaling:

`orientation_diff_deg`, `headhead_dist_norm`, `apex_angle_deg`,
`strahler_order_diff`, `proximity_profile_norm`.

Dimensionless-by-design is the core of the Earth→Mars transfer-learning idea:
Mars valley networks are larger and lower-resolution, but the *geometry* of a
confluence is scale-free.

## Model variants

| Variant | Features | Artifact | Use |
|---------|----------|----------|-----|
| **Geometric-only** | 5 dimensionless | `xgb_geom_only.json` | most portable baseline |
| **Geom + CNN embedding** | 5 geom + 4-d CNN embedding | `xgb_geom_plus_cnn_emb.json` | adds raster shape context |
| **Geom + CNN logit** | 5 geom + 1 CNN logit | `xgb_geom_plus_cnn_logit.json` | lighter CNN signal |
| **Production** | 5 dimensionless | `xgb_touching_classifier.json` | **frozen**, threshold 0.577406 |
| **Regime A/B/C** | per-pruning-regime retrains | `*_reg{A,B,C}.json`, `cnn_outlet_reg*.pt` | drainage-density calibration |

The **CNN** (`cnn_outlet_final.pt`) encodes a 128×128 **5-class** patch
(BACKGROUND/BRANCH_A/BRANCH_B/OTHER_STREAMS/CONFLUENCE_MARKER) into a 4-d
embedding. Mars patches **must stay 5-class** so the Earth-trained CNN applies
without retraining. Compare variants with
`channel_heads.models.comparison.compare_predictions`.

## The threshold issue (important for Mars)

The Earth **F1-optimal** threshold (0.577406) is tuned to maximise F1 on the
*Earth* label distribution. It should **not** be copied blindly to Mars:

- Mars has **no ground-truth labels**, so F1 cannot be computed there.
- Mars valley networks differ in base rate of true couplings and in feature
  distribution (resolution, network size), so an Earth-balanced threshold can
  badly mis-calibrate Mars precision/recall.
- For Mars we therefore prefer a **deliberately chosen operating threshold**:
  precision-oriented (`models.thresholds.max_precision_threshold`) and/or
  cross-checked against an independent signal such as **drainage-density
  calibration** (`channel_heads.dd_calibration`) and regime pruning.

Use the threshold-sensitivity and Dd-calibration notebooks
([notebooks.md](notebooks.md)) to pick and justify the Mars operating
threshold — treat it as a scientific decision, not a copied constant.

## Known scientific risks

The geometric/units risk register (S1–S8, e.g. the `upstream_distance()` unit
assumption behind ΔL) lives in
[ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md); units are centralised in
`channel_heads/units.py`.
