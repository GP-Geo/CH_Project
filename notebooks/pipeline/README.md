# Pipeline notebooks (one per stage, 00–14)

A clean, **enumerated walkthrough of the whole pipeline** — one notebook per
stage in `docs/PIPELINE_DESIGN.md` / `STAGE_ASSET_MAP.md`. Each notebook is
**lightweight and executable**: it imports the canonical `channel_heads` package
and loads the artifacts already on disk to verify/visualize that stage. It does
**not** re-run heavy compute — use the `channel-heads` CLI for that
(`python -m channel_heads <command>`).

These are the canonical map of the pipeline. The pre-existing themed notebooks
(`analysis/`, `training/`, `mars/`, `regime/`, `diagnostics/`, `interpretation/`,
`presentation/`) remain as **deep-dive references** and are linked from the
relevant stage.

| Stage | Notebook | Builds with (CLI) |
|---|---|---|
| 0 | `00_project_setup_and_assumptions.ipynb` | — (paths/regimes/units) |
| 1 | `01_earth_source_data_exploration.ipynb` | — (RAW_KEEP DEMs) |
| 2 | `02_earth_interactive_network_exploration.ipynb` | — |
| 3 | `03_mars_interactive_network_exploration.ipynb` | `run-mars-pipeline --stage topology` |
| 4 | `04_earth_mars_regime_calibration.ipynb` | — (frozen regimes) |
| 5 | `05_final_earth_network_generation_and_qa.ipynb` | `build-earth-features --regime <r>` |
| 6 | `06_earth_pair_and_label_generation.ipynb` | (pairing/labeling) |
| 7 | `07_earth_model_input_construction.ipynb` | `build-cnn-patches --regime <r>` |
| 8 | `08_model_training.ipynb` | `train-cnn-regime`, `train-combined-xgb-regime` |
| 9 | `09_earth_model_validation_and_tuning.ipynb` | `eval-lobo-cv`, `retune-threshold-regime` |
| 10 | `10_final_mars_model_input_generation.ipynb` | `run-mars-pipeline --stage patches/embeddings` |
| 11 | `11_mars_inference.ipynb` | `run-mars-combined-regime --regime <r>` |
| 12 | `12_mars_threshold_and_prediction_analysis.ipynb` | (threshold sweep) |
| 13 | `13_scientific_interpretation.ipynb` | (interpretation) |
| 14 | `14_figures_poster_and_reporting.ipynb` | `make-result-figures`, `generate-poster-figures` |

## Regenerating

`_build_pipeline_notebooks.py` regenerates this set (overwrites executed
outputs). After regenerating, execute with:

```bash
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 notebooks/pipeline/*.ipynb
```
