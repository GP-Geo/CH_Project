# PIPELINE_DESIGN.md

High-level pipeline design for the Channel Heads project refactor.

This document defines the intended scientific and computational workflow before
committing to package architecture, notebook structure, or script archival.

The guiding principle is:

> Explore and calibrate first, generate final data second, train models third,
> transfer to Mars fourth, interpret last.

---

## Core design principles

1. **Package-first architecture**
   - Core logic should live in `channel_heads/`.
   - Scripts should become thin wrappers or archive candidates.
   - Notebooks should call package functions, not hide production logic.

2. **Exploration and production are separate**
   - Exploratory stages support visualization, parameter testing, and regime selection.
   - Production stages generate final model-ready data from frozen decisions.

3. **Calibration comes before final Earth training data**
   - The Earth network regime defines the training distribution.
   - Threshold, pruning, and complexity-reduction choices must be decided before final Earth pair generation and model training.

4. **QA is part of the pipeline**
   - Visual inspection, outlier detection, aggressive-pruning checks, patch inspection, and feature-distribution analysis are formal pipeline responsibilities.

5. **Caching is required**
   - Expensive stages such as pair generation, hard-negative filtering, patch generation, and embedding extraction should avoid unnecessary recomputation.

6. **Mars is not an afterthought**
   - Mars source-data exploration and Earth-Mars calibration occur early because Mars defines the transfer target.

---

## Pipeline overview

```text
0. Project setup and assumptions

1. Earth source-data exploration
2. Earth interactive network exploration
3. Mars interactive network exploration
4. Earth-Mars regime calibration

5. Final Earth network generation and QA
6. Earth pair and label generation
7. Earth model-input construction

8. Model training
9. Earth model validation and tuning

10. Final Mars model-input generation
11. Mars inference
12. Mars threshold and prediction analysis

13. Scientific interpretation
14. Figures, poster, and reporting
```

---

## Stage 0 - Project setup and assumptions

### Purpose

Define the shared project rules before processing starts.

### Input / output

Input:
- Existing project structure
- Current data folders
- Current model assumptions
- Existing audit and agent documentation

Output:
- Agreed project conventions

### Decision or artifact

A stable baseline for:
- Paths
- CRS assumptions
- Naming conventions
- Data-stage conventions
- Model contracts
- Reproducibility rules
- Refactor boundaries

---

## Stage 1 - Earth source-data exploration

### Purpose

Understand the terrestrial basins and verify that the raw Earth data is usable.

### Input / output

Input:
- Earth basins
- DEMs
- Flow products
- Existing TopoToolbox outputs, if available

Output:
- Basic Earth data overview
- Source-data QA notes
- Sample basin visualizations

### Decision or artifact

Confirm that the Earth source data is valid enough to serve as the reference domain.

---

## Stage 2 - Earth interactive network exploration

### Purpose

Provide a user-friendly notebook for exploring Earth basin networks before final regime selection.

The notebook should allow users to:
- Browse different basins
- Test different stream-extraction thresholds
- Inspect how parameters affect network structure
- Visualize basin boundaries, extracted networks, and basic network properties

### Input / output

Input:
- Earth source data
- Candidate TopoToolbox parameters
- Default threshold settings for preview

Output:
- Interactive or easily configurable visualization notebook
- Sample Earth network maps
- Basic network summaries

### Decision or artifact

Initial intuition about Earth network behavior under different extraction parameters.

---

## Stage 3 - Mars interactive network exploration

### Purpose

Understand the mapped Martian valley networks before using them as the transfer target.

This stage should be similar in spirit to Stage 2, but focused on Mars.

The notebook should support:
- Browsing Martian networks
- Visualizing mapped valley networks
- Inspecting network structure and spatial quality
- Displaying networks over MOLA maps where possible

### Input / output

Input:
- Mars valley-network vectors
- MOLA or other Mars raster support layers, where available
- Existing Mars topology outputs, if available

Output:
- Mars network overview figures
- Sample network maps
- MOLA-overlay visualizations
- Basic Mars network summaries

### Decision or artifact

Confirm what the Mars target domain looks like and identify any obvious source-data issues.

---

## Stage 4 - Earth-Mars regime calibration

### Purpose

Choose the Earth network-generation regime that best matches Mars while preserving useful Earth network information.

This is the main early decision gate.

The stage should compare:
- Stream-extraction thresholds
- Pruning strategies
- Delta-pruning parameters
- Drainage-density similarity
- Network complexity
- Amount of network length lost
- Visual and geomorphic plausibility

### Input / output

Input:
- Earth exploratory networks
- Mars mapped networks
- Candidate threshold and pruning settings
- Drainage-density and complexity metrics

Output:
- Regime-comparison results
- Visual comparison outputs
- Candidate regime ranking or shortlist

### Decision or artifact

Selected Earth analog regime, including:
- Extraction threshold
- Pruning strategy
- Complexity-reduction logic
- Rationale for the choice

This decision should be frozen before final Earth data generation.

---

## Stage 5 - Final Earth network generation and QA

### Purpose

Generate the final Earth drainage networks using the selected calibrated regime.

This is the final stage where network quality can still be improved before model-input generation.

### Input / output

Input:
- Earth source data
- Selected Earth-Mars regime

Output:
- Final Earth networks
- Final Earth topology-ready network data
- QA outputs for strongly affected basins

### Decision or artifact

Frozen Earth network representation for the rest of the project.

QA should include:
- Outlier basin visualization
- Aggressively pruned basin inspection
- Basins or outlets with large network-length loss
- Cases where the selected regime may have damaged the network structure

---

## Stage 6 - Earth pair and label generation

### Purpose

Generate the supervised Earth pair dataset from the final Earth networks.

### Input / output

Input:
- Final Earth networks
- Earth topology information
- Labeling rules
- Hard-negative filtering rules

Output:
- Earth channel-head pairs
- Pair confluences
- Branch paths
- Touching / non-touching labels
- Filtering audit
- Cached intermediate results
- Visual samples of touching and non-touching pairs

### Decision or artifact

Final Earth labeled pair dataset.

Notes:
- The hard-negative filter should be applied as early as possible to reduce downstream runtime.
- Caching should prevent repeated expensive pair and path regeneration.
- Visual samples should support both QA and scientific interpretation.

---

## Stage 7 - Earth model-input construction

### Purpose

Convert Earth pairs into model-ready inputs.

### Input / output

Input:
- Earth labeled pairs
- Branch paths
- Final Earth networks

Output:
- Geometric features
- CNN raster patches
- Patch manifest
- Cached outputs
- Patch visualizations
- Feature-distribution summaries

### Decision or artifact

Complete Earth model-input dataset.

This stage should help verify that:
- Patches are visually meaningful
- Features differ in interpretable ways between touching and non-touching pairs
- Data quality is high enough for model training

---

## Stage 8 - Model training

### Purpose

Train predictive models on the final Earth dataset.

### Input / output

Input:
- Earth geometric features
- Earth raster patches
- CNN embeddings or logits
- Earth labels

Output:
- Trained CNN model
- XGBoost models
- Potential additional ML model baselines
- Feature columns
- Threshold files
- Training summaries

### Decision or artifact

Candidate trained model variants ready for validation and comparison.

The main model families may include:
- Geometric-only models
- Geometry + CNN embedding models
- Geometry + CNN logit models
- Additional classical ML baselines, if useful

---

## Stage 9 - Earth model validation and tuning

### Purpose

Evaluate whether the trained models are robust enough for Mars transfer.

### Input / output

Input:
- Trained models
- Earth validation strategy
- Earth labels and model inputs

Output:
- Hyperparameter tuning results
- ROC curves
- Precision-recall curves
- Confusion matrices
- Feature importance
- Threshold-dependent performance
- LOBO or basin-level validation results
- Model-comparison summary

### Decision or artifact

Decision on which model or models are credible for Mars inference.

---

## Stage 10 - Final Mars model-input generation

### Purpose

Generate Mars model inputs using contracts compatible with the Earth-trained models.

### Input / output

Input:
- Mars valley networks
- Mars topology
- Earth-compatible processing contracts
- Hard-negative filtering rules

Output:
- Mars topology
- Mars channel-head pairs
- Mars branch paths
- Mars geometric features
- Mars CNN raster patches
- Mars embeddings or logits
- QA visualizations

### Decision or artifact

Final Mars model-input dataset.

Notes:
- The hard-negative filter should also be applied here.
- This stage should end with clear visualization outputs for QA and scientific interpretation.

---

## Stage 11 - Mars inference

### Purpose

Apply the selected Earth-trained models to Mars.

### Input / output

Input:
- Trained Earth models
- Mars model-input dataset
- Feature-column contracts
- Model thresholds, where relevant

Output:
- Mars prediction probabilities
- Mars touching / non-touching predictions
- Model-agreement summaries
- Network-level summaries
- Prediction maps

### Decision or artifact

Mars prediction dataset.

---

## Stage 12 - Mars threshold and prediction analysis

### Purpose

Analyze how Mars prediction interpretation changes under different threshold choices.

### Input / output

Input:
- Mars prediction probabilities
- Earth validation behavior
- Candidate operating thresholds

Output:
- Threshold-comparison summaries
- Probability-distribution plots
- Touching / non-touching pair visualizations
- Channel-head pair distribution summaries
- Network-level statistics
- High-confidence prediction sets

### Decision or artifact

Chosen interpretation strategy for Mars predictions.

This stage should not only produce a binary result, but also expose the uncertainty and sensitivity of the Mars interpretation.

---

## Stage 13 - Scientific interpretation

### Purpose

Translate model outputs into geomorphological meaning.

### Input / output

Input:
- Mars predictions
- Network summaries
- Earth-Mars regime calibration context
- Threshold analysis
- Visual outputs

Output:
- Scientific interpretation notes
- Main findings
- Candidate figures
- Open questions

### Decision or artifact

Main scientific story of the project.

Status:
- Still intentionally flexible.
- Should be refined after Mars inference and threshold analysis are reviewed.

---

## Stage 14 - Figures, poster, and reporting

### Purpose

Convert the project outputs into presentation-ready material.

### Input / output

Input:
- Final figures
- Final tables
- Scientific interpretation
- Poster or report requirements

Output:
- Poster figures
- Report figures
- Presentation notebooks
- Final summaries

### Decision or artifact

Communication-ready project package.

Status:
- Still intentionally flexible.
- Should be refined after the scientific interpretation stage becomes clearer.

---

## Integration with existing refactor

This design should be integrated gradually.

Recommended transition:

```text
1. Keep existing scripts working.
2. Map each existing script/notebook to one pipeline stage.
3. Move reusable logic into `channel_heads/` stage by stage.
4. Keep root scripts as thin wrappers while the package API stabilizes.
5. Rebuild notebooks around the pipeline stages.
6. Archive legacy scripts only after the package commands and notebooks replace them.
```

---

## Immediate next steps

1. Add this file as `PIPELINE_DESIGN.md`.
2. Create or update a stage-to-asset map that links existing scripts, notebooks, and outputs to the stages above.
3. Use the stage map to prioritize the next refactor work.
4. Start with the high-leverage stages:
   - Stage 4 - Earth-Mars regime calibration
   - Stage 5 - Final Earth network generation and QA
   - Stage 6 - Earth pair and label generation
   - Stage 7 - Earth model-input construction

---

## Current open design questions

1. What exactly is the final Earth-Mars regime-selection scheme?
2. Which visual QA outputs are mandatory before model training?
3. What caching strategy should be used for expensive pair, feature, and patch generation?
4. Which model families should be included beyond XGBoost and CNN-based variants?
5. How should Mars operating thresholds be selected and justified?
6. What should be the final scientific interpretation framework?
7. What figures are needed for the poster and final reporting?
