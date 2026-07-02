# HANDOFF — read this first

This page is for the student taking over the project. Everything below is
either actionable on day 1 or points at the one document that is.

## What this is

A framework for detecting **channel-head coupling** in drainage networks,
based on Goren & Shelef (2024, [doi:10.5194/esurf-12-1347-2024](https://doi.org/10.5194/esurf-12-1347-2024)).
It pairs channel heads that meet at confluences, decides whether their basins
are spatially coupled, and predicts coupling with an ML classifier whose
features are dimensionless — so a model **trained on Earth DEMs transfers to
Martian valley networks** without retraining. You are inheriting the
*framework*, not just its results: the pipeline is data-agnostic, and you can
feed it your own per-basin Earth DEM GeoTIFFs and your own Mars DEM + valley
vectors, then extend the science from there.

## Ownership & contacts

| Role | Who | Contact |
|---|---|---|
| Owner / sole committer | Guy Pinkas (GitHub **GP-Geo**) | <guy.pinkas123@gmail.com> |
| Supervisor | Prof. Liran Goren, Dept. of Earth & Environmental Sciences, Ben-Gurion University of the Negev | *TO BE FILLED BY OWNER* |
| Method-paper authors / stakeholders | Liran Goren & Eitan Shelef | via supervisor |

Decision authority (merges into `main`, repo visibility, data publication,
paper authorship): **owner + supervisor — TO BE CONFIRMED.**

## Day-1 orientation

1. **Clone** <https://github.com/GP-Geo/CH_Project>. **Note:** the active
   branch is `refactor/package-first-architecture` — the GitHub default
   branch `main` is stale until the owner fast-forwards it. Check out the
   active branch before doing anything else.
2. **Install** per the [README](README.md) Install section (conda env +
   `pip install -e ".[dev,geo,viz,cnn,ml]"`).
3. **Verify the install** with the README's 3-step check: the
   `CouplingAnalyzer` import one-liner, `channel-heads --help`, and
   `conda run -n ch-heads pytest -q` (608 tests, fully synthetic fixtures —
   no data download needed).
4. **Reading order:** [README.md](README.md) →
   [docs/architecture.md](docs/architecture.md) →
   [notebooks/pipeline/](notebooks/pipeline/) `00`–`14` →
   [STAGE_ASSET_MAP.md](STAGE_ASSET_MAP.md).

## What you get vs what you must obtain

**In the repo:** all code, all docs, the trained models in `models/`
(tracked in git, ~3 MB — provenance and checksums in
[models/MANIFEST.md](models/MANIFEST.md)), and **one example DEM**
(`data/cropped_DEMs/Inyo_strm_crop.tif`) so the README quickstart works on a
fresh clone.

**Not in the repo:** the ~18 GB `data/` tree (raw inputs + derived outputs).
The originals live on the owner's machine; provenance and re-download
pointers are in [docs/DATA_STATUS.md](docs/DATA_STATUS.md). You can also
bring entirely new data — the framework does not depend on the original
basins.

## State of the science at handoff

- **Three frozen complexity regimes** (`regA/B/C`) bracket the Earth→Mars
  comparison; see [docs/regimes_summary.md](docs/regimes_summary.md). The
  spread across regimes *is* the calibration-uncertainty estimate.
- **Mars coupling ≈ 47–60 %** depending on regime (Jun-13-2026 predictions:
  regA 49.6 % / regB 59.7 % / regC 46.6 %).
- **Validation honesty:** true cross-basin leave-one-basin-out (LOBO),
  geometry-only features, gives **pooled AUC ≈ 0.77–0.78** (per-basin mean
  ≈ 0.70–0.74). The older headline **≈ 0.91 is within-basin** held-out test
  AUC — always label which statistic you are quoting. Details and the risk
  register: [docs/ROADMAP_AND_RISKS.md](docs/ROADMAP_AND_RISKS.md).
- **One outstanding validation run:**
  `channel-heads lobo-validate --mode per_fold_cnn` (retrains the CNN per
  fold; command and compute cost are in the ROADMAP). It has never been
  executed — no outputs exist on disk yet.

## Owner checklist before departure

- [ ] Add the student as a GitHub collaborator (requires GP-Geo login).
- [ ] Decide: fast-forward `main` to the active branch, or switch the
      default branch.
- [ ] Confirm repo visibility (currently **PUBLIC**).
- [ ] Fill in Prof. Goren's contact above + the authorship plan (also update
      the commented note in `CITATION.cff`).
- [ ] Get a courtesy OK from Prof. Goren about sharing the `DEMsG&S24`
      source crops.
- [ ] Brain-dump the trim/delta order-reversal experiment into the
      ROADMAP ledger.
- [ ] Confirm the ~7–10× Earth/Mars drainage-density figure recorded in the
      ledger.
- [ ] Copy `data/` + `GuyPinkasLiran/DEMsG&S24/` to institutional storage
      and record the location in [docs/DATA_STATUS.md](docs/DATA_STATUS.md).
