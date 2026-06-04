#!/usr/bin/env python
"""CLI: run the Mars cross-planet pipeline (Phase 1 → 6C).

Thin wrapper — all logic lives in :mod:`channel_heads.pipelines`. Run a single
stage with ``--stage`` or the whole chain with ``--stage all`` (default).

    python scripts/cli/run_mars_pipeline.py --stage topology
    python scripts/cli/run_mars_pipeline.py --stage all
"""

from __future__ import annotations

import argparse

from channel_heads import pipelines

STAGES = {
    "topology": pipelines.build_mars_topology,
    "pairs": pipelines.extract_mars_pairs,
    "features": pipelines.build_mars_features,
    "xgb": pipelines.run_mars_xgb_inference,
    "patches": pipelines.build_mars_cnn_patches,
    "embeddings": pipelines.extract_mars_cnn_embeddings,
    "combined": pipelines.run_mars_combined_inference,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = ap.parse_args(argv)
    if args.stage == "all":
        pipelines.run_full_mars_pipeline()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
