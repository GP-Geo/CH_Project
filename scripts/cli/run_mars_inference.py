#!/usr/bin/env python
"""CLI: run Mars model inference (combined geom+CNN variants, Phase 6C).

Thin wrapper over :func:`channel_heads.pipelines.run_mars_combined_inference`.
Assumes the upstream stages (topology → pairs → features → patches →
embeddings) have already produced their artifacts.
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.run_mars_combined_inference()


if __name__ == "__main__":
    main()
