#!/usr/bin/env python
"""Phase 2B — Mars first-meet pair extraction (thin wrapper).

The real logic now lives in :mod:`channel_heads.mars.pairs`. Prefer
``python scripts/cli/run_mars_pipeline.py --stage pairs`` or calling
``channel_heads.pipelines.extract_mars_pairs()`` directly.

Outputs:
  data/Mars/topology/mars_vn_pairs.gpkg
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.extract_mars_pairs()


if __name__ == "__main__":
    main()
