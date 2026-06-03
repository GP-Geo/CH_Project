#!/usr/bin/env python
"""Phase 1 — Mars valley-network topology build (thin wrapper).

The real logic now lives in :mod:`channel_heads.mars.topology`. Prefer
``python scripts/cli/run_mars_pipeline.py --stage topology`` or calling
``channel_heads.pipelines.build_mars_topology()`` directly.

Outputs:
  data/Mars/topology/mars_vn_topology_model_ready.gpkg
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.build_mars_topology()


if __name__ == "__main__":
    main()
