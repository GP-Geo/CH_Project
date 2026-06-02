"""Mars cross-planet pipeline logic.

Package home for what used to live in the heavy ``scripts/build_mars_*`` and
``scripts/extract_mars_*`` files:

* :mod:`channel_heads.mars.topology` — Phase 1: valley-network → graph topology
  GeoPackage (nodes / segments / outlets / channel heads / confluences).
* :mod:`channel_heads.mars.pairs` — Phase 2B: first-meet channel-head pair
  extraction on the Mars directed graph.

High-level orchestration across stages lives in
:mod:`channel_heads.pipelines`.
"""

from channel_heads.mars import pairs, topology
from channel_heads.mars.pairs import extract_first_meet_pairs
from channel_heads.mars.topology import build_and_write_topology, build_topology

__all__ = [
    "topology",
    "pairs",
    "build_topology",
    "build_and_write_topology",
    "extract_first_meet_pairs",
]
