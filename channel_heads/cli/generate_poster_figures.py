#!/usr/bin/env python
"""CLI: generate poster / report figures into data/results/final_figures.

Thin wrapper over :func:`channel_heads.pipelines.generate_poster_figures`. The
documented, canonical path is the presentation notebooks (see
``docs/notebooks.md``); this is the headless batch entry point.
"""

from __future__ import annotations

from channel_heads import pipelines


def main(argv=None) -> None:
    out = pipelines.generate_poster_figures()
    print(f"Figures written under: {out}")


if __name__ == "__main__":
    main()
