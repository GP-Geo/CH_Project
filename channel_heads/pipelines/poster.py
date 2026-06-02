"""Poster / report figure generation.

The canonical, documented path is the presentation notebooks (see
``docs/notebooks.md``); this function is the headless batch entry point that
writes the figure set to :data:`channel_heads.io.paths.POSTER_FIGURES_DIR`.
"""

from __future__ import annotations

from pathlib import Path

from channel_heads.io import paths
from channel_heads.pipelines._delegate import run_script


def generate_poster_figures(output_dir: Path | None = None) -> Path:
    """Generate the poster/report figure set.

    TRANSITIONAL — runs ``scripts/make_result_figures.py``. Returns the output
    directory (defaults to ``data/results/final_figures``).
    """
    out = Path(output_dir) if output_dir else paths.poster_figures_dir()
    run_script("make_result_figures.py")
    return out


__all__ = ["generate_poster_figures"]
