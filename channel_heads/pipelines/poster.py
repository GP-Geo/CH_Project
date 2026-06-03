"""Poster / report figure generation.

The canonical, documented path is the presentation notebooks (see
``docs/notebooks.md``); this function is the headless batch entry point that
writes the figure set to :data:`channel_heads.io.paths.POSTER_FIGURES_DIR`.
"""

from __future__ import annotations

from pathlib import Path

import runpy
import sys

from channel_heads.io import paths


def generate_poster_figures(output_dir: Path | None = None) -> Path:
    """Generate the poster/report figure set.

    Runs ``scripts/cli/make_result_figures.py``. Returns the output
    directory (defaults to ``data/results/final_figures``).
    """
    out = Path(output_dir) if output_dir else paths.poster_figures_dir()
    script = paths.PROJECT_ROOT / "scripts" / "cli" / "make_result_figures.py"
    old_argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
    return out


__all__ = ["generate_poster_figures"]
