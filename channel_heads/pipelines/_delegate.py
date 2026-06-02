"""Transitional bridge for non-Mars stages whose logic still lives in scripts.

Mars inference no longer uses this bridge; its stages are package-resident.
Earth/regime training and poster helpers still delegate here pending separate
extraction slices. These stages are marked ``TRANSITIONAL`` in their docstrings;
see ``docs/architecture.md`` for the extraction backlog.
"""

from __future__ import annotations

import runpy
import sys

from channel_heads.io import paths
from channel_heads.logging_config import get_logger

log = get_logger("pipelines.delegate")


def run_script(name: str, argv: list[str] | None = None) -> None:
    """Execute ``scripts/<name>`` as ``__main__`` with optional ``argv``."""
    script = paths.PROJECT_ROOT / "scripts" / name
    if not script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script}")
    log.info("[transitional] running scripts/%s", name)
    old_argv = sys.argv
    sys.argv = [str(script), *(argv or [])]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


__all__ = ["run_script"]
