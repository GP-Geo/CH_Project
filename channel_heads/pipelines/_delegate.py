"""Transitional bridge for pipeline stages whose logic still lives in scripts.

The Phase-1/2B Mars stages (topology, pairs) are fully migrated into
:mod:`channel_heads.mars`. The remaining heavy stages (feature tables, CNN
patches, embeddings, combined inference) still hold their logic in ``scripts/``
and are scheduled for extraction. Until then, the public pipeline functions run
the corresponding script **in-process** so the API is complete and runnable.

These are marked ``TRANSITIONAL`` in their docstrings; see
``docs/architecture.md`` for the extraction backlog.
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
