"""Safe cleanup of *generated* data — manifest + dry-run.

Encodes the classification in ``docs/data_management.md`` / ``docs/DATA_STATUS.md``
so a clean rebuild can reclaim disk space without ever touching source inputs or
trained models.

Hard safety rules:

* ``RAW_KEEP`` paths (DEMs, valley vectors, MOLA) are **never** listed.
* ``models/`` is **never** listed (trained artifacts are deleted only by hand).
* Nothing is removed unless ``clean(..., dry_run=False)`` is called explicitly.

Typical use::

    from channel_heads.io import cleanup
    print(cleanup.format_manifest(cleanup.scan()))      # dry-run report
    cleanup.clean(tags={"STALE_AFTER_RASTER_FIX"}, dry_run=False)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from channel_heads.io import paths

# (relative-glob-under-DATA_DIR, tag, regeneration hint)
_RULES: list[tuple[str, str, str]] = [
    ("results/_rasters_regA", "STALE_AFTER_RASTER_FIX", "channel-heads run-mars-pipeline (regA patches)"),
    ("results/_rasters_regB", "STALE_AFTER_RASTER_FIX", "channel-heads run-mars-pipeline (regB patches)"),
    ("results/_rasters_regC", "STALE_AFTER_RASTER_FIX", "channel-heads run-mars-pipeline (regC patches)"),
    ("results/*/rasters", "STALE_AFTER_RASTER_FIX", "Earth rasterization stage"),
    ("Mars/model_inputs/cnn_patches_5class", "STALE_AFTER_RASTER_FIX", "pipelines.build_mars_cnn_patches"),
    ("Mars/model_outputs/figures_combined", "REPORT", "presentation notebooks"),
    ("results/figures_models", "REPORT", "presentation notebooks"),
    ("outputs", "LEGACY", "duplicate of data/results — archive"),
]


@dataclass
class CleanupItem:
    path: Path
    tag: str
    regen: str
    size_bytes: int = field(default=0)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def scan() -> list[CleanupItem]:
    """Return existing generated artifacts eligible for cleanup."""
    items: list[CleanupItem] = []
    for rel, tag, regen in _RULES:
        for match in sorted(paths.DATA_DIR.glob(rel)):
            if match.exists():
                items.append(CleanupItem(match, tag, regen, _dir_size(match)))
    return items


def format_manifest(items: list[CleanupItem]) -> str:
    """Human-readable manifest table."""
    if not items:
        return "No generated artifacts found to clean."
    lines = [
        f"{'TAG':<26} {'SIZE(MB)':>10}  PATH",
        "-" * 80,
    ]
    total = 0
    for it in sorted(items, key=lambda x: (-x.size_bytes)):
        rel = it.path.relative_to(paths.DATA_DIR.parent)
        lines.append(f"{it.tag:<26} {it.size_mb:>10.1f}  {rel}")
        total += it.size_bytes
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':<26} {total / 1e6:>10.1f}  MB across {len(items)} item(s)")
    lines.append("")
    lines.append("Regeneration hints:")
    for it in items:
        lines.append(f"  - {it.path.name}: {it.regen}")
    return "\n".join(lines)


def clean(
    *,
    tags: set[str] | None = None,
    dry_run: bool = True,
    archive_to: Path | None = None,
) -> list[Path]:
    """Delete (or archive) generated artifacts.

    Parameters
    ----------
    tags : set[str], optional
        Only act on items with these tags. Default: all eligible items.
    dry_run : bool
        If True (default) nothing is removed; the would-be targets are returned.
    archive_to : Path, optional
        If given, items are *moved* here instead of deleted.

    Returns the list of paths acted on (or that would be).
    """
    items = scan()
    if tags is not None:
        items = [it for it in items if it.tag in tags]
    acted: list[Path] = []
    for it in items:
        acted.append(it.path)
        if dry_run:
            continue
        if archive_to is not None:
            dest = Path(archive_to) / it.path.relative_to(paths.DATA_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(it.path), str(dest))
        elif it.path.is_dir():
            shutil.rmtree(it.path)
        else:
            it.path.unlink()
    return acted


__all__ = ["CleanupItem", "scan", "format_manifest", "clean"]
