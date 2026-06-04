"""Enable ``python -m channel_heads <command>`` → the CLI dispatcher."""

from __future__ import annotations

from channel_heads.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
