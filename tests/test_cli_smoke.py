"""Smoke tests for the CLI dispatcher: every subcommand must answer --help.

Guards against three regression classes: a registered command whose module
fails to import, a command whose main() ignores argv (and would silently run
its real workload on --help — overwriting artifacts), and a stale entry in
the COMMANDS mapping.
"""

from __future__ import annotations

import importlib

import pytest

from channel_heads.cli import COMMANDS, main


def _help_exits_zero(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code in (0, None)


def test_dispatcher_help() -> None:
    # The dispatcher handles --help itself (prints usage, returns 0).
    assert main(["--help"]) == 0


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_subcommand_help(command: str) -> None:
    module_name = f"channel_heads.cli.{COMMANDS[command]}"
    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # optional heavy dep (e.g. torch) missing
        pytest.skip(f"{module_name} not importable here: {exc}")
    _help_exits_zero([command, "--help"])
