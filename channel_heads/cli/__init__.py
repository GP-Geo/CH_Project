"""Channel-heads command-line interface (canonical entry point).

Single dispatcher over every pipeline command. Each subcommand lives in its own
module under :mod:`channel_heads.cli` and exposes ``main(argv=None) -> int``.

Run::

    channel-heads <command> [args]      # console-script
    python -m channel_heads <command>   # module form

Use ``channel-heads --help`` to list commands, or
``channel-heads <command> --help`` for a command's own options.
"""

from __future__ import annotations

import importlib
import sys

# subcommand (kebab-case)  ->  module under channel_heads.cli
COMMANDS: dict[str, str] = {
    "analyze": "_analyze",
    "build-earth-features": "build_earth_features_regime",
    "build-cnn-patches": "build_cnn_patches_regime",
    "train-cnn-regime": "train_cnn_regime",
    "train-cnn-baseline": "train_cnn_baseline",
    "train-cnn-multiseed": "train_cnn_multiseed",
    "train-combined-xgb-regime": "train_combined_xgb_regime",
    "train-combined-xgb-phase6b": "train_combined_xgb_phase6b",
    "eval-lobo-cv": "eval_lobo_cv",
    "retune-threshold-regime": "retune_threshold_regime",
    "run-mars-pipeline": "run_mars_pipeline",
    "run-mars-combined-regime": "run_mars_combined_regime",
    "make-result-figures": "make_result_figures",
    "generate-poster-figures": "generate_poster_figures",
}


def _usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["channel-heads <command> [args]", "", "commands:"]
    lines += [f"  {c:<{width}}  (channel_heads.cli.{m})" for c, m in COMMANDS.items()]
    lines += ["", "Run 'channel-heads <command> --help' for a command's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv[0]`` to the matching subcommand's ``main(rest)``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 1
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    module = importlib.import_module(f"channel_heads.cli.{COMMANDS[cmd]}")
    rc = module.main(rest)
    return 0 if rc is None else int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
