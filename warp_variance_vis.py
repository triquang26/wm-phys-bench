"""
warp_variance_vis.py  —  DEPRECATED shim

This file was formerly a 300-line standalone script.  It is now a thin
backward-compatibility shim that delegates to ``python -m warp_score``.

Migration guide
---------------
Old usage::

    PYTHONPATH=... python warp_variance_vis.py \\
      --query ../image_no_bg/low/.../frame.png \\
      --setting turbo --device cuda

New equivalent::

    python -m warp_score detect \\
      --query path/to/frame.png \\
      --setting turbo --device cuda

Subcommand mapping
------------------
* ``--query``     present  →  ``warp_score detect``
* ``--calibrate`` present  →  ``warp_score calibrate``
* anything else             →  argv[1] is taken as the subcommand directly

To suppress the deprecation warning set the environment variable
``WARP_SCORE_NO_DEPRECATION_WARN=1``.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    # ------------------------------------------------------------------
    # Deprecation warning (one line, to stderr)
    # ------------------------------------------------------------------
    if not os.environ.get("WARP_SCORE_NO_DEPRECATION_WARN"):
        print(
            "DeprecationWarning: warp_variance_vis.py → use `python -m warp_score detect` instead",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Import warp_score; give a helpful message if not installed
    # ------------------------------------------------------------------
    try:
        from warp_score.cli import main as _cli_main
    except ImportError as exc:
        print(
            f"Error: could not import warp_score ({exc}).\n"
            "Install it with:  pip install -e .  (from the repo root)\n"
            "or:               pip install warp-score",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Translate old argv to new subcommand-based argv
    # ------------------------------------------------------------------
    argv = sys.argv[1:]  # strip the script name

    if "--query" in argv:
        # detect subcommand
        sys.argv = [sys.argv[0], "detect"] + argv
    elif "--calibrate" in argv:
        # calibrate subcommand; remove the bare --calibrate flag because
        # the new CLI uses `calibrate` as the positional subcommand
        argv_without_flag = [a for a in argv if a != "--calibrate"]
        sys.argv = [sys.argv[0], "calibrate"] + argv_without_flag
    else:
        # Pass argv as-is; first element should already be the subcommand
        sys.argv = [sys.argv[0]] + argv

    _cli_main()


if __name__ == "__main__":
    main()
